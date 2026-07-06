"""
Claim & obligation construction (AVG graph-construction + obligation-generation).

This module realises the deterministic parts of AVG stages 3-4 over an already
compiled HTIR graph:

  * lift observable effects into first-class ``EvidenceNode`` objects;
  * induce ``ClaimNode`` objects from step outcomes, artifact provenance, and
    final answers;
  * generate ``Obligation`` objects from the domain spec's templates
    (universal / domain / trajectory-triggered) plus trajectory triggers;
  * wire the support (E_sup), constraint (E_cons), and validation (E_val) edges.

Checking (running mechanical/semantic checkers to fill ``Obligation.result``)
is intentionally *not* done here — obligations are emitted with
``checker`` routed to a class and ``status = PENDING`` so the checking stage
can be added on top without reshaping the graph.
"""

from __future__ import annotations

from harnessfix.models.domain import Constraint, DomainSpec
from harnessfix.models.htir import (
    HTIR,
    ArtifactEffect,
    CheckerType,
    ClaimNode,
    ClaimStatus,
    ConstraintLink,
    EvidenceNode,
    EvidenceType,
    ExecutionStatus,
    Obligation,
    ObligationScope,
    SupportLink,
    SupportPolarity,
    TraceStep,
    ValidationKind,
    ValidationLink,
)
from harnessfix.utils.io import truncate

# Operation-type names treated as validations / edits / final answers.
# These are heuristics over the domain vocabulary; specialised domains can use
# different names and still match via substring (e.g. 'run_test', 'edit_file').
_VALIDATION_HINTS = ("validation", "test")
_EDIT_HINTS = ("edit", "artifact_editing", "write")
_FINAL_HINTS = ("final_submission", "final_answer")


def _is_role(role: str, hints: tuple[str, ...]) -> bool:
    r = role.lower()
    return any(h in r for h in hints)


def _checker_for_evidence(ev: EvidenceType) -> CheckerType:
    """Route an obligation to a checker class based on its required evidence."""
    if ev in (EvidenceType.EXECUTABLE, EvidenceType.SCHEMA, EvidenceType.ARTIFACT):
        return CheckerType.MECHANICAL
    if ev in (EvidenceType.SEMANTIC, EvidenceType.POLICY):
        return CheckerType.SEMANTIC
    return CheckerType.UNASSIGNED


def _evidence_type_for_effect(effect: ArtifactEffect) -> EvidenceType:
    if effect in (ArtifactEffect.ARTIFACT_CHANGE, ArtifactEffect.MIXED):
        return EvidenceType.ARTIFACT
    if effect == ArtifactEffect.STATE_CHANGE:
        return EvidenceType.LOG
    return EvidenceType.NONE


def build_claims_and_obligations(htir: HTIR, spec: DomainSpec) -> HTIR:
    """
    Populate ``htir`` in place with evidence, claim, and obligation nodes and
    the support/constraint/validation edges implied by its steps and artifacts,
    using the templates in ``spec``. Returns the same graph for chaining.
    """
    _ev_id = _Counter()
    _claim_id = _Counter()
    _ob_id = _Counter()

    artifact_by_ident = {a.identifier: a for a in htir.artifacts}

    # Evidence attached to each step (for candidate-evidence wiring).
    evidence_by_step: dict[int, list[int]] = {}

    # ------------------------------------------------------------------
    # 1. Evidence nodes from observable effects.
    # ------------------------------------------------------------------
    for step in htir.steps_in_order():
        for eff in step.artifact_state_effects:
            artifact = artifact_by_ident.get(eff.affected_resource)
            ev = EvidenceNode(
                evidence_id=_ev_id.next(),
                evidence_type=_evidence_type_for_effect(eff.effect_category),
                description=eff.observed_change or eff.affected_resource,
                content=truncate(eff.supporting_evidence, 500),
                artifact_ids=[artifact.artifact_id] if artifact else [],
                step_ids=[step.step_id],
            )
            htir.evidence.append(ev)
            evidence_by_step.setdefault(step.step_id, []).append(ev.evidence_id)

        # Executable evidence for validation/tool steps with a known status.
        if step.execution_status != ExecutionStatus.UNKNOWN and (
            _is_role(step.role, _VALIDATION_HINTS) or step.role == "tool_invocation"
        ):
            ev = EvidenceNode(
                evidence_id=_ev_id.next(),
                evidence_type=EvidenceType.EXECUTABLE,
                description=f"step {step.step_id} reported status={step.execution_status.value}",
                content=truncate(step.response_message, 500),
                step_ids=[step.step_id],
            )
            htir.evidence.append(ev)
            evidence_by_step.setdefault(step.step_id, []).append(ev.evidence_id)

    # ------------------------------------------------------------------
    # 2. Claim nodes.
    # ------------------------------------------------------------------
    claim_by_step: dict[int, list[int]] = {}

    def _add_claim(statement: str, claim_type: str, step_id: int | None,
                   artifact_ids: list[int] | None = None,
                   status: ClaimStatus = ClaimStatus.UNVERIFIED) -> ClaimNode:
        claim = ClaimNode(
            claim_id=_claim_id.next(),
            statement=statement,
            claim_type=claim_type,
            source_step_id=step_id,
            artifact_ids=artifact_ids or [],
            status=status,
        )
        htir.claims.append(claim)
        if step_id is not None:
            claim_by_step.setdefault(step_id, []).append(claim.claim_id)
        return claim

    # Execution-outcome and provenance claims.
    for step in htir.steps_in_order():
        if step.execution_status in (ExecutionStatus.SUCCESS, ExecutionStatus.FAILURE):
            _add_claim(
                f"Step {step.step_id} ({step.role}) completed with status "
                f"'{step.execution_status.value}'.",
                claim_type="execution_status",
                step_id=step.step_id,
            )
        for art_id in step.produced_artifact_ids:
            art = htir.get_artifact(art_id)
            ident = art.identifier if art else str(art_id)
            _add_claim(
                f"Artifact '{ident}' was produced/modified by step {step.step_id}.",
                claim_type="artifact_provenance",
                step_id=step.step_id,
                artifact_ids=[art_id],
            )
        if _is_role(step.role, _FINAL_HINTS):
            _add_claim(
                f"Final answer at step {step.step_id} is supported by trajectory evidence.",
                claim_type="final_answer_support",
                step_id=step.step_id,
                status=ClaimStatus.UNRESOLVED,
            )

    # ------------------------------------------------------------------
    # 3. Support edges (E_sup): step-local evidence supports step-local claims.
    # ------------------------------------------------------------------
    for step_id, claim_ids in claim_by_step.items():
        ev_ids = evidence_by_step.get(step_id, [])
        for claim_id in claim_ids:
            for ev_id in ev_ids:
                htir.support_links.append(
                    SupportLink(
                        evidence_id=ev_id,
                        claim_id=claim_id,
                        polarity=SupportPolarity.SUPPORTS,
                    )
                )

    # ------------------------------------------------------------------
    # 4. Validation edges (E_val): each validation op validates the most recent
    #    prior artifact-producing step.
    # ------------------------------------------------------------------
    ordered = htir.steps_in_order()
    for i, step in enumerate(ordered):
        if not _is_role(step.role, _VALIDATION_HINTS):
            continue
        target = _last_producer_before(ordered, i)
        kind = ValidationKind.FULL if "test" in step.role.lower() else ValidationKind.TARGETED
        htir.validation_links.append(
            ValidationLink(
                source_id=step.step_id,
                target_step_id=target.step_id if target else None,
                target_artifact_id=(target.produced_artifact_ids[0]
                                    if target and target.produced_artifact_ids else None),
                validation_kind=kind,
                outcome=step.execution_status,
            )
        )

    # ------------------------------------------------------------------
    # 5. Constraint edges (E_cons): tie domain constraints to governed steps.
    # ------------------------------------------------------------------
    for constraint in spec.constraints:
        _link_constraint(htir, constraint)

    # ------------------------------------------------------------------
    # 6. Obligations from templates + trajectory triggers.
    # ------------------------------------------------------------------
    def _emit(template, claim_id: int, step_id: int, scope: ObligationScope) -> None:
        htir.obligations.append(
            Obligation(
                obligation_id=_ob_id.next(),
                claim_id=claim_id,
                required_evidence=template.required_evidence,
                candidate_evidence_ids=evidence_by_step.get(step_id, []),
                checker=_checker_for_evidence(template.required_evidence),
                severity=template.severity,
                escalation=template.escalation,
                scope=scope,
                template_id=template.template_id,
                description=template.claim_template,
            )
        )

    for template in spec.obligation_templates:
        for step in htir.steps_in_order():
            if not _template_triggers(template, step):
                continue
            # Anchor the obligation on a relevant claim from this step, or a
            # synthetic claim if the step produced none.
            claim_ids = claim_by_step.get(step.step_id)
            if claim_ids:
                target_claim = claim_ids[-1]
            else:
                target_claim = _add_claim(
                    template.claim_template, claim_type=template.template_id,
                    step_id=step.step_id,
                ).claim_id
            _emit(template, target_claim, step.step_id, template.scope)

    return htir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Counter:
    def __init__(self, start: int = 1):
        self._n = start

    def next(self) -> int:
        v = self._n
        self._n += 1
        return v


def _last_producer_before(ordered: list[TraceStep], idx: int) -> TraceStep | None:
    for j in range(idx - 1, -1, -1):
        if ordered[j].produced_artifact_ids:
            return ordered[j]
    return None


def _template_triggers(template, step: TraceStep) -> bool:
    trig = template.trigger
    if not trig:
        return True
    if trig == "artifact_edit":
        return bool(step.produced_artifact_ids) or _is_role(step.role, _EDIT_HINTS)
    if trig == "failed_step":
        return step.execution_status in (
            ExecutionStatus.FAILURE, ExecutionStatus.TIMEOUT, ExecutionStatus.BLOCKED
        )
    # Otherwise the trigger names an operation type (exact or substring match).
    return step.role == trig or trig in step.role.lower()


def _link_constraint(htir: HTIR, constraint: Constraint) -> None:
    applies = constraint.applies_to_operations
    for step in htir.steps:
        if applies and step.role not in applies:
            continue
        htir.constraint_links.append(
            ConstraintLink(
                constraint_id=constraint.constraint_id,
                step_id=step.step_id,
                satisfied=None,  # resolved by the (future) checking stage
                note=constraint.description,
            )
        )
