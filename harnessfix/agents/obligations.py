"""
Claim & obligation construction (AVG graph-construction + obligation-generation).

This module realises the deterministic parts of AVG stage 4 (obligation
generation) over an already compiled *and enriched* HTIR graph:

  * lift observable effects into first-class ``EvidenceNode`` objects;
  * induce ``ClaimNode`` objects from step outcomes, artifact provenance, and
    final answers;
  * wire the support edge set (E_sup);
  * generate ``Obligation`` objects from the domain spec's templates
    (universal / domain / trajectory-triggered) plus trajectory triggers;
  * seed additional ``Obligation`` objects from unresolved well-formedness /
    analysis-module issues (``htir.wellformedness``).

Graph *enrichment* -- the E_val (validation), E_cons (constraint), and
E_causal (dependency) edge sets, well-formedness checks, and the six AVG
analysis modules (avg.tex Sec. 3.4-3.5) -- is owned by
``harnessfix.agents.analysis`` and is expected to have already run over
``htir`` by the time this function is called (see
``TraceAbstractionAgent.compile``). This function only *consumes* that
enrichment.

Checking (running mechanical/semantic checkers to fill ``Obligation.result``)
is intentionally *not* done here — obligations are emitted with
``checker`` routed to a class and ``status = PENDING`` so the checking stage
can be added on top without reshaping the graph.

NOTE on id spaces: evidence, claim, and obligation ids are each assigned by
their own ``_Counter`` starting at 1, so they overlap with each other and
with step/artifact ids. This is safe today because every cross-reference is
typed by field name (e.g. ``Obligation.claim_id`` vs. ``candidate_evidence_ids``),
but do not mix ids from different node kinds in a shared untyped container
(cf. ``WellFormednessIssue.offending_node_ids``, which already mixes
step/artifact ids by necessity).
"""

from __future__ import annotations

from harnessfix.models.domain import DomainSpec, ObligationTemplate
from harnessfix.models.htir import (
    HTIR,
    ArtifactEffect,
    CheckerType,
    ClaimNode,
    ClaimStatus,
    EscalationRule,
    EvidenceNode,
    EvidenceType,
    ExecutionStatus,
    Obligation,
    ObligationScope,
    Severity,
    SupportLink,
    SupportPolarity,
    TraceStep,
    WellFormednessIssue,
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


def _escalation_for_issue(issue: WellFormednessIssue) -> EscalationRule:
    """
    Map a well-formedness/analysis-module issue to an escalation rule (alpha_i)
    for its seeded obligation. Integrity findings (tampering: modifying tests
    directly, deleting required artifacts) and CRITICAL-severity issues veto
    rather than merely requesting more evidence; other HIGH-severity issues
    escalate for review; everything else requests evidence.
    """
    if issue.rule_id.startswith("integrity_"):
        return EscalationRule.VETO
    if issue.severity == Severity.CRITICAL:
        return EscalationRule.VETO
    if issue.severity == Severity.HIGH:
        return EscalationRule.ESCALATE
    return EscalationRule.REQUEST_EVIDENCE


def _checker_for_evidence(ev: EvidenceType) -> CheckerType:
    """
    Route an obligation to a checker class based on its required evidence
    (avg.tex Sec. 3.8). LOG is observability evidence a mechanical checker can
    read directly (e.g. exit codes / log excerpts); MANUAL evidence needs a
    human, so it abstains rather than staying unrouted. Only genuinely no
    evidence (NONE) stays UNASSIGNED.
    """
    if ev in (EvidenceType.EXECUTABLE, EvidenceType.SCHEMA, EvidenceType.ARTIFACT, EvidenceType.LOG):
        return CheckerType.MECHANICAL
    if ev in (EvidenceType.SEMANTIC, EvidenceType.POLICY):
        return CheckerType.SEMANTIC
    if ev == EvidenceType.MANUAL:
        return CheckerType.ABSTENTION
    return CheckerType.UNASSIGNED


def _evidence_type_for_effect(effect: ArtifactEffect) -> EvidenceType:
    if effect in (ArtifactEffect.ARTIFACT_CHANGE, ArtifactEffect.MIXED):
        return EvidenceType.ARTIFACT
    if effect == ArtifactEffect.STATE_CHANGE:
        return EvidenceType.LOG
    return EvidenceType.NONE


def _evidence_supports(
    htir: HTIR, claim: ClaimNode, ev: EvidenceNode
) -> SupportPolarity | None:
    """
    Decide whether ``ev`` is related to ``claim`` at all and, if so, whether it
    supports or refutes it (E_sup, avg.tex Sec. 3.6/3.9). Returns ``None`` when
    the pair is unrelated (different step/artifact and no type match), so
    unrelated step-local evidence is not wired to a claim just because they
    happen to share a step.
    """
    shares_step = claim.source_step_id is not None and claim.source_step_id in ev.step_ids
    shares_artifact = bool(set(claim.artifact_ids) & set(ev.artifact_ids))
    if not (shares_step or shares_artifact):
        return None

    if claim.claim_type == "execution_status":
        # Only executable/log evidence about *this* step speaks to whether the
        # step's execution outcome claim holds; other step-local evidence
        # (e.g. an unrelated artifact effect) does not.
        if ev.evidence_type not in (EvidenceType.EXECUTABLE, EvidenceType.LOG) or not shares_step:
            return None
        step = htir.get_step(claim.source_step_id) if claim.source_step_id is not None else None
        if step is None:
            return None
        if step.execution_status == ExecutionStatus.SUCCESS:
            return SupportPolarity.SUPPORTS
        if step.execution_status in (
            ExecutionStatus.FAILURE, ExecutionStatus.TIMEOUT, ExecutionStatus.BLOCKED,
        ):
            return SupportPolarity.REFUTES
        return None

    if claim.claim_type == "artifact_provenance":
        # Provenance claims are about the artifact itself; only evidence that
        # names the same artifact (or is emitted at the producing step)
        # supports it.
        if not (shares_artifact or (shares_step and ev.evidence_type == EvidenceType.ARTIFACT)):
            return None
        return SupportPolarity.SUPPORTS

    if claim.claim_type == "final_answer_support":
        return SupportPolarity.SUPPORTS if (shares_step or shares_artifact) else None

    # Synthetic / template-seeded claim types: fall back to step/artifact
    # co-location, since there is no richer semantic to match on.
    return SupportPolarity.SUPPORTS


def build_claims_and_obligations(htir: HTIR, spec: DomainSpec) -> HTIR:
    """
    Populate ``htir`` in place with evidence, claim, and obligation nodes and
    the support/constraint/validation edges implied by its steps and artifacts,
    using the templates in ``spec``. Returns the same graph for chaining.

    Idempotent: everything this function produces (``claims``, ``evidence``,
    ``obligations``, ``support_links``) is deterministically re-derivable from
    ``htir.steps`` / ``htir.artifacts`` / ``htir.wellformedness`` and ``spec``,
    so a second call clears and rebuilds those lists rather than appending to
    them (a naive re-run would otherwise double every node).
    """
    htir.claims.clear()
    htir.evidence.clear()
    htir.obligations.clear()
    htir.support_links.clear()

    _ev_id = _Counter()
    _claim_id = _Counter()
    _ob_id = _Counter()

    artifact_by_ident = {a.identifier: a for a in htir.artifacts}

    # Evidence attached to each step (for candidate-evidence wiring).
    evidence_by_step: dict[int, list[int]] = {}
    evidence_by_id: dict[int, EvidenceNode] = {}

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
            evidence_by_id[ev.evidence_id] = ev
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
            evidence_by_id[ev.evidence_id] = ev
            evidence_by_step.setdefault(step.step_id, []).append(ev.evidence_id)

    # ------------------------------------------------------------------
    # 2. Claim nodes.
    # ------------------------------------------------------------------
    claim_by_step: dict[int, list[int]] = {}
    claims_by_id: dict[int, ClaimNode] = {}

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
        claims_by_id[claim.claim_id] = claim
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
    # 3. Support edges (E_sup): step-local evidence supports/refutes
    #    step-local claims of the *same* claim, not the full step-local
    #    cross-product -- see ``_evidence_supports`` for the match + polarity
    #    rules.
    # ------------------------------------------------------------------
    for step_id, claim_ids in claim_by_step.items():
        ev_ids = evidence_by_step.get(step_id, [])
        for claim_id in claim_ids:
            claim = claims_by_id[claim_id]
            for ev_id in ev_ids:
                polarity = _evidence_supports(htir, claim, evidence_by_id[ev_id])
                if polarity is None:
                    continue
                htir.support_links.append(
                    SupportLink(evidence_id=ev_id, claim_id=claim_id, polarity=polarity)
                )

    # ------------------------------------------------------------------
    # 4. Obligations from templates + trajectory triggers.
    #
    # NOTE: E_val (validation), E_cons (constraint), and the E_causal
    # first-cut dependency links are no longer wired here. They are graph
    # *enrichment*, owned by the Step-3 analysis layer
    # (harnessfix.agents.analysis.enrich), which the pipeline runs before
    # this function so ``htir.validation_links`` / ``constraint_links`` /
    # ``dependency_links`` are already populated by the time obligations
    # are generated.
    # ------------------------------------------------------------------
    def _candidate_evidence(template: ObligationTemplate, step_id: int) -> list[int]:
        """E_i: candidate evidence at this step *of the required type* r_i."""
        return [
            ev_id for ev_id in evidence_by_step.get(step_id, [])
            if evidence_by_id[ev_id].evidence_type == template.required_evidence
        ]

    def _emit(template: ObligationTemplate, claim_id: int, step_id: int, scope: ObligationScope) -> None:
        htir.obligations.append(
            Obligation(
                obligation_id=_ob_id.next(),
                claim_id=claim_id,
                required_evidence=template.required_evidence,
                candidate_evidence_ids=_candidate_evidence(template, step_id),
                checker=_checker_for_evidence(template.required_evidence),
                severity=template.severity,
                escalation=template.escalation,
                scope=scope,
                template_id=template.template_id,
                description=template.claim_template,
            )
        )

    def _target_claim(template: ObligationTemplate, step: TraceStep) -> int:
        """
        Anchor on the claim at this step whose ``claim_type`` matches the
        template's ``target_claim_type``, rather than an arbitrary
        (positional) claim from the step. Synthesizes a claim typed after the
        template when no existing claim matches (or the template declares no
        target type).
        """
        wanted = template.target_claim_type
        if wanted:
            for claim_id in claim_by_step.get(step.step_id, []):
                if claims_by_id[claim_id].claim_type == wanted:
                    return claim_id
        return _add_claim(
            template.claim_template,
            claim_type=wanted or template.template_id,
            step_id=step.step_id,
        ).claim_id

    _emitted_pairs: set[tuple[int, str]] = set()
    for template in spec.obligation_templates:
        for step in htir.steps_in_order():
            if not _template_triggers(template, step):
                continue
            target_claim = _target_claim(template, step)
            key = (target_claim, template.template_id)
            if key in _emitted_pairs:
                continue
            _emitted_pairs.add(key)
            _emit(template, target_claim, step.step_id, template.scope)

    # ------------------------------------------------------------------
    # 5. Unresolved obligations seeded by well-formedness / analysis-module
    #    issues (avg.tex Sec. 3.4): a failure there does not mean the task
    #    failed, it means evidence is missing, so it becomes an UNRESOLVED
    #    claim routed to the ABSTENTION checker rather than a task failure.
    # ------------------------------------------------------------------
    for issue in htir.wellformedness:
        seed_claim = _add_claim(
            issue.message or f"Well-formedness rule '{issue.rule_id}' is unresolved.",
            claim_type=f"wellformedness:{issue.rule_id}",
            step_id=None,
            status=ClaimStatus.UNRESOLVED,
        )
        htir.obligations.append(
            Obligation(
                obligation_id=_ob_id.next(),
                claim_id=seed_claim.claim_id,
                required_evidence=EvidenceType.NONE,
                candidate_evidence_ids=[],
                checker=CheckerType.ABSTENTION,
                severity=issue.severity,
                escalation=_escalation_for_issue(issue),
                scope=ObligationScope.TRAJECTORY_TRIGGERED,
                template_id=f"wellformedness:{issue.rule_id}",
                description=issue.message,
            )
        )

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


def _template_triggers(template: ObligationTemplate, step: TraceStep) -> bool:
    """
    Whether ``template.trigger`` fires for ``step``. Two event names are
    reserved and always checked as events, never as operation-type names:
    ``artifact_edit`` and ``failed_step`` (plus ``""`` = always). Any other
    trigger must name an operation type *exactly* -- no substring matching --
    so e.g. a template with ``trigger: decision`` does not spuriously fire on
    the ``orchestration_decision`` operation type. ``step.role`` is already
    validated against the domain vocabulary (``spec.operation_type_names()``)
    at trace-abstraction time, so an exact string match here is sufficient
    and correctly case-sensitive.
    """
    trig = template.trigger
    if not trig:
        return True
    if trig == "artifact_edit":
        return bool(step.produced_artifact_ids) or _is_role(step.role, _EDIT_HINTS)
    if trig == "failed_step":
        return step.execution_status in (
            ExecutionStatus.FAILURE, ExecutionStatus.TIMEOUT, ExecutionStatus.BLOCKED
        )
    return step.role == trig


# _link_constraint (E_cons) and _link_dependencies (E_causal first cut) have
# moved to harnessfix.agents.analysis (Step-3 graph-enrichment layer); see
# ``link_constraints`` / ``link_dependencies`` there.

