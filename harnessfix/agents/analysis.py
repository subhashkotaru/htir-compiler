"""
Analysis modules  (avg.tex Sec. 3.4 "Well-Formedness Checks" and Sec. 3.5
"Analysis Modules").

This module is the Step-3 graph-*enrichment* layer: it sits between graph
construction (``TraceAbstractionAgent._extract_artifacts`` / temporal /
reuse / control-flow) and obligation generation
(``harnessfix.agents.obligations.build_claims_and_obligations``). Per the
pipeline, it now *owns*:

  * well-formedness checks (domain-independent structural validation);
  * provenance analysis (final-answer -> source-artifact linking, on top of
    the E_prov edges already built in ``_extract_artifacts``);
  * dependency analysis (E_causal), including the deterministic first cut
    formerly in ``obligations._link_dependencies`` plus dependencies with no
    explicit artifact link;
  * validation-edge wiring (E_val), formerly in
    ``obligations.build_claims_and_obligations``;
  * state-transition analysis;
  * policy-linking analysis, including the constraint-edge wiring (E_cons)
    formerly in ``obligations._link_constraint``;
  * integrity analysis.

Coverage analysis is the exception: it reports over ``HTIR.obligations``, so
it cannot run until *after* ``build_claims_and_obligations``. Call
``compute_coverage`` explicitly once obligations exist (see
``TraceAbstractionAgent.compile``).

All mechanical (deterministic) passes always run. Passes that need model
judgement are gated behind ``use_semantic`` (default off), mirroring
``TraceAbstractionAgent.compile(attach_harness_layers=...)``: mechanical
results must stay reproducible byte-for-byte so the checked-in
``data/htir_outputs/*.json`` fixtures don't drift.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from harnessfix.models.domain import DomainSpec
from harnessfix.models.htir import (
    ArtifactEffect,
    ConstraintLink,
    CoverageReport,
    DependencyLink,
    ExecutionStatus,
    HTIR,
    Severity,
    StateTransitionPattern,
    TraceStep,
    ValidationKind,
    ValidationLink,
    WellFormednessIssue,
)
from harnessfix.agents.obligations import _EDIT_HINTS, _FINAL_HINTS, _VALIDATION_HINTS, _is_role
from harnessfix.utils.io import truncate
from harnessfix.utils.llm import DEFAULT_MODEL, chat_json, system, user

# Artifact type used to recognise policy documents/rules among an HTIR's
# artifacts. Domains are free to declare an artifact type named "policy" in
# their R_d (see harnessfix/domains/*.yaml); if none of the compiled
# artifacts use it, policy-linking degrades gracefully (nothing to link, but
# unresolved obligations are still emitted for policy-sensitive steps).
_POLICY_ARTIFACT_TYPE = "policy"

# Substrings taken as evidence of tampering/deletion in an observed-change
# description (Integrity analysis).
_DELETION_HINTS = ("delete", "deleted", "deletion", "remove", "removed", "rm ")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def enrich(htir: HTIR, spec: DomainSpec, *, use_semantic: bool = False) -> HTIR:
    """
    Run the Step-3 analysis layer over an already graph-constructed
    ``htir`` (``_extract_artifacts`` + temporal links must already be
    populated). Mutates ``htir`` in place and returns it for chaining.

    Order:
      1. Well-formedness checks (domain-independent, self-contained over
         steps/artifacts/provenance -- does not depend on any of the passes
         below).
      2. Provenance analysis (final-answer -> source-artifact linking).
      3. Dependency analysis (E_causal).
      4. Validation-edge wiring (E_val) -- needed by state-transition.
      5. State-transition analysis.
      6. Policy-linking analysis (+ E_cons constraint wiring).
      7. Integrity analysis.

    Coverage analysis is intentionally excluded; call ``compute_coverage``
    after ``build_claims_and_obligations``.
    """
    check_wellformedness(htir, spec)
    link_provenance_to_final_answer(htir, use_semantic=use_semantic)
    link_dependencies(htir)
    link_validations(htir)
    analyze_state_transitions(htir)
    link_policy(htir, spec, use_semantic=use_semantic)
    check_integrity(htir)
    return htir


# ---------------------------------------------------------------------------
# 3.0 Well-formedness checks (avg.tex Sec. 3.4)
# ---------------------------------------------------------------------------

def check_wellformedness(htir: HTIR, spec: DomainSpec) -> list[WellFormednessIssue]:
    """
    Domain-independent structural validation. A failure does not mean the
    task failed -- it means the trace is missing evidence needed for
    confident verification, so each failure is appended to
    ``htir.wellformedness`` and later becomes an UNRESOLVED claim +
    ABSTENTION obligation (see ``obligations.build_claims_and_obligations``),
    never a task failure.

    Fully self-contained over ``htir.steps`` / ``htir.artifacts`` /
    ``htir.provenance_links`` and ``spec`` so it can run first, before any
    other analysis pass.
    """
    ordered = htir.steps_in_order()
    issues: list[WellFormednessIssue] = []

    # (a) every tool output is linked to a tool invocation.
    for artifact in htir.artifacts:
        if artifact.artifact_type == "tool_result" and artifact.produced_by_step_id is None:
            issues.append(
                WellFormednessIssue(
                    rule_id="tool_output_unlinked",
                    severity=Severity.MEDIUM,
                    offending_node_ids=[artifact.artifact_id],
                    message=f"Tool-result artifact '{artifact.identifier}' has no producing tool invocation.",
                )
            )

    # (b) every artifact mutation has producer provenance + before/after
    #     when available.
    effects_by_step_and_resource: dict[tuple[int, str], str] = {
        (s.step_id, eff.affected_resource): eff.observed_change
        for s in ordered
        for eff in s.artifact_state_effects
    }
    for link in htir.provenance_links:
        if link.relation.value not in ("created", "modified"):
            continue
        artifact = htir.get_artifact(link.artifact_id)
        if artifact is None or artifact.produced_by_step_id is None:
            issues.append(
                WellFormednessIssue(
                    rule_id="artifact_mutation_missing_provenance",
                    severity=Severity.HIGH,
                    offending_node_ids=[link.artifact_id, link.step_id],
                    message=f"Artifact {link.artifact_id} mutated at step {link.step_id} has no producer provenance.",
                )
            )
            continue
        observed = effects_by_step_and_resource.get((link.step_id, artifact.identifier), "")
        if not observed.strip():
            issues.append(
                WellFormednessIssue(
                    rule_id="artifact_mutation_missing_before_after",
                    severity=Severity.LOW,
                    offending_node_ids=[link.artifact_id, link.step_id],
                    message=(
                        f"Artifact '{artifact.identifier}' mutated at step {link.step_id} "
                        "has no recorded before/after change description."
                    ),
                )
            )

    # (c) every final-success claim depends on >=1 evidence/validation node.
    for step in ordered:
        if not (_is_role(step.role, _FINAL_HINTS) and step.execution_status == ExecutionStatus.SUCCESS):
            continue
        has_prior_validation = any(
            s.step_id < step.step_id and _is_role(s.role, _VALIDATION_HINTS)
            and s.execution_status == ExecutionStatus.SUCCESS
            for s in ordered
        )
        has_evidence = bool(step.artifact_state_effects) or bool(step.consumed_artifact_ids)
        if not (has_prior_validation or has_evidence):
            issues.append(
                WellFormednessIssue(
                    rule_id="final_success_without_evidence",
                    severity=Severity.HIGH,
                    offending_node_ids=[step.step_id],
                    message=f"Final-success step {step.step_id} depends on no evidence or validation node.",
                )
            )

    # (d) every validation claim declares full/targeted/manual. A bare role
    #     of exactly "validation" (no more specific keyword) is too generic
    #     to classify.
    _kind_hints = ("test", "target", "manual", "full")
    for step in ordered:
        if not _is_role(step.role, _VALIDATION_HINTS):
            continue
        if step.role.strip().lower() == "validation" and not any(h in step.role.lower() for h in _kind_hints):
            issues.append(
                WellFormednessIssue(
                    rule_id="validation_kind_undeclared",
                    severity=Severity.LOW,
                    offending_node_ids=[step.step_id],
                    message=f"Validation step {step.step_id} does not declare full/targeted/manual granularity.",
                )
            )

    # (e) every policy-sensitive action links to a policy artifact or is
    #     marked unresolved.
    _emit_policy_unlinked_issues(htir, spec)

    # (f) every failed validation that motivates a later edit connects to a
    #     later validation attempt or an unresolved obligation.
    for i, step in enumerate(ordered):
        if not (_is_role(step.role, _VALIDATION_HINTS) and step.execution_status == ExecutionStatus.FAILURE):
            continue
        has_later_validation = any(
            s.step_id > step.step_id and _is_role(s.role, _VALIDATION_HINTS) for s in ordered
        )
        if not has_later_validation:
            issues.append(
                WellFormednessIssue(
                    rule_id="failed_validation_unresolved",
                    severity=Severity.MEDIUM,
                    offending_node_ids=[step.step_id],
                    message=f"Failed validation at step {step.step_id} has no later validation attempt.",
                )
            )

    htir.wellformedness.extend(issues)
    return issues


def _policy_sensitive_steps(htir: HTIR, spec: DomainSpec) -> list[TraceStep]:
    """Steps governed by a domain constraint that explicitly names their role."""
    governed_roles = {role for c in spec.constraints for role in c.applies_to_operations}
    if not governed_roles:
        return []
    return [s for s in htir.steps_in_order() if s.role in governed_roles]


def _emit_policy_unlinked_issues(htir: HTIR, spec: DomainSpec) -> None:
    """
    Shared by ``check_wellformedness`` (rule e) and ``link_policy`` (the
    module that actually performs policy linking) so calling it from either
    (or both) call sites is idempotent -- no duplicate issues per step.
    """
    policy_artifact_ids = {a.artifact_id for a in htir.artifacts if a.artifact_type == _POLICY_ARTIFACT_TYPE}
    already_flagged = {
        issue.offending_node_ids[0]
        for issue in htir.wellformedness
        if issue.rule_id == "policy_action_unlinked" and issue.offending_node_ids
    }
    for step in _policy_sensitive_steps(htir, spec):
        if step.step_id in already_flagged:
            continue
        if policy_artifact_ids & set(step.consumed_artifact_ids):
            continue
        htir.wellformedness.append(
            WellFormednessIssue(
                rule_id="policy_action_unlinked",
                severity=Severity.HIGH,
                offending_node_ids=[step.step_id],
                message=(
                    f"Step {step.step_id} ({step.role}) is governed by a domain constraint "
                    "but is not linked to any policy artifact."
                ),
            )
        )
        already_flagged.add(step.step_id)


# ---------------------------------------------------------------------------
# 3.1 Provenance analysis
# ---------------------------------------------------------------------------

class _CitedArtifact(BaseModel):
    artifact_identifier: str
    rationale: str = ""


class _CitedArtifactList(BaseModel):
    artifacts: list[_CitedArtifact] = Field(default_factory=list)


def link_provenance_to_final_answer(htir: HTIR, *, use_semantic: bool = False, model: str = DEFAULT_MODEL) -> None:
    """
    Provenance analysis (avg.tex Sec. 3.5): artifact <-> operation linking is
    already done deterministically by
    ``TraceAbstractionAgent._extract_artifacts`` (E_prov). This adds the
    final-answer -> source-artifact link the paper calls out separately.

    Mechanical (always on): final-answer steps that explicitly consumed an
    artifact get a dependency edge to it. Free-text citations (an artifact
    referenced only in the response prose, with no explicit consumption
    link) require model judgement and are gated on ``use_semantic``.
    """
    ordered = htir.steps_in_order()
    final_steps = [s for s in ordered if _is_role(s.role, _FINAL_HINTS)]

    existing_targets = {
        (lk.source_step_id, lk.target_artifact_id)
        for lk in htir.dependency_links
        if lk.target_artifact_id is not None
    }

    for final_step in final_steps:
        for artifact_id in final_step.consumed_artifact_ids:
            key = (final_step.step_id, artifact_id)
            if key in existing_targets:
                continue
            artifact = htir.get_artifact(artifact_id)
            if artifact is None:
                continue
            htir.dependency_links.append(
                DependencyLink(
                    source_step_id=final_step.step_id,
                    target_artifact_id=artifact_id,
                    reason=f"final answer cites source artifact '{artifact.identifier}'",
                )
            )
            existing_targets.add(key)

        if use_semantic:
            _link_final_answer_semantic(htir, final_step, existing_targets, model=model)


def _link_final_answer_semantic(
    htir: HTIR, final_step: TraceStep, existing_targets: set[tuple[int, int]], *, model: str
) -> None:
    """Ask the model which *other* artifacts the final answer's text cites."""
    candidates = [
        a for a in htir.artifacts if (final_step.step_id, a.artifact_id) not in existing_targets
    ]
    if not candidates:
        return

    artifacts_desc = "\n".join(f"- {a.identifier}: {a.description}" for a in candidates)
    msgs = [
        system(
            "You are an expert trace analyst specialising in provenance. "
            "Given a final-answer step and a list of candidate artifacts, "
            "identify which artifacts the final answer's content is actually "
            "supported by, even if not explicitly consumed."
        ),
        user(
            f"Final answer (step {final_step.step_id}):\n{truncate(final_step.response_message, 1000)}\n\n"
            f"Candidate artifacts:\n{artifacts_desc}\n\n"
            "Return artifacts: a list of {artifact_identifier, rationale} for artifacts that "
            "genuinely support the final answer's content."
        ),
    ]
    result = chat_json(msgs, _CitedArtifactList, model=model)
    by_ident = {a.identifier: a for a in htir.artifacts}
    for cited in result.artifacts:
        artifact = by_ident.get(cited.artifact_identifier)
        if artifact is None:
            continue
        key = (final_step.step_id, artifact.artifact_id)
        if key in existing_targets:
            continue
        htir.dependency_links.append(
            DependencyLink(
                source_step_id=final_step.step_id,
                target_artifact_id=artifact.artifact_id,
                reason=cited.rationale or f"final answer semantically cites artifact '{artifact.identifier}'",
            )
        )
        existing_targets.add(key)


# ---------------------------------------------------------------------------
# 3.2 Dependency analysis (E_causal)
# ---------------------------------------------------------------------------

def link_dependencies(htir: HTIR) -> None:
    """
    Dependency analysis (avg.tex Sec. 3.5): promotes the deterministic first
    cut formerly in ``obligations._link_dependencies`` (consumer step ->
    producer step of a consumed artifact) and adds the dependencies the
    paper notes have no explicit artifact link:

      * an edit that depends on the most recent prior *failing* validation;
      * a final answer that depends on a policy artifact, when one exists.

    All mechanical/deterministic -- no LLM call.
    """
    ordered = htir.steps_in_order()

    # First cut: artifact-consumption dependencies.
    for step in ordered:
        for artifact_id in step.consumed_artifact_ids:
            artifact = htir.get_artifact(artifact_id)
            if artifact is None or artifact.produced_by_step_id is None:
                continue
            if artifact.produced_by_step_id == step.step_id:
                continue
            htir.dependency_links.append(
                DependencyLink(
                    source_step_id=step.step_id,
                    target_step_id=artifact.produced_by_step_id,
                    target_artifact_id=artifact_id,
                    reason=f"consumes artifact '{artifact.identifier}' produced by step {artifact.produced_by_step_id}",
                )
            )

    existing_pairs = {(lk.source_step_id, lk.target_step_id) for lk in htir.dependency_links}

    # Extended cut: edit depends on the most recent failing validation.
    last_failed_validation: TraceStep | None = None
    for step in ordered:
        if _is_role(step.role, _VALIDATION_HINTS) and step.execution_status == ExecutionStatus.FAILURE:
            last_failed_validation = step
            continue
        if _is_role(step.role, _EDIT_HINTS) and last_failed_validation is not None:
            key = (step.step_id, last_failed_validation.step_id)
            if key not in existing_pairs:
                htir.dependency_links.append(
                    DependencyLink(
                        source_step_id=step.step_id,
                        target_step_id=last_failed_validation.step_id,
                        reason=f"edit follows failing validation at step {last_failed_validation.step_id}",
                    )
                )
                existing_pairs.add(key)
            last_failed_validation = None  # consumed by this edit

    # Extended cut: final answer depends on a policy artifact, if any exists.
    policy_artifacts = [a for a in htir.artifacts if a.artifact_type == _POLICY_ARTIFACT_TYPE]
    if policy_artifacts:
        existing_artifact_targets = {
            (lk.source_step_id, lk.target_artifact_id)
            for lk in htir.dependency_links
            if lk.target_artifact_id is not None
        }
        for step in ordered:
            if not _is_role(step.role, _FINAL_HINTS):
                continue
            for artifact in policy_artifacts:
                key = (step.step_id, artifact.artifact_id)
                if key in existing_artifact_targets:
                    continue
                htir.dependency_links.append(
                    DependencyLink(
                        source_step_id=step.step_id,
                        target_artifact_id=artifact.artifact_id,
                        reason=f"final answer depends on policy artifact '{artifact.identifier}'",
                    )
                )
                existing_artifact_targets.add(key)


# ---------------------------------------------------------------------------
# Validation-edge wiring (E_val) -- moved from obligations.py
# ---------------------------------------------------------------------------

def link_validations(htir: HTIR) -> None:
    """
    E_val: each validation operation validates the most recent prior
    artifact-producing step. Moved here (Step-3 analysis layer) from
    ``obligations.build_claims_and_obligations`` so validation edges exist
    before obligation generation reads them, and so state-transition
    analysis (below) can use them.
    """
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


def _last_producer_before(ordered: list[TraceStep], idx: int) -> TraceStep | None:
    for j in range(idx - 1, -1, -1):
        if ordered[j].produced_artifact_ids:
            return ordered[j]
    return None


# ---------------------------------------------------------------------------
# 3.3 State-transition analysis
# ---------------------------------------------------------------------------

_STATE_TRANSITION_PATTERN = "failing_validation_edit_revalidation"


def analyze_state_transitions(htir: HTIR) -> None:
    """
    State-transition analysis (avg.tex Sec. 3.5): recognise the
    failing-validation -> relevant-edit -> post-edit-validation pattern (its
    domain analogue e.g. source table -> filtered query -> aggregation ->
    exported report is left to specialised domains). Mechanical: uses only
    execution status and temporal order, no LLM call.
    """
    ordered = htir.steps_in_order()

    for val in ordered:
        if not (_is_role(val.role, _VALIDATION_HINTS) and val.execution_status == ExecutionStatus.FAILURE):
            continue

        later = [s for s in ordered if s.step_id > val.step_id]
        edit_step = next((s for s in later if _is_role(s.role, _EDIT_HINTS)), None)
        if edit_step is None:
            htir.state_transitions.append(
                StateTransitionPattern(
                    pattern_name=_STATE_TRANSITION_PATTERN,
                    step_ids=[val.step_id],
                    matched=False,
                    note=f"validation failed at step {val.step_id} with no subsequent edit",
                )
            )
            continue

        revalidation = next(
            (s for s in later if s.step_id > edit_step.step_id and _is_role(s.role, _VALIDATION_HINTS)),
            None,
        )
        if revalidation is None:
            htir.state_transitions.append(
                StateTransitionPattern(
                    pattern_name=_STATE_TRANSITION_PATTERN,
                    step_ids=[val.step_id, edit_step.step_id],
                    matched=False,
                    note=f"edit at step {edit_step.step_id} not followed by a revalidation",
                )
            )
            continue

        matched = revalidation.execution_status == ExecutionStatus.SUCCESS
        htir.state_transitions.append(
            StateTransitionPattern(
                pattern_name=_STATE_TRANSITION_PATTERN,
                step_ids=[val.step_id, edit_step.step_id, revalidation.step_id],
                matched=matched,
                note=(
                    f"step {val.step_id} failed, step {edit_step.step_id} edited, "
                    f"step {revalidation.step_id} {'passed' if matched else 'did not pass'} revalidation"
                ),
            )
        )


# ---------------------------------------------------------------------------
# 3.4 Policy-linking analysis (+ E_cons constraint wiring)
# ---------------------------------------------------------------------------

def link_policy(htir: HTIR, spec: DomainSpec, *, use_semantic: bool = False, model: str = DEFAULT_MODEL) -> None:
    """
    Policy-linking analysis (avg.tex Sec. 3.5): connects policy-sensitive
    operations to policy artifacts they consumed; if a policy-sensitive step
    has no linked policy artifact, an unresolved obligation is emitted (see
    ``_emit_policy_unlinked_issues``, shared with ``check_wellformedness``
    rule (e) so calling both is idempotent).

    Also wires the constraint edges (E_cons), formerly
    ``obligations._link_constraint``: ties every domain constraint (S_d.K_d)
    to the steps it governs.
    """
    link_constraints(htir, spec)

    policy_artifacts = [a for a in htir.artifacts if a.artifact_type == _POLICY_ARTIFACT_TYPE]

    if use_semantic and policy_artifacts:
        _link_policy_semantic(htir, spec, policy_artifacts, model=model)

    # Mechanical unresolved-obligation emission always runs, regardless of
    # whether the semantic pass above found additional (soft) relevance --
    # it only suppresses steps that ended up with an explicit consumption
    # link to a policy artifact.
    _emit_policy_unlinked_issues(htir, spec)


def link_constraints(htir: HTIR, spec: DomainSpec) -> None:
    """E_cons: tie every domain constraint (S_d.K_d) to the steps it governs."""
    for constraint in spec.constraints:
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


class _PolicyRelevance(BaseModel):
    relevant: bool = False
    rationale: str = ""


def _link_policy_semantic(htir: HTIR, spec: DomainSpec, policy_artifacts: list, *, model: str) -> None:
    """
    Optional semantic relevance judgement for policy-sensitive steps that
    have no *explicit* consumption link to a policy artifact, so a soft
    (semantically-relevant-but-not-consumed) link can still be recorded as a
    dependency edge before the mechanical unresolved-obligation check runs.
    """
    policy_artifact_ids = {a.artifact_id for a in policy_artifacts}
    artifacts_desc = "\n".join(f"- {a.identifier}: {a.description}" for a in policy_artifacts)

    for step in _policy_sensitive_steps(htir, spec):
        if policy_artifact_ids & set(step.consumed_artifact_ids):
            continue
        msgs = [
            system(
                "You are an expert compliance analyst. Decide whether a step is "
                "governed by any of the listed policy artifacts even though no "
                "explicit consumption link exists."
            ),
            user(
                f"Step {step.step_id} ({step.role}):\n"
                f"REQUEST: {truncate(step.request_message, 500)}\n"
                f"RESPONSE: {truncate(step.response_message, 500)}\n\n"
                f"Policy artifacts:\n{artifacts_desc}\n\n"
                "Return relevant (bool) and rationale (string)."
            ),
        ]
        result = chat_json(msgs, _PolicyRelevance, model=model)
        if not result.relevant:
            continue
        for artifact in policy_artifacts:
            if artifact.artifact_id not in step.consumed_artifact_ids:
                step.consumed_artifact_ids.append(artifact.artifact_id)
            htir.dependency_links.append(
                DependencyLink(
                    source_step_id=step.step_id,
                    target_artifact_id=artifact.artifact_id,
                    reason=result.rationale or f"step {step.step_id} is semantically governed by policy artifact '{artifact.identifier}'",
                )
            )


# ---------------------------------------------------------------------------
# 3.5 Integrity analysis
# ---------------------------------------------------------------------------

def check_integrity(htir: HTIR) -> None:
    """
    Integrity analysis (avg.tex Sec. 3.5): detect shortcut/tamper behaviour.
    Mostly mechanical; findings are appended to ``htir.wellformedness`` with
    HIGH severity so they seed unresolved obligations (severity-aware
    aggregation is a later step, per avg.tex Sec. 3.6+).
    """
    ordered = htir.steps_in_order()

    # Tests modified directly, rather than the source they validate.
    test_artifact_ids = {
        a.artifact_id for a in htir.artifacts
        if "test" in a.artifact_type.lower() or "test" in a.identifier.lower()
    }
    for step in ordered:
        if not _is_role(step.role, _EDIT_HINTS):
            continue
        touched_tests = set(step.produced_artifact_ids) & test_artifact_ids
        if touched_tests:
            htir.wellformedness.append(
                WellFormednessIssue(
                    rule_id="integrity_test_modified",
                    severity=Severity.HIGH,
                    offending_node_ids=[step.step_id, *sorted(touched_tests)],
                    message=(
                        f"Step {step.step_id} ({step.role}) modified a test artifact "
                        "directly instead of the source it validates."
                    ),
                )
            )

    # Required-artifact deletion.
    for step in ordered:
        for eff in step.artifact_state_effects:
            if eff.effect_category not in (ArtifactEffect.STATE_CHANGE, ArtifactEffect.ARTIFACT_CHANGE):
                continue
            if any(kw in eff.observed_change.lower() for kw in _DELETION_HINTS):
                htir.wellformedness.append(
                    WellFormednessIssue(
                        rule_id="integrity_artifact_deleted",
                        severity=Severity.HIGH,
                        offending_node_ids=[step.step_id],
                        message=f"Step {step.step_id} deleted/removed '{eff.affected_resource}'.",
                    )
                )

    # Final claims not traceable to any source artifact.
    for step in ordered:
        if not _is_role(step.role, _FINAL_HINTS):
            continue
        has_link = bool(step.consumed_artifact_ids) or any(
            lk.source_step_id == step.step_id for lk in htir.dependency_links
        )
        if not has_link:
            htir.wellformedness.append(
                WellFormednessIssue(
                    rule_id="integrity_untraceable_final_claim",
                    severity=Severity.HIGH,
                    offending_node_ids=[step.step_id],
                    message=f"Final answer at step {step.step_id} cites no tracked source artifact.",
                )
            )


# ---------------------------------------------------------------------------
# 3.6 Coverage analysis -- depends on obligations, run after
# obligations.build_claims_and_obligations (see TraceAbstractionAgent.compile).
# ---------------------------------------------------------------------------

def compute_coverage(htir: HTIR) -> CoverageReport:
    """
    Coverage analysis (avg.tex Sec. 3.5): evidence coverage by obligation
    type x evidence type. Must run after
    ``obligations.build_claims_and_obligations`` since it reports over
    ``htir.obligations``.
    """
    report = CoverageReport(total_obligations=len(htir.obligations))
    evidence_type_by_id = {e.evidence_id: e.evidence_type.value for e in htir.evidence}

    for ob in htir.obligations:
        key = ob.template_id or ob.scope.value
        report.by_obligation_type[key] = report.by_obligation_type.get(key, 0) + 1
        if ob.candidate_evidence_ids:
            report.covered_obligations += 1
        for ev_id in ob.candidate_evidence_ids:
            et = evidence_type_by_id.get(ev_id, "none")
            report.by_evidence_type[et] = report.by_evidence_type.get(et, 0) + 1

    htir.coverage = report
    return report
