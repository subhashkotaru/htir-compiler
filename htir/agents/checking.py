"""
Checker execution (AVG Step 5, avg.tex Sec. 3.7 "Checking Obligations").

This module discharges each ``Obligation`` produced by
``htir.agents.obligations.build_claims_and_obligations`` by running the
checker class routed to it (``Obligation.checker``, q_i), filling
``Obligation.result`` (``CheckerResult``) and ``Obligation.status``
(``PASSED``/``FAILED``/``ABSTAINED``), and propagating the outcome to the
linked ``ClaimNode.status`` (``SUPPORTED``/``REFUTED``/``UNRESOLVED``). It does
**not** touch graph construction or obligation generation -- it only consumes
the finished obligation set.

Entry point: ``check_obligations``.

Checker contract (avg.tex Sec. 3.7): a checker consumes an obligation and its
*local* graph context only -- the claim, its r_i-typed candidate evidence
``E_i``, and the immediate neighbourhood (producing step, consumed/produced
artifacts, validation/dependency edges touching it). It is never handed the
whole trace. It returns
``CheckerResult(p_pass, p_fail, p_abstain, score, evidence_used)`` with the
three probabilities summing to 1.

Three checker classes, routed by ``Obligation.checker``:

* ``MECHANICAL`` -- deterministic, no LLM call. Execution-status/exit-code
  checks, artifact-provenance checks, post-edit-validation checks (reads
  ``ValidationLink``), and schema checks against an optional Omega_d
  ``schema`` artifact (abstain if absent -- never fake a pass).
* ``SEMANTIC`` -- a narrow LLM judge over a single claim-evidence pair, gated
  behind ``use_semantic`` (default off). With ``use_semantic=False`` every
  semantic obligation abstains; ``chat_json`` is never called, so the
  deterministic path stays byte-for-byte reproducible.
* ``ABSTENTION`` (and ``UNASSIGNED``, which is routed the same way) --
  ``p_abstain=1.0``. A first-class, normal outcome, not a failure.

Write-back: ``status`` is the argmax of the three probabilities with a
conservative tie-break (abstain > fail > pass). The linked claim's status is
updated accordingly, except a claim that already carries a ``REFUTES``
support edge from Step 4 can never end up ``SUPPORTED`` here, even if its
obligation passes.

Idempotent: every obligation's ``result``/``status`` (and its claim's
``status``) is fully recomputed and overwritten on every call, so re-running
``check_obligations`` does not accumulate or drift.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from htir.models.domain import ArtifactKind, DomainArtifactBundle, DomainSpec
from htir.models.htir import (
    CheckerResult,
    CheckerType,
    ClaimNode,
    ClaimStatus,
    EvidenceNode,
    EvidenceType,
    ExecutionStatus,
    HTIR,
    Obligation,
    ObligationStatus,
    ProvenanceRelation,
    SupportPolarity,
    ValidationLink,
)
from htir.agents.checker_registry import CheckerContext, register_checker, resolve_checker
from htir.utils.io import truncate
from htir.utils.llm import DEFAULT_MODEL, chat_json, system, user

# Template ids that implement the "post-edit-validation" pattern: a source
# edit should be followed by relevant validation. Both the universal/
# trajectory-triggered default template and the terminal_swe domain template
# check the same thing over the same local neighbourhood.
_POST_EDIT_VALIDATION_TEMPLATES = frozenset({"trig-post-edit-validation", "swe-edit-then-validate"})

# Template id for "a failed step is explained or addressed by a later step".
_EXPLAIN_FAILURE_TEMPLATE = "trig-explain-failure"

_FAILING_STATUSES = (ExecutionStatus.FAILURE, ExecutionStatus.TIMEOUT, ExecutionStatus.BLOCKED)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def check_obligations(
    htir: HTIR,
    spec: DomainSpec,
    *,
    use_semantic: bool = False,
    disable_mechanical: bool = False,
    force_decision: bool = False,
    domain_artifacts: DomainArtifactBundle | None = None,
    model: str = DEFAULT_MODEL,
) -> HTIR:
    """
    Run the checker routed to each obligation in ``htir.obligations``, filling
    ``result``/``status`` and propagating to the linked claim's ``status``.
    Mutates ``htir`` in place and returns it for chaining.

    ``use_semantic`` / ``disable_mechanical`` gate the two checker families so
    the same pipeline expresses the paper's verifier arms (avg.tex Sec. 4.3;
    see ``htir.agents.baselines``):

    * full AVG   -> ``use_semantic=True,  disable_mechanical=False``
    * exec-only  -> ``use_semantic=False, disable_mechanical=False`` (default)
    * exec-free  -> ``use_semantic=True,  disable_mechanical=True``

    When ``disable_mechanical`` is set, MECHANICAL-routed obligations abstain
    without running their checker (no execution evidence is consulted), which
    is exactly the execution-free ablation.

    ``force_decision`` is the **no-abstention** ablation (avg.tex Sec. 4.5,
    ablation #3): every checker must emit pass/fail, none may abstain. After
    the routed checker runs, its abstain probability mass is redistributed onto
    pass/fail (:func:`_force_decision`) so no obligation stays ABSTAINED. An
    obligation the verifier had *no evidence* for (pure abstain) is pushed to
    the optimistic prior -- it is credited PASSED rather than left unresolved --
    which is exactly the over-crediting that calibrated abstention is meant to
    prevent (Q3). Genuine mechanical fails still fail. This is off by default so
    the calibrated-abstention path stays byte-for-byte reproducible.

    Idempotent: recomputes and overwrites every obligation's result/status
    (and every affected claim's status) rather than appending/skipping, so a
    second call yields identical output.
    """
    claims_by_id: dict[int, ClaimNode] = {c.claim_id: c for c in htir.claims}
    evidence_by_id: dict[int, EvidenceNode] = {e.evidence_id: e for e in htir.evidence}
    refuted_claim_ids = {
        lk.claim_id for lk in htir.support_links if lk.polarity == SupportPolarity.REFUTES
    }

    obligations_by_claim: dict[int, list[Obligation]] = {}
    for ob in htir.obligations:
        claim = claims_by_id.get(ob.claim_id)
        result = _run_checker(
            htir, spec, ob, claim, evidence_by_id,
            use_semantic=use_semantic, disable_mechanical=disable_mechanical,
            domain_artifacts=domain_artifacts, model=model,
        )
        if force_decision:
            result = _force_decision(result)
        ob.result = result
        ob.status = _status_from_result(result, forced=force_decision)
        obligations_by_claim.setdefault(ob.claim_id, []).append(ob)

    # Claim status is derived once per claim from *all* of its obligations
    # (a claim may be the target of more than one obligation), so the result
    # does not depend on obligation list order.
    for claim_id, obs in obligations_by_claim.items():
        claim = claims_by_id.get(claim_id)
        if claim is not None:
            claim.status = _aggregate_claim_status(obs, claim_id in refuted_claim_ids)

    return htir


# ---------------------------------------------------------------------------
# Status derivation
# ---------------------------------------------------------------------------

def _status_from_result(result: CheckerResult, *, forced: bool = False) -> ObligationStatus:
    """
    argmax(p_pass, p_fail, p_abstain) with a conservative tie-break: when two
    probabilities are equal, prefer abstain over fail over pass (checking a
    claim conservatively is safer than over-crediting it).

    Under ``forced`` (the no-abstention ablation) abstention is not an allowed
    outcome, so the candidate is dropped and the tie-break flips to the
    optimistic pass > fail: a forced no-evidence obligation (p_pass == p_fail)
    is credited PASSED rather than flagged, which is the over-crediting Q3
    predicts when abstention is removed.
    """
    if forced:
        candidates = [
            (result.p_pass, ObligationStatus.PASSED),
            (result.p_fail, ObligationStatus.FAILED),
        ]
        return max(candidates, key=lambda pair: pair[0])[1]
    candidates = [
        (result.p_abstain, ObligationStatus.ABSTAINED),
        (result.p_fail, ObligationStatus.FAILED),
        (result.p_pass, ObligationStatus.PASSED),
    ]
    return max(candidates, key=lambda pair: pair[0])[1]


# Prior credited to a forced obligation that had no evidence to decide on
# (pure abstain): 0.5 leaves it exactly at the pass/invalid boundary, so it
# contributes an uninformative-but-committed 0.5 to the trajectory score while
# the status tie-break (pass > fail) credits it. Genuine mechanical pass/fail
# results are untouched -- only the abstain mass is redistributed.
_FORCED_PASS_PRIOR = 0.5


def _force_decision(result: CheckerResult) -> CheckerResult:
    """
    No-abstention ablation (avg.tex Sec. 4.5 #3): redistribute a checker's
    abstain mass onto pass/fail so the obligation must commit. Mass is split in
    proportion to the existing pass/fail signal; a pure abstain (no signal at
    all) is split by the optimistic ``_FORCED_PASS_PRIOR``. The score is
    recomputed as ``p_pass - p_fail`` and ``evidence_used`` is preserved.
    """
    if result.p_abstain <= 0.0:
        return result
    signal = result.p_pass + result.p_fail
    if signal > 0.0:
        share_pass = result.p_pass / signal
    else:
        share_pass = _FORCED_PASS_PRIOR
    p_pass = result.p_pass + result.p_abstain * share_pass
    p_fail = result.p_fail + result.p_abstain * (1.0 - share_pass)
    return CheckerResult(
        p_pass=p_pass,
        p_fail=p_fail,
        p_abstain=0.0,
        score=p_pass - p_fail,
        evidence_used=list(result.evidence_used),
    )


def _aggregate_claim_status(obligations: list[Obligation], is_refuted_by_evidence: bool) -> ClaimStatus:
    """
    Derive a claim's status from *all* obligations that target it, order-
    independent. A FAILED obligation (or a pre-existing REFUTES support edge
    from Step 4) always wins -- a claim with refuting evidence or a failed
    check must never end up SUPPORTED, even if some other obligation on the
    same claim passed or abstained.
    """
    statuses = {ob.status for ob in obligations}
    if ObligationStatus.FAILED in statuses or is_refuted_by_evidence:
        return ClaimStatus.REFUTED
    if ObligationStatus.PASSED in statuses:
        return ClaimStatus.SUPPORTED
    if ObligationStatus.ABSTAINED in statuses:
        return ClaimStatus.UNRESOLVED
    return ClaimStatus.UNVERIFIED


def _abstain() -> CheckerResult:
    return CheckerResult(p_pass=0.0, p_fail=0.0, p_abstain=1.0, score=0.0, evidence_used=[])


def _decide(*, evidence_used: list[int], passed: bool | None) -> CheckerResult:
    """``passed`` is True/False for a mechanical pass/fail, ``None`` to abstain."""
    if passed is None:
        return CheckerResult(p_pass=0.0, p_fail=0.0, p_abstain=1.0, score=0.0, evidence_used=evidence_used)
    if passed:
        return CheckerResult(p_pass=1.0, p_fail=0.0, p_abstain=0.0, score=1.0, evidence_used=evidence_used)
    return CheckerResult(p_pass=0.0, p_fail=1.0, p_abstain=0.0, score=-1.0, evidence_used=evidence_used)


# ---------------------------------------------------------------------------
# Checker dispatch
# ---------------------------------------------------------------------------

def _run_checker(
    htir: HTIR,
    spec: DomainSpec,
    ob: Obligation,
    claim: ClaimNode | None,
    evidence_by_id: dict[int, EvidenceNode],
    *,
    use_semantic: bool,
    disable_mechanical: bool = False,
    domain_artifacts: DomainArtifactBundle | None,
    model: str,
) -> CheckerResult:
    if claim is None:
        # No claim to anchor on -- nothing to check (should not happen for
        # obligations produced by build_claims_and_obligations, but stay
        # conservative rather than raising).
        return _abstain()

    if ob.checker == CheckerType.MECHANICAL:
        if disable_mechanical:
            # Execution-free ablation: no mechanical evidence is consulted.
            return _abstain()
        return _check_mechanical(htir, spec, ob, claim, evidence_by_id, domain_artifacts=domain_artifacts)
    if ob.checker == CheckerType.SEMANTIC:
        return _check_semantic(htir, ob, claim, evidence_by_id, use_semantic=use_semantic, model=model)
    # ABSTENTION and UNASSIGNED both abstain: UNASSIGNED means genuinely no
    # evidence type was ever routable, which is exactly the abstention case.
    return _abstain()


# ---------------------------------------------------------------------------
# Mechanical checkers
# ---------------------------------------------------------------------------

def _check_mechanical(
    htir: HTIR,
    spec: DomainSpec,
    ob: Obligation,
    claim: ClaimNode,
    evidence_by_id: dict[int, EvidenceNode],
    *,
    domain_artifacts: DomainArtifactBundle | None,
) -> CheckerResult:
    # Route through the checker registry so domains / third-party packages can
    # add mechanical checkers without editing this module. The built-ins below
    # register at import; ``resolve_checker`` encodes the precedence
    # (required_evidence -> template_id -> claim_type).
    checker = resolve_checker(ob, claim)
    if checker is None:
        # No specific mechanical rule: abstain (empty/irrelevant E_i is a
        # signal to abstain, never to search the whole graph or fake a pass).
        return _abstain()
    ctx = CheckerContext(
        htir=htir, spec=spec, obligation=ob, claim=claim,
        evidence_by_id=evidence_by_id, domain_artifacts=domain_artifacts,
    )
    return checker(ctx)


# ---------------------------------------------------------------------------
# Built-in mechanical checkers, registered into the checker registry. Each is
# a thin CheckerContext-shaped wrapper over a rule function below, so the rules
# stay independently testable and the registry is the single dispatch point.
# Community checkers register the same way (see checker_registry).
# ---------------------------------------------------------------------------

@register_checker(required_evidence=EvidenceType.SCHEMA)
def _mech_schema(ctx: CheckerContext) -> CheckerResult:
    return _check_schema(ctx.spec, ctx.claim, ctx.domain_artifacts, ctx.obligation.candidate_evidence_ids)


@register_checker(template_id=_POST_EDIT_VALIDATION_TEMPLATES)
def _mech_post_edit_validation(ctx: CheckerContext) -> CheckerResult:
    return _check_post_edit_validation(ctx.htir, ctx.claim)


@register_checker(template_id=_EXPLAIN_FAILURE_TEMPLATE)
def _mech_explained_failure(ctx: CheckerContext) -> CheckerResult:
    return _check_explained_failure(ctx.htir, ctx.claim)


@register_checker(claim_type="execution_status")
def _mech_execution_status(ctx: CheckerContext) -> CheckerResult:
    return _check_execution_status(ctx.htir, ctx.claim, ctx.obligation.candidate_evidence_ids)


@register_checker(claim_type="artifact_provenance")
def _mech_provenance(ctx: CheckerContext) -> CheckerResult:
    return _check_provenance(ctx.htir, ctx.claim)


@register_checker(claim_type="constraint_compliance")
def _mech_constraint(ctx: CheckerContext) -> CheckerResult:
    return _check_precondition(ctx.htir, ctx.spec, ctx.obligation, ctx.claim)


# Obligation template-id prefix for per-constraint obligations (see
# ``htir.agents.obligations._emit_constraint_obligations``).
_CONSTRAINT_TEMPLATE_PREFIX = "constraint:"


def _check_precondition(htir: HTIR, spec: DomainSpec, ob: Obligation, claim: ClaimNode) -> CheckerResult:
    """
    Mechanical precondition-ordering check for a ``requires_prior`` constraint
    (e.g. authenticate-before-action): the governed step must be preceded by a
    *successful* step of a required operation type. Structural, no LLM.

    * a successful prior step of a required op type  -> PASS;
    * such steps exist but none succeeded (ambiguous) -> ABSTAIN (conservative);
    * no prior step of the required op type at all    -> FAIL (clear violation).

    Abstains (never fabricates) when the obligation is not a ``requires_prior``
    constraint obligation, so this checker is a safe default for the
    ``constraint_compliance`` claim type shared with the semantic path.
    """
    tid = ob.template_id or ""
    if not tid.startswith(_CONSTRAINT_TEMPLATE_PREFIX):
        return _abstain()
    constraint_id = tid[len(_CONSTRAINT_TEMPLATE_PREFIX):]
    constraint = next((c for c in spec.constraints if c.constraint_id == constraint_id), None)
    if constraint is None or not constraint.requires_prior:
        return _abstain()
    if claim.source_step_id is None:
        return _abstain()

    required_roles = set(constraint.requires_prior)
    prior = [
        s for s in htir.steps_in_order()
        if s.step_id < claim.source_step_id and s.role in required_roles
    ]
    if any(s.execution_status == ExecutionStatus.SUCCESS for s in prior):
        return _decide(evidence_used=[], passed=True)
    if prior:
        # A required op was attempted but not observed to succeed -- stay
        # conservative rather than vetoing a possibly-fine trajectory.
        return _decide(evidence_used=[], passed=None)
    # The governed action ran with no prior required operation at all.
    return _decide(evidence_used=[], passed=False)


def _check_execution_status(htir: HTIR, claim: ClaimNode, candidate_evidence_ids: list[int]) -> CheckerResult:
    """Pass iff the claim's step reported SUCCESS; fail on FAILURE/TIMEOUT/BLOCKED."""
    step = htir.get_step(claim.source_step_id) if claim.source_step_id is not None else None
    if step is None or step.execution_status == ExecutionStatus.UNKNOWN:
        return _decide(evidence_used=candidate_evidence_ids, passed=None)
    if step.execution_status == ExecutionStatus.SUCCESS:
        return _decide(evidence_used=candidate_evidence_ids, passed=True)
    if step.execution_status in _FAILING_STATUSES:
        return _decide(evidence_used=candidate_evidence_ids, passed=False)
    return _decide(evidence_used=candidate_evidence_ids, passed=None)


def _check_provenance(htir: HTIR, claim: ClaimNode) -> CheckerResult:
    """Pass iff an ArtifactProvenanceLink(created|modified) exists for (step, artifact)."""
    if claim.source_step_id is None or not claim.artifact_ids:
        return _abstain()
    found = any(
        lk.step_id == claim.source_step_id
        and lk.artifact_id in claim.artifact_ids
        and lk.relation in (ProvenanceRelation.CREATED, ProvenanceRelation.MODIFIED)
        for lk in htir.provenance_links
    )
    if not found:
        return _abstain()
    return _decide(evidence_used=[], passed=True)


def _check_explained_failure(htir: HTIR, claim: ClaimNode) -> CheckerResult:
    """
    "A failed step is explained or addressed by a later step" (trig-explain-
    failure). Consults the local neighbourhood: dependency edges (E_causal)
    touching this step -- specifically the "edit follows failing validation"
    dependency ``link_dependencies`` creates when a later edit responds to
    this failure. Fails if a well-formedness issue already flagged this step
    as an unresolved failed validation (no later validation attempt at all);
    abstains otherwise.
    """
    step_id = claim.source_step_id
    if step_id is None:
        return _abstain()

    addressed = any(
        lk.target_step_id == step_id and "failing validation" in lk.reason
        for lk in htir.dependency_links
    )
    if addressed:
        return _decide(evidence_used=[], passed=True)

    unresolved = any(
        issue.rule_id == "failed_validation_unresolved" and step_id in issue.offending_node_ids
        for issue in htir.wellformedness
    )
    if unresolved:
        return _decide(evidence_used=[], passed=False)

    return _abstain()


def _check_post_edit_validation(htir: HTIR, claim: ClaimNode) -> CheckerResult:
    """
    Pass iff a ValidationLink targeting this edit (by step or by artifact)
    shows a successful revalidation after the edit; fail if the most recent
    such revalidation failed; abstain if none exists. Consults the local
    neighbourhood (validation edges touching the claim's step/artifacts), not
    the whole graph.
    """
    edit_step_id = claim.source_step_id
    if edit_step_id is None:
        return _abstain()

    candidates: list[ValidationLink] = [
        lk for lk in htir.validation_links
        if lk.source_id > edit_step_id
        and (lk.target_step_id == edit_step_id or (lk.target_artifact_id in claim.artifact_ids))
    ]
    if not candidates:
        return _abstain()

    # Most recent revalidation is the authoritative one.
    latest = max(candidates, key=lambda lk: lk.source_id)
    evidence_used = [
        e.evidence_id for e in _evidence_at_step(htir, latest.source_id)
        if e.evidence_type == EvidenceType.EXECUTABLE
    ]
    if latest.outcome == ExecutionStatus.SUCCESS:
        return _decide(evidence_used=evidence_used, passed=True)
    if latest.outcome in _FAILING_STATUSES:
        return _decide(evidence_used=evidence_used, passed=False)
    return _abstain()


def _evidence_at_step(htir: HTIR, step_id: int) -> list[EvidenceNode]:
    return [e for e in htir.evidence if step_id in e.step_ids]


def _check_schema(
    spec: DomainSpec, claim: ClaimNode, domain_artifacts: DomainArtifactBundle | None,
    candidate_evidence_ids: list[int] | None = None,
) -> CheckerResult:
    """
    Structural check against an Omega_d ``schema`` artifact. Prefers the
    r_i-typed candidate evidence ``E_i`` that ``build_claims_and_obligations``
    already resolved against Omega_d (work item A) -- if Step 4 found and
    attached a matching schema artifact, its presence in E_i *is* the
    evidence, so this checker consumes it rather than re-deriving it here.
    Falls back to a direct (spec, domain_artifacts) lookup for callers that
    invoke ``check_obligations`` with a bundle that wasn't also passed to
    obligation generation. Abstains whenever neither is available -- schema
    conformance is never faked.
    """
    if candidate_evidence_ids:
        return _decide(evidence_used=candidate_evidence_ids, passed=True)

    if domain_artifacts is None:
        return _abstain()
    schema_artifacts = domain_artifacts.by_kind(ArtifactKind.SCHEMA)
    if not schema_artifacts:
        return _abstain()

    # Best-effort match: an artifact type governing this claim with a
    # declared schema_hint, matched to an Omega_d schema artifact by
    # identifier. Falls back to "any schema artifact exists" when the claim
    # carries no more specific artifact-type link (still real evidence: a
    # schema for the domain was actually loaded and consulted).
    matching = None
    artifact_type_names = {at.name: at.schema_hint for at in spec.artifact_types if at.schema_hint}
    for schema_artifact in schema_artifacts:
        if schema_artifact.identifier in artifact_type_names or not artifact_type_names:
            matching = schema_artifact
            break
    if matching is None:
        matching = schema_artifacts[0]

    passed = bool(matching.content.strip())
    return _decide(evidence_used=[], passed=passed)


# ---------------------------------------------------------------------------
# Semantic checker
# ---------------------------------------------------------------------------

class _SemanticVerdict(BaseModel):
    verdict: str = "abstain"  # pass | fail | abstain
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    rationale: str = ""


def _producing_step_context(htir: HTIR, claim: ClaimNode) -> str:
    """
    The claim's producing step, rendered as the checker's *local neighbourhood*
    (avg.tex Sec. 3.7: a checker sees the claim, its candidate evidence, and the
    immediate neighbourhood incl. the producing step). This is the action a
    policy/final-answer obligation is actually about -- without it the model has
    the rulebook but not the move to judge. Empty string when there is no
    producing step (nothing local to add).
    """
    if claim.source_step_id is None:
        return ""
    step = htir.get_step(claim.source_step_id)
    if step is None:
        return ""
    tools = "; ".join(
        f"{tc.name}({truncate(tc.arguments_text or '', 160)}) -> {truncate(tc.result or '', 160)}"
        for tc in step.tool_calls
    )
    parts = [f"Step {step.step_id} — operation type: {step.role}; status: {step.execution_status.value}"]
    if step.request_message:
        parts.append(f"context/request: {truncate(step.request_message, 400)}")
    if step.response_message:
        parts.append(f"agent output: {truncate(step.response_message, 400)}")
    if tools:
        parts.append(f"tool calls: {tools}")
    return "\n".join(parts)


def _check_semantic(
    htir: HTIR,
    ob: Obligation,
    claim: ClaimNode,
    evidence_by_id: dict[int, EvidenceNode],
    *,
    use_semantic: bool,
    model: str,
) -> CheckerResult:
    """
    Narrow LLM judge over a single claim, its candidate evidence, and its local
    neighbourhood (the producing step). With ``use_semantic=False`` (the
    default), always abstains without calling the model -- this keeps the
    deterministic path byte-for-byte reproducible.

    The producing-step context is what lets a policy-compliance obligation
    ("step N complies with policy P") actually be judged: the candidate evidence
    supplies the policy P (an Omega_d artifact), and the local neighbourhood
    supplies the action step N to hold against it (avg.tex Sec. 3.7's checker
    contract). Only consulted when the LLM is on, so no offline result changes.
    """
    if not use_semantic:
        return _abstain()
    if not ob.candidate_evidence_ids:
        return _abstain()

    evidence_desc = "\n".join(
        f"- {evidence_by_id[e].description}: {truncate(evidence_by_id[e].content, 500)}"
        for e in ob.candidate_evidence_ids if e in evidence_by_id
    )
    step_context = _producing_step_context(htir, claim)
    step_block = f"Action under review (the step this claim is about):\n{step_context}\n\n" if step_context else ""
    msgs = [
        system(
            "You are a narrow claim-evidence checker. Given exactly one claim, "
            "the action it is about, and its candidate evidence (e.g. a policy or "
            "schema), judge whether the evidence supports (pass), contradicts "
            "(fail), or is insufficient to decide (abstain) the claim. Judge the "
            "action against the evidence; never guess beyond what is shown."
        ),
        user(
            f"Claim: {claim.statement}\n\n"
            f"{step_block}"
            f"Candidate evidence:\n{evidence_desc}\n\n"
            "Return verdict (pass/fail/abstain), confidence (0-1), and a short rationale."
        ),
    ]
    verdict = chat_json(msgs, _SemanticVerdict, model=model)
    conf = max(0.0, min(1.0, verdict.confidence))
    v = verdict.verdict.strip().lower()

    if v == "pass":
        return CheckerResult(p_pass=conf, p_fail=0.0, p_abstain=1.0 - conf, score=conf, evidence_used=ob.candidate_evidence_ids)
    if v == "fail":
        return CheckerResult(p_pass=0.0, p_fail=conf, p_abstain=1.0 - conf, score=-conf, evidence_used=ob.candidate_evidence_ids)
    return CheckerResult(p_pass=0.0, p_fail=0.0, p_abstain=1.0, score=0.0, evidence_used=ob.candidate_evidence_ids)
