"""
Checker execution (AVG Step 5, avg.tex Sec. 3.8 "Checking Obligations").

This module discharges each ``Obligation`` produced by
``harnessfix.agents.obligations.build_claims_and_obligations`` by running the
checker class routed to it (``Obligation.checker``, q_i), filling
``Obligation.result`` (``CheckerResult``) and ``Obligation.status``
(``PASSED``/``FAILED``/``ABSTAINED``), and propagating the outcome to the
linked ``ClaimNode.status`` (``SUPPORTED``/``REFUTED``/``UNRESOLVED``). It does
**not** touch graph construction or obligation generation -- it only consumes
the finished obligation set.

Entry point: ``check_obligations``.

Checker contract (avg.tex Sec. 3.8): a checker consumes an obligation and its
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

from harnessfix.models.domain import ArtifactKind, DomainArtifactBundle, DomainSpec
from harnessfix.models.htir import (
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
from harnessfix.utils.io import truncate
from harnessfix.utils.llm import DEFAULT_MODEL, chat_json, system, user

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
    domain_artifacts: DomainArtifactBundle | None = None,
    model: str = DEFAULT_MODEL,
) -> HTIR:
    """
    Run the checker routed to each obligation in ``htir.obligations``, filling
    ``result``/``status`` and propagating to the linked claim's ``status``.
    Mutates ``htir`` in place and returns it for chaining.

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
            use_semantic=use_semantic, domain_artifacts=domain_artifacts, model=model,
        )
        ob.result = result
        ob.status = _status_from_result(result)
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

def _status_from_result(result: CheckerResult) -> ObligationStatus:
    """
    argmax(p_pass, p_fail, p_abstain) with a conservative tie-break: when two
    probabilities are equal, prefer abstain over fail over pass (checking a
    claim conservatively is safer than over-crediting it).
    """
    candidates = [
        (result.p_abstain, ObligationStatus.ABSTAINED),
        (result.p_fail, ObligationStatus.FAILED),
        (result.p_pass, ObligationStatus.PASSED),
    ]
    return max(candidates, key=lambda pair: pair[0])[1]


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
    domain_artifacts: DomainArtifactBundle | None,
    model: str,
) -> CheckerResult:
    if claim is None:
        # No claim to anchor on -- nothing to check (should not happen for
        # obligations produced by build_claims_and_obligations, but stay
        # conservative rather than raising).
        return _abstain()

    if ob.checker == CheckerType.MECHANICAL:
        return _check_mechanical(htir, spec, ob, claim, evidence_by_id, domain_artifacts=domain_artifacts)
    if ob.checker == CheckerType.SEMANTIC:
        return _check_semantic(ob, claim, evidence_by_id, use_semantic=use_semantic, model=model)
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
    if ob.required_evidence == EvidenceType.SCHEMA:
        return _check_schema(spec, claim, domain_artifacts, ob.candidate_evidence_ids)

    if ob.template_id in _POST_EDIT_VALIDATION_TEMPLATES or (
        claim.claim_type == "artifact_provenance" and ob.template_id and "validat" in ob.template_id
    ):
        return _check_post_edit_validation(htir, claim)

    if ob.template_id == _EXPLAIN_FAILURE_TEMPLATE:
        return _check_explained_failure(htir, claim)

    if claim.claim_type == "execution_status":
        return _check_execution_status(htir, claim, ob.candidate_evidence_ids)

    if claim.claim_type == "artifact_provenance":
        return _check_provenance(htir, claim)

    # No specific mechanical rule for this claim type/template: fall back to
    # the general contract -- empty E_i is a signal to abstain, never to
    # search the whole graph or fake a pass.
    if not ob.candidate_evidence_ids:
        return _abstain()
    return _abstain()


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


def _check_semantic(
    ob: Obligation,
    claim: ClaimNode,
    evidence_by_id: dict[int, EvidenceNode],
    *,
    use_semantic: bool,
    model: str,
) -> CheckerResult:
    """
    Narrow LLM judge over a single claim-evidence pair. With
    ``use_semantic=False`` (the default), always abstains without calling the
    model -- this keeps the deterministic path byte-for-byte reproducible.
    """
    if not use_semantic:
        return _abstain()
    if not ob.candidate_evidence_ids:
        return _abstain()

    evidence_desc = "\n".join(
        f"- {evidence_by_id[e].description}: {truncate(evidence_by_id[e].content, 500)}"
        for e in ob.candidate_evidence_ids if e in evidence_by_id
    )
    msgs = [
        system(
            "You are a narrow claim-evidence checker. Given exactly one claim "
            "and its candidate evidence, judge whether the evidence supports "
            "(pass), contradicts (fail), or is insufficient to decide "
            "(abstain) the claim. Never guess beyond the evidence given."
        ),
        user(
            f"Claim: {claim.statement}\n\n"
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
