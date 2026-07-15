"""
Dynamic escalation loop (avg.tex "Checking Obligations" -> "Abstention and
escalation", and "Online Intervention" -> the ``request-evidence`` action).
(Cited by section name, not number: the paper's subsection numbering and the
rest of the codebase's ``Sec. 3.x`` references are currently out of sync -- see
``docs/experiment-consistency.md``.)

Base \\AVG checking is a *single static pass*: every obligation's checker runs
once and the trajectory is aggregated once. When a high-severity obligation
*abstains* -- common for policy-compliance, where the narrow checker is handed
the policy text but only the one producing step -- that abstention is final, so
the trajectory stays ``uncertain`` even though the evidence to resolve it exists
elsewhere in the graph.

This module implements the escalation the proposal already specifies but the
static path never exercised: when the LLM is available, an abstaining
high-severity obligation is **re-checked against a broadened local
neighbourhood** (the ``request-evidence`` intervention -- gather more of the
already-recorded evidence, then re-judge), and the trajectory is re-aggregated.
The loop repeats, widening the window each round, until no abstention flips or a
round budget is hit.

Method-preserving. It adds no node type, edge type, checker class, obligation
template, or aggregation rule. It re-runs the *existing* semantic checker over a
wider slice of the *existing* graph (the "immediate neighbourhood" the Sec. 3.7
checker contract already grants) and re-runs the *existing* ``aggregate``. With
``use_llm=False`` it is a no-op that returns the static verdict unchanged, so
every offline result is byte-for-byte identical.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from htir.agents.checking import (
    _aggregate_claim_status,
    _status_from_result,
    check_obligations,
)
from htir.agents.witness import aggregate
from htir.models.domain import DomainArtifactBundle, DomainSpec
from htir.models.htir import (
    AggregateResult,
    CheckerResult,
    CheckerType,
    ClaimNode,
    EvidenceNode,
    HTIR,
    Obligation,
    ObligationStatus,
    Severity,
    SupportPolarity,
)
from htir.utils.io import truncate
from htir.utils.llm import DEFAULT_MODEL, chat_json, system, user

# Severities that warrant escalation when they abstain (a low/medium abstention
# is left as an honest "insufficient evidence", per the paper's abstention-as-
# feature stance). Escalation rules that mean "gather more evidence / retry".
_ESCALATABLE_SEVERITIES = frozenset({Severity.HIGH, Severity.CRITICAL})
_ESCALATABLE_RULES = frozenset({"request-evidence", "repair", "escalate"})


class _SemanticVerdict(BaseModel):
    verdict: str = "abstain"  # pass | fail | abstain
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    rationale: str = ""


class EscalationEvent(BaseModel):
    """One obligation re-checked under escalation, for the audit trail."""
    obligation_id: int
    claim_id: int
    round: int
    window: int
    from_status: str
    to_status: str
    rationale: str = ""


class EscalationResult(BaseModel):
    """Outcome of the dynamic loop: the final verdict and what it took to get there."""
    aggregate: AggregateResult
    rounds: int = 0
    n_escalated: int = Field(0, description="obligations re-checked under escalation")
    n_resolved: int = Field(0, description="abstentions the loop flipped to pass/fail")
    llm_calls: int = 0
    events: list[EscalationEvent] = Field(default_factory=list)


def _neighbourhood_context(htir: HTIR, claim: ClaimNode, window: int) -> str:
    """
    The claim's producing step plus the ``window`` steps before it -- the
    broadened local neighbourhood escalation gathers. For a policy-compliance
    claim on a mutation, this is what surfaces the preceding authentication and
    user-confirmation turns the SOP requires, which the single-step context
    lacked.
    """
    if claim.source_step_id is None:
        return ""
    ordered = htir.steps_in_order()
    idx = next((i for i, s in enumerate(ordered) if s.step_id == claim.source_step_id), None)
    if idx is None:
        return ""
    lo = max(0, idx - window)
    lines: list[str] = []
    for s in ordered[lo: idx + 1]:
        marker = "  >> ACTION UNDER REVIEW: " if s.step_id == claim.source_step_id else "  - "
        tools = "; ".join(
            f"{tc.name}({truncate(tc.arguments_text or '', 120)}) -> {truncate(tc.result or '', 120)}"
            for tc in s.tool_calls
        )
        seg = f"{marker}step {s.step_id} [{s.role}/{s.execution_status.value}]"
        if s.request_message:
            seg += f" | user/context: {truncate(s.request_message, 200)}"
        if s.response_message:
            seg += f" | agent: {truncate(s.response_message, 200)}"
        if tools:
            seg += f" | tools: {tools}"
        lines.append(seg)
    return "\n".join(lines)


def _escalated_semantic_check(
    htir: HTIR,
    ob: Obligation,
    claim: ClaimNode,
    evidence_by_id: dict[int, EvidenceNode],
    *,
    window: int,
    model: str,
) -> CheckerResult | None:
    """
    Re-run the narrow semantic judge for one obligation over a *broadened* local
    neighbourhood (``window`` preceding steps) plus its candidate evidence.
    Returns a fresh ``CheckerResult`` or ``None`` if the LLM is unavailable
    (leaving the caller to keep the prior abstention).
    """
    evidence_desc = "\n".join(
        f"- {evidence_by_id[e].description}: {truncate(evidence_by_id[e].content, 600)}"
        for e in ob.candidate_evidence_ids if e in evidence_by_id
    )
    context = _neighbourhood_context(htir, claim, window)
    msgs = [
        system(
            "You are a narrow claim-evidence checker re-examining a claim after "
            "gathering more of the surrounding trace (an escalation). Judge the "
            "action under review against the candidate evidence (e.g. a policy) "
            "using the broadened context -- e.g. whether required prior steps "
            "(authentication, explicit user confirmation) actually occurred. "
            "Return pass, fail, or abstain; only abstain if the broadened context "
            "still cannot settle it. Never guess beyond what is shown."
        ),
        user(
            f"Claim: {claim.statement}\n\n"
            f"Surrounding trace (most recent last):\n{context}\n\n"
            f"Candidate evidence:\n{evidence_desc}\n\n"
            "Return verdict (pass/fail/abstain), confidence (0-1), and a short rationale."
        ),
    ]
    try:
        v = chat_json(msgs, _SemanticVerdict, model=model)
    except (EnvironmentError, ImportError):
        return None
    except Exception:
        return None

    conf = max(0.0, min(1.0, v.confidence))
    verdict = v.verdict.strip().lower()
    ev = list(ob.candidate_evidence_ids)
    if verdict == "pass":
        return CheckerResult(p_pass=conf, p_fail=0.0, p_abstain=1.0 - conf, score=conf, evidence_used=ev)
    if verdict == "fail":
        return CheckerResult(p_pass=0.0, p_fail=conf, p_abstain=1.0 - conf, score=-conf, evidence_used=ev)
    return CheckerResult(p_pass=0.0, p_fail=0.0, p_abstain=1.0, score=0.0, evidence_used=ev)


def _recompute_claim_statuses(htir: HTIR) -> None:
    """Recompute every claim's status from all its obligations (as check_obligations does)."""
    claims_by_id = {c.claim_id: c for c in htir.claims}
    refuted = {lk.claim_id for lk in htir.support_links if lk.polarity == SupportPolarity.REFUTES}
    by_claim: dict[int, list[Obligation]] = {}
    for ob in htir.obligations:
        by_claim.setdefault(ob.claim_id, []).append(ob)
    for claim_id, obs in by_claim.items():
        c = claims_by_id.get(claim_id)
        if c is not None:
            c.status = _aggregate_claim_status(obs, claim_id in refuted)


def verify_with_escalation(
    htir: HTIR,
    spec: DomainSpec,
    *,
    use_llm: bool = False,
    model: str = DEFAULT_MODEL,
    domain_artifacts: DomainArtifactBundle | None = None,
    max_rounds: int = 2,
    base_window: int = 4,
    commit_threshold: float = 0.0,
    log: Any = None,
) -> EscalationResult:
    """
    Verify ``htir`` with the dynamic escalation loop and return the final
    aggregate plus an audit trail.

    Runs the standard static check once, then -- only when ``use_llm`` -- looks
    for high-severity SEMANTIC obligations that abstained and whose escalation
    rule asks to gather evidence, re-checks each over a neighbourhood that widens
    by round, and re-aggregates. Offline (``use_llm=False``) it is exactly the
    static verdict (no LLM re-judge is possible), so nothing offline changes.

    ``commit_threshold`` gates how confident a re-check must be to flip an
    abstention: an escalated pass/fail is only committed when its probability
    mass ``max(p_pass, p_fail) >= commit_threshold``. ``0.0`` (default) commits
    any non-abstain verdict (maximal coverage); raising it (e.g. ``0.6``) keeps
    only confident resolutions, trading coverage for resolved-accuracy -- the
    knob for the coverage/precision tradeoff the loop exposes.
    """
    check_obligations(htir, spec, use_semantic=use_llm, domain_artifacts=domain_artifacts, model=model)
    agg = aggregate(htir)
    result = EscalationResult(aggregate=agg)
    if not use_llm:
        return result

    claims_by_id = {c.claim_id: c for c in htir.claims}
    evidence_by_id = {e.evidence_id: e for e in htir.evidence}

    for rnd in range(1, max_rounds + 1):
        window = base_window + (rnd - 1) * 4
        targets = [
            ob for ob in htir.obligations
            if ob.checker == CheckerType.SEMANTIC
            and ob.status == ObligationStatus.ABSTAINED
            and ob.severity in _ESCALATABLE_SEVERITIES
            and ob.escalation.value in _ESCALATABLE_RULES
            and ob.candidate_evidence_ids
        ]
        if not targets:
            break

        changed = False
        for ob in targets:
            claim = claims_by_id.get(ob.claim_id)
            if claim is None:
                continue
            new = _escalated_semantic_check(htir, ob, claim, evidence_by_id, window=window, model=model)
            result.llm_calls += 1
            if new is None:
                continue
            new_status = _status_from_result(new)
            confident = max(new.p_pass, new.p_fail) >= commit_threshold
            if new_status != ObligationStatus.ABSTAINED and confident:
                result.events.append(EscalationEvent(
                    obligation_id=ob.obligation_id, claim_id=ob.claim_id, round=rnd, window=window,
                    from_status=ob.status.value, to_status=new_status.value,
                ))
                ob.result = new
                ob.status = new_status
                result.n_resolved += 1
                changed = True
            result.n_escalated += 1

        if changed:
            _recompute_claim_statuses(htir)
            agg = aggregate(htir)
            result.aggregate = agg
        result.rounds = rnd
        if not changed:
            break
        if log is not None:
            print(f"[escalation] round {rnd}: window={window}, resolved so far={result.n_resolved}", file=log)

    return result
