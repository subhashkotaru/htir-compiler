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
already-recorded trace, then re-judge), and the trajectory is re-aggregated.
The loop repeats, widening the window each round, until no abstention flips or a
round budget is hit.

**Dynamic obligation update (``relocalize``, default on).** Escalation does not
merely re-score an obligation; it *updates the obligation itself*. The gathered
neighbourhood is lifted into a first-class ``EvidenceNode``, appended to the
graph, and added to the obligation's candidate-evidence set ``E_i`` (with a new
``E_sup`` support edge to the claim on resolution). So the obligation's
localization -- not just its verdict -- reflects the evidence the re-check
actually used, and the verification witness ``E_W`` surfaces it. This was chosen
over verdict-only re-scoring (which leaves ``E_i`` empty, so the witness cannot
show *why* the verdict changed) after an on-domain measurement; the same
measurement showed that resolving the intrinsically trace-unverifiable
``final_answer_support`` obligations (whose ``E_i`` is empty by design, because
the task-success signal is not in the trace) *doubles* the false-valid rate, so
those are deliberately excluded from escalation.

Method-preserving. It adds no node *type*, edge *type*, checker class,
obligation template, or aggregation rule. It grows the *existing* evidence /
support / obligation node sets the generator already produces, re-runs the
*existing* semantic checker over a wider slice of the *existing* graph (the
"immediate neighbourhood" the checker contract already grants), and re-runs the
*existing* ``aggregate``. With ``use_llm=False`` it is a no-op that returns the
static verdict unchanged, so every offline result is byte-for-byte identical.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from htir.agents.checking import (
    _aggregate_claim_status,
    _status_from_result,
    check_obligations,
)
from htir.agents.witness import aggregate, build_witness
from htir.models.domain import DomainArtifactBundle, DomainSpec
from htir.models.htir import (
    AggregateResult,
    CheckerResult,
    CheckerType,
    ClaimNode,
    EvidenceNode,
    EvidenceType,
    HTIR,
    Obligation,
    ObligationStatus,
    Severity,
    SupportLink,
    SupportPolarity,
)
from htir.utils.io import truncate
from htir.utils.llm import DEFAULT_MODEL, chat_json, system, user

# Severities that warrant escalation when they abstain (a low/medium abstention
# is left as an honest "insufficient evidence", per the paper's abstention-as-
# feature stance). Escalation rules that mean "gather more evidence / retry".
_ESCALATABLE_SEVERITIES = frozenset({Severity.HIGH, Severity.CRITICAL})
_ESCALATABLE_RULES = frozenset({"request-evidence", "repair", "escalate"})
# Claim types never escalated: an intrinsically trace-unverifiable claim whose
# task-success signal is not in the trace, so a semantic re-check systematically
# over-passes it (measured to double false-valid). It stays abstained.
_NON_ESCALATABLE_CLAIM_TYPES = frozenset({"final_answer_support"})


def _escalatable_claim(claim: ClaimNode | None) -> bool:
    """A target obligation's claim must exist and not be an excluded type."""
    return claim is not None and claim.claim_type not in _NON_ESCALATABLE_CLAIM_TYPES


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
    evidence_added: list[int] = Field(default_factory=list, description="evidence node ids gathered onto E_i")
    rationale: str = ""


class EscalationResult(BaseModel):
    """Outcome of the dynamic loop: the final verdict and what it took to get there."""
    aggregate: AggregateResult
    rounds: int = 0
    n_escalated: int = Field(0, description="obligations re-checked under escalation")
    n_resolved: int = Field(0, description="abstentions the loop flipped to pass/fail")
    n_evidence_added: int = Field(0, description="EvidenceNodes gathered onto obligations' E_i")
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


def _next_evidence_id(htir: HTIR) -> int:
    """The next free evidence-node id (evidence ids are their own space)."""
    return max((e.evidence_id for e in htir.evidence), default=0) + 1


def _gather_neighbourhood_evidence(htir: HTIR, claim: ClaimNode, window: int) -> EvidenceNode | None:
    """
    Lift the claim's broadened local neighbourhood into a first-class
    ``EvidenceNode`` (the concrete "gather evidence" of the ``request-evidence``
    escalation), append it to the graph, and return it. ``None`` when there is
    no producing step / no context to gather (nothing to localize onto).
    """
    context = _neighbourhood_context(htir, claim, window)
    if not context:
        return None
    ordered = htir.steps_in_order()
    idx = next((i for i, s in enumerate(ordered) if s.step_id == claim.source_step_id), None)
    step_ids = (
        [s.step_id for s in ordered[max(0, idx - window): idx + 1]] if idx is not None else []
    )
    ev = EvidenceNode(
        evidence_id=_next_evidence_id(htir),
        # Gathered context is a semantic/observational bundle, not executable /
        # schema / artifact evidence -- SEMANTIC keeps the type contract honest.
        evidence_type=EvidenceType.SEMANTIC,
        description=f"gathered neighbourhood evidence (window={window}) for step {claim.source_step_id}",
        content=truncate(context, 1500),
        step_ids=step_ids,
    )
    htir.evidence.append(ev)
    return ev


def _escalated_semantic_check(
    ob: Obligation,
    claim: ClaimNode,
    evidence_by_id: dict[int, EvidenceNode],
    *,
    model: str,
    context: str = "",
    exclude_evidence_id: int | None = None,
) -> CheckerResult | None:
    """
    Re-run the narrow semantic judge for one obligation over its candidate
    evidence ``E_i`` plus the gathered ``context`` (the broadened neighbourhood,
    always shown under "Surrounding trace" so the prompt is identical whether or
    not the neighbourhood was also persisted as an evidence node).
    ``exclude_evidence_id`` drops the just-gathered neighbourhood node from the
    evidence bullet list so it is not duplicated (it already appears as the
    context). Returns a fresh ``CheckerResult`` (``evidence_used`` = the current
    ``E_i``) or ``None`` if the LLM is unavailable.
    """
    evidence_desc = "\n".join(
        f"- {evidence_by_id[e].description}: {truncate(evidence_by_id[e].content, 1500)}"
        for e in ob.candidate_evidence_ids
        if e in evidence_by_id and e != exclude_evidence_id
    )
    context_block = f"Surrounding trace (most recent last):\n{context}\n\n" if context else ""
    msgs = [
        system(
            "You are a narrow claim-evidence checker re-examining a claim after "
            "gathering more of the surrounding trace (an escalation). Judge the "
            "action under review against the candidate evidence (e.g. a policy) "
            "using the gathered context -- e.g. whether required prior steps "
            "(authentication, explicit user confirmation) actually occurred. "
            "Return pass, fail, or abstain; only abstain if the gathered context "
            "still cannot settle it. Never guess beyond what is shown."
        ),
        user(
            f"Claim: {claim.statement}\n\n"
            f"{context_block}"
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
    relocalize: bool = True,
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

    ``relocalize`` (default on) makes the escalation *update the obligation*, not
    just its verdict: the gathered neighbourhood is lifted into an
    ``EvidenceNode``, appended to the graph and to the obligation's ``E_i``, with
    a support edge added on resolution -- so ``E_i`` / the witness ``E_W`` show
    the evidence the re-check used. Set ``relocalize=False`` for the verdict-only
    ablation (re-score over transient context, ``E_i`` untouched).

    ``commit_threshold`` gates how confident a re-check must be to flip an
    abstention: an escalated pass/fail is only committed when its probability
    mass ``max(p_pass, p_fail) >= commit_threshold``. ``0.0`` (default) commits
    any non-abstain verdict; raising it keeps only confident resolutions, trading
    coverage for resolved-accuracy.

    ``final_answer_support`` obligations are deliberately never escalated: their
    ``E_i`` is empty by design (the task-success signal is not in the trace), and
    an on-domain measurement showed resolving them *doubles* the false-valid
    rate. They stay abstained, as the paper intends.
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
            and _escalatable_claim(claims_by_id.get(ob.claim_id))
        ]
        if not targets:
            break

        changed = False
        for ob in targets:
            claim = claims_by_id[ob.claim_id]

            # request-evidence: gather the broadened neighbourhood. Under
            # ``relocalize`` it becomes a first-class EvidenceNode the obligation
            # is localized onto (E_i grows); either way it is shown to the
            # re-check as context, so the verdict is prompt-identical to the
            # verdict-only path -- the update adds provenance, not a new prompt.
            gathered: EvidenceNode | None = None
            if relocalize:
                gathered = _gather_neighbourhood_evidence(htir, claim, window)
                if gathered is not None:
                    evidence_by_id[gathered.evidence_id] = gathered
                    ob.candidate_evidence_ids.append(gathered.evidence_id)
                    result.n_evidence_added += 1
            context = gathered.content if gathered is not None else _neighbourhood_context(htir, claim, window)

            new = _escalated_semantic_check(
                ob, claim, evidence_by_id, model=model, context=context,
                exclude_evidence_id=(gathered.evidence_id if gathered is not None else None),
            )
            result.llm_calls += 1
            if new is None:
                continue
            new_status = _status_from_result(new)
            confident = max(new.p_pass, new.p_fail) >= commit_threshold
            if new_status != ObligationStatus.ABSTAINED and confident:
                # Resolve: overwrite result/status and wire the E_sup edge from
                # the gathered evidence to the claim (supports on pass, refutes
                # on fail), so the update is auditable, not just a flipped bit.
                ob.result = new
                ob.status = new_status
                result.n_resolved += 1
                changed = True
                if gathered is not None:
                    htir.support_links.append(SupportLink(
                        evidence_id=gathered.evidence_id,
                        claim_id=claim.claim_id,
                        polarity=(SupportPolarity.SUPPORTS if new_status == ObligationStatus.PASSED
                                  else SupportPolarity.REFUTES),
                    ))
                result.events.append(EscalationEvent(
                    obligation_id=ob.obligation_id, claim_id=ob.claim_id, round=rnd, window=window,
                    from_status=ObligationStatus.ABSTAINED.value, to_status=new_status.value,
                    evidence_added=[gathered.evidence_id] if gathered is not None else [],
                ))
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

    # Refresh the witness so E_W reflects the grown E_i / support edges.
    build_witness(htir)
    return result
