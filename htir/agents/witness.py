"""
Aggregation z_tau + Verification witness W_tau (AVG Step 6, avg.tex Sec.
3.9-3.10 "Aggregating Obligation Results" / "Verification Witness").

This module collapses the checked obligation set (``htir.agents.
checking.check_obligations`` must already have run) into a trajectory-level
status and the verification witness that is AVG's stated output. It does not
run any checkers itself and makes no LLM calls -- aggregation and the review
recommendation are both mechanical and deterministic, per avg.tex Sec. 3.10
("Keep it mechanical (no LLM) so it is reproducible; a semantic prose upgrade
can be gated behind use_semantic later").

Entry points: ``aggregate`` (z_tau, avg.tex Sec. 3.9) and ``build_witness``
(W_tau, avg.tex Sec. 3.10).

Aggregation is severity-aware (avg.tex Sec. 3.9):

* A **failed** obligation whose severity is HIGH or CRITICAL vetoes the
  trajectory -> ``predicted_status = "invalid"``, even if many low-severity
  obligations pass (the paper's "modified tests but suite passes" example).
* Otherwise, if there is no veto but many HIGH/CRITICAL obligations abstained
  -> ``predicted_status = "uncertain"``, not "valid".
* Otherwise -> ``predicted_status = "valid"``.

The thresholds that decide "many" are explicit module-level constants so
they are auditable and testable (see ``HIGH_SEVERITIES``,
``UNCERTAIN_ABSTAIN_COUNT_THRESHOLD``, ``UNCERTAIN_ABSTAIN_FRACTION_THRESHOLD``).
"""

from __future__ import annotations

from htir.models.htir import (
    AggregateResult,
    HTIR,
    Obligation,
    ObligationStatus,
    Severity,
    VerificationWitness,
)

# ---------------------------------------------------------------------------
# Aggregation thresholds (explicit, auditable, testable)
# ---------------------------------------------------------------------------

# Severities that can veto a trajectory when FAILED, or make it "uncertain"
# when many of them ABSTAIN.
HIGH_SEVERITIES: frozenset[Severity] = frozenset({Severity.HIGH, Severity.CRITICAL})

# A trajectory with no veto is still "uncertain" (not "valid") if at least
# this many high-severity obligations abstained, ...
UNCERTAIN_ABSTAIN_COUNT_THRESHOLD = 2

# ... or if the *fraction* of high-severity obligations that abstained is at
# least this much (catches the case where there are few high-severity
# obligations overall but most of them abstained).
UNCERTAIN_ABSTAIN_FRACTION_THRESHOLD = 0.5

# Severity weights used for the uncertainty score u_hat (severity-weighted
# abstention mass) -- higher severities contribute more to uncertainty.
SEVERITY_WEIGHT: dict[Severity, float] = {
    Severity.LOW: 1.0,
    Severity.MEDIUM: 2.0,
    Severity.HIGH: 3.0,
    Severity.CRITICAL: 4.0,
}

STATUS_VALID = "valid"
STATUS_INVALID = "invalid"
STATUS_UNCERTAIN = "uncertain"


# ---------------------------------------------------------------------------
# z_tau: aggregation (avg.tex Sec. 3.9)
# ---------------------------------------------------------------------------

def aggregate(htir: HTIR) -> AggregateResult:
    """
    Aggregate ``htir.obligations`` (already checked) into
    ``z_tau = (y_hat, u_hat, c_hat, eta_hat)``. Mutates ``htir.aggregate`` and
    returns the same result for chaining. Idempotent: fully recomputed from
    ``htir.obligations`` / ``htir.coverage`` on every call.
    """
    obligations = htir.obligations

    failed_high = [
        o for o in obligations if o.status == ObligationStatus.FAILED and o.severity in HIGH_SEVERITIES
    ]
    high_severity = [o for o in obligations if o.severity in HIGH_SEVERITIES]
    abstained_high = [o for o in high_severity if o.status == ObligationStatus.ABSTAINED]

    if failed_high:
        predicted_status = STATUS_INVALID
    elif high_severity and (
        len(abstained_high) >= UNCERTAIN_ABSTAIN_COUNT_THRESHOLD
        or (len(abstained_high) / len(high_severity)) >= UNCERTAIN_ABSTAIN_FRACTION_THRESHOLD
    ):
        predicted_status = STATUS_UNCERTAIN
    else:
        predicted_status = STATUS_VALID

    uncertainty = _uncertainty(obligations)
    evidence_coverage = _evidence_coverage(htir)
    aggregated_evidence_ids = sorted({
        ev_id for o in obligations if o.result is not None for ev_id in o.result.evidence_used
    })

    result = AggregateResult(
        predicted_status=predicted_status,
        uncertainty=uncertainty,
        evidence_coverage=evidence_coverage,
        aggregated_evidence_ids=aggregated_evidence_ids,
    )
    htir.aggregate = result
    return result


def _uncertainty(obligations: list[Obligation]) -> float:
    """Severity-weighted abstention mass in [0, 1]."""
    if not obligations:
        return 0.0
    total_weight = sum(SEVERITY_WEIGHT[o.severity] for o in obligations)
    if total_weight == 0.0:
        return 0.0
    abstained_weight = sum(
        SEVERITY_WEIGHT[o.severity] for o in obligations if o.status == ObligationStatus.ABSTAINED
    )
    return abstained_weight / total_weight


def _evidence_coverage(htir: HTIR) -> float:
    report = htir.coverage
    if report.total_obligations == 0:
        return 1.0
    return report.covered_obligations / report.total_obligations


# ---------------------------------------------------------------------------
# W_tau: verification witness (avg.tex Sec. 3.10)
# ---------------------------------------------------------------------------

def build_witness(htir: HTIR) -> VerificationWitness:
    """
    Build ``W_tau = (O+, O-, O-empty, E_W, R_W)`` from ``htir.obligations``
    (already checked; ``aggregate`` should have already run so
    ``htir.aggregate`` is available for the review recommendation, but this
    function tolerates ``htir.aggregate is None`` by aggregating on the fly).
    Mutates ``htir.witness`` and returns the same result for chaining.
    Idempotent: fully recomputed from ``htir.obligations`` on every call.
    """
    obligations = htir.obligations
    agg = htir.aggregate or aggregate(htir)

    passed = [o for o in obligations if o.status == ObligationStatus.PASSED]
    failed = [o for o in obligations if o.status == ObligationStatus.FAILED]
    abstained = [o for o in obligations if o.status == ObligationStatus.ABSTAINED]

    witness_evidence_ids: set[int] = set()
    for o in failed + abstained:
        if o.result is not None:
            witness_evidence_ids.update(o.result.evidence_used)

    vetoing = [o for o in failed if o.severity in HIGH_SEVERITIES]
    for o in vetoing:
        # Evidence behind the vetoing obligation(s), even if evidence_used
        # was empty for some reason -- fall back to the candidate evidence.
        witness_evidence_ids.update(o.candidate_evidence_ids)

    review_recommendation = _build_review_recommendation(agg, vetoing, failed, abstained)

    witness = VerificationWitness(
        passed_obligation_ids=sorted(o.obligation_id for o in passed),
        failed_obligation_ids=sorted(o.obligation_id for o in failed),
        abstained_obligation_ids=sorted(o.obligation_id for o in abstained),
        witness_evidence_ids=sorted(witness_evidence_ids),
        review_recommendation=review_recommendation,
    )
    htir.witness = witness
    return witness


def _build_review_recommendation(
    agg: AggregateResult, vetoing: list[Obligation], failed: list[Obligation], abstained: list[Obligation],
) -> str:
    """
    A short, deterministic (no LLM) template string: overall status, the
    vetoing obligation(s) if any, the count/kind of unresolved high-severity
    obligations, ending with "inspect: <single most important unresolved/
    failed obligation>".
    """
    parts = [f"Status: {agg.predicted_status}."]

    if vetoing:
        ids = ", ".join(str(o.obligation_id) for o in sorted(vetoing, key=lambda o: o.obligation_id))
        parts.append(f"Vetoed by {len(vetoing)} failed high-severity obligation(s): [{ids}].")

    unresolved_high = [o for o in abstained if o.severity in HIGH_SEVERITIES]
    if unresolved_high:
        parts.append(f"{len(unresolved_high)} unresolved high-severity obligation(s) abstained.")

    most_important = _most_important(vetoing, failed, abstained)
    if most_important is not None:
        label = most_important.template_id or most_important.description or f"#{most_important.obligation_id}"
        parts.append(f"Inspect: obligation {most_important.obligation_id} ({label}).")
    else:
        parts.append("Inspect: nothing outstanding.")

    return " ".join(parts)


_SEVERITY_ORDER = {Severity.CRITICAL: 3, Severity.HIGH: 2, Severity.MEDIUM: 1, Severity.LOW: 0}


def _most_important(
    vetoing: list[Obligation], failed: list[Obligation], abstained: list[Obligation],
) -> Obligation | None:
    """
    The single most important obligation to inspect: a vetoing obligation
    first, then any other failed obligation, then abstained obligations --
    each tier ordered by severity (descending) then obligation id (ascending)
    for determinism.
    """
    for tier in (vetoing, failed, abstained):
        if tier:
            return min(tier, key=lambda o: (-_SEVERITY_ORDER[o.severity], o.obligation_id))
    return None
