"""
Online intervention iota_t (AVG Step 7, avg.tex Sec. 3.11 "Online
Intervention").

During execution, AVG monitors obligations on the *partial* graph G_{tau<=t}
and, when a high-severity obligation fails or abstains, the harness chooses
an intervention

    iota_t in {accept, request-evidence, rerank, veto, repair, clarify, escalate}

selected by

    iota_t* = argmax_iota E[r_hat_t^iota - beta*Cost(iota) - gamma*Risk(iota)]

This module implements that loop over a *recorded* trace replayed prefix by
prefix (see ``TraceAbstractionAgent.compile_prefix``), not a live agent --
this keeps it testable and deterministic, per the Step-7 handoff.

Entry points:

* ``active_obligations(htir_prefix)`` -- the high-severity obligations that
  are failing or abstaining in the current partial graph.
* ``select_intervention(obligation, htir_prefix)`` -- picks iota_t for one
  active obligation. First cut: a deterministic policy driven by
  ``Obligation.escalation`` (alpha_i, fixed at obligation-generation time) --
  the per-obligation escalation rule *is* the default iota. This is computed
  through the paper's argmax form via pluggable ``benefit``/``cost``/``risk``
  functions with simple constant defaults, so the interface already matches
  the eventual learned-estimator upgrade without changing callers.
* ``run_intervention_loop(agent, task_id, raw_steps, harness_snippets, ...)``
  -- replays the whole trace prefix by prefix, recording an
  ``InterventionLogEntry`` for every active obligation at every step.

Purely a recommendation trace: nothing here drives an agent. Keeps the
alpha_i (fixed, per-obligation) vs. iota_t (chosen online, per-step)
distinction documented at ``htir/models/htir.py:118-154``.
"""

from __future__ import annotations

from typing import Any, Callable

from htir.agents.witness import HIGH_SEVERITIES
from htir.models.htir import (
    HTIR,
    InterventionAction,
    InterventionLogEntry,
    Obligation,
    ObligationStatus,
)

BenefitFn = Callable[[InterventionAction, Obligation, HTIR], float]
CostFn = Callable[[InterventionAction], float]
RiskFn = Callable[[InterventionAction], float]

# Cost/risk weights in the paper's iota_t* = argmax E[r_hat - beta*Cost - gamma*Risk].
BETA = 1.0
GAMMA = 1.0

# Simple constant defaults for Cost/Risk, pluggable via select_intervention's
# cost_fn/risk_fn -- placeholders for a learned estimator, not tuned values.
# Ordered roughly by how disruptive/irreversible the action is.
_DEFAULT_COST: dict[InterventionAction, float] = {
    InterventionAction.ACCEPT: 0.0,
    InterventionAction.REQUEST_EVIDENCE: 0.1,
    InterventionAction.RERANK: 0.1,
    InterventionAction.CLARIFY: 0.2,
    InterventionAction.REPAIR: 0.3,
    InterventionAction.ESCALATE: 0.5,
    InterventionAction.VETO: 0.4,
}
_DEFAULT_RISK: dict[InterventionAction, float] = {
    InterventionAction.ACCEPT: 0.0,
    InterventionAction.REQUEST_EVIDENCE: 0.0,
    InterventionAction.RERANK: 0.05,
    InterventionAction.CLARIFY: 0.05,
    InterventionAction.REPAIR: 0.1,
    InterventionAction.ESCALATE: 0.1,
    InterventionAction.VETO: 0.2,
}


def _default_benefit(action: InterventionAction, obligation: Obligation, htir_prefix: HTIR) -> float:
    """
    r_hat_t^iota placeholder: 1.0 for the obligation's own escalation rule
    (alpha_i), 0.0 for every other action. Until a learned benefit estimator
    exists, this makes the argmax reduce to "follow alpha_i" -- the
    documented first cut -- while keeping the same call signature a learned
    estimator would use (it may consult ``htir_prefix`` for richer context).
    """
    return 1.0 if action.value == obligation.escalation.value else 0.0


def _default_cost(action: InterventionAction) -> float:
    return _DEFAULT_COST.get(action, 0.0)


def _default_risk(action: InterventionAction) -> float:
    return _DEFAULT_RISK.get(action, 0.0)


def active_obligations(htir_prefix: HTIR) -> list[Obligation]:
    """
    The active obligation set at step t (avg.tex Sec. 3.11): high-severity
    (HIGH/CRITICAL) obligations that are FAILED or ABSTAINED in the current
    partial graph ``htir_prefix`` (G_{tau<=t}). Obligations must already be
    checked (e.g. via ``TraceAbstractionAgent.compile_prefix``, which runs
    ``check_obligations`` by default) -- a still-PENDING obligation is not
    "active", it simply hasn't been evaluated yet.
    """
    return [
        o for o in htir_prefix.obligations
        if o.severity in HIGH_SEVERITIES and o.status in (ObligationStatus.FAILED, ObligationStatus.ABSTAINED)
    ]


def select_intervention(
    obligation: Obligation,
    htir_prefix: HTIR,
    *,
    benefit_fn: BenefitFn = _default_benefit,
    cost_fn: CostFn = _default_cost,
    risk_fn: RiskFn = _default_risk,
    beta: float = BETA,
    gamma: float = GAMMA,
) -> InterventionAction:
    """
    iota_t* = argmax_iota E[r_hat - beta*Cost(iota) - gamma*Risk(iota)]
    (avg.tex Sec. 3.11), evaluated over every ``InterventionAction``. With the
    default ``benefit_fn``/``cost_fn``/``risk_fn`` this deterministically
    picks ``obligation.escalation`` (alpha_i): swapping in a learned
    ``benefit_fn`` upgrades the policy without changing this interface.
    """
    def score(action: InterventionAction) -> float:
        return benefit_fn(action, obligation, htir_prefix) - beta * cost_fn(action) - gamma * risk_fn(action)

    return max(InterventionAction, key=score)


def _rationale(obligation: Obligation, action: InterventionAction) -> str:
    label = obligation.template_id or obligation.description or f"obligation #{obligation.obligation_id}"
    return (
        f"Obligation {obligation.obligation_id} ({label}) is {obligation.status.value} "
        f"(severity={obligation.severity.value}); escalation alpha_i={obligation.escalation.value} "
        f"-> iota_t={action.value}."
    )


def run_intervention_loop(
    agent: Any,
    task_id: str,
    raw_steps: list[dict[str, Any]],
    harness_snippets: dict[str, str],
    **compile_kwargs: Any,
) -> HTIR:
    """
    Replay ``raw_steps`` prefix by prefix over a *recorded* trace (not a live
    agent), computing the active obligation set and the selected
    intervention at every step t (avg.tex Sec. 3.11). Returns the fully
    compiled ``HTIR`` (over the whole trace) with ``intervention_log``
    populated with one ``InterventionLogEntry`` per (step, active obligation)
    pair observed along the way. Purely a recommendation trace -- does not
    drive an agent.

    ``agent`` is a ``TraceAbstractionAgent``; ``compile_kwargs`` are forwarded
    to ``agent.compile_prefix`` at every step (e.g. ``domain_artifacts``,
    ``use_semantic_analysis``).
    """
    log: list[InterventionLogEntry] = []
    final_htir: HTIR | None = None

    for t in range(1, len(raw_steps) + 1):
        htir_prefix = agent.compile_prefix(task_id, raw_steps, harness_snippets, t, **compile_kwargs)
        for ob in active_obligations(htir_prefix):
            action = select_intervention(ob, htir_prefix)
            log.append(
                InterventionLogEntry(
                    step_id=t,
                    obligation_id=ob.obligation_id,
                    action=action,
                    rationale=_rationale(ob, action),
                )
            )
        final_htir = htir_prefix

    if final_htir is None:
        final_htir = agent.compile_prefix(task_id, raw_steps, harness_snippets, 0, **compile_kwargs)
    final_htir.intervention_log = log
    return final_htir
