"""
Verifier baselines / arms (avg.tex Sec. 4.3, "Baselines"; Sec. 4.6 ablations
1 and 3).

The experiments compare full \\AVG against several reduced verifiers. Rather
than reimplement the pipeline per arm, this module expresses each arm as a
configuration over the *same* compiled graph:

* ``AVG_FULL``  -- graph obligations, mechanical + semantic checkers.
* ``EXEC_ONLY`` -- graph obligations, mechanical checkers only (no semantic).
* ``EXEC_FREE`` -- graph obligations, semantic checkers only (no execution/
  mechanical evidence): the execution-free ablation.
* ``NO_ABSTENTION`` -- full-AVG graph + evidence, but every checker is forced
  to emit pass/fail (``force_decision``): the no-abstention ablation (Sec. 4.6
  #3) used by SA-3 to isolate the effect of calibrated abstention.
* ``MONOLITHIC`` -- the anti-thesis baseline: a *single* scalar judge over the
  whole trajectory, with no obligation graph, no evidence localization, and no
  abstention. This is what \\AVG's factorization is meant to beat.

The three graph arms reuse ``check_obligations`` (gated by its ``use_semantic``
/ ``disable_mechanical`` flags) and ``aggregate``. The monolithic arm is a
deliberately endpoint-oriented judge: deterministically it trusts the last
observable validation outcome (the blind spot AVG exposes), or, with
``use_llm=True`` and an API key, one LLM pass over the truncated trace.

Every arm returns an ``AggregateResult`` so downstream metrics (false-valid
rate, resolved accuracy, ...) compare arms uniformly. ``run_arm`` works on a
copy of the graph by default, so running several arms over one trace does not
let one arm's checker write-backs leak into another.
"""

from __future__ import annotations

from enum import Enum

from htir.agents.checking import check_obligations
from htir.agents.witness import STATUS_INVALID, STATUS_UNCERTAIN, STATUS_VALID, aggregate
from htir.models.domain import DomainArtifactBundle, DomainSpec
from htir.models.htir import (
    AggregateResult,
    ExecutionStatus,
    HTIR,
)
from htir.utils.io import truncate
from htir.utils.llm import DEFAULT_MODEL

# Roles (substring, case-insensitive) that represent an observable validation
# whose outcome an endpoint judge would trust.
_VALIDATION_ROLE_HINTS = ("test", "validation")
_FAILING = (ExecutionStatus.FAILURE, ExecutionStatus.TIMEOUT, ExecutionStatus.BLOCKED)


class VerifierArm(str, Enum):
    AVG_FULL = "avg_full"
    EXEC_ONLY = "exec_only"
    EXEC_FREE = "exec_free"
    NO_ABSTENTION = "no_abstention"
    MONOLITHIC = "monolithic"


# Flag configuration for the graph-based arms. ``force_decision`` is the
# no-abstention ablation (avg.tex Sec. 4.6 #3): the same graph + evidence as
# full AVG, but every checker must commit to pass/fail instead of abstaining.
_ARM_FLAGS: dict[VerifierArm, dict[str, bool]] = {
    VerifierArm.AVG_FULL: {"use_semantic": True, "disable_mechanical": False, "force_decision": False},
    VerifierArm.EXEC_ONLY: {"use_semantic": False, "disable_mechanical": False, "force_decision": False},
    VerifierArm.EXEC_FREE: {"use_semantic": True, "disable_mechanical": True, "force_decision": False},
    VerifierArm.NO_ABSTENTION: {"use_semantic": True, "disable_mechanical": False, "force_decision": True},
}


def run_arm(
    htir: HTIR,
    spec: DomainSpec,
    arm: VerifierArm,
    *,
    use_llm: bool = False,
    domain_artifacts: DomainArtifactBundle | None = None,
    model: str = DEFAULT_MODEL,
    in_place: bool = False,
) -> AggregateResult:
    """
    Evaluate ``htir`` (already compiled through obligation generation) under a
    single verifier ``arm`` and return its ``AggregateResult``.

    By default this operates on a deep copy so the arm's checker write-backs do
    not mutate the caller's graph (set ``in_place=True`` to opt out). ``use_llm``
    gates the semantic checker (AVG_FULL / EXEC_FREE) and the monolithic LLM
    judge; without it those fall back to abstain / the deterministic endpoint
    heuristic so every arm runs fully offline.
    """
    graph = htir if in_place else htir.model_copy(deep=True)

    if arm == VerifierArm.MONOLITHIC:
        return monolithic_judge(graph, use_llm=use_llm, model=model)

    flags = _ARM_FLAGS[arm]
    check_obligations(
        graph, spec,
        use_semantic=flags["use_semantic"] and use_llm,
        disable_mechanical=flags["disable_mechanical"],
        force_decision=flags["force_decision"],
        domain_artifacts=domain_artifacts,
        model=model,
    )
    return aggregate(graph)


def run_all_arms(
    htir: HTIR,
    spec: DomainSpec,
    *,
    arms: list[VerifierArm] | None = None,
    use_llm: bool = False,
    domain_artifacts: DomainArtifactBundle | None = None,
    model: str = DEFAULT_MODEL,
) -> dict[VerifierArm, AggregateResult]:
    """Run several arms over one trace (each on its own copy) for comparison."""
    arms = arms or list(VerifierArm)
    return {
        arm: run_arm(htir, spec, arm, use_llm=use_llm, domain_artifacts=domain_artifacts, model=model)
        for arm in arms
    }


# ---------------------------------------------------------------------------
# Monolithic single-scalar judge (the baseline AVG is meant to beat)
# ---------------------------------------------------------------------------

def monolithic_judge(
    htir: HTIR, *, use_llm: bool = False, model: str = DEFAULT_MODEL,
) -> AggregateResult:
    """
    A single scalar verdict over the whole trajectory -- no obligation graph,
    no evidence localization, no abstention (avg.tex Sec. 4.3 monolithic
    baseline; ablation 1, "graph vs. monolith").

    Deterministic (default): trust the *last observable validation outcome*,
    the endpoint-only heuristic a monolithic judge tends to collapse to -- a
    passing final test reads as ``valid`` even when required process steps were
    skipped, which is exactly the failure mode AVG's factorization surfaces.
    With ``use_llm=True`` and an API key, one LLM pass over the truncated trace
    produces the verdict instead; any LLM failure falls back to the heuristic.
    """
    if use_llm:
        verdict = _llm_monolithic_verdict(htir, model=model)
        if verdict is not None:
            return verdict
    return _endpoint_monolithic_verdict(htir)


def _endpoint_monolithic_verdict(htir: HTIR) -> AggregateResult:
    steps = htir.steps_in_order()

    def _is_validation(role: str) -> bool:
        r = role.lower()
        return any(h in r for h in _VALIDATION_ROLE_HINTS)

    # Prefer the last validation step with a known status; else the last step
    # with any known status (still endpoint-oriented, no localization).
    validations = [s for s in steps if _is_validation(s.role) and s.execution_status != ExecutionStatus.UNKNOWN]
    fallback = [s for s in steps if s.execution_status != ExecutionStatus.UNKNOWN]
    ranked = validations or fallback

    if not ranked:
        status = STATUS_UNCERTAIN
    else:
        last = ranked[-1]
        if last.execution_status == ExecutionStatus.SUCCESS:
            status = STATUS_VALID
        elif last.execution_status in _FAILING:
            status = STATUS_INVALID
        else:
            status = STATUS_UNCERTAIN

    # A monolith reports no evidence coverage and no per-obligation uncertainty
    # (it has no obligation graph); uncertainty is 1.0 only when it had nothing
    # observable to judge.
    return AggregateResult(
        predicted_status=status,
        uncertainty=0.0 if ranked else 1.0,
        evidence_coverage=0.0,
        aggregated_evidence_ids=[],
    )


def _llm_monolithic_verdict(htir: HTIR, *, model: str) -> AggregateResult | None:
    """
    One LLM pass over the truncated trace -> valid/invalid/uncertain. Returns
    ``None`` (so the caller falls back to the deterministic heuristic) if the
    LLM/openai extra/API key is unavailable, keeping the offline path intact.
    """
    from pydantic import BaseModel, Field

    class Verdict(BaseModel):
        status: str = Field("uncertain", description="valid | invalid | uncertain")
        confidence: float = 0.0

    try:
        from htir.utils.llm import chat_json, system, user
    except Exception:
        return None

    transcript = "\n".join(
        f"[{s.step_id}] ({s.role}/{s.execution_status.value}) "
        f"{truncate(s.request_message, 200)} -> {truncate(s.response_message, 400)}"
        for s in htir.steps_in_order()
    )
    msgs = [
        system(
            "You are a single monolithic trajectory judge. Read the whole agent "
            "trace and return one overall verdict: 'valid' (task correctly "
            "completed), 'invalid' (incorrect/violated), or 'uncertain'. You do "
            "not get per-step evidence; judge holistically."
        ),
        user(f"Trace:\n{truncate(transcript, 6000)}\n\nReturn status and confidence."),
    ]
    try:
        v = chat_json(msgs, Verdict, model=model)
    except (EnvironmentError, ImportError):
        return None
    except Exception:
        return None

    status = v.status.strip().lower()
    if status not in (STATUS_VALID, STATUS_INVALID, STATUS_UNCERTAIN):
        status = STATUS_UNCERTAIN
    conf = max(0.0, min(1.0, v.confidence))
    return AggregateResult(
        predicted_status=status,
        uncertainty=1.0 - conf,
        evidence_coverage=0.0,
        aggregated_evidence_ids=[],
    )
