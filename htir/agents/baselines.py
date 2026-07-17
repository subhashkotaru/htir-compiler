"""
Verifier baselines / arms (avg.tex Sec. 4.3, "Baselines"; Sec. 4.5 ablations
1 and 3).

The experiments compare full \\AVG against several reduced verifiers. Rather
than reimplement the pipeline per arm, this module expresses each arm as a
configuration over the *same* compiled graph:

* ``AVG_FULL``  -- graph obligations, mechanical + semantic checkers.
* ``EXEC_ONLY`` -- graph obligations, mechanical checkers only (no semantic).
* ``EXEC_FREE`` -- graph obligations, semantic checkers only (no execution/
  mechanical evidence): the execution-free ablation.
* ``NO_ABSTENTION`` -- full-AVG graph + evidence, but every checker is forced
  to emit pass/fail (``force_decision``): the no-abstention ablation (Sec. 4.5
  #3) used by SA-3 to isolate the effect of calibrated abstention.
* ``MONOLITHIC`` -- the anti-thesis baseline: a *single* scalar judge over the
  whole trajectory, with no obligation graph, no evidence localization, and no
  abstention. This is what \\AVG's factorization is meant to beat.
* ``PRM`` -- a **process reward model** baseline (SA-8): score every step and
  aggregate (min/mean threshold) to a trajectory verdict. Offline it is a
  deterministic step-heuristic scorer (from parsed execution signals); with
  ``use_llm`` an LLM step-critic scores each step. A PRM must score *every*
  step, including weak-label ones with no clear signal, so it *over-commits*
  (never abstains) -- the failure mode AVG's calibrated abstention avoids.
* ``AGENT_JUDGE`` -- an **Agent-as-a-Judge** baseline (SA-8): a judge that may
  *gather* evidence (multi-hop over step outcomes / artifacts) before
  committing, vs. the one-shot ``MONOLITHIC`` judge. It is LLM-backed; offline
  (no key) it degrades to a deterministic execution-evidence gather that still
  commits valid/invalid -- so it is still fooled by plausible-but-invalid long
  traces whose visible steps all pass. The richer multi-hop reasoning
  (reading artifacts/policies) needs a key; the offline fallback is honest
  about being a proxy, and it never crashes without one.

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
    PRM = "prm"
    AGENT_JUDGE = "agent_judge"


# Flag configuration for the graph-based arms. ``force_decision`` is the
# no-abstention ablation (avg.tex Sec. 4.5 #3): the same graph + evidence as
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
    if arm == VerifierArm.PRM:
        return prm_verdict(graph, use_llm=use_llm, model=model)
    if arm == VerifierArm.AGENT_JUDGE:
        return agent_judge_verdict(graph, use_llm=use_llm, model=model)

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


# ---------------------------------------------------------------------------
# Process reward model (PRM) baseline (SA-8)
# ---------------------------------------------------------------------------
#
# A process/step reward model scores every step and aggregates to a trajectory
# verdict. The competitive-baseline point (avg.tex SA-8) is that a PRM must
# assign a score to *every* step -- including weak-label steps with no clear
# execution signal -- and then commit, so it *over-commits*: it never abstains,
# and it credits a structurally-clean-but-actually-failed trajectory ``valid``.
# AVG's calibrated abstention is exactly the discipline a PRM lacks.

# Deterministic per-step reward for the offline step-heuristic PRM, keyed on the
# already-parsed execution status (no LLM, byte-deterministic). UNKNOWN is the
# neutral prior a PRM is forced to emit when a step carries no execution signal.
_PRM_STEP_SCORE: dict[ExecutionStatus, float] = {
    ExecutionStatus.SUCCESS: 1.0,
    ExecutionStatus.FAILURE: 0.0,
    ExecutionStatus.TIMEOUT: 0.0,
    ExecutionStatus.BLOCKED: 0.0,
    ExecutionStatus.UNKNOWN: 0.5,
}
# Shallow error markers that pull an otherwise-unknown step's score down -- a
# deterministic stand-in for a learned step critic reading the step's output.
_PRM_ERROR_MARKERS = ("traceback", "error:", "exception", "assertionerror", "fatal:", " failed")
_PRM_DEFAULT_THRESHOLD = 0.5


def _prm_step_score(step) -> float:
    """Deterministic reward in [0, 1] for one step (offline step-heuristic PRM)."""
    base = _PRM_STEP_SCORE.get(step.execution_status, 0.5)
    if step.execution_status == ExecutionStatus.UNKNOWN:
        text = (step.response_message or "").lower()
        if any(m in text for m in _PRM_ERROR_MARKERS):
            return 0.25
    return base


def prm_verdict(
    htir: HTIR, *,
    aggregation: str = "mean",
    threshold: float = _PRM_DEFAULT_THRESHOLD,
    use_llm: bool = False,
    model: str = DEFAULT_MODEL,
) -> AggregateResult:
    """
    Process reward model verdict: score each step, aggregate (``mean`` or
    ``min``), threshold to a trajectory status (avg.tex SA-8).

    Deterministic (default): score every step from its parsed execution signal
    (:func:`_prm_step_score`) and threshold the aggregate. The PRM always
    commits (``valid``/``invalid``, never ``uncertain``) -- the over-commitment
    a PRM cannot avoid. With ``use_llm=True`` an LLM step-critic scores each
    step instead; any LLM failure falls back to the deterministic heuristic so
    the offline path is intact.
    """
    if use_llm:
        verdict = _llm_prm_verdict(htir, aggregation=aggregation, threshold=threshold, model=model)
        if verdict is not None:
            return verdict

    steps = htir.steps_in_order()
    scores = [_prm_step_score(s) for s in steps]
    if not scores:
        # No steps to score at all: the PRM has nothing to commit on.
        return AggregateResult(
            predicted_status=STATUS_UNCERTAIN, uncertainty=1.0,
            evidence_coverage=0.0, aggregated_evidence_ids=[],
        )

    agg = min(scores) if aggregation == "min" else (sum(scores) / len(scores))
    status = STATUS_VALID if agg >= threshold else STATUS_INVALID
    # Confidence is the margin from the decision boundary; a PRM commits, so its
    # uncertainty never reaches the abstain-worthy range AVG would flag.
    uncertainty = max(0.0, 1.0 - 2.0 * abs(agg - 0.5))
    return AggregateResult(
        predicted_status=status,
        uncertainty=uncertainty,
        evidence_coverage=0.0,
        aggregated_evidence_ids=[],
    )


def _llm_prm_verdict(
    htir: HTIR, *, aggregation: str, threshold: float, model: str,
) -> AggregateResult | None:
    """
    LLM step-critic PRM: one narrow call per step scoring its quality in [0, 1],
    aggregated exactly like the deterministic heuristic. Returns ``None`` (so the
    caller falls back to the heuristic) if the LLM/openai extra/API key is
    unavailable, keeping the offline path intact.
    """
    from pydantic import BaseModel, Field

    class StepScore(BaseModel):
        score: float = Field(0.5, description="quality of this step in [0,1]")

    try:
        from htir.utils.llm import chat_json, system, user
    except Exception:
        return None

    steps = htir.steps_in_order()
    if not steps:
        return None
    scores: list[float] = []
    for s in steps:
        msgs = [
            system(
                "You are a process reward model. Score the quality of ONE agent "
                "step in [0,1]: 1.0 = clearly correct/productive, 0.0 = clearly "
                "wrong/harmful. Judge only this step."
            ),
            user(
                f"Step ({s.role}/{s.execution_status.value}): "
                f"{truncate(s.request_message, 200)} -> {truncate(s.response_message, 400)}"
            ),
        ]
        try:
            v = chat_json(msgs, StepScore, model=model)
        except (EnvironmentError, ImportError):
            return None
        except Exception:
            return None
        scores.append(max(0.0, min(1.0, v.score)))

    agg = min(scores) if aggregation == "min" else (sum(scores) / len(scores))
    status = STATUS_VALID if agg >= threshold else STATUS_INVALID
    uncertainty = max(0.0, 1.0 - 2.0 * abs(agg - 0.5))
    return AggregateResult(
        predicted_status=status, uncertainty=uncertainty,
        evidence_coverage=0.0, aggregated_evidence_ids=[],
    )


# ---------------------------------------------------------------------------
# Agent-as-a-Judge baseline (SA-8)
# ---------------------------------------------------------------------------
#
# An Agent-as-a-Judge may *gather* evidence (multi-hop over steps/artifacts)
# before committing to a verdict, unlike the one-shot MONOLITHIC judge. It is
# an LLM-backed baseline; offline (no key) it degrades to a deterministic
# execution-evidence gather that still emits a single committed verdict -- so it
# is still fooled by plausible-but-invalid long traces whose visible steps all
# pass. The separation from the monolith (and its real multi-hop reasoning over
# artifacts/policies) needs a key; the offline fallback is an honest proxy and
# never crashes without one.


def agent_judge_verdict(
    htir: HTIR, *, use_llm: bool = False, model: str = DEFAULT_MODEL, max_hops: int = 3,
) -> AggregateResult:
    """
    Agent-as-a-Judge verdict (avg.tex SA-8). With ``use_llm`` and a key, an LLM
    judge gathers evidence over up to ``max_hops`` rounds before committing;
    without a key it falls back to :func:`_evidence_gather_verdict`, a
    deterministic multi-hop scan of step outcomes that still commits.
    """
    if use_llm:
        verdict = _llm_agent_judge_verdict(htir, max_hops=max_hops, model=model)
        if verdict is not None:
            return verdict
    return _evidence_gather_verdict(htir)


def _evidence_gather_verdict(htir: HTIR) -> AggregateResult:
    """
    Deterministic evidence-gathering judge: scan *all* observable step outcomes
    (multi-hop, not just the endpoint the monolith reads) and commit a verdict.

    Unlike the monolith it flags ``invalid`` when a failing step is never
    resolved by a later success anywhere in the trace; but lacking the
    obligation-graph abstention discipline it still *commits* ``valid`` on a
    structurally-clean trace whose visible steps all pass -- the plausible-but-
    invalid long trace it is fooled by.
    """
    steps = htir.steps_in_order()
    observable = [s for s in steps if s.execution_status != ExecutionStatus.UNKNOWN]
    if not observable:
        return AggregateResult(
            predicted_status=STATUS_UNCERTAIN, uncertainty=1.0,
            evidence_coverage=0.0, aggregated_evidence_ids=[],
        )

    # Multi-hop scan: is there a failing step with no later success to resolve it?
    unresolved_failure = False
    for i, s in enumerate(observable):
        if s.execution_status in _FAILING:
            if not any(o.execution_status == ExecutionStatus.SUCCESS for o in observable[i + 1:]):
                unresolved_failure = True
                break

    last = observable[-1]
    if last.execution_status in _FAILING or unresolved_failure:
        status = STATUS_INVALID
    elif last.execution_status == ExecutionStatus.SUCCESS:
        status = STATUS_VALID
    else:
        status = STATUS_UNCERTAIN

    coverage = len(observable) / len(steps) if steps else 0.0
    return AggregateResult(
        predicted_status=status,
        uncertainty=0.0,
        evidence_coverage=coverage,
        aggregated_evidence_ids=[],
    )


def _llm_agent_judge_verdict(htir: HTIR, *, max_hops: int, model: str) -> AggregateResult | None:
    """
    LLM Agent-as-a-Judge: the judge is shown the trace and may request up to
    ``max_hops`` targeted evidence pulls (specific step/artifact detail) before
    committing a verdict. Returns ``None`` (caller falls back to the
    deterministic gather) if the LLM/openai extra/API key is unavailable.

    This is a single-call approximation of the agentic loop (the trace is
    provided up front with an instruction to reason multi-hop) to keep the token
    budget matched to the monolithic judge and AVG's semantic checker; a fuller
    tool-calling loop is left for the LLM campaign.
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
            "You are an Agent-as-a-Judge. Before committing, gather evidence by "
            "reasoning multi-hop over the trace: check that each claimed success is "
            "supported by a produced artifact and a real validation, and that no "
            "required step was skipped or its failure left unresolved. Then return "
            "one verdict: 'valid', 'invalid', or 'uncertain'. Do not trust the "
            "final step's outcome alone."
        ),
        user(f"Trace:\n{truncate(transcript, 6000)}\n\nGather evidence, then return status and confidence."),
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
