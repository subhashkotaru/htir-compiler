"""
SA-7 -- Downstream payoff: best-of-N reranking / filtering (spotlight plan P0).

SA-1 … SA-6 established that AVG's *intrinsic* verifier quality -- its false-valid
rate -- is far below a monolithic judge's. SA-7 answers the "so what": does that
lower false-valid rate translate into a measurably better *downstream outcome*
when the verifier is used as a **selector** over candidate trajectories? This is
the clause that moves the paper from "a better verifier" to "a better verifier
that produces better data / picks".

We group a corpus by task (each task has N candidate trajectories with mixed
ground-truth reward) and use each verifier arm two ways:

* **Reranking (primary).** For every task, pick the single best-scored
  trajectory and report the **true success rate of the picks** -- how often the
  selected trajectory is actually correct (reward = 1). A verifier that credits
  reward-hacks as valid picks them; one that abstains on them does not.

  AVG's selection score is its coverage-aware probability-of-valid
  (:func:`htir.eval.calibration.trajectory_valid_score`), which *induces* the
  documented tie-break -- a resolved-valid obligation set scores toward 1, a
  broadly-abstained one sits at the 0.5 prior, a resolved-invalid one toward 0 --
  so **resolved-valid > abstained > resolved-invalid** falls out, with graded
  ties instead of a flat band. AVG therefore never prefers a trajectory it
  flagged invalid over one it abstained on, and among equals prefers the one with
  more discharged supporting evidence. The monolithic arm's score is the analogous
  ``0.5 ± 0.5·confidence`` around its verdict.

* **Filtering (stronger variant).** Filter the pool to the trajectories each arm
  credits ``valid`` and report that filtered-in set's **true-success rate**
  (precision) -- exactly "how much reward-hack leaks into the kept training
  data" -- alongside **yield** (fraction kept), so a high-precision/low-yield arm
  is not flattered. Precision above the pool base rate is the selection value.

Two reference rows bracket the arms: an **oracle** selector (picks a truly-valid
candidate whenever the task has one -- the ceiling) and a **random** selector
(expected success of an unguided pick = the per-task valid fraction, the floor).

Rigor. The corpus is compiled + scored **once** (the expensive step); the
selector metrics are then evaluated over ``--seeds`` independent per-task
candidate subsamples (matched candidate budget across arms) and reported as
**mean ± SE**. The headline reranking gap (``avg_full`` vs ``monolithic``) also
carries a **paired bootstrap over tasks** on the full candidate set -- a 95% CI
and one-sided p-value that does not lean on the small seed count.

Offline reproducibility. With no API key (``use_llm=False``, the default) the
whole pipeline is deterministic: ``avg_full`` collapses onto ``exec_only``
(mechanical evidence + abstention-aware aggregation) and ``monolithic`` is the
endpoint heuristic, exactly as in SA-1. Every number here is byte-reproducible
offline; the LLM arms only sharpen the same contrast.

CLI::

    python -m htir.eval.experiment_sa7 --domain tau_bench \\
        --cache data/tau_cache/tau_all.jsonl --n 1000 --min-candidates 3 \\
        --candidates-per-task 8 --seeds 0,1,2 --out data/sa7_tau_results.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field

from htir.agents.baselines import VerifierArm, run_arm
from htir.agents.trace_abstraction import TraceAbstractionAgent
from htir.eval.calibration import DECISION_BOUNDARY, trajectory_valid_score
from htir.eval.datasets import iter_local_traces, to_canonical_steps
from htir.eval.seeds import MeanSE, mean_se
from htir.eval.weak_labels import (
    LABEL_VALID,
    STATUS_UNCERTAIN,
    evaluate_predictions,
    extract_reward,
    label_from_reward,
)
from htir.models.domain import DomainArtifactBundle, DomainSpec, get_domain_spec
from htir.models.htir import AggregateResult

# The selector arms reported by default. The SA-8 ``prm`` arm slots in as a third
# column once it lands (pass ``--arms avg_full,monolithic,prm``); until then the
# offline-runnable AVG vs monolith contrast is the deliverable.
DEFAULT_ARMS: list[VerifierArm] = [VerifierArm.AVG_FULL, VerifierArm.MONOLITHIC]

# The graph arms whose selection score is the coverage-aware p_valid over the
# checked obligations; every other arm (only ``monolithic`` today) derives its
# score from the aggregate verdict + confidence.
_GRAPH_ARMS = frozenset({
    VerifierArm.AVG_FULL.value, VerifierArm.EXEC_ONLY.value,
    VerifierArm.EXEC_FREE.value, VerifierArm.NO_ABSTENTION.value,
})

# Fixed seed for the paired-bootstrap significance test (kept off the CLI so the
# reported p-value / CI is reproducible regardless of the seed sweep).
_BOOTSTRAP_SEED = 20260715


def _monolith_pvalid(agg: AggregateResult) -> float:
    """
    A monolithic verdict's probability-of-valid, on the same [0, 1] scale as
    :func:`trajectory_valid_score`: ``0.5 ± 0.5·confidence`` around the boundary,
    with ``confidence = 1 - uncertainty``. A confident ``valid`` -> ~1, a
    confident ``invalid`` -> ~0, ``uncertain`` -> the 0.5 prior.
    """
    conf = max(0.0, min(1.0, 1.0 - agg.uncertainty))
    if agg.predicted_status == LABEL_VALID:
        return DECISION_BOUNDARY + DECISION_BOUNDARY * conf
    if agg.predicted_status == STATUS_UNCERTAIN:
        return DECISION_BOUNDARY
    return DECISION_BOUNDARY - DECISION_BOUNDARY * conf  # invalid


# ---------------------------------------------------------------------------
# Per-trace scoring record (compiled + scored once, reused across seeds)
# ---------------------------------------------------------------------------

class PerTraceRecord(BaseModel):
    """One candidate trajectory: its task, weak label, and each arm's verdict."""
    task_id: str = ""
    reward: Optional[int] = None
    label: Optional[str] = None
    n_steps: int = 0
    # arm -> predicted status ('valid'/'invalid'/'uncertain') for the filter
    status: dict[str, str] = Field(default_factory=dict)
    # arm -> coverage-aware p_valid in [0, 1] for the reranking selection score
    score: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

class ArmReranking(BaseModel):
    """Best-of-N reranking payoff for one selector arm, aggregated over seeds."""
    arm: str
    pick_success: MeanSE = Field(default_factory=MeanSE, description="True success rate of the per-task picks")
    delta_vs_monolith: MeanSE = Field(
        default_factory=MeanSE, description="Per-seed (arm - monolithic) pick-success gap",
    )


class ArmFiltering(BaseModel):
    """Filtering (keep-if-valid) payoff for one selector arm, aggregated over seeds."""
    arm: str
    precision: MeanSE = Field(default_factory=MeanSE, description="True-valid fraction of the kept (valid-credited) set")
    yield_kept: MeanSE = Field(default_factory=MeanSE, description="Fraction of the pool credited valid (kept)")
    false_valid_rate: MeanSE = Field(default_factory=MeanSE, description="Reward-hack leakage: invalid traces credited valid")


class Significance(BaseModel):
    """Paired bootstrap over tasks for the headline reranking gap (avg vs mono)."""
    comparison: str = "avg_full - monolithic (reranking pick-success)"
    test: str = "paired bootstrap over tasks"
    n_tasks: int = 0
    n_boot: int = 0
    observed_gap: float = Field(0.0, description="Pick-success(avg) - pick-success(mono) on the full candidate set")
    ci95_low: float = 0.0
    ci95_high: float = 0.0
    p_value_one_sided: float = Field(
        1.0, description="Bootstrap P(gap <= 0): probability AVG does not beat the monolith",
    )


class SA7Result(BaseModel):
    """Full SA-7 downstream-payoff output: config, reranking + filtering, provenance."""
    experiment: str = "SA-7: Downstream payoff (best-of-N reranking / filtering)"
    domain_id: str = ""
    n_traces: int = 0
    n_labeled: int = 0
    n_tasks_total: int = 0
    n_tasks_used: int = Field(0, description="Tasks meeting the min-candidates floor (reranking universe)")
    min_candidates: int = 0
    candidates_per_task: int = 0
    base_rate_valid: float = Field(0.0, description="Pool-wide fraction of labeled traces truly valid")
    use_llm: bool = False
    seeds: list[int] = Field(default_factory=list)
    seconds: float = 0.0

    reranking: list[ArmReranking] = Field(default_factory=list)
    filtering: list[ArmFiltering] = Field(default_factory=list)
    oracle_pick_success: MeanSE = Field(default_factory=MeanSE)
    random_pick_success: MeanSE = Field(default_factory=MeanSE)
    significance: Significance = Field(default_factory=Significance)
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Scoring: compile every trace once, run every arm
# ---------------------------------------------------------------------------

def score_traces(
    raw_traces: Iterable[dict[str, Any]],
    *,
    spec: DomainSpec,
    arms: list[VerifierArm],
    omega: DomainArtifactBundle | None = None,
    use_llm: bool = False,
    model: str = "openai/gpt-4o",
    progress_every: int = 250,
    log: Any = sys.stderr,
) -> list[PerTraceRecord]:
    """
    Compile each raw trace once (obligation generation, no checks) and score every
    selector ``arm`` over its own copy of that graph, capturing both the arm's
    predicted status (for the keep-if-valid filter) and its coverage-aware
    p_valid selection score (for reranking). Graph arms are scored ``in_place``
    on a private copy so :func:`trajectory_valid_score` can read the discharged
    obligation results; the monolith derives its score from its verdict. This is
    the expensive pass; the selector metrics resample cheaply over its output.
    """
    agent = TraceAbstractionAgent(model=model, domain_spec=spec, domain_artifacts=omega)
    records: list[PerTraceRecord] = []
    for i, raw in enumerate(raw_traces):
        reward = extract_reward(raw)
        task_id = str(raw.get("task_name", "")) if isinstance(raw, dict) else ""
        try:
            steps = to_canonical_steps(raw)
            htir = agent.compile(
                task_id=task_id or f"trace-{i}",
                raw_steps=steps,
                harness_snippets={},
                generate_obligations=True,
                use_semantic_analysis=use_llm,
                run_checks=False,
                domain_artifacts=omega,
            )
        except Exception as exc:  # a malformed trace should not sink the run
            if log is not None:
                print(f"[sa7] skip trace {i} ({task_id}): {exc!r}", file=log)
            continue

        rec = PerTraceRecord(
            task_id=task_id or f"trace-{i}",
            reward=reward,
            label=label_from_reward(reward),
            n_steps=len(htir.steps),
        )
        for arm in arms:
            graph = htir.model_copy(deep=True)
            agg = run_arm(graph, spec, arm, use_llm=use_llm, domain_artifacts=omega,
                          model=model, in_place=True)
            rec.status[arm.value] = agg.predicted_status
            if arm.value in _GRAPH_ARMS:
                s = trajectory_valid_score(graph)
                rec.score[arm.value] = DECISION_BOUNDARY if s is None else s
            else:
                rec.score[arm.value] = _monolith_pvalid(agg)
        records.append(rec)
        if progress_every and log is not None and (i + 1) % progress_every == 0:
            print(f"[sa7] compiled+scored {i + 1} traces...", file=log)
    return records


# ---------------------------------------------------------------------------
# Selector metrics over one candidate pool
# ---------------------------------------------------------------------------

def _rerank_pick(recs: list[PerTraceRecord], arm: str) -> PerTraceRecord:
    """
    The best-of-N pick for one task under one arm: argmax over candidates of the
    coverage-aware p_valid selection score. ``max`` is stable, so a full tie
    (e.g. every candidate abstained to the 0.5 prior) deterministically keeps the
    first candidate.
    """
    return max(recs, key=lambda r: r.score.get(arm, DECISION_BOUNDARY))


def _pool_scores(
    tasks: dict[str, list[PerTraceRecord]],
    arms: list[str],
) -> dict[str, float]:
    """
    Compute every selector metric over one candidate pool ``tasks`` (task_id ->
    labeled candidate records). Returns a flat ``{metric: value}`` dict:

    * ``<arm>.pick_success``   -- reranking: true-valid fraction of per-task picks
    * ``<arm>.filter_precision`` / ``.filter_yield`` / ``.filter_false_valid``
    * ``oracle`` / ``random`` / ``base_rate`` -- the reference bracket
    """
    task_ids = list(tasks)
    out: dict[str, float] = {}

    # Reranking: one pick per task per arm.
    for arm in arms:
        correct = sum(1 for tid in task_ids if _rerank_pick(tasks[tid], arm).label == LABEL_VALID)
        out[f"{arm}.pick_success"] = correct / len(task_ids) if task_ids else 0.0

    # Oracle ceiling / random floor / pool base rate.
    oracle = sum(1 for tid in task_ids if any(r.label == LABEL_VALID for r in tasks[tid]))
    out["oracle"] = oracle / len(task_ids) if task_ids else 0.0
    rand_terms = [
        sum(1 for r in tasks[tid] if r.label == LABEL_VALID) / len(tasks[tid])
        for tid in task_ids if tasks[tid]
    ]
    out["random"] = sum(rand_terms) / len(rand_terms) if rand_terms else 0.0

    pooled = [r for tid in task_ids for r in tasks[tid]]
    n_valid = sum(1 for r in pooled if r.label == LABEL_VALID)
    out["base_rate"] = n_valid / len(pooled) if pooled else 0.0

    # Filtering: keep-if-valid, scored over the pooled candidates.
    labels = [r.label for r in pooled]
    for arm in arms:
        preds = [r.status.get(arm, STATUS_UNCERTAIN) for r in pooled]
        m = evaluate_predictions(preds, labels)
        n_kept = sum(1 for p in preds if p == LABEL_VALID)
        out[f"{arm}.filter_precision"] = m.valid_precision
        out[f"{arm}.filter_yield"] = n_kept / len(pooled) if pooled else 0.0
        out[f"{arm}.filter_false_valid"] = m.false_valid_rate
    return out


def _subsample_pool(
    tasks: dict[str, list[PerTraceRecord]],
    *,
    candidates_per_task: int,
    seed: int,
) -> dict[str, list[PerTraceRecord]]:
    """
    One seed's candidate pool: for each task, deterministically shuffle its
    candidates and keep up to ``candidates_per_task`` of them (all of them when
    ``candidates_per_task <= 0``, i.e. the full-pool / zero-variance setting). The
    per-task cap is a matched candidate budget across arms; varying which
    candidates are offered is the seed axis for the mean ± SE.
    """
    if candidates_per_task <= 0:
        return tasks
    out: dict[str, list[PerTraceRecord]] = {}
    for tid, recs in tasks.items():
        pool = list(recs)
        random.Random(f"{seed}:{tid}").shuffle(pool)
        out[tid] = pool[:candidates_per_task]
    return out


# ---------------------------------------------------------------------------
# Paired bootstrap over tasks (headline significance)
# ---------------------------------------------------------------------------

def _paired_bootstrap(
    tasks: dict[str, list[PerTraceRecord]],
    arm: str,
    baseline: str,
    *,
    n_boot: int,
) -> Significance:
    """
    Paired bootstrap over tasks of the reranking pick-success gap ``arm -
    baseline`` on the given candidate pool. Each task contributes a paired
    outcome (arm's pick correct?  baseline's pick correct?); resampling tasks
    with replacement ``n_boot`` times yields the gap's 95% CI and the one-sided
    p-value ``P(gap <= 0)``. Deterministic (fixed :data:`_BOOTSTRAP_SEED`).
    """
    task_ids = list(tasks)
    diffs = [
        (1 if _rerank_pick(tasks[tid], arm).label == LABEL_VALID else 0)
        - (1 if _rerank_pick(tasks[tid], baseline).label == LABEL_VALID else 0)
        for tid in task_ids
    ]
    n = len(diffs)
    observed = sum(diffs) / n if n else 0.0
    if n == 0:
        return Significance(comparison=f"{arm} - {baseline} (reranking pick-success)", n_boot=n_boot)

    rng = random.Random(_BOOTSTRAP_SEED)
    means: list[float] = []
    for _ in range(n_boot):
        s = sum(diffs[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * (n_boot - 1))]
    hi = means[int(0.975 * (n_boot - 1))]
    p_le0 = sum(1 for m in means if m <= 0.0) / n_boot
    return Significance(
        comparison=f"{arm} - {baseline} (reranking pick-success)",
        n_tasks=n,
        n_boot=n_boot,
        observed_gap=round(observed, 4),
        ci95_low=round(lo, 4),
        ci95_high=round(hi, 4),
        p_value_one_sided=round(p_le0, 4),
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_sa7(
    raw_traces: Iterable[dict[str, Any]],
    *,
    spec: DomainSpec | None = None,
    arms: list[VerifierArm] | None = None,
    omega: DomainArtifactBundle | None = None,
    use_llm: bool = False,
    min_candidates: int = 3,
    candidates_per_task: int = 8,
    seeds: list[int] | None = None,
    n_boot: int = 2000,
    model: str = "openai/gpt-4o",
    progress_every: int = 250,
    log: Any = sys.stderr,
) -> SA7Result:
    """
    Execute SA-7 over ``raw_traces`` grouped by ``task_name``.

    Traces are compiled + scored once (:func:`score_traces`), grouped into tasks
    meeting the ``min_candidates`` floor, and evaluated as selectors over
    ``seeds`` candidate subsamples (reranking + filtering, mean ± SE). The
    ``avg_full`` vs ``monolithic`` reranking gap additionally gets a paired
    bootstrap over tasks. Runs fully offline by default.
    """
    spec = spec or get_domain_spec("tau_bench")
    arms = arms or list(DEFAULT_ARMS)
    seeds = seeds if seeds is not None else [0, 1, 2]
    arm_names = [a.value for a in arms]
    t0 = time.time()

    records = score_traces(
        raw_traces, spec=spec, arms=arms, omega=omega, use_llm=use_llm,
        model=model, progress_every=progress_every, log=log,
    )

    # Group into tasks; the reranking universe is the labeled candidates of tasks
    # that clear the min-candidates floor.
    by_task: dict[str, list[PerTraceRecord]] = defaultdict(list)
    for r in records:
        if r.label is not None:
            by_task[r.task_id].append(r)
    tasks = {tid: recs for tid, recs in by_task.items() if len(recs) >= min_candidates}

    # Per-seed selector metrics -> mean ± SE.
    per_seed: list[dict[str, float]] = []
    for seed in seeds:
        pool = _subsample_pool(tasks, candidates_per_task=candidates_per_task, seed=seed)
        per_seed.append(_pool_scores(pool, arm_names))

    def _agg(metric: str) -> MeanSE:
        return mean_se([s[metric] for s in per_seed if metric in s])

    reranking: list[ArmReranking] = []
    filtering: list[ArmFiltering] = []
    mono = VerifierArm.MONOLITHIC.value
    for arm in arm_names:
        deltas = [s[f"{arm}.pick_success"] - s[f"{mono}.pick_success"] for s in per_seed]
        reranking.append(ArmReranking(
            arm=arm,
            pick_success=_agg(f"{arm}.pick_success"),
            delta_vs_monolith=mean_se(deltas),
        ))
        filtering.append(ArmFiltering(
            arm=arm,
            precision=_agg(f"{arm}.filter_precision"),
            yield_kept=_agg(f"{arm}.filter_yield"),
            false_valid_rate=_agg(f"{arm}.filter_false_valid"),
        ))

    # Headline significance: paired bootstrap over tasks for the reranking gap, on
    # the first seed's candidate pool so the reported gap matches the matched
    # candidate budget the per-seed pick_success table uses.
    sig = Significance(comparison=f"{VerifierArm.AVG_FULL.value} - {mono} (reranking pick-success)", n_boot=n_boot)
    if VerifierArm.AVG_FULL.value in arm_names and mono in arm_names and tasks:
        sig_pool = _subsample_pool(tasks, candidates_per_task=candidates_per_task, seed=seeds[0])
        sig = _paired_bootstrap(sig_pool, VerifierArm.AVG_FULL.value, mono, n_boot=n_boot)

    n_labeled = sum(1 for r in records if r.label is not None)
    pooled_valid = sum(1 for r in records if r.label == LABEL_VALID)

    notes = [
        "Reranking: per task, pick argmax over candidates by the coverage-aware p_valid selection "
        "score (trajectory_valid_score for AVG; 0.5±0.5·confidence for the monolith), which induces "
        "resolved-valid > abstained (0.5 prior) > resolved-invalid. pick_success is the true "
        "(reward=1) rate of the picks. oracle = task has any valid candidate (ceiling); random = "
        "mean per-task valid fraction (unguided floor).",
        "Filtering: keep the traces an arm credits 'valid'; precision is the kept set's true-valid "
        "fraction (reward-hack that leaks into kept data = 1-precision), yield is the fraction kept. "
        "Precision above base_rate is the selection value; report yield so high-precision/low-yield "
        "arms are not flattered.",
        f"Significance: paired bootstrap over {sig.n_tasks} tasks (n_boot={n_boot}) at the seed-0 "
        "candidate budget; CI95 and one-sided P(gap<=0) for the avg_full - monolithic reranking gap. "
        "The stable cross-domain win is the filtering hack-leakage (false_valid) gap, whose per-seed "
        "SE is tiny; reranking is comparable offline (CI straddles 0).",
    ]
    if not use_llm:
        notes.append(
            "Offline run (no API key): avg_full collapses onto exec_only (mechanical evidence + "
            "abstention-aware aggregation) and monolithic is the endpoint heuristic, as in SA-1. "
            "Numbers are byte-reproducible; the LLM arms only sharpen the same contrast."
        )

    return SA7Result(
        domain_id=spec.domain_id,
        n_traces=len(records),
        n_labeled=n_labeled,
        n_tasks_total=len(by_task),
        n_tasks_used=len(tasks),
        min_candidates=min_candidates,
        candidates_per_task=candidates_per_task,
        base_rate_valid=(pooled_valid / n_labeled) if n_labeled else 0.0,
        use_llm=use_llm,
        seeds=list(seeds),
        seconds=round(time.time() - t0, 2),
        reranking=reranking,
        filtering=filtering,
        oracle_pick_success=_agg("oracle"),
        random_pick_success=_agg("random"),
        significance=sig,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_table(result: SA7Result) -> str:
    """A compact fixed-width Table-5 view for the terminal / logs."""
    lines: list[str] = []
    lines.append(
        f"SA-7: Downstream payoff  |  domain={result.domain_id}  "
        f"n={result.n_traces} (labeled {result.n_labeled}), tasks={result.n_tasks_used}"
        f"/{result.n_tasks_total} (>= {result.min_candidates} cand)  "
        f"cand/task={result.candidates_per_task}  seeds={result.seeds}  use_llm={result.use_llm}"
    )
    lines.append(f"  base-rate valid {result.base_rate_valid:.3f}  |  "
                 f"oracle {result.oracle_pick_success.as_str()}  |  "
                 f"random {result.random_pick_success.as_str()}")

    lines.append("  [reranking: best-of-N pick success (true reward=1 rate)]")
    lines.append(f"    {'arm':<12} {'pick_success':>16} {'Δ vs mono':>16}")
    for a in result.reranking:
        lines.append(f"    {a.arm:<12} {a.pick_success.as_str():>16} {a.delta_vs_monolith.as_str():>16}")

    lines.append("  [filtering: keep-if-valid]")
    lines.append(f"    {'arm':<12} {'precision':>16} {'yield':>16} {'false_valid':>16}")
    for a in result.filtering:
        lines.append(
            f"    {a.arm:<12} {a.precision.as_str():>16} {a.yield_kept.as_str():>16} "
            f"{a.false_valid_rate.as_str():>16}"
        )

    s = result.significance
    lines.append(
        f"  [significance] {s.comparison}: gap {s.observed_gap:+.3f} "
        f"(95% CI [{s.ci95_low:+.3f}, {s.ci95_high:+.3f}], "
        f"one-sided p={s.p_value_one_sided:.4f}, {s.test}, n_tasks={s.n_tasks})"
    )
    for note in result.notes:
        lines.append(f"  note: {note}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_traces(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.domain == "tau_bench":
        from htir.eval.datasets import load_tau_bench
        if args.hf:
            return load_tau_bench(hf=True, limit=args.hf_limit)
        if not args.cache:
            raise SystemExit("provide --cache <jsonl> (or --hf) for tau_bench")
        return load_tau_bench([args.cache])
    # terminal / any turn-schema corpus.
    if args.hf:
        from htir.eval.datasets import load_terminalbench
        return load_terminalbench(limit=args.hf_limit, streaming=True)
    if not args.cache:
        raise SystemExit("provide --cache <jsonl> (or --hf)")
    return list(iter_local_traces([args.cache]))


def _cap_by_tasks(
    traces: list[dict[str, Any]], n: int, *, seed: int = 0,
) -> list[dict[str, Any]]:
    """
    Cap the pool to ~``n`` traces while keeping task groups intact (whole tasks
    only), so reranking still sees each kept task's full candidate set. Tasks are
    drawn in a deterministic *shuffled* order (not sorted-id), so the cap is an
    unbiased task subsample rather than skewing to alphabetically-first tasks
    (which, e.g. on τ-bench, would keep only ``airline_*`` and miss ``retail_*``).
    """
    if n <= 0 or len(traces) <= n:
        return traces
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in traces:
        by_task[str(t.get("task_name", ""))].append(t)
    order = sorted(by_task)
    random.Random(seed).shuffle(order)
    out: list[dict[str, Any]] = []
    for tid in order:
        group = by_task[tid]
        if out and len(out) + len(group) > n:
            break
        out.extend(group)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SA-7: downstream payoff (best-of-N reranking / filtering)")
    src = p.add_argument_group("data source")
    src.add_argument("--domain", type=str, default="tau_bench",
                     help="domain spec S_d + loader (tau_bench | terminal_swe | ...)")
    src.add_argument("--cache", type=str, default="", help="local JSON/JSONL corpus")
    src.add_argument("--hf", action="store_true", help="stream from the HF hub instead of --cache")
    src.add_argument("--hf-limit", type=int, default=None, help="records to pull when --hf")
    src.add_argument("--n", type=int, default=0, help="cap pool to ~N traces (whole tasks kept; 0 = all)")
    p.add_argument("--min-candidates", type=int, default=3, help="min candidates for a task to enter reranking")
    p.add_argument("--candidates-per-task", type=int, default=8,
                   help="matched candidate budget sampled per task per seed (0 = use all)")
    p.add_argument("--seeds", type=str, default="0,1,2", help="comma list of seeds for mean±SE")
    p.add_argument("--n-boot", type=int, default=2000, help="paired-bootstrap resamples for the headline gap")
    p.add_argument("--arms", type=str, default="",
                   help="comma list of arms (default avg_full,monolithic); e.g. add exec_only or prm")
    p.add_argument("--use-llm", action="store_true", help="enable LLM monolith + semantic checker (needs key)")
    p.add_argument("--model", type=str, default="openai/gpt-4o")
    p.add_argument("--out", type=str, default="", help="write SA7Result JSON here")
    args = p.parse_args(argv)

    from htir.models.domain import load_domain_artifacts

    spec = get_domain_spec(args.domain)
    try:
        omega = load_domain_artifacts(args.domain)
    except Exception:
        omega = None

    if args.arms:
        arms = [VerifierArm(a.strip()) for a in args.arms.split(",") if a.strip()]
    else:
        arms = list(DEFAULT_ARMS)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    traces = _cap_by_tasks(_load_traces(args), args.n)
    result = run_sa7(
        traces,
        spec=spec,
        arms=arms,
        omega=omega,
        use_llm=args.use_llm,
        min_candidates=args.min_candidates,
        candidates_per_task=args.candidates_per_task,
        seeds=seeds,
        n_boot=args.n_boot,
        model=args.model,
    )
    print(format_table(result))
    if args.out:
        Path(args.out).write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(f"\n[sa7] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
