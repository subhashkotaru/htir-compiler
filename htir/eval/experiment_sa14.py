"""
SA-14 -- Track M: live harness-optimization baseline (Meta-Harness vs. a plain
Base Agent) on Terminal-Bench 2.0.

Named baselines the paper cites but never runs -- SkillOpt, Meta-Harness,
Life-Harness -- were previously deferred to Related Work (see
``docs/build-roadmap.md`` B4: "no runnable comparators exist"). This experiment
closes that gap for Meta-Harness (Lee et al. 2026, arXiv:2603.28052): its
released Terminal-Bench 2.0 scaffold ("meta-harness-tbench2-artifact") is a
*fixed, already-discovered* agent harness, so it can be run as a baseline arm
without reproducing its (expensive) outer-loop search.

Unlike SA-1's verifier arms (which re-score *already-recorded* trajectories),
this is a harness-optimization comparison: two *harnesses* (``meta_harness``,
Lee et al.'s discovered scaffold, vs. ``base_agent``, a plain baseline) are
each *executed* against live Terminal-Bench 2.0 tasks under a matched model
(default ``openai/gpt-4o-mini`` -- note this is an out-of-distribution
transfer test: Meta-Harness's scaffold was discovered/reported against Claude
Opus 4.6, not gpt-4o-mini, so a gap here is not directly comparable to their
paper's 76.4% headline). See ``scripts/live_meta_harness_tb2.py`` for the
capture step (external, live, costs real API/sandbox spend -- not run by this
module or by CI) and ``htir.eval.datasets.load_meta_harness_tb2`` for the
ingestion of its output.

Given captured live trajectories (grouped by the ``harness`` field), this
module reports, per harness:

* **Task-outcome metric** (Meta-Harness's own headline type): the real
  Terminal-Bench 2.0 task-success rate.
* **Verifier-quality metrics** (AVG's own headline type, as in SA-1): false-
  valid rate, resolved accuracy/fraction, and abstention rate for each
  requested verifier arm (default ``avg_full``, ``exec_only``, ``monolithic``),
  scored against that same real pass/fail outcome.

Rigor. Each captured trial is compiled + scored once; ``--seeds`` independent
**task-level bootstrap resamples** (not independent live reruns -- those would
require re-spending the live budget) give a mean +/- SE per metric, plus a
paired t-test (meta_harness - base_agent) on task-success rate and on
avg_full's false-valid rate, over the same per-seed resamples. This is an
honest resampling-based SE over one captured run, not a claim of independent
experimental replicates; the notes disclose this.

CLI::

    python -m htir.eval.experiment_sa14 \\
        --cache data/live_traces/meta_harness_tb2/meta_harness_gpt-4o-mini.jsonl \\
        --cache data/live_traces/meta_harness_tb2/base_agent_gpt-4o-mini.jsonl \\
        --seeds 0,1,2 --out data/sa14_results.json
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field

from htir.agents.baselines import VerifierArm, run_arm
from htir.agents.trace_abstraction import TraceAbstractionAgent
from htir.eval.datasets import iter_local_traces, load_meta_harness_tb2, to_canonical_steps
from htir.eval.seeds import MeanSE, PairedGap, mean_se, paired_t_test
from htir.eval.weak_labels import (
    VerifierMetrics,
    evaluate_predictions,
    extract_reward,
    label_from_reward,
)
from htir.models.domain import TERMINAL_DOMAIN_SPEC, DomainSpec, get_domain_spec

HARNESS_META = "meta_harness"
HARNESS_BASE = "base_agent"

DEFAULT_ARMS: list[VerifierArm] = [
    VerifierArm.AVG_FULL,
    VerifierArm.EXEC_ONLY,
    VerifierArm.MONOLITHIC,
]


# ---------------------------------------------------------------------------
# Per-trace scoring record (compiled + scored once, resampled across seeds)
# ---------------------------------------------------------------------------

class PerTraceRecord(BaseModel):
    """One captured trial: its task/harness/model, weak label, and each arm's verdict."""
    task_id: str = ""
    harness: str = "unknown"
    model: str = ""
    reward: Optional[int] = None
    label: Optional[str] = None
    n_steps: int = 0
    predicted: dict[str, str] = Field(default_factory=dict)


def score_traces(
    raw_traces: Iterable[dict[str, Any]],
    *,
    spec: DomainSpec,
    arms: list[VerifierArm],
    use_llm: bool = False,
    model: str = "openai/gpt-4o",
    progress_every: int = 100,
    log: Any = sys.stderr,
) -> list[PerTraceRecord]:
    """Compile each captured trial once and score every verifier ``arm`` over it."""
    agent = TraceAbstractionAgent(model=model, domain_spec=spec)
    records: list[PerTraceRecord] = []
    for i, raw in enumerate(raw_traces):
        reward = extract_reward(raw)
        task_id = str(raw.get("task_name", "")) if isinstance(raw, dict) else ""
        harness = str(raw.get("harness") or "unknown") if isinstance(raw, dict) else "unknown"
        trial_model = str(raw.get("model") or "") if isinstance(raw, dict) else ""
        try:
            steps = to_canonical_steps(raw)
            htir = agent.compile(
                task_id=task_id or f"trace-{i}",
                raw_steps=steps,
                harness_snippets={},
                generate_obligations=True,
                run_checks=False,
            )
        except Exception as exc:  # a malformed trace should not sink the run
            if log is not None:
                print(f"[sa14] skip trace {i} ({task_id}): {exc!r}", file=log)
            continue

        rec = PerTraceRecord(
            task_id=task_id or f"trace-{i}",
            harness=harness,
            model=trial_model,
            reward=reward,
            label=label_from_reward(reward),
            n_steps=len(htir.steps),
        )
        for arm in arms:
            agg = run_arm(htir, spec, arm, use_llm=use_llm, model=model)
            rec.predicted[arm.value] = agg.predicted_status
        records.append(rec)
        if progress_every and log is not None and (i + 1) % progress_every == 0:
            print(f"[sa14] compiled+scored {i + 1} trials...", file=log)
    return records


# ---------------------------------------------------------------------------
# Task-level bootstrap resampling (SE over one captured run, not live reruns)
# ---------------------------------------------------------------------------

def _by_task(records: list[PerTraceRecord]) -> dict[str, list[PerTraceRecord]]:
    out: dict[str, list[PerTraceRecord]] = defaultdict(list)
    for r in records:
        out[r.task_id].append(r)
    return out


def _resample_tasks(tasks: dict[str, list[PerTraceRecord]], *, seed: int) -> list[PerTraceRecord]:
    """
    One bootstrap resample: draw ``len(tasks)`` task ids with replacement and
    pool all of that task's captured trials (attempts). Resampling by *task*
    (not by trace) avoids pseudo-replication when a task has multiple harbor
    ``--n-attempts``.
    """
    task_ids = list(tasks)
    if not task_ids:
        return []
    rng = random.Random(seed)
    chosen = [task_ids[rng.randrange(len(task_ids))] for _ in range(len(task_ids))]
    return [r for tid in chosen for r in tasks[tid]]


def _metrics_for_pool(pool: list[PerTraceRecord], arm_names: list[str]) -> dict[str, float]:
    labels = [r.label for r in pool]
    out: dict[str, float] = {
        "task_success_rate": (
            sum(1 for r in pool if r.label == "valid") / len(labels) if labels else 0.0
        ),
    }
    for arm in arm_names:
        preds = [r.predicted.get(arm, "uncertain") for r in pool]
        m = evaluate_predictions(preds, labels)
        out[f"{arm}.false_valid_rate"] = m.false_valid_rate
        out[f"{arm}.resolved_accuracy"] = m.resolved_accuracy
        out[f"{arm}.resolved_fraction"] = m.resolved_fraction
        out[f"{arm}.abstention_rate"] = m.abstention_rate
    return out


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

class ArmMetrics(BaseModel):
    """One verifier arm's metrics for one harness, mean +/- SE over seeds."""
    arm: str
    false_valid_rate: MeanSE = Field(default_factory=MeanSE)
    resolved_accuracy: MeanSE = Field(default_factory=MeanSE)
    resolved_fraction: MeanSE = Field(default_factory=MeanSE)
    abstention_rate: MeanSE = Field(default_factory=MeanSE)


class HarnessReport(BaseModel):
    """Task-outcome + verifier-quality report for one harness variant."""
    harness: str
    model: str = ""
    n_traces: int = 0
    n_tasks: int = 0
    n_labeled: int = 0
    task_success_rate: MeanSE = Field(default_factory=MeanSE, description="Real TB2 pass-rate, task-bootstrapped")
    arms: list[ArmMetrics] = Field(default_factory=list)


class SA14Result(BaseModel):
    """Full SA-14 output: config, per-harness reports, and provenance."""
    experiment: str = "SA-14: Track M -- Meta-Harness vs. Base Agent (Terminal-Bench 2.0)"
    domain_id: str = ""
    use_llm: bool = False
    seeds: list[int] = Field(default_factory=list)
    seconds: float = 0.0
    harnesses: list[HarnessReport] = Field(default_factory=list)
    success_gap: PairedGap = Field(default_factory=PairedGap)
    false_valid_gap: PairedGap = Field(default_factory=PairedGap)
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_sa14(
    raw_traces: Iterable[dict[str, Any]],
    *,
    spec: DomainSpec | None = None,
    arms: list[VerifierArm] | None = None,
    use_llm: bool = False,
    seeds: list[int] | None = None,
    model: str = "openai/gpt-4o-mini",
    progress_every: int = 100,
    log: Any = sys.stderr,
) -> SA14Result:
    """
    Execute SA-14 over captured live Terminal-Bench 2.0 trials, grouped by their
    ``harness`` field (``meta_harness`` / ``base_agent``). Runs fully offline
    over already-captured data -- no live calls are issued here; see
    ``scripts/live_meta_harness_tb2.py`` for the capture step.
    """
    spec = spec or TERMINAL_DOMAIN_SPEC
    arms = arms or list(DEFAULT_ARMS)
    arm_names = [a.value for a in arms]
    seeds = seeds if seeds is not None else [0, 1, 2]
    t0 = time.time()

    records = score_traces(
        raw_traces, spec=spec, arms=arms, use_llm=use_llm, model=model,
        progress_every=progress_every, log=log,
    )

    by_harness: dict[str, list[PerTraceRecord]] = defaultdict(list)
    for r in records:
        by_harness[r.harness].append(r)

    per_seed_by_harness: dict[str, list[dict[str, float]]] = {}
    reports: list[HarnessReport] = []
    for harness, recs in sorted(by_harness.items()):
        tasks = _by_task(recs)
        per_seed = [_metrics_for_pool(_resample_tasks(tasks, seed=s), arm_names) for s in seeds]
        per_seed_by_harness[harness] = per_seed

        def _agg(metric: str) -> MeanSE:
            return mean_se([s[metric] for s in per_seed if metric in s])

        arm_metrics = [
            ArmMetrics(
                arm=arm,
                false_valid_rate=_agg(f"{arm}.false_valid_rate"),
                resolved_accuracy=_agg(f"{arm}.resolved_accuracy"),
                resolved_fraction=_agg(f"{arm}.resolved_fraction"),
                abstention_rate=_agg(f"{arm}.abstention_rate"),
            )
            for arm in arm_names
        ]
        n_labeled = sum(1 for r in recs if r.label is not None)
        models = {r.model for r in recs if r.model}
        reports.append(HarnessReport(
            harness=harness,
            model=next(iter(models), model) if len(models) <= 1 else "/".join(sorted(models)),
            n_traces=len(recs),
            n_tasks=len(tasks),
            n_labeled=n_labeled,
            task_success_rate=_agg("task_success_rate"),
            arms=arm_metrics,
        ))

    success_gap = PairedGap()
    false_valid_gap = PairedGap()
    if HARNESS_META in per_seed_by_harness and HARNESS_BASE in per_seed_by_harness:
        meta_seed = per_seed_by_harness[HARNESS_META]
        base_seed = per_seed_by_harness[HARNESS_BASE]
        success_gap = paired_t_test(
            [s["task_success_rate"] for s in meta_seed],
            [s["task_success_rate"] for s in base_seed],
            label="task_success_rate", a=HARNESS_META, b=HARNESS_BASE,
        )
        avg_key = f"{VerifierArm.AVG_FULL.value}.false_valid_rate"
        if all(avg_key in s for s in meta_seed) and all(avg_key in s for s in base_seed):
            false_valid_gap = paired_t_test(
                [s[avg_key] for s in meta_seed],
                [s[avg_key] for s in base_seed],
                label="avg_full.false_valid_rate", a=HARNESS_META, b=HARNESS_BASE,
            )

    notes = [
        "seeds are task-level BOOTSTRAP RESAMPLES of one captured live run, not "
        "independent live re-executions (each re-run would re-spend real API/sandbox "
        "budget); the reported SE therefore reflects resampling variance over the "
        "captured trials, not run-to-run variance of harbor/the model.",
        "task_success_rate is the real Terminal-Bench 2.0 outcome (Meta-Harness's own "
        "headline metric type); the arm rows are AVG's verifier-quality read on the same "
        "trials (SA-1's headline metric type), so the two halves can be sanity-checked "
        "against each other -- e.g. does AVG's false_valid_rate spike on the harness with "
        "a lower real success rate.",
        "Meta-Harness's scaffold was discovered/reported against Claude Opus 4.6 on "
        "Terminal-Bench 2.0 (76.4%); running it under a different model (default here: "
        "openai/gpt-4o-mini) is an out-of-distribution transfer test of the discovered "
        "harness, not a reproduction of their reported number.",
    ]
    if not use_llm:
        notes.append(
            "use_llm=False: avg_full collapses onto exec_only (mechanical evidence + "
            "abstention-aware aggregation) and monolithic is the endpoint heuristic, as in SA-1."
        )

    return SA14Result(
        domain_id=spec.domain_id,
        use_llm=use_llm,
        seeds=list(seeds),
        seconds=round(time.time() - t0, 2),
        harnesses=reports,
        success_gap=success_gap,
        false_valid_gap=false_valid_gap,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_table(result: SA14Result) -> str:
    """A compact fixed-width results table for the terminal / logs."""
    lines: list[str] = []
    lines.append(
        f"SA-14: Track M -- Meta-Harness vs. Base Agent (TB2)  |  "
        f"domain={result.domain_id}  use_llm={result.use_llm}  seeds={result.seeds}"
    )
    for h in result.harnesses:
        lines.append(
            f"  [{h.harness}]  model={h.model}  n={h.n_traces} (tasks={h.n_tasks}, "
            f"labeled={h.n_labeled})  task_success={h.task_success_rate.as_str()}"
        )
        lines.append(f"    {'arm':<12} {'false_valid':>13} {'res_acc':>13} {'res_frac':>13} {'abstain':>13}")
        for a in h.arms:
            lines.append(
                f"    {a.arm:<12} {a.false_valid_rate.as_str():>13} {a.resolved_accuracy.as_str():>13} "
                f"{a.resolved_fraction.as_str():>13} {a.abstention_rate.as_str():>13}"
            )
    if result.success_gap.n_seeds:
        lines.append(f"  [gap] task_success_rate: {result.success_gap.as_str()}")
    if result.false_valid_gap.n_seeds:
        lines.append(f"  [gap] avg_full.false_valid_rate: {result.false_valid_gap.as_str()}")
    for note in result.notes:
        lines.append(f"  note: {note}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SA-14: Track M -- Meta-Harness vs. Base Agent (Terminal-Bench 2.0)")
    p.add_argument("--cache", action="append", default=[],
                    help="local JSON/JSONL capture (repeat --cache for each harness's file)")
    p.add_argument("--domain", type=str, default="terminal_swe")
    p.add_argument("--arms", type=str, default="",
                    help="comma list of arms (default avg_full,exec_only,monolithic)")
    p.add_argument("--seeds", type=str, default="0,1,2", help="comma list of bootstrap seeds for mean±SE")
    p.add_argument("--use-llm", action="store_true", help="enable LLM monolith + semantic checker (needs key)")
    p.add_argument("--model", type=str, default="openai/gpt-4o-mini")
    p.add_argument("--out", type=str, default="", help="write SA14Result JSON here")
    args = p.parse_args(argv)

    if not args.cache:
        raise SystemExit("provide at least one --cache <jsonl> (one per captured harness file)")

    spec = get_domain_spec(args.domain)
    arms = [VerifierArm(a.strip()) for a in args.arms.split(",") if a.strip()] if args.arms else list(DEFAULT_ARMS)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    traces = load_meta_harness_tb2(args.cache)
    result = run_sa14(
        traces, spec=spec, arms=arms, use_llm=args.use_llm, seeds=seeds, model=args.model,
    )
    print(format_table(result))
    if args.out:
        Path(args.out).write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(f"\n[sa14] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
