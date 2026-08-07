"""
SA-15 -- Track S: SkillOpt-trained skill vs. no-skill baseline on
Terminal-Bench 2.0.

Companion to SA-14 (Track M / Meta-Harness). SkillOpt (Yang et al. 2026,
arXiv:2605.23904) trains a portable ``best_skill.md`` via a held-out-gated
text-space loop; there is no pre-discovered Terminal-Bench artifact to just
run (unlike Meta-Harness), so Track S's live half is the training driver
``scripts/skillopt_train_tb2.py`` plus a matched ``no_skill`` capture.

This module is the *offline* analysis half: given captured live trajectories
grouped by ``harness`` in ``{skillopt, no_skill}``, it reports the same
task-outcome + verifier-quality metrics as SA-14. Capture / training are
deliberately not invoked here.

CLI::

    python -m htir.eval.experiment_sa15 \\
        --cache data/live_traces/skillopt_tb2/skillopt_openai_gpt-4o-mini.jsonl \\
        --cache data/live_traces/skillopt_tb2/no_skill_openai_gpt-4o-mini.jsonl \\
        --seeds 0,1,2 --out data/sa15_results.json
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field

from htir.agents.baselines import VerifierArm
from htir.eval.datasets import load_skillopt_tb2
from htir.eval.experiment_sa14 import (
    DEFAULT_ARMS,
    ArmMetrics,
    HarnessReport,
    PerTraceRecord,
    _by_task,
    _metrics_for_pool,
    _resample_tasks,
    score_traces,
)
from htir.eval.seeds import MeanSE, PairedGap, mean_se, paired_t_test
from htir.models.domain import TERMINAL_DOMAIN_SPEC, DomainSpec, get_domain_spec

HARNESS_SKILLOPT = "skillopt"
HARNESS_NO_SKILL = "no_skill"


class SA15Result(BaseModel):
    """Track S offline analysis output (SkillOpt vs. no-skill on TB2)."""
    experiment: str = "SA-15: Track S -- SkillOpt vs. No-Skill (TB2)"
    domain_id: str = "terminal_swe"
    use_llm: bool = False
    seeds: list[int] = Field(default_factory=list)
    seconds: float = 0.0
    harnesses: list[HarnessReport] = Field(default_factory=list)
    success_gap: PairedGap = Field(default_factory=PairedGap)
    false_valid_gap: PairedGap = Field(default_factory=PairedGap)
    notes: list[str] = Field(default_factory=list)


def run_sa15(
    raw_traces: Iterable[dict[str, Any]],
    *,
    spec: DomainSpec | None = None,
    arms: list[VerifierArm] | None = None,
    use_llm: bool = False,
    seeds: list[int] | None = None,
    model: str = "openai/gpt-4o-mini",
    progress_every: int = 100,
    log: Any = sys.stderr,
) -> SA15Result:
    """
    Execute SA-15 over captured live TB2 trials grouped by ``harness``
    (``skillopt`` / ``no_skill``). Fully offline over already-captured data.
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

        def _agg(metric: str, _per_seed: list[dict[str, float]] = per_seed) -> MeanSE:
            return mean_se([s[metric] for s in _per_seed if metric in s])

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
    if HARNESS_SKILLOPT in per_seed_by_harness and HARNESS_NO_SKILL in per_seed_by_harness:
        skill_seed = per_seed_by_harness[HARNESS_SKILLOPT]
        base_seed = per_seed_by_harness[HARNESS_NO_SKILL]
        success_gap = paired_t_test(
            [s["task_success_rate"] for s in skill_seed],
            [s["task_success_rate"] for s in base_seed],
            label="task_success_rate", a=HARNESS_SKILLOPT, b=HARNESS_NO_SKILL,
        )
        avg_key = f"{VerifierArm.AVG_FULL.value}.false_valid_rate"
        if all(avg_key in s for s in skill_seed) and all(avg_key in s for s in base_seed):
            false_valid_gap = paired_t_test(
                [s[avg_key] for s in skill_seed],
                [s[avg_key] for s in base_seed],
                label="avg_full.false_valid_rate", a=HARNESS_SKILLOPT, b=HARNESS_NO_SKILL,
            )

    notes = [
        "seeds are task-level BOOTSTRAP RESAMPLES of one captured live run, not "
        "independent live re-executions.",
        "skillopt harness = Terminal-Bench 2.0 trials run with SkillOpt's "
        "exported best_skill.md injected via harbor --skill; no_skill = the "
        "same agent/model/tasks with no skill injected.",
        "Training (scripts/skillopt_train_tb2.py) is separate from this offline "
        "analysis module and is not invoked here.",
    ]
    if not use_llm:
        notes.append(
            "use_llm=False: avg_full collapses onto exec_only; monolithic is the "
            "endpoint heuristic (same as SA-1 / SA-14 offline)."
        )

    return SA15Result(
        domain_id=spec.domain_id,
        use_llm=use_llm,
        seeds=list(seeds),
        seconds=round(time.time() - t0, 2),
        harnesses=reports,
        success_gap=success_gap,
        false_valid_gap=false_valid_gap,
        notes=notes,
    )


def format_sa15(result: SA15Result) -> str:
    lines = [
        f"SA-15: Track S -- SkillOpt vs. No-Skill (TB2)  |  "
        f"domain={result.domain_id}  use_llm={result.use_llm}  seeds={result.seeds}"
    ]
    for h in result.harnesses:
        lines.append(
            f"  [{h.harness}]  model={h.model}  n={h.n_traces} "
            f"(tasks={h.n_tasks}, labeled={h.n_labeled})  "
            f"task_success={h.task_success_rate.mean:.3f}±{h.task_success_rate.se:.3f}"
        )
        lines.append(
            f"    {'arm':14s} {'false_valid':>12s} {'res_acc':>12s} "
            f"{'res_frac':>12s} {'abstain':>12s}"
        )
        for a in h.arms:
            lines.append(
                f"    {a.arm:14s} "
                f"{a.false_valid_rate.mean:6.3f}±{a.false_valid_rate.se:<5.3f} "
                f"{a.resolved_accuracy.mean:6.3f}±{a.resolved_accuracy.se:<5.3f} "
                f"{a.resolved_fraction.mean:6.3f}±{a.resolved_fraction.se:<5.3f} "
                f"{a.abstention_rate.mean:6.3f}±{a.abstention_rate.se:<5.3f}"
            )
    if result.success_gap.n_seeds:
        g = result.success_gap
        lines.append(
            f"  paired gap task_success ({g.a} - {g.b}): "
            f"{g.mean_diff:+.3f}±{g.se_diff:.3f}  t={g.t_stat:.2f} p={g.p_value:.4g}"
        )
    for note in result.notes:
        lines.append(f"  note: {note}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SA-15: SkillOpt vs. no-skill on TB2")
    p.add_argument("--cache", action="append", default=[],
                   help="local JSON/JSONL capture (repeat --cache for each harness file)")
    p.add_argument("--domain", type=str, default="terminal_swe")
    p.add_argument("--arms", type=str, default="",
                   help="comma list of arms (default avg_full,exec_only,monolithic)")
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--use-llm", action="store_true")
    p.add_argument("--model", type=str, default="openai/gpt-4o-mini")
    p.add_argument("--out", type=str, default="")
    args = p.parse_args(argv)

    if not args.cache:
        raise SystemExit("provide at least one --cache <jsonl>")

    spec = get_domain_spec(args.domain)
    arms = (
        [VerifierArm(a.strip()) for a in args.arms.split(",") if a.strip()]
        if args.arms.strip() else None
    )
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    traces = load_skillopt_tb2(args.cache)
    result = run_sa15(
        traces, spec=spec, arms=arms, use_llm=args.use_llm, seeds=seeds, model=args.model,
    )
    print(format_sa15(result))
    if args.out:
        Path(args.out).write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
        print(f"\n[sa15] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
