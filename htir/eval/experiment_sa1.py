"""
SA-1 -- Q1: Graph vs. Monolith (verifier factorization).

Runs the experiment from ``docs/experiment-plan.md`` (SA-1) / ``avg.tex`` Sec. 4
(question Q1, ablation #1). It compiles a balanced Terminal-Bench sample once
per trace, then scores four verifier arms over the *same* compiled graph:

* ``avg_full``   -- graph obligations, mechanical + semantic checkers (full AVG).
* ``exec_only``  -- graph obligations, mechanical checkers only.
* ``exec_free``  -- graph obligations, semantic checkers only (no execution).
* ``monolithic`` -- a single scalar judge over the whole trace (no graph, no
  evidence localization, no abstention) -- the baseline AVG is meant to beat.

Each arm's ``predicted_status`` is scored against the weak reward label
(``reward in {0, 1}``) with :func:`htir.eval.weak_labels.evaluate_predictions`.
The headline contrast is **false-valid rate**: how often a failed trajectory
(reward = 0) is nonetheless credited ``valid``. We also report resolved
accuracy vs. the 50% balanced base rate, failure-flag precision/recall, and
abstention rate, overall and on a long-horizon slice (Q1 predicts the graph's
advantage grows with horizon).

Offline reproducibility. With no API key (``use_llm=False``, the default) the
whole pipeline is deterministic: the semantic checker abstains rather than
calling an LLM, so ``avg_full`` collapses onto ``exec_only`` (mechanical
evidence only) and ``exec_free`` abstains everywhere. The meaningful offline
contrast is therefore **AVG (mechanical + abstention-aware aggregation) vs.
monolithic (endpoint heuristic)**; the semantic arms need a key to separate.
This equivalence is reported explicitly rather than hidden.

Cost. Because the offline run issues zero real LLM calls, we report a
*cost proxy* -- the number of LLM invocations each arm would issue if its
semantic checker ran: 0 for ``exec_only``; one narrow claim-evidence call per
SEMANTIC-routed obligation for ``avg_full`` / ``exec_free``; and one
full-trace judge call for ``monolithic``. This is the compute axis for the
cost-normalized performance curve (avg.tex Sec. 4.7).

CLI::

    python -m htir.eval.experiment_sa1 --cache sa1_sample.jsonl --out sa1_results.json
    python -m htir.eval.experiment_sa1 --hf --n 3000 --out sa1_results.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field

from htir.agents.baselines import VerifierArm, run_arm
from htir.agents.trace_abstraction import TraceAbstractionAgent
from htir.eval.datasets import iter_local_traces, to_canonical_steps
from htir.eval.seeds import PairedGap, format_aggregate, paired_t_test, run_multiseed
from htir.eval.weak_labels import (
    VerifierMetrics,
    evaluate_predictions,
    extract_reward,
    label_from_reward,
)
from htir.models.domain import TERMINAL_DOMAIN_SPEC, DomainSpec, get_domain_spec
from htir.models.htir import CheckerType, HTIR

# Arms reported, in the order they appear in the results table. SA-8 adds the
# two competitive-baseline categories the 2026 field demands -- a process reward
# model (``prm``) and an Agent-as-a-Judge (``agent_judge``) -- alongside the
# strawman monolith.
DEFAULT_ARMS: list[VerifierArm] = [
    VerifierArm.AVG_FULL,
    VerifierArm.EXEC_ONLY,
    VerifierArm.EXEC_FREE,
    VerifierArm.MONOLITHIC,
    VerifierArm.PRM,
    VerifierArm.AGENT_JUDGE,
]

# A trajectory is "long-horizon" (Q1's stress regime) if it has at least this
# many operation steps. The default is the median-ish knee for Terminal-Bench
# turn traces; overridable from the CLI.
DEFAULT_LONG_HORIZON_STEPS = 20


class PerTraceRecord(BaseModel):
    """One trace's label, size, and each arm's predicted status + cost proxy."""
    task_id: str = ""
    reward: Optional[int] = None
    label: Optional[str] = None
    n_steps: int = 0
    n_obligations: int = 0
    n_semantic_obligations: int = 0
    predicted: dict[str, str] = Field(default_factory=dict)
    llm_calls: dict[str, int] = Field(default_factory=dict)


class ArmReport(BaseModel):
    """Scored metrics for one arm, overall and on the long-horizon slice."""
    arm: str
    overall: VerifierMetrics
    long_horizon: VerifierMetrics
    total_llm_calls: int = 0
    mean_llm_calls: float = 0.0


class SA1Result(BaseModel):
    """Full SA-1 experiment output: config, per-arm reports, and provenance."""
    experiment: str = "SA-1: Graph vs. Monolith (Q1)"
    n_traces: int = 0
    n_labeled: int = 0
    base_rate_valid: float = Field(0.0, description="Fraction of labeled traces with label 'valid'")
    long_horizon_steps: int = DEFAULT_LONG_HORIZON_STEPS
    n_long_horizon: int = 0
    use_llm: bool = False
    use_semantic: bool = False
    domain_id: str = ""
    seconds: float = 0.0
    arms: list[ArmReport] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def _semantic_obligation_count(htir: HTIR) -> int:
    """How many obligations were routed to the SEMANTIC checker (cost proxy)."""
    return sum(1 for o in htir.obligations if o.checker == CheckerType.SEMANTIC)


def _llm_calls_for_arm(arm: VerifierArm, n_semantic: int, n_steps: int, use_llm: bool) -> int:
    """
    LLM invocations an arm would issue on one trace (see module docstring):
    exec_only 0; avg_full / exec_free one per SEMANTIC obligation; monolithic
    one full-trace judge. SA-8: ``agent_judge`` one full-trace judge pass
    (token-budget-matched to the monolith and AVG's semantic checker), and
    ``prm`` one narrow step-critic call per step. Counted as the *would-issue*
    cost even when ``use_llm`` is off (offline run), which is the cost axis.
    """
    if arm == VerifierArm.EXEC_ONLY:
        return 0
    if arm in (VerifierArm.MONOLITHIC, VerifierArm.AGENT_JUDGE):
        return 1
    if arm == VerifierArm.PRM:
        return n_steps
    # avg_full / exec_free: one narrow call per semantic-routed obligation.
    return n_semantic


def run_sa1(
    raw_traces: Iterable[dict[str, Any]],
    *,
    spec: DomainSpec | None = None,
    arms: list[VerifierArm] | None = None,
    use_llm: bool = False,
    use_semantic: bool = False,
    long_horizon_steps: int = DEFAULT_LONG_HORIZON_STEPS,
    model: str = "openai/gpt-4o",
    progress_every: int = 250,
    log: Any = sys.stderr,
) -> SA1Result:
    """
    Execute SA-1 over ``raw_traces`` (turn-schema dicts with a ``reward``).

    Each trace is compiled once through obligation generation (deterministic,
    no checks), then every arm is scored on its own copy of that graph via
    :func:`htir.agents.baselines.run_arm`. Returns a fully populated
    :class:`SA1Result`. Runs offline by default; ``use_llm`` / ``use_semantic``
    turn on the LLM monolith and the semantic checker respectively.
    """
    spec = spec or TERMINAL_DOMAIN_SPEC
    arms = arms or list(DEFAULT_ARMS)
    agent = TraceAbstractionAgent(model=model, domain_spec=spec)

    records: list[PerTraceRecord] = []
    t0 = time.time()

    for i, raw in enumerate(raw_traces):
        reward = extract_reward(raw)
        label = label_from_reward(reward)
        task_id = str(raw.get("task_name", "")) if isinstance(raw, dict) else ""

        try:
            steps = to_canonical_steps(raw)
            htir = agent.compile(
                task_id=task_id or f"trace-{i}",
                raw_steps=steps,
                harness_snippets={},
                generate_obligations=True,
                use_semantic_analysis=use_semantic and use_llm,
                run_checks=False,
            )
        except Exception as exc:  # a malformed trace should not sink the run
            if log is not None:
                print(f"[sa1] skip trace {i} ({task_id}): {exc!r}", file=log)
            continue

        n_semantic = _semantic_obligation_count(htir)
        rec = PerTraceRecord(
            task_id=task_id,
            reward=reward,
            label=label,
            n_steps=len(htir.steps),
            n_obligations=len(htir.obligations),
            n_semantic_obligations=n_semantic,
        )

        for arm in arms:
            agg = run_arm(
                htir, spec, arm,
                use_llm=use_llm,
                model=model,
            )
            rec.predicted[arm.value] = agg.predicted_status
            rec.llm_calls[arm.value] = _llm_calls_for_arm(arm, n_semantic, len(htir.steps), use_llm)

        records.append(rec)
        if progress_every and log is not None and (i + 1) % progress_every == 0:
            print(f"[sa1] compiled+scored {i + 1} traces...", file=log)

    return _assemble(records, arms, spec, use_llm, use_semantic, long_horizon_steps, time.time() - t0)


def _assemble(
    records: list[PerTraceRecord],
    arms: list[VerifierArm],
    spec: DomainSpec,
    use_llm: bool,
    use_semantic: bool,
    long_horizon_steps: int,
    seconds: float,
) -> SA1Result:
    labels = [r.label for r in records]
    n_labeled = sum(1 for l in labels if l is not None)
    n_valid = sum(1 for l in labels if l == "valid")
    long_idx = [i for i, r in enumerate(records) if r.n_steps >= long_horizon_steps]

    arm_reports: list[ArmReport] = []
    for arm in arms:
        preds = [r.predicted.get(arm.value, "uncertain") for r in records]
        overall = evaluate_predictions(preds, labels)
        lh_preds = [preds[i] for i in long_idx]
        lh_labels = [labels[i] for i in long_idx]
        long_horizon = evaluate_predictions(lh_preds, lh_labels)
        calls = [r.llm_calls.get(arm.value, 0) for r in records]
        arm_reports.append(
            ArmReport(
                arm=arm.value,
                overall=overall,
                long_horizon=long_horizon,
                total_llm_calls=sum(calls),
                mean_llm_calls=statistics.fmean(calls) if calls else 0.0,
            )
        )

    notes: list[str] = []
    if not use_llm:
        notes.append(
            "Offline run (no API key): the semantic checker abstains, so avg_full "
            "collapses onto exec_only and exec_free abstains everywhere. The valid "
            "offline contrast is avg_full/exec_only (mechanical + abstention-aware "
            "aggregation) vs. monolithic (endpoint heuristic)."
        )
        notes.append(
            "SA-8 baselines offline: prm is the deterministic step-heuristic process "
            "reward model (score every step, mean-threshold); it abstains only on a "
            "trace with no steps to score, so on any scoreable trace it commits -- "
            "over-committing on weak-label steps and, by rewarding locally-plausible "
            "steps, crediting even more failed traces than the endpoint monolith. "
            "agent_judge falls back to a deterministic multi-hop step-outcome gather "
            "that still commits -- an honest proxy for the LLM Agent-as-a-Judge, which "
            "(like the semantic checker) needs a key to show its real evidence-gathering. "
            "Both remain fooled by structurally-clean-but-failed traces; AVG abstains instead."
        )
    notes.append(
        "llm_calls are a would-issue cost proxy (0 real calls offline): the compute "
        "axis for the cost-normalized curve (avg.tex Sec. 4.7)."
    )

    return SA1Result(
        n_traces=len(records),
        n_labeled=n_labeled,
        base_rate_valid=(n_valid / n_labeled) if n_labeled else 0.0,
        long_horizon_steps=long_horizon_steps,
        n_long_horizon=len(long_idx),
        use_llm=use_llm,
        use_semantic=use_semantic,
        domain_id=spec.domain_id,
        seconds=round(seconds, 2),
        arms=arm_reports,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_table(result: SA1Result) -> str:
    """A compact fixed-width results table for the terminal / logs."""
    lines: list[str] = []
    lines.append(
        f"SA-1: Graph vs. Monolith  |  n={result.n_traces} "
        f"(labeled {result.n_labeled}, base-rate valid {result.base_rate_valid:.2f})  |  "
        f"domain={result.domain_id}  use_llm={result.use_llm}"
    )
    header = (
        f"{'arm':<12} {'false_valid':>11} {'res_acc':>8} {'res_frac':>8} "
        f"{'abstain':>8} {'ff_prec':>8} {'ff_rec':>7} {'cost/tr':>8}"
    )

    def _rows(scope: str) -> list[str]:
        out = [f"  [{scope}]"]
        out.append("  " + header)
        for a in result.arms:
            m = getattr(a, scope)
            out.append(
                f"  {a.arm:<12} {m.false_valid_rate:>11.3f} {m.resolved_accuracy:>8.3f} "
                f"{m.resolved_fraction:>8.3f} {m.abstention_rate:>8.3f} "
                f"{m.failure_flag_precision:>8.3f} {m.failure_flag_recall:>7.3f} "
                f"{a.mean_llm_calls:>8.2f}"
            )
        return out

    lines += _rows("overall")
    lines.append(
        f"  [long-horizon >= {result.long_horizon_steps} steps]  "
        f"n={result.n_long_horizon}"
    )
    lines += _rows("long_horizon")[1:]  # skip duplicate scope banner
    for note in result.notes:
        lines.append(f"  note: {note}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Multi-seed sweep + significance (avg.tex Sec. 4.7 statistical reporting)
# ---------------------------------------------------------------------------

# Per-seed metrics aggregated to mean±SE (their raw per-seed values feed the
# paired significance test). Iterates ``res.arms`` so any arm (incl. SA-8's
# prm / agent_judge) flows through automatically.
def sa1_seed_metrics(res: SA1Result) -> dict[str, float]:
    out: dict[str, float] = {"base_rate_valid": res.base_rate_valid}
    for a in res.arms:
        out[f"{a.arm}.false_valid"] = a.overall.false_valid_rate
        out[f"{a.arm}.false_valid.long"] = a.long_horizon.false_valid_rate
        out[f"{a.arm}.resolved_frac"] = a.overall.resolved_fraction
        out[f"{a.arm}.resolved_acc"] = a.overall.resolved_accuracy
    return out


# Key gaps whose significance we report: each weaker baseline's false-valid rate
# vs. full AVG (the headline contrast). SA-8 adds prm and agent_judge.
SIG_GAPS: list[tuple[str, str]] = [
    ("monolithic", "avg_full"),
    ("prm", "avg_full"),
    ("agent_judge", "avg_full"),
]


def _significance(aggregate: dict[str, Any]) -> list[PairedGap]:
    """Paired t-tests on the SIG_GAPS false-valid contrasts, using the raw
    per-seed values that :func:`htir.eval.seeds.aggregate` retained."""
    gaps: list[PairedGap] = []
    for a_arm, b_arm in SIG_GAPS:
        a_key, b_key = f"{a_arm}.false_valid", f"{b_arm}.false_valid"
        if a_key not in aggregate or b_key not in aggregate:
            continue
        gaps.append(paired_t_test(
            aggregate[a_key].values, aggregate[b_key].values,
            label=f"{a_arm}_vs_{b_arm}.false_valid", a=a_arm, b=b_arm,
        ))
    return gaps


def _sample_pool(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Load the trace pool that each seed draws a balanced subsample from."""
    if args.hf:
        from htir.eval.datasets import load_terminalbench
        return list(load_terminalbench(limit=args.hf_limit, streaming=True))
    if not args.cache:
        raise SystemExit("provide --cache <jsonl> or --hf")
    return list(iter_local_traces([args.cache]))


def run_sa1_multiseed(
    pool: list[dict[str, Any]], *, spec: DomainSpec, seeds: list[int], n: int,
    long_horizon_steps: int, model: str, log: Any = sys.stderr,
) -> dict[str, Any]:
    """
    Run SA-1 over ``len(seeds)`` independent balanced subsamples of ``pool`` and
    package mean±SE + a paired-t significance statement on the key false-valid
    gaps (avg.tex Sec. 4.7). Mirrors the τ-bench driver's multi-seed shape.
    """
    from htir.eval.datasets import balanced_sample

    def sample_fn(seed: int) -> list[dict[str, Any]]:
        return balanced_sample(pool, n, seed=seed)

    def run_fn(sample: list[dict[str, Any]]) -> SA1Result:
        return run_sa1(sample, spec=spec, long_horizon_steps=long_horizon_steps,
                       model=model, progress_every=0, log=log)

    summary, per_seed = run_multiseed(sample_fn, run_fn, seeds, extract=sa1_seed_metrics, log=log)
    gaps = _significance(summary.aggregate)
    return {
        "experiment": "SA-1 + SA-8: Graph vs. Monolith + PRM / Agent-as-a-Judge (Q1)",
        "domain": spec.domain_id,
        "seeds": seeds,
        "n_per_seed": summary.n_per_seed,
        "use_llm": False,
        "aggregate": {k: v.model_dump() for k, v in summary.aggregate.items()},
        "significance": [g.model_dump() for g in gaps],
        "per_seed": [r.model_dump() for r in per_seed],
        "notes": per_seed[0].notes if per_seed else [],
    }


def format_multiseed(out: dict[str, Any]) -> str:
    """Compact report of the multi-seed aggregate + significance statements."""
    lines: list[str] = []
    lines.append(
        f"{out['experiment']}  |  domain={out['domain']}  "
        f"seeds={out['seeds']}  n/seed={out['n_per_seed']}"
    )
    # False-valid mean±SE per arm (overall + long-horizon).
    from htir.eval.seeds import MeanSE
    agg = {k: MeanSE(**v) for k, v in out["aggregate"].items()}
    lines.append("  [false_valid mean±SE over seeds]")
    lines.append(f"    {'arm':<12} {'overall':>16} {'long_horizon':>16} {'res_frac':>16}")
    for arm in [a.value for a in DEFAULT_ARMS]:
        ov = agg.get(f"{arm}.false_valid")
        lh = agg.get(f"{arm}.false_valid.long")
        rf = agg.get(f"{arm}.resolved_frac")
        if ov is None:
            continue
        lines.append(
            f"    {arm:<12} {ov.as_str():>16} "
            f"{(lh.as_str() if lh else '-'):>16} {(rf.as_str() if rf else '-'):>16}"
        )
    lines.append("  [significance -- paired t-test on false_valid gap vs full AVG]")
    for g in out["significance"]:
        gap = PairedGap(**g)
        lines.append(f"    {gap.as_str()}")
    return "\n".join(lines)


def _load_traces(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.hf:
        from htir.eval.datasets import balanced_sample, load_terminalbench
        raw = load_terminalbench(limit=args.hf_limit, streaming=True)
        return balanced_sample(raw, args.n, seed=args.seed)
    if not args.cache:
        raise SystemExit("provide --cache <jsonl> or --hf")
    traces = list(iter_local_traces([args.cache]))
    if args.n and len(traces) > args.n:
        traces = traces[: args.n]
    return traces


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SA-1: Graph vs. Monolith (Q1)")
    src = p.add_argument_group("data source")
    src.add_argument("--cache", type=str, default="", help="local JSON/JSONL sample (turn schema)")
    src.add_argument("--hf", action="store_true", help="pull from the HF terminalbench dataset")
    src.add_argument("--hf-limit", type=int, default=9000, help="records to stream when --hf")
    src.add_argument("--n", type=int, default=3000, help="target balanced sample size (per seed)")
    src.add_argument("--seed", type=int, default=0, help="seed for the single-seed path")
    src.add_argument("--seeds", type=str, default="0,1,2",
                     help="comma list of seeds for the multi-seed sweep (mean±SE + significance)")
    p.add_argument("--single-seed", action="store_true",
                   help="run one seed and write a flat SA1Result (legacy schema) instead of the sweep")
    p.add_argument("--use-llm", action="store_true", help="enable LLM monolith + semantic checker")
    p.add_argument("--domain", type=str, default="terminal_swe",
                   help="domain spec S_d to verify under (e.g. terminal_swe, tau_bench)")
    p.add_argument("--long-horizon-steps", type=int, default=DEFAULT_LONG_HORIZON_STEPS)
    p.add_argument("--model", type=str, default="openai/gpt-4o")
    p.add_argument("--out", type=str, default="", help="write result JSON here")
    args = p.parse_args(argv)

    spec = get_domain_spec(args.domain)

    # Multi-seed sweep (default): the rigor path -- mean±SE over >=3 seeds and a
    # paired-t significance statement on the key false-valid gaps. Offline only
    # (the sweep is byte-deterministic); use --single-seed --use-llm for an LLM run.
    if not args.single_seed and not args.use_llm:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
        pool = _sample_pool(args)
        out = run_sa1_multiseed(
            pool, spec=spec, seeds=seeds, n=args.n,
            long_horizon_steps=args.long_horizon_steps, model=args.model,
        )
        print(format_multiseed(out))
        if args.out:
            Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
            print(f"\n[sa1] wrote {args.out}", file=sys.stderr)
        return 0

    traces = _load_traces(args)
    result = run_sa1(
        traces,
        spec=spec,
        use_llm=args.use_llm,
        use_semantic=args.use_llm,
        long_horizon_steps=args.long_horizon_steps,
        model=args.model,
    )
    print(format_table(result))
    if args.out:
        Path(args.out).write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(f"\n[sa1] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
