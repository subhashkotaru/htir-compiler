"""
SA-3 -- Q3: Calibrated Abstention (avg.tex Sec. 4 question Q3; ablation #3).

Q3 asks whether *calibrated abstention* -- letting a checker decline rather
than force a pass/fail -- reduces harmful false positives and yields better-
calibrated confidence. This experiment compiles a balanced Terminal-Bench
sample once per trace, then scores two arms over the *same* obligation graph:

* ``avg_full``      -- full AVG with calibrated abstention (the default): an
  obligation with insufficient evidence ABSTAINS, and a trajectory with no
  positively discharged evidence aggregates to ``uncertain`` instead of being
  credited.
* ``no_abstention`` -- the ablation (avg.tex Sec. 4.5 #3): the *same* graph and
  evidence, but every checker is forced to emit pass/fail
  (``force_decision``). A no-evidence obligation is credited on the optimistic
  prior, so the trajectory is pushed to a valid/invalid verdict with no
  ``uncertain`` escape hatch.

Both arms share one coverage-aware probability-of-valid ``p_valid``
(:func:`htir.eval.calibration.trajectory_valid_score`) -- identical by
construction, because the no-abstention prior and an abstention both sit on the
0.5 boundary -- and differ only in their *decision policy*
(:func:`htir.agents.witness.aggregate` over the calibrated vs. forced graph).
SA-3 therefore isolates abstention: given the same confidence, does declining
to answer on the low-confidence traces beat committing on all of them? We
report:

* **false-valid reduction** -- the headline: how much abstention cuts the rate
  at which failed (reward = 0) trajectories are credited ``valid``.
* **abstention-calibrated AUROC** -- tie-aware AUROC of the shared score over
  only the traces each arm commits to (vs. the global AUROC over all traces).
* **ECE + reliability diagram** -- calibration of the shared ``p_valid``
  against the empirical valid rate, globally and per committed set.
* **precision at fixed abstention budgets** -- the risk-coverage curve of the
  shared score.

Offline reproducibility. With no API key (``use_llm=False``, the default) the
semantic checker abstains, so ``avg_full`` is mechanical-evidence-only and
``no_abstention`` forces those same mechanical checks to commit. The contrast
is therefore *calibrated abstention vs. forced commitment over identical
mechanical evidence* -- exactly ablation #3 -- computed with zero LLM calls.
Calibration truth here is the weak trajectory-level reward label; the step/
obligation-level gold slice (avg.tex P0) would sharpen the diagram but is not
required for these trajectory-level numbers. Both caveats are reported, not
hidden.

CLI::

    python -m htir.eval.experiment_sa3 --cache sa3_sample.jsonl --out sa3_results.json
    python -m htir.eval.experiment_sa3 --hf --n 3000 --out sa3_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field

from htir.agents.baselines import VerifierArm, _ARM_FLAGS
from htir.agents.checking import check_obligations
from htir.agents.trace_abstraction import TraceAbstractionAgent
from htir.agents.witness import aggregate
from htir.eval.calibration import (
    DECISION_BOUNDARY,
    ReliabilityBin,
    RiskCoveragePoint,
    expected_calibration_error,
    reliability_bins,
    risk_coverage_curve,
    roc_auc,
    trajectory_valid_score,
)
from htir.eval.datasets import iter_local_traces, to_canonical_steps
from htir.eval.weak_labels import (
    LABEL_VALID,
    STATUS_UNCERTAIN,
    VerifierMetrics,
    evaluate_predictions,
    extract_reward,
    label_from_reward,
)
from htir.models.domain import TERMINAL_DOMAIN_SPEC, DomainSpec, get_domain_spec

# The two arms SA-3 contrasts, in table order: calibrated abstention (may
# abstain -> 'uncertain') vs. the no-abstention ablation (force_decision -> must
# commit). They share one calibrated score; only the decision policy differs.
SA3_ARMS: list[VerifierArm] = [VerifierArm.AVG_FULL, VerifierArm.NO_ABSTENTION]
_RELIABILITY_BINS = 10


class PerTraceRecord(BaseModel):
    """One trace's label, size, shared calibrated score, and per-arm decision."""
    task_id: str = ""
    reward: Optional[int] = None
    label: Optional[str] = None
    n_steps: int = 0
    n_obligations: int = 0
    score: Optional[float] = Field(None, description="Shared coverage-aware p_valid (None if no obligation had a result)")
    decision: dict[str, str] = Field(default_factory=dict)


class ArmDecisionReport(BaseModel):
    """Decision quality + abstention-calibrated metrics for one arm's policy."""
    arm: str
    decision_metrics: VerifierMetrics
    n_committed: int = 0
    coverage: float = Field(0.0, description="Fraction of labeled traces the arm commits to (not 'uncertain')")
    auroc_committed: Optional[float] = Field(None, description="Abstention-calibrated AUROC over committed traces")
    ece_committed: float = Field(0.0, description="ECE of the shared p_valid over the arm's committed traces")


class SA3Result(BaseModel):
    """Full SA-3 output: config, shared calibration, per-arm policies, headline."""
    experiment: str = "SA-3: Calibrated Abstention (Q3)"
    n_traces: int = 0
    n_labeled: int = 0
    base_rate_valid: float = Field(0.0, description="Fraction of labeled traces with label 'valid'")
    use_llm: bool = False
    domain_id: str = ""
    seconds: float = 0.0
    # Global calibration of the shared p_valid score over all labeled traces.
    auroc_all: Optional[float] = Field(None, description="AUROC of the shared score over all labeled traces")
    ece_all: float = Field(0.0, description="ECE of the shared score over all labeled traces")
    reliability: list[ReliabilityBin] = Field(default_factory=list)
    risk_coverage: list[RiskCoveragePoint] = Field(default_factory=list)
    false_valid_reduction: dict[str, float] = Field(
        default_factory=dict,
        description="abstain vs no-abstain false-valid rate: absolute and relative reduction",
    )
    arms: list[ArmDecisionReport] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def _decide_arm(htir_compiled, spec: DomainSpec, arm: VerifierArm, *, use_llm: bool, model: str) -> str:
    """Check ``htir_compiled`` under ``arm`` on its own copy and return its
    aggregated decision (valid/invalid/uncertain). ``force_decision`` (the
    no-abstention arm) removes the 'uncertain' escape hatch."""
    graph = htir_compiled.model_copy(deep=True)
    flags = _ARM_FLAGS[arm]
    check_obligations(
        graph, spec,
        use_semantic=flags["use_semantic"] and use_llm,
        disable_mechanical=flags["disable_mechanical"],
        force_decision=flags["force_decision"],
        model=model,
    )
    return aggregate(graph).predicted_status


def _trace_score(htir_compiled, spec: DomainSpec, *, use_llm: bool, model: str) -> Optional[float]:
    """The shared coverage-aware p_valid, computed once from the calibrated
    (non-forced) checked graph."""
    graph = htir_compiled.model_copy(deep=True)
    check_obligations(graph, spec, use_semantic=use_llm, model=model)
    return trajectory_valid_score(graph)


def run_sa3(
    raw_traces: Iterable[dict[str, Any]],
    *,
    spec: DomainSpec | None = None,
    use_llm: bool = False,
    model: str = "openai/gpt-4o",
    progress_every: int = 250,
    log: Any = sys.stderr,
) -> SA3Result:
    """
    Execute SA-3 over ``raw_traces`` (turn-schema dicts with a ``reward``).

    Compiles each trace once (deterministic, no checks), then scores the
    calibrated-abstention and no-abstention arms on independent copies. Returns
    a fully populated :class:`SA3Result`. Offline by default (no LLM calls).
    """
    spec = spec or TERMINAL_DOMAIN_SPEC
    agent = TraceAbstractionAgent(model=model, domain_spec=spec)

    records: list[PerTraceRecord] = []
    t0 = time.time()

    for i, raw in enumerate(raw_traces):
        reward = extract_reward(raw)
        label = label_from_reward(reward)
        task_id = str(raw.get("task_name", "")) if isinstance(raw, dict) else ""

        try:
            steps = to_canonical_steps(raw)
            compiled = agent.compile(
                task_id=task_id or f"trace-{i}",
                raw_steps=steps,
                harness_snippets={},
                generate_obligations=True,
                use_semantic_analysis=False,
                run_checks=False,
            )
        except Exception as exc:  # a malformed trace should not sink the run
            if log is not None:
                print(f"[sa3] skip trace {i} ({task_id}): {exc!r}", file=log)
            continue

        rec = PerTraceRecord(
            task_id=task_id,
            reward=reward,
            label=label,
            n_steps=len(compiled.steps),
            n_obligations=len(compiled.obligations),
            score=_trace_score(compiled, spec, use_llm=use_llm, model=model),
        )
        for arm in SA3_ARMS:
            rec.decision[arm.value] = _decide_arm(compiled, spec, arm, use_llm=use_llm, model=model)
        records.append(rec)

        if progress_every and log is not None and (i + 1) % progress_every == 0:
            print(f"[sa3] compiled+scored {i + 1} traces...", file=log)

    return _assemble(records, spec, use_llm, time.time() - t0)


def _assemble(
    records: list[PerTraceRecord], spec: DomainSpec, use_llm: bool, seconds: float,
) -> SA3Result:
    labeled = [r for r in records if r.label is not None]
    n_valid = sum(1 for r in labeled if r.label == LABEL_VALID)

    # Shared calibrated score over all labeled traces (0.5-imputed when a trace
    # carried no scoreable obligation), and its label 1 = valid / 0 = invalid.
    y = [1 if r.label == LABEL_VALID else 0 for r in labeled]
    scores = [r.score if r.score is not None else DECISION_BOUNDARY for r in labeled]

    arm_reports: list[ArmDecisionReport] = []
    false_valid_by_arm: dict[str, float] = {}
    for arm in SA3_ARMS:
        preds = [r.decision.get(arm.value, STATUS_UNCERTAIN) for r in records]
        labels_str = [r.label for r in records]
        metrics = evaluate_predictions(preds, labels_str)
        false_valid_by_arm[arm.value] = metrics.false_valid_rate

        # Committed = traces this arm's policy did NOT abstain on. The shared
        # score restricted to that set gives the abstention-calibrated AUROC/ECE.
        committed = [
            (s, yi) for r, s, yi in zip(labeled, scores, y)
            if r.decision.get(arm.value, STATUS_UNCERTAIN) != STATUS_UNCERTAIN
        ]
        c_scores = [s for s, _ in committed]
        c_labels = [yi for _, yi in committed]
        arm_reports.append(ArmDecisionReport(
            arm=arm.value,
            decision_metrics=metrics,
            n_committed=len(committed),
            coverage=(len(committed) / len(labeled)) if labeled else 0.0,
            auroc_committed=roc_auc(c_scores, c_labels) if committed else None,
            ece_committed=expected_calibration_error(c_scores, c_labels, n_bins=_RELIABILITY_BINS),
        ))

    fv_abstain = false_valid_by_arm.get(VerifierArm.AVG_FULL.value, 0.0)
    fv_forced = false_valid_by_arm.get(VerifierArm.NO_ABSTENTION.value, 0.0)
    false_valid_reduction = {
        "no_abstention": fv_forced,
        "avg_full": fv_abstain,
        "absolute": fv_forced - fv_abstain,
        "relative": ((fv_forced - fv_abstain) / fv_forced) if fv_forced else 0.0,
    }

    notes: list[str] = []
    if not use_llm:
        notes.append(
            "Offline run (no API key): the semantic checker abstains, so avg_full is "
            "mechanical-evidence-only and no_abstention forces those same mechanical "
            "checks to commit. The contrast is calibrated abstention vs. forced "
            "commitment over identical evidence (ablation #3)."
        )
    notes.append(
        "Calibration truth is the weak trajectory-level reward label; the step-level "
        "gold slice (avg.tex P0) would sharpen the reliability diagram but is not "
        "required for these trajectory-level metrics."
    )
    notes.append(
        "auroc_committed is the abstention-calibrated AUROC: the shared score "
        "restricted to the traces each arm commits to. auroc_all/ece_all/reliability/"
        "risk_coverage are over all labeled traces (0.5-imputed when a trace had no "
        "scoreable obligation)."
    )

    return SA3Result(
        n_traces=len(records),
        n_labeled=len(labeled),
        base_rate_valid=(n_valid / len(labeled)) if labeled else 0.0,
        use_llm=use_llm,
        domain_id=spec.domain_id,
        seconds=round(seconds, 2),
        auroc_all=roc_auc(scores, y) if labeled else None,
        ece_all=expected_calibration_error(scores, y, n_bins=_RELIABILITY_BINS),
        reliability=reliability_bins(scores, y, n_bins=_RELIABILITY_BINS),
        risk_coverage=risk_coverage_curve(scores, y) if labeled else [],
        false_valid_reduction=false_valid_reduction,
        arms=arm_reports,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_table(result: SA3Result) -> str:
    """A compact fixed-width results table for the terminal / logs."""
    lines: list[str] = []
    lines.append(
        f"SA-3: Calibrated Abstention  |  n={result.n_traces} "
        f"(labeled {result.n_labeled}, base-rate valid {result.base_rate_valid:.2f})  |  "
        f"domain={result.domain_id}  use_llm={result.use_llm}"
    )
    aa = f"{result.auroc_all:.3f}" if result.auroc_all is not None else "n/a"
    lines.append(
        f"  shared calibrated score:  AUROC(all)={aa}  ECE(all)={result.ece_all:.3f}"
    )

    header = (
        f"{'arm':<14} {'false_valid':>11} {'abstain':>8} {'coverage':>9} "
        f"{'auroc_cmt':>10} {'ece_cmt':>8} {'res_acc':>8}"
    )
    lines.append("  " + header)
    for a in result.arms:
        m = a.decision_metrics
        ac = f"{a.auroc_committed:.3f}" if a.auroc_committed is not None else "  n/a"
        lines.append(
            f"  {a.arm:<14} {m.false_valid_rate:>11.3f} {m.abstention_rate:>8.3f} "
            f"{a.coverage:>9.3f} {ac:>10} {a.ece_committed:>8.3f} {m.resolved_accuracy:>8.3f}"
        )

    fv = result.false_valid_reduction
    lines.append(
        f"  false-valid reduction (no_abstention {fv.get('no_abstention', 0):.3f} -> "
        f"avg_full {fv.get('avg_full', 0):.3f}): "
        f"abs {fv.get('absolute', 0):.3f}, rel {fv.get('relative', 0):.1%}"
    )

    if result.risk_coverage:
        lines.append("  [precision @ abstention budget -- shared score]")
        lines.append(f"    {'budget':>7} {'coverage':>9} {'accuracy':>9} {'false_valid':>11} {'valid_prec':>10}")
        for p in result.risk_coverage:
            lines.append(
                f"    {p.abstention_budget:>7.2f} {p.coverage:>9.3f} {p.accuracy:>9.3f} "
                f"{p.false_valid_rate:>11.3f} {p.valid_precision:>10.3f}"
            )
    for note in result.notes:
        lines.append(f"  note: {note}")
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
    p = argparse.ArgumentParser(description="SA-3: Calibrated Abstention (Q3)")
    src = p.add_argument_group("data source")
    src.add_argument("--cache", type=str, default="", help="local JSON/JSONL sample (turn schema)")
    src.add_argument("--hf", action="store_true", help="pull from the HF terminalbench dataset")
    src.add_argument("--hf-limit", type=int, default=9000, help="records to stream when --hf")
    src.add_argument("--n", type=int, default=3000, help="target balanced sample size")
    src.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-llm", action="store_true", help="enable the semantic checker")
    p.add_argument("--domain", type=str, default="terminal_swe",
                   help="domain spec S_d to verify under (e.g. terminal_swe, tau_bench)")
    p.add_argument("--model", type=str, default="openai/gpt-4o")
    p.add_argument("--out", type=str, default="", help="write SA3Result JSON here")
    args = p.parse_args(argv)

    traces = _load_traces(args)
    result = run_sa3(traces, spec=get_domain_spec(args.domain), use_llm=args.use_llm, model=args.model)
    print(format_table(result))
    if args.out:
        Path(args.out).write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(f"\n[sa3] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
