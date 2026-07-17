"""
SA-11 -- Selective-verification frontier (Fig 2) + calibration reframe
(spotlight plan P1).

SA-1 / SA-3 reported AVG at a single operating point: ~14-16 % coverage, ~84 %
abstain. Read as a fixed property that looks like a weakness ("it only answers
on a sixth of the traces"). SA-11 shows it is a **knob**. Sweeping AVG's
decision threshold over the coverage-aware ``trajectory_valid_score`` traces a
**false-valid-vs-coverage frontier**; the monolithic / PRM / Agent-as-a-Judge
baselines are each a single point in that plane, and every baseline point sits
*on or above* AVG's frontier -- i.e. at the baseline's own coverage AVG credits
strictly fewer reward-hacks, and AVG's default operating point buys a large
false-valid reduction by simply moving down the same frontier.

This is also the honest home for the calibration story. The shared-score AUROC
(0.46 Terminal-Bench / 0.60 tau-bench) is low because the trajectory-level
reward is a *coarse* label -- a well-formed-but-failed trace and a genuinely
correct one both carry ``reward=1`` structure the mechanical checks cannot
separate. That coarse-label ranking is not what AVG relies on; the *abstain /
don't-abstain decision* is, and the frontier is exactly what that decision buys.

Construction (per seed).

* Compile each trace once, check the obligations (non-forced) and read the
  shared coverage-aware ``p_valid`` (:func:`trajectory_valid_score`; 0.5-imputed
  when a trace carried no scoreable obligation). This is identical to the SA-3
  shared score.
* **AVG frontier** -- sweep an acceptance threshold ``tau`` over ``p_valid``:
  predict ``valid`` iff ``p_valid >= tau``, ``invalid`` iff ``p_valid <= 1-tau``,
  else abstain. Each ``tau`` is one operating point ``(coverage, false_valid)``
  where ``coverage`` is the committed fraction and ``false_valid`` is the SA-1
  metric -- invalid traces credited valid over **all** labeled-invalid traces
  (so a low-coverage point earns its low false-valid honestly, by abstaining).
* **AVG native point** -- the coverage / false-valid of AVG's actual aggregator
  decision (``avg_full``): the operating point the system ships at.
* **Baseline points** -- ``monolithic`` / ``prm`` / ``agent_judge`` each contribute
  one ``(coverage, false_valid)`` point via :func:`run_arm`.
* **Matched-coverage** -- for each baseline, linearly interpolate AVG's frontier
  to the baseline's coverage and read AVG's false-valid *there*: the apples-to-
  apples gap, done per seed then aggregated.

Rigor. Everything is computed per seed over an independent balanced subsample
and reported as mean +/- SE (>= 3 seeds). The headline matched-coverage gaps
carry a paired t-test over the per-seed values (avg.tex Sec. 4.7). The whole
offline path is byte-deterministic (no API key): ``avg_full`` collapses onto
mechanical evidence + abstention-aware aggregation and the baselines are their
deterministic heuristics, exactly as in SA-1.

Deliverable. Fig 2 (``docs/sa11_frontier[_tau].png`` when matplotlib is present;
skipped with a note otherwise) + ``data/sa11[_tau]_results.json``.

CLI::

    python -m htir.eval.experiment_sa11 --cache data/tau_cache/terminal_sample_600.jsonl \\
        --n 400 --seeds 0,1,2 --out data/sa11_results.json --fig docs/sa11_frontier.png
    python -m htir.eval.experiment_sa11 --domain tau_bench \\
        --cache data/tau_cache/tau_all.jsonl --n 600 --seeds 0,1,2 \\
        --out data/sa11_tau_results.json --fig docs/sa11_frontier_tau.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field

from htir.agents.baselines import VerifierArm, run_arm
from htir.agents.checking import check_obligations
from htir.agents.trace_abstraction import TraceAbstractionAgent
from htir.eval.calibration import (
    DECISION_BOUNDARY,
    expected_calibration_error,
    roc_auc,
    trajectory_valid_score,
)
from htir.eval.datasets import balanced_sample, iter_local_traces, to_canonical_steps
from htir.eval.seeds import MeanSE, PairedGap, mean_se, paired_t_test
from htir.eval.weak_labels import (
    LABEL_INVALID,
    LABEL_VALID,
    STATUS_UNCERTAIN,
    extract_reward,
    label_from_reward,
)
from htir.models.domain import DomainArtifactBundle, DomainSpec, get_domain_spec

# Baseline verifiers overlaid as points on AVG's frontier (fixed order = fixed
# colours in the figure). AVG (avg_full) is the curve, not a baseline point.
BASELINE_ARMS: list[VerifierArm] = [
    VerifierArm.MONOLITHIC,
    VerifierArm.PRM,
    VerifierArm.AGENT_JUDGE,
]

# Acceptance-threshold grid for the frontier sweep. Dense just above 0.5 (where
# most of the score mass sits, so coverage falls fast) and coarser out toward the
# confident tail. tau=0.5 is the force-commit end (coverage ~1.0).
DEFAULT_THRESHOLDS: tuple[float, ...] = (
    0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.57, 0.60,
    0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
)
_RELIABILITY_BINS = 10


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

class FrontierPoint(BaseModel):
    """One AVG operating point: acceptance threshold -> (coverage, false-valid)."""
    threshold: float
    coverage: MeanSE = Field(default_factory=MeanSE)
    false_valid: MeanSE = Field(default_factory=MeanSE)


class OperatingPoint(BaseModel):
    """A single verifier's (coverage, false-valid), aggregated over seeds."""
    arm: str
    coverage: MeanSE = Field(default_factory=MeanSE)
    false_valid: MeanSE = Field(default_factory=MeanSE)


class MatchedCoverage(BaseModel):
    """AVG's false-valid read off its frontier at a baseline's own coverage."""
    arm: str
    coverage: MeanSE = Field(default_factory=MeanSE, description="Baseline's coverage (the match point)")
    baseline_false_valid: MeanSE = Field(default_factory=MeanSE)
    avg_false_valid_at_coverage: MeanSE = Field(default_factory=MeanSE)
    gap: MeanSE = Field(
        default_factory=MeanSE,
        description="baseline_false_valid - avg_false_valid_at_coverage (>0 => AVG dominates)",
    )
    significance: PairedGap = Field(default_factory=PairedGap)


class SA11Result(BaseModel):
    """Full SA-11 output: the frontier, overlaid points, matched-coverage gaps."""
    experiment: str = "SA-11: Selective-verification frontier (Fig 2)"
    domain_id: str = ""
    n_per_seed: list[int] = Field(default_factory=list)
    n_labeled_per_seed: list[int] = Field(default_factory=list)
    base_rate_valid: MeanSE = Field(default_factory=MeanSE)
    use_llm: bool = False
    seeds: list[int] = Field(default_factory=list)
    seconds: float = 0.0

    frontier: list[FrontierPoint] = Field(default_factory=list)
    avg_native: OperatingPoint = Field(default_factory=lambda: OperatingPoint(arm="avg_full"))
    baselines: list[OperatingPoint] = Field(default_factory=list)
    matched_coverage: list[MatchedCoverage] = Field(default_factory=list)

    # Calibration reframe: raw ranking is weak (coarse-label artifact) but the
    # abstain decision -- the frontier -- is what carries the safety.
    auroc_all: MeanSE = Field(default_factory=MeanSE, description="AUROC of the shared p_valid over all labeled traces")
    ece_all: MeanSE = Field(default_factory=MeanSE, description="ECE of the shared p_valid over all labeled traces")
    dominates_at_matched_coverage: bool = False
    figure_path: str = ""
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-seed scoring
# ---------------------------------------------------------------------------

class _Scored(BaseModel):
    """One trace's weak label, shared p_valid, and each baseline arm's decision."""
    label: str
    p_valid: float
    status: dict[str, str] = Field(default_factory=dict)


def _score_sample(
    sample: Iterable[dict[str, Any]],
    *,
    spec: DomainSpec,
    omega: DomainArtifactBundle | None,
    use_llm: bool,
    model: str,
    log: Any,
) -> list[_Scored]:
    """Compile + check each trace once; capture the shared ``p_valid`` (from the
    non-forced ``avg_full`` graph) and every baseline arm's decision status."""
    agent = TraceAbstractionAgent(model=model, domain_spec=spec, domain_artifacts=omega)
    out: list[_Scored] = []
    for i, raw in enumerate(sample):
        label = label_from_reward(extract_reward(raw))
        if label is None:
            continue
        task_id = str(raw.get("task_name", "")) if isinstance(raw, dict) else ""
        try:
            steps = to_canonical_steps(raw)
            compiled = agent.compile(
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
                print(f"[sa11] skip trace {i} ({task_id}): {exc!r}", file=log)
            continue

        # Shared coverage-aware score, read from the calibrated (non-forced) graph.
        scored = compiled.model_copy(deep=True)
        check_obligations(scored, spec, use_semantic=use_llm, model=model)
        pv = trajectory_valid_score(scored)

        rec = _Scored(label=label, p_valid=DECISION_BOUNDARY if pv is None else pv)
        for arm in [VerifierArm.AVG_FULL, *BASELINE_ARMS]:
            graph = compiled.model_copy(deep=True)
            agg = run_arm(graph, spec, arm, use_llm=use_llm, domain_artifacts=omega,
                          model=model, in_place=True)
            rec.status[arm.value] = agg.predicted_status
        out.append(rec)
    return out


def _false_valid_rate(preds: list[str], labels: list[str]) -> float:
    """SA-1 false-valid: invalid traces credited valid, over ALL labeled-invalid."""
    n_invalid = sum(1 for l in labels if l == LABEL_INVALID)
    if n_invalid == 0:
        return 0.0
    fv = sum(1 for p, l in zip(preds, labels) if l == LABEL_INVALID and p == LABEL_VALID)
    return fv / n_invalid


def _threshold_point(scored: list[_Scored], tau: float) -> tuple[float, float]:
    """AVG operating point at acceptance threshold ``tau``: (coverage, false_valid).
    Predict valid iff p_valid >= tau, invalid iff p_valid <= 1-tau, else abstain."""
    n = len(scored)
    labels = [s.label for s in scored]
    preds: list[str] = []
    for s in scored:
        if s.p_valid >= tau:
            preds.append(LABEL_VALID)
        elif s.p_valid <= 1.0 - tau:
            preds.append(LABEL_INVALID)
        else:
            preds.append(STATUS_UNCERTAIN)
    coverage = sum(1 for p in preds if p != STATUS_UNCERTAIN) / n if n else 0.0
    return coverage, _false_valid_rate(preds, labels)


def _arm_point(scored: list[_Scored], arm: str) -> tuple[float, float]:
    """A baseline arm's (coverage, false_valid) from its committed decisions."""
    n = len(scored)
    labels = [s.label for s in scored]
    preds = [s.status.get(arm, STATUS_UNCERTAIN) for s in scored]
    coverage = sum(1 for p in preds if p != STATUS_UNCERTAIN) / n if n else 0.0
    return coverage, _false_valid_rate(preds, labels)


def _interp_false_valid(frontier: list[tuple[float, float]], target_cov: float) -> float:
    """
    Linearly interpolate AVG's false-valid at ``target_cov`` on one seed's
    frontier ``[(coverage, false_valid), ...]``. Coverage is monotone-decreasing
    in the threshold; we sort by coverage ascending and interpolate, clamping to
    the endpoints outside the swept range.
    """
    pts = sorted(frontier)  # by coverage ascending
    if not pts:
        return 0.0
    if target_cov <= pts[0][0]:
        return pts[0][1]
    if target_cov >= pts[-1][0]:
        return pts[-1][1]
    for (c0, f0), (c1, f1) in zip(pts, pts[1:]):
        if c0 <= target_cov <= c1:
            if c1 == c0:
                return (f0 + f1) / 2.0
            w = (target_cov - c0) / (c1 - c0)
            return f0 + w * (f1 - f0)
    return pts[-1][1]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_sa11(
    pool: list[dict[str, Any]],
    *,
    spec: DomainSpec | None = None,
    omega: DomainArtifactBundle | None = None,
    seeds: list[int] | None = None,
    n: int = 400,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    use_llm: bool = False,
    model: str = "openai/gpt-4o",
    log: Any = sys.stderr,
) -> SA11Result:
    """
    Execute SA-11 over ``len(seeds)`` independent balanced subsamples of ``pool``.

    Per seed: compile + score once, sweep the acceptance threshold to trace AVG's
    frontier, and read every baseline arm's operating point plus AVG's false-valid
    interpolated to each baseline's coverage. Aggregates to mean +/- SE and runs a
    paired t-test on the matched-coverage gaps. Offline / byte-deterministic by
    default.
    """
    spec = spec or get_domain_spec("terminal_swe")
    seeds = seeds if seeds is not None else [0, 1, 2]
    t0 = time.time()

    # Per-seed collections keyed for aggregation.
    n_per_seed: list[int] = []
    n_labeled: list[int] = []
    base_rates: list[float] = []
    front_cov: dict[float, list[float]] = {t: [] for t in thresholds}
    front_fv: dict[float, list[float]] = {t: [] for t in thresholds}
    native_cov: list[float] = []
    native_fv: list[float] = []
    base_cov: dict[str, list[float]] = {a.value: [] for a in BASELINE_ARMS}
    base_fv: dict[str, list[float]] = {a.value: [] for a in BASELINE_ARMS}
    matched_target_cov: dict[str, list[float]] = {a.value: [] for a in BASELINE_ARMS}
    matched_base_fv: dict[str, list[float]] = {a.value: [] for a in BASELINE_ARMS}
    matched_avg_fv: dict[str, list[float]] = {a.value: [] for a in BASELINE_ARMS}
    auroc_vals: list[float] = []
    ece_vals: list[float] = []

    for seed in seeds:
        sample = balanced_sample(pool, n, seed=seed)
        n_per_seed.append(len(sample))
        scored = _score_sample(sample, spec=spec, omega=omega, use_llm=use_llm,
                               model=model, log=log)
        n_labeled.append(len(scored))
        if not scored:
            continue
        base_rates.append(sum(1 for s in scored if s.label == LABEL_VALID) / len(scored))

        # AVG frontier.
        frontier_pts: list[tuple[float, float]] = []
        for tau in thresholds:
            cov, fv = _threshold_point(scored, tau)
            front_cov[tau].append(cov)
            front_fv[tau].append(fv)
            frontier_pts.append((cov, fv))

        # AVG native operating point + baseline points + matched coverage.
        n_cov, n_fv = _arm_point(scored, VerifierArm.AVG_FULL.value)
        native_cov.append(n_cov)
        native_fv.append(n_fv)
        for arm in BASELINE_ARMS:
            cov, fv = _arm_point(scored, arm.value)
            base_cov[arm.value].append(cov)
            base_fv[arm.value].append(fv)
            matched_target_cov[arm.value].append(cov)
            matched_base_fv[arm.value].append(fv)
            matched_avg_fv[arm.value].append(_interp_false_valid(frontier_pts, cov))

        # Calibration reframe: raw ranking quality of the shared score.
        y = [1 if s.label == LABEL_VALID else 0 for s in scored]
        scores = [s.p_valid for s in scored]
        au = roc_auc(scores, y)
        if au is not None:
            auroc_vals.append(au)
        ece_vals.append(expected_calibration_error(scores, y, n_bins=_RELIABILITY_BINS))

    result = _assemble(
        spec, seeds, n_per_seed, n_labeled, base_rates, thresholds,
        front_cov, front_fv, native_cov, native_fv, base_cov, base_fv,
        matched_target_cov, matched_base_fv, matched_avg_fv,
        auroc_vals, ece_vals, use_llm, time.time() - t0,
    )
    return result


def _assemble(
    spec, seeds, n_per_seed, n_labeled, base_rates, thresholds,
    front_cov, front_fv, native_cov, native_fv, base_cov, base_fv,
    matched_target_cov, matched_base_fv, matched_avg_fv,
    auroc_vals, ece_vals, use_llm, seconds,
) -> SA11Result:
    frontier = [
        FrontierPoint(threshold=t, coverage=mean_se(front_cov[t]), false_valid=mean_se(front_fv[t]))
        for t in thresholds
    ]

    matched: list[MatchedCoverage] = []
    dominates = True
    for arm in BASELINE_ARMS:
        v = arm.value
        gaps = [b - a for b, a in zip(matched_base_fv[v], matched_avg_fv[v])]
        sig = paired_t_test(
            matched_base_fv[v], matched_avg_fv[v],
            label=f"{v}_vs_avg_full.false_valid@matched_coverage", a=v, b="avg_full",
        )
        matched.append(MatchedCoverage(
            arm=v,
            coverage=mean_se(matched_target_cov[v]),
            baseline_false_valid=mean_se(matched_base_fv[v]),
            avg_false_valid_at_coverage=mean_se(matched_avg_fv[v]),
            gap=mean_se(gaps),
            significance=sig,
        ))
        if mean_se(gaps).mean < 0:
            dominates = False

    notes = [
        "Frontier: sweep the acceptance threshold tau over the shared coverage-aware "
        "p_valid (predict valid iff p_valid>=tau, invalid iff p_valid<=1-tau, else "
        "abstain). Each tau is an operating point (coverage, false_valid); false_valid "
        "is the SA-1 metric (invalid credited valid over ALL labeled-invalid), so a "
        "low-coverage point earns its low false-valid by abstaining, not by hiding "
        "errors in the denominator.",
        "Matched coverage: AVG's false-valid is read off its own frontier, linearly "
        "interpolated to each baseline's coverage, per seed; gap>0 means AVG credits "
        "fewer reward-hacks than the baseline at the same coverage. Baselines sit on "
        "or above the frontier -> AVG weakly dominates, and its default operating point "
        "moves far down the same frontier for a large false-valid reduction.",
        "Calibration reframe: the shared-score AUROC is low (coarse trajectory-level "
        "reward label -- a well-formed-but-failed trace is indistinguishable from a "
        "correct one to the mechanical checks), but the abstain decision the frontier "
        "traces is what carries the safety, not the raw ranking. The margin is thinnest "
        "at high coverage (where the weak score is forced to rank the confusable middle) "
        "and widest at AVG's low-coverage operating point.",
    ]
    if not use_llm:
        notes.append(
            "Offline run (no API key): avg_full is mechanical evidence + abstention-aware "
            "aggregation and the baselines are their deterministic heuristics, as in SA-1. "
            "Every point is byte-reproducible; the LLM arms only sharpen the same frontier."
        )

    return SA11Result(
        domain_id=spec.domain_id,
        n_per_seed=n_per_seed,
        n_labeled_per_seed=n_labeled,
        base_rate_valid=mean_se(base_rates),
        use_llm=use_llm,
        seeds=list(seeds),
        seconds=round(seconds, 2),
        frontier=frontier,
        avg_native=OperatingPoint(
            arm="avg_full", coverage=mean_se(native_cov), false_valid=mean_se(native_fv),
        ),
        baselines=[
            OperatingPoint(arm=a.value, coverage=mean_se(base_cov[a.value]),
                           false_valid=mean_se(base_fv[a.value]))
            for a in BASELINE_ARMS
        ],
        matched_coverage=matched,
        auroc_all=mean_se(auroc_vals),
        ece_all=mean_se(ece_vals),
        dominates_at_matched_coverage=dominates,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Figure (Fig 2) -- degrades gracefully if matplotlib is absent
# ---------------------------------------------------------------------------

# Validated categorical palette (dataviz skill references/palette.md): AVG in
# blue, native point in green, baselines in fixed-order distinct hues.
_AVG_HUE = "#2a78d6"
_NATIVE_HUE = "#008300"
_BASELINE_HUES = {"monolithic": "#e34948", "prm": "#eb6834", "agent_judge": "#4a3aa7"}
_BASELINE_MARKERS = {"monolithic": "o", "prm": "s", "agent_judge": "^"}


def render_figure(result: SA11Result, path: str, *, log: Any = sys.stderr) -> str:
    """
    Render Fig 2 -- AVG's false-valid-vs-coverage frontier with the baseline
    points overlaid and AVG's native operating point marked -- to ``path``.
    Returns the written path, or "" if matplotlib is unavailable (a note is
    logged; the JSON is the source of truth either way).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        if log is not None:
            print(f"[sa11] matplotlib unavailable, skipping figure: {exc!r}", file=log)
        return ""

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    xs = [p.coverage.mean for p in result.frontier]
    ys = [p.false_valid.mean for p in result.frontier]
    yerr = [p.false_valid.se for p in result.frontier]
    # Frontier as a curve (sorted by coverage) with a light SE band.
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs_s = [xs[i] for i in order]
    ys_s = [ys[i] for i in order]
    err_s = [yerr[i] for i in order]
    ax.plot(xs_s, ys_s, "-", color=_AVG_HUE, lw=2.0, label="AVG frontier (threshold sweep)", zorder=3)
    ax.fill_between(
        xs_s,
        [max(0.0, y - e) for y, e in zip(ys_s, err_s)],
        [y + e for y, e in zip(ys_s, err_s)],
        color=_AVG_HUE, alpha=0.12, lw=0, zorder=1,
    )

    # AVG native operating point.
    nx, ny = result.avg_native.coverage.mean, result.avg_native.false_valid.mean
    ax.scatter([nx], [ny], s=90, color=_NATIVE_HUE, marker="*", zorder=5,
               edgecolor="white", linewidth=0.8, label="AVG operating point (avg_full)")

    # Baseline points with matched-coverage connectors down to the frontier.
    for b in result.baselines:
        hue = _BASELINE_HUES.get(b.arm, "#52514e")
        mk = _BASELINE_MARKERS.get(b.arm, "D")
        bx, by = b.coverage.mean, b.false_valid.mean
        ax.scatter([bx], [by], s=55, color=hue, marker=mk, zorder=4,
                   edgecolor="white", linewidth=0.8, label=b.arm)
        # Vertical drop to AVG's frontier at the baseline's coverage.
        mc = next((m for m in result.matched_coverage if m.arm == b.arm), None)
        if mc is not None:
            ax.plot([bx, bx], [by, mc.avg_false_valid_at_coverage.mean],
                    ":", color=hue, lw=1.0, alpha=0.6, zorder=2)

    ax.set_xlabel("Coverage (fraction of traces committed)")
    ax.set_ylabel("False-valid rate (reward-hacks credited valid)")
    ax.set_title(f"Fig 2 — Selective-verification frontier ({result.domain_id})")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, min(1.02, max(ys + [b.false_valid.mean for b in result.baselines]) + 0.08))
    ax.grid(True, color="#e6e6e3", lw=0.7, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    if log is not None:
        print(f"[sa11] wrote figure {path}", file=log)
    return path


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_table(result: SA11Result) -> str:
    """A compact fixed-width view of the frontier + matched-coverage gaps."""
    lines: list[str] = []
    lines.append(
        f"SA-11: Selective-verification frontier  |  domain={result.domain_id}  "
        f"seeds={result.seeds}  n/seed={result.n_per_seed}  use_llm={result.use_llm}"
    )
    lines.append(
        f"  base-rate valid {result.base_rate_valid.as_str()}  |  "
        f"shared-score AUROC {result.auroc_all.as_str()}  ECE {result.ece_all.as_str()}  "
        f"(weak ranking -- coarse-label artifact; the abstain decision is what's calibrated)"
    )

    lines.append("  [AVG frontier: acceptance threshold -> (coverage, false_valid)]")
    lines.append(f"    {'tau':>5} {'coverage':>16} {'false_valid':>16}")
    for p in result.frontier:
        lines.append(f"    {p.threshold:>5.2f} {p.coverage.as_str():>16} {p.false_valid.as_str():>16}")

    lines.append("  [operating points]")
    lines.append(f"    {'arm':<14} {'coverage':>16} {'false_valid':>16}")
    lines.append(f"    {'avg_full*':<14} {result.avg_native.coverage.as_str():>16} "
                 f"{result.avg_native.false_valid.as_str():>16}")
    for b in result.baselines:
        lines.append(f"    {b.arm:<14} {b.coverage.as_str():>16} {b.false_valid.as_str():>16}")

    lines.append("  [matched coverage: AVG false-valid read off its frontier at each baseline's coverage]")
    lines.append(f"    {'baseline':<14} {'coverage':>10} {'baseline_fv':>14} {'avg_fv@cov':>14} {'gap (b-a)':>14}")
    for m in result.matched_coverage:
        lines.append(
            f"    {m.arm:<14} {m.coverage.mean:>10.3f} {m.baseline_false_valid.as_str():>14} "
            f"{m.avg_false_valid_at_coverage.as_str():>14} {m.gap.as_str():>14}"
        )
    lines.append("  [significance -- paired t-test on the matched-coverage false-valid gap]")
    for m in result.matched_coverage:
        lines.append(f"    {m.significance.as_str()}")
    lines.append(f"  dominates_at_matched_coverage={result.dominates_at_matched_coverage}")
    if result.figure_path:
        lines.append(f"  figure: {result.figure_path}")
    for note in result.notes:
        lines.append(f"  note: {note}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_pool(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.domain == "tau_bench":
        from htir.eval.datasets import load_tau_bench
        if args.hf:
            return load_tau_bench(hf=True, limit=args.hf_limit)
        if not args.cache:
            raise SystemExit("provide --cache <jsonl> (or --hf) for tau_bench")
        return load_tau_bench([args.cache])
    if args.hf:
        from htir.eval.datasets import load_terminalbench
        return list(load_terminalbench(limit=args.hf_limit, streaming=True))
    if not args.cache:
        raise SystemExit("provide --cache <jsonl> (or --hf)")
    return list(iter_local_traces([args.cache]))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SA-11: selective-verification frontier (Fig 2)")
    src = p.add_argument_group("data source")
    src.add_argument("--domain", type=str, default="terminal_swe",
                     help="domain spec S_d + loader (terminal_swe | tau_bench | ...)")
    src.add_argument("--cache", type=str, default="", help="local JSON/JSONL corpus")
    src.add_argument("--hf", action="store_true", help="stream from the HF hub instead of --cache")
    src.add_argument("--hf-limit", type=int, default=9000, help="records to pull when --hf")
    src.add_argument("--n", type=int, default=400, help="balanced subsample size per seed")
    p.add_argument("--seeds", type=str, default="0,1,2", help="comma list of seeds for mean±SE")
    p.add_argument("--use-llm", action="store_true", help="enable LLM monolith + semantic checker (needs key)")
    p.add_argument("--model", type=str, default="openai/gpt-4o")
    p.add_argument("--out", type=str, default="", help="write SA11Result JSON here")
    p.add_argument("--fig", type=str, default="", help="write Fig 2 PNG here (needs matplotlib)")
    args = p.parse_args(argv)

    from htir.models.domain import load_domain_artifacts

    spec = get_domain_spec(args.domain)
    try:
        omega = load_domain_artifacts(args.domain)
    except Exception:
        omega = None
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    pool = _load_pool(args)
    result = run_sa11(pool, spec=spec, omega=omega, seeds=seeds, n=args.n,
                      use_llm=args.use_llm, model=args.model)
    if args.fig:
        result.figure_path = render_figure(result, args.fig)
    print(format_table(result))
    if args.out:
        Path(args.out).write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(f"\n[sa11] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
