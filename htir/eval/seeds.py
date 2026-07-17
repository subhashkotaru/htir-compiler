"""
Multi-seed sweep + mean±SE aggregation (WP-0.1 / avg.tex Sec. 4.7 statistical
reporting).

Every ``experiment_sa*`` runner already takes a ``seed`` that draws a fresh
``balanced_sample`` and is otherwise deterministic, so a *k*-seed sweep is just
"run the experiment on *k* independent balanced subsamples and report the mean
and standard error of each metric." This module is that thin, experiment-
agnostic wrapper:

* :func:`run_multiseed` runs ``run_fn(sample_fn(seed))`` for each seed and
  collects the per-seed result objects (any type).
* An ``extract`` callable flattens one result into a ``{metric_name: value}``
  dict, and :func:`aggregate` reduces the per-seed dicts to a
  ``{metric_name: MeanSE}`` map.

Nothing here knows about a specific experiment's schema -- the caller supplies
the ``extract`` that names the handful of metrics it wants aggregated (false-
valid rate per arm, AUROC, ...), so the same harness serves SA-1 … SA-6.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Callable, Iterable, Sequence, TypeVar

from pydantic import BaseModel, Field

R = TypeVar("R")


class MeanSE(BaseModel):
    """A metric aggregated over seeds: mean, standard error, and the raw values."""
    mean: float = 0.0
    se: float = Field(0.0, description="Standard error of the mean = stdev / sqrt(n)")
    stdev: float = 0.0
    n: int = 0
    values: list[float] = Field(default_factory=list)

    def as_str(self, digits: int = 3) -> str:
        return f"{self.mean:.{digits}f}±{self.se:.{digits}f}"


def mean_se(values: Sequence[float]) -> MeanSE:
    """Mean, sample standard error, and stdev of ``values`` (SE=0 for n<2)."""
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return MeanSE()
    mean = statistics.fmean(vals)
    stdev = statistics.stdev(vals) if n > 1 else 0.0
    se = stdev / math.sqrt(n) if n > 1 else 0.0
    return MeanSE(mean=mean, se=se, stdev=stdev, n=n, values=vals)


def aggregate(
    results: Iterable[R],
    extract: Callable[[R], dict[str, float]],
) -> dict[str, MeanSE]:
    """
    Aggregate a metric map across per-seed results.

    ``extract(result)`` returns a flat ``{metric_name: value}`` dict for one
    seed; the union of metric names across seeds is aggregated, so a metric
    missing in some seed (e.g. an AUROC that was ``None``) is averaged over only
    the seeds that reported it.
    """
    per_metric: dict[str, list[float]] = {}
    for res in results:
        for name, val in extract(res).items():
            if val is None:
                continue
            per_metric.setdefault(name, []).append(float(val))
    return {name: mean_se(vals) for name, vals in per_metric.items()}


class MultiSeedRun(BaseModel):
    """The output of a seed sweep: the seeds, per-seed results, and aggregate."""
    seeds: list[int] = Field(default_factory=list)
    n_per_seed: list[int] = Field(default_factory=list)
    aggregate: dict[str, MeanSE] = Field(default_factory=dict)
    # Per-seed result objects are kept out of the schema (arbitrary types); the
    # caller holds them via the returned ``.results`` attribute if needed.
    model_config = {"arbitrary_types_allowed": True}


def run_multiseed(
    sample_fn: Callable[[int], list[dict[str, Any]]],
    run_fn: Callable[[list[dict[str, Any]]], R],
    seeds: Sequence[int],
    *,
    extract: Callable[[R], dict[str, float]] | None = None,
    log: Any = None,
) -> tuple[MultiSeedRun, list[R]]:
    """
    Run ``run_fn`` over ``k = len(seeds)`` independent balanced subsamples and
    aggregate. ``sample_fn(seed)`` draws the sample for one seed; ``run_fn``
    executes the experiment on it. Returns ``(summary, per_seed_results)`` where
    ``summary.aggregate`` is populated when an ``extract`` is given.
    """
    results: list[R] = []
    n_per_seed: list[int] = []
    for seed in seeds:
        sample = sample_fn(seed)
        n_per_seed.append(len(sample))
        if log is not None:
            print(f"[seeds] seed={seed}: n={len(sample)}", file=log)
        results.append(run_fn(sample))

    agg = aggregate(results, extract) if extract is not None else {}
    summary = MultiSeedRun(seeds=list(seeds), n_per_seed=n_per_seed, aggregate=agg)
    return summary, results


def format_aggregate(agg: dict[str, MeanSE], *, digits: int = 3, title: str = "") -> str:
    """A compact ``metric   mean±se  (values)`` block for logs."""
    lines: list[str] = []
    if title:
        lines.append(title)
    width = max((len(k) for k in agg), default=0)
    for name in sorted(agg):
        m = agg[name]
        vals = ", ".join(f"{v:.{digits}f}" for v in m.values)
        lines.append(f"  {name:<{width}}  {m.as_str(digits)}  (n={m.n}: {vals})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Significance testing on a key gap (avg.tex Sec. 4.7 statistical reporting)
# ---------------------------------------------------------------------------

class PairedGap(BaseModel):
    """A paired significance test of ``a - b`` over per-seed metric values."""
    label: str = ""
    a: str = ""
    b: str = ""
    mean_a: float = 0.0
    mean_b: float = 0.0
    mean_diff: float = 0.0
    se_diff: float = 0.0
    t_stat: float = 0.0
    df: int = 0
    p_value: float = Field(1.0, description="Two-sided paired t-test p-value")
    n_seeds: int = 0
    diffs: list[float] = Field(default_factory=list)

    def as_str(self, digits: int = 3) -> str:
        return (
            f"{self.a} - {self.b} = {self.mean_diff:+.{digits}f}"
            f"±{self.se_diff:.{digits}f}  (t={self.t_stat:.2f}, df={self.df}, "
            f"p={self.p_value:.4f}, n={self.n_seeds})"
        )


def paired_t_test(a_values: Sequence[float], b_values: Sequence[float], *, label: str = "",
                  a: str = "a", b: str = "b") -> PairedGap:
    """
    Two-sided **paired t-test** of ``a - b`` over per-seed values (each seed is a
    matched pair: same balanced subsample scored by both arms). With <2 seeds the
    test is undefined and ``p_value`` stays 1.0. ``scipy`` is used for the exact
    t-distribution p-value when available; otherwise a normal approximation.
    """
    va = [float(x) for x in a_values]
    vb = [float(x) for x in b_values]
    n = min(len(va), len(vb))
    diffs = [va[i] - vb[i] for i in range(n)]
    mean_a = statistics.fmean(va) if va else 0.0
    mean_b = statistics.fmean(vb) if vb else 0.0
    if n < 2:
        return PairedGap(
            label=label, a=a, b=b, mean_a=mean_a, mean_b=mean_b,
            mean_diff=(mean_a - mean_b), n_seeds=n, diffs=diffs,
        )
    mean_d = statistics.fmean(diffs)
    sd_d = statistics.stdev(diffs)
    se_d = sd_d / math.sqrt(n) if sd_d > 0 else 0.0
    df = n - 1
    if se_d == 0.0:
        # No within-pair variance: a nonzero constant gap is as significant as
        # the data can express (p->0); a zero gap is not (p=1).
        t_stat = math.inf if mean_d != 0.0 else 0.0
        p_value = 0.0 if mean_d != 0.0 else 1.0
    else:
        t_stat = mean_d / se_d
        try:
            from scipy import stats  # exact Student-t
            p_value = float(2.0 * stats.t.sf(abs(t_stat), df))
        except Exception:
            # Normal approximation fallback (no scipy).
            p_value = float(math.erfc(abs(t_stat) / math.sqrt(2.0)))
    return PairedGap(
        label=label, a=a, b=b, mean_a=mean_a, mean_b=mean_b,
        mean_diff=mean_d, se_diff=se_d, t_stat=t_stat, df=df,
        p_value=p_value, n_seeds=n, diffs=diffs,
    )
