"""
Calibration + abstention metrics (avg.tex Sec. 4.4 "Verifier metrics"; SA-3 /
Q3 "calibrated abstention").

SA-1 scored a verifier by the *discrete* decision it committed to
(valid / invalid / uncertain). Q3 asks a sharper question: is the verifier's
*confidence* well calibrated, and does declining to answer (abstention) buy
robustness? Answering it needs a continuous per-trajectory score plus metrics
that reward a verifier for being uncertain exactly where it is wrong.

This module is pure Python (no numpy / sklearn) so the offline path keeps no
new dependency, and every metric is deterministic.

Scores. Each trajectory gets one coverage-aware probability-of-valid
``p_valid in [0, 1]`` (:func:`trajectory_valid_score`): the severity-weighted
mean over its obligations of ``p_pass + 0.5 * p_abstain`` -- a discharged pass
contributes 1, a fail 0, and an *abstention the uninformative 0.5*. Abstention
therefore pulls confidence toward the decision boundary rather than being
silently dropped, so the score is consistent with the aggregator's decision
(a broadly-abstained trajectory scores near 0.5, i.e. "uncertain", instead of
being credited on one incidental pass).

Because the no-abstention ablation forces each abstaining checker onto the same
0.5 prior, both arms share this identical score by construction: the ablation
is purely an *abstention policy* over one calibrated score. That is what SA-3
isolates -- given the same confidence, does declining to answer on the
low-confidence traces (the calibrated arm) beat committing on all of them?

Metrics.
* :func:`roc_auc` -- tie-aware ROC AUROC (Mann-Whitney form). Reported both
  over *all* traces and over only the *committed* traces
  (abstention-calibrated AUROC): the calibrated arm is allowed to withhold a
  verdict on the traces it abstains on, which is the whole point of abstention.
* :func:`reliability_bins` / :func:`expected_calibration_error` -- the
  reliability diagram of ``p_valid`` against the empirical valid rate, and its
  ECE (the gap area). This is calibration of the probability itself, which is
  what the reliability-diagram deliverable plots.
* :func:`risk_coverage_curve` -- precision / false-valid rate as a function of
  how much the verifier is allowed to abstain (rank by confidence
  ``|p_valid - 0.5|``, drop the least-confident budget). This is "precision at
  fixed abstention budgets": it shows abstention removing exactly the traces a
  verifier is least sure of.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

from pydantic import BaseModel, Field

from htir.agents.witness import SEVERITY_WEIGHT
from htir.models.htir import HTIR

# The decision boundary a probability-of-valid score is thresholded at, and the
# uninformative prior an abstained obligation / fully-uncertain trace sits on.
DECISION_BOUNDARY = 0.5


# ---------------------------------------------------------------------------
# Per-trajectory continuous score
# ---------------------------------------------------------------------------

def trajectory_valid_score(htir: HTIR) -> Optional[float]:
    """
    Aggregate a checked ``htir`` into a coverage-aware probability-of-valid in
    [0, 1]: the severity-weighted mean over its obligations of
    ``p_pass + 0.5 * p_abstain``. A mechanically discharged pass contributes 1,
    a fail 0, and an abstention the uninformative 0.5 -- so broad abstention
    pulls the score toward the boundary (honest uncertainty) instead of letting
    a lone incidental pass credit the trajectory.

    Returns ``None`` only when no obligation carries a result at all (nothing to
    score); callers impute the 0.5 boundary for such traces.
    """
    num = 0.0
    den = 0.0
    for ob in htir.obligations:
        if ob.result is None:
            continue
        w = SEVERITY_WEIGHT[ob.severity]
        num += w * (ob.result.p_pass + 0.5 * ob.result.p_abstain)
        den += w
    if den == 0.0:
        return None
    return num / den


# ---------------------------------------------------------------------------
# ROC AUROC (tie-aware, Mann-Whitney U form)
# ---------------------------------------------------------------------------

def _average_ranks(values: Sequence[float]) -> list[float]:
    """1-based ranks of ``values`` ascending, ties assigned their mean rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # mean of 1-based ranks i+1..j+1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    """
    ROC AUROC of ``scores`` against binary ``labels`` (1 = valid, 0 = invalid)
    via the tie-corrected Mann-Whitney U statistic. Returns ``None`` unless
    both classes are present (AUROC is undefined otherwise). Higher = better
    ranking of valid above invalid; 0.5 = chance.
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must be the same length")
    n_pos = sum(1 for l in labels if l == 1)
    n_neg = sum(1 for l in labels if l == 0)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _average_ranks(scores)
    sum_pos = sum(r for r, l in zip(ranks, labels) if l == 1)
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# ---------------------------------------------------------------------------
# Reliability diagram + ECE
# ---------------------------------------------------------------------------

class ReliabilityBin(BaseModel):
    """One equal-width confidence bin of a reliability diagram."""
    lo: float
    hi: float
    count: int = 0
    mean_score: float = 0.0        # mean predicted p_valid in the bin
    empirical_valid: float = 0.0   # observed fraction valid in the bin
    gap: float = 0.0               # |mean_score - empirical_valid|


def reliability_bins(
    scores: Sequence[float], labels: Sequence[int], *, n_bins: int = 10,
) -> list[ReliabilityBin]:
    """
    Bin ``scores`` into ``n_bins`` equal-width [0, 1] buckets and, per bucket,
    report the mean predicted ``p_valid`` vs. the empirical valid rate -- the
    points of a reliability diagram. A perfectly calibrated verifier has
    ``mean_score == empirical_valid`` in every populated bin.
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must be the same length")
    edges = [i / n_bins for i in range(n_bins + 1)]
    bins: list[ReliabilityBin] = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        # last bin is closed on the right so score == 1.0 lands somewhere.
        idx = [
            i for i, s in enumerate(scores)
            if (s >= lo and s < hi) or (b == n_bins - 1 and s == hi)
        ]
        if idx:
            ms = sum(scores[i] for i in idx) / len(idx)
            ev = sum(labels[i] for i in idx) / len(idx)
            bins.append(ReliabilityBin(
                lo=lo, hi=hi, count=len(idx), mean_score=ms,
                empirical_valid=ev, gap=abs(ms - ev),
            ))
        else:
            bins.append(ReliabilityBin(lo=lo, hi=hi, count=0))
    return bins


def expected_calibration_error(
    scores: Sequence[float], labels: Sequence[int], *, n_bins: int = 10,
) -> float:
    """
    Expected calibration error: the count-weighted mean over reliability bins
    of ``|mean_score - empirical_valid|``. 0 = perfectly calibrated ``p_valid``;
    larger = more over/under-confident.
    """
    n = len(scores)
    if n == 0:
        return 0.0
    bins = reliability_bins(scores, labels, n_bins=n_bins)
    return sum(b.count / n * b.gap for b in bins)


# ---------------------------------------------------------------------------
# Risk-coverage: precision at fixed abstention budgets
# ---------------------------------------------------------------------------

class RiskCoveragePoint(BaseModel):
    """Verifier quality when allowed to abstain on the least-confident budget."""
    abstention_budget: float = Field(..., description="Fraction of traces abstained on")
    coverage: float = Field(..., description="Fraction of traces committed to (1 - budget)")
    n_kept: int = 0
    accuracy: float = 0.0          # of kept traces, fraction correctly classified
    false_valid_rate: float = 0.0  # of kept invalid traces, fraction called valid
    valid_precision: float = 0.0   # of kept traces called valid, fraction truly valid


def risk_coverage_curve(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    budgets: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
) -> list[RiskCoveragePoint]:
    """
    Precision / accuracy / false-valid rate as a function of abstention budget.

    Traces are ranked by confidence ``|p_valid - 0.5|`` (distance from the
    decision boundary); at budget ``b`` the least-confident ``b`` fraction is
    abstained on and the metrics are computed over the retained, thresholded
    predictions (``valid`` iff ``p_valid >= 0.5``). A verifier whose confidence
    is meaningful sees accuracy/precision rise and false-valid rate fall as the
    budget grows -- abstention removing exactly the traces it is least sure of.
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must be the same length")
    n = len(scores)
    order = sorted(range(n), key=lambda i: abs(scores[i] - DECISION_BOUNDARY), reverse=True)
    points: list[RiskCoveragePoint] = []
    for b in budgets:
        keep = n - int(math.floor(b * n))
        kept = order[:keep]
        if not kept:
            points.append(RiskCoveragePoint(abstention_budget=b, coverage=0.0))
            continue
        correct = kept_invalid = kept_false_valid = pred_valid = pred_valid_correct = 0
        for i in kept:
            pred_valid_i = scores[i] >= DECISION_BOUNDARY
            true_valid = labels[i] == 1
            if pred_valid_i == true_valid:
                correct += 1
            if not true_valid:
                kept_invalid += 1
                if pred_valid_i:
                    kept_false_valid += 1
            if pred_valid_i:
                pred_valid += 1
                if true_valid:
                    pred_valid_correct += 1
        points.append(RiskCoveragePoint(
            abstention_budget=round(b, 3),
            coverage=round(len(kept) / n, 4),
            n_kept=len(kept),
            accuracy=correct / len(kept),
            false_valid_rate=(kept_false_valid / kept_invalid) if kept_invalid else 0.0,
            valid_precision=(pred_valid_correct / pred_valid) if pred_valid else 0.0,
        ))
    return points
