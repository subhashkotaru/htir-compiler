"""
Weak labels + verifier metrics (avg.tex Sec. 4.4, "Verifier metrics").

Recorded agent traces rarely carry step-level gold labels, but they usually
carry *trajectory-level* supervision: a ``reward in {0, 1}`` (solved / not
solved) and, per step, an exit code or error marker. This module turns that
weak supervision into labels and scores a verifier's ``predicted_status``
against them.

The headline number is the **false-valid rate**: the fraction of *failed*
trajectories (reward = 0) a verifier nonetheless credits as ``valid``. Driving
this to zero is the entire point of the aggregation fix (avg.tex Sec. 3.9) and
the reason abstention exists -- a verifier should say ``uncertain`` rather than
hand out unsupported credit. ``resolved_accuracy`` then measures how often the
verifier is *right* when it does commit to valid/invalid, and
``abstention_rate`` how often it declines.

These are trajectory-level metrics computable today from ``reward`` alone; the
step/obligation-level metrics (AUROC, ECE, evidence-localization quality) need
the hand-labeled gold slice and are out of scope here.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from pydantic import BaseModel, Field

from htir.models.htir import ExecutionStatus, HTIR

# Canonical trajectory labels derived from reward.
LABEL_VALID = "valid"
LABEL_INVALID = "invalid"
# The verifier statuses we treat as a committed (resolved) decision vs. a
# declined one.
_RESOLVED_STATUSES = frozenset({LABEL_VALID, LABEL_INVALID})
STATUS_UNCERTAIN = "uncertain"


class TraceLabel(BaseModel):
    """A weak, trajectory-level label derived from recorded supervision."""
    task_id: str = ""
    reward: Optional[int] = Field(None, description="Trajectory reward in {0, 1} if recorded")
    label: Optional[str] = Field(None, description="'valid'/'invalid' derived from reward, or None")
    source: str = Field("reward", description="Where the label came from")


def label_from_reward(reward: Any) -> Optional[str]:
    """
    Map a trajectory reward to a weak label: ``1`` (or any truthy non-zero
    numeric) -> ``valid``; ``0`` -> ``invalid``; ``None``/unparseable -> None
    (unknown, excluded from labeled metrics rather than guessed).
    """
    if reward is None:
        return None
    try:
        r = float(reward)
    except (TypeError, ValueError):
        return None
    return LABEL_VALID if r != 0.0 else LABEL_INVALID


def extract_reward(raw_trace: Any) -> Optional[int]:
    """Read the trajectory ``reward`` from a raw trace dict (turn schema)."""
    if isinstance(raw_trace, dict):
        r = raw_trace.get("reward")
        if r is None:
            return None
        try:
            return int(round(float(r)))
        except (TypeError, ValueError):
            return None
    return None


def trace_label(raw_trace: Any, *, task_id: str = "") -> TraceLabel:
    """Build a :class:`TraceLabel` from a raw trace dict."""
    reward = extract_reward(raw_trace)
    tid = task_id or (str(raw_trace.get("task_name", "")) if isinstance(raw_trace, dict) else "")
    return TraceLabel(task_id=tid, reward=reward, label=label_from_reward(reward))


def weak_step_labels(htir: HTIR) -> dict[int, str]:
    """
    Per-step weak outcome labels from parsed execution status: ``success`` /
    ``failure`` for steps whose status is known, omitting UNKNOWN steps. Useful
    as weak step-level truth (e.g. for intervention precision/recall) until the
    gold slice exists.
    """
    labels: dict[int, str] = {}
    for step in htir.steps_in_order():
        if step.execution_status == ExecutionStatus.SUCCESS:
            labels[step.step_id] = "success"
        elif step.execution_status in (
            ExecutionStatus.FAILURE, ExecutionStatus.TIMEOUT, ExecutionStatus.BLOCKED,
        ):
            labels[step.step_id] = "failure"
    return labels


class VerifierMetrics(BaseModel):
    """Trajectory-level verifier quality against weak reward labels."""
    n: int = 0
    n_labeled: int = 0

    false_valid_rate: float = Field(
        0.0, description="P(predicted 'valid' | label 'invalid') -- the headline metric",
    )
    false_invalid_rate: float = Field(
        0.0, description="P(predicted 'invalid' | label 'valid')",
    )
    resolved_accuracy: float = Field(
        0.0, description="Accuracy among labeled traces the verifier resolved (valid/invalid)",
    )
    resolved_fraction: float = Field(
        0.0, description="Fraction of labeled traces the verifier resolved rather than abstaining",
    )
    abstention_rate: float = Field(
        0.0, description="Fraction predicted 'uncertain'",
    )
    valid_precision: float = Field(
        0.0, description="Of traces predicted 'valid', fraction truly valid",
    )
    valid_recall: float = Field(
        0.0, description="Of truly valid traces, fraction predicted 'valid'",
    )
    confusion: dict[str, int] = Field(
        default_factory=dict,
        description="'{predicted}|{label}' -> count over labeled traces",
    )


def evaluate_predictions(
    predicted_statuses: Sequence[str],
    labels: Sequence[Optional[str]],
) -> VerifierMetrics:
    """
    Score verifier ``predicted_statuses`` against weak ``labels`` (each
    ``'valid'``/``'invalid'``/``None``). Traces whose label is ``None`` count
    toward ``n`` and ``abstention_rate`` but are excluded from the
    label-conditioned rates. All rates are guarded against division by zero.
    """
    if len(predicted_statuses) != len(labels):
        raise ValueError("predicted_statuses and labels must be the same length")

    n = len(predicted_statuses)
    abstain = sum(1 for p in predicted_statuses if p == STATUS_UNCERTAIN)

    confusion: dict[str, int] = {}
    labeled_pairs = [(p, l) for p, l in zip(predicted_statuses, labels) if l is not None]
    n_labeled = len(labeled_pairs)
    for p, l in labeled_pairs:
        confusion[f"{p}|{l}"] = confusion.get(f"{p}|{l}", 0) + 1

    def _rate(numer: int, denom: int) -> float:
        return numer / denom if denom else 0.0

    n_invalid = sum(1 for _, l in labeled_pairs if l == LABEL_INVALID)
    n_valid = sum(1 for _, l in labeled_pairs if l == LABEL_VALID)

    false_valid = sum(1 for p, l in labeled_pairs if l == LABEL_INVALID and p == LABEL_VALID)
    false_invalid = sum(1 for p, l in labeled_pairs if l == LABEL_VALID and p == LABEL_INVALID)

    resolved = [(p, l) for p, l in labeled_pairs if p in _RESOLVED_STATUSES]
    resolved_correct = sum(1 for p, l in resolved if p == l)

    predicted_valid = sum(1 for p, l in labeled_pairs if p == LABEL_VALID)
    true_valid_pred_valid = sum(1 for p, l in labeled_pairs if p == LABEL_VALID and l == LABEL_VALID)

    return VerifierMetrics(
        n=n,
        n_labeled=n_labeled,
        false_valid_rate=_rate(false_valid, n_invalid),
        false_invalid_rate=_rate(false_invalid, n_valid),
        resolved_accuracy=_rate(resolved_correct, len(resolved)),
        resolved_fraction=_rate(len(resolved), n_labeled),
        abstention_rate=_rate(abstain, n),
        valid_precision=_rate(true_valid_pred_valid, predicted_valid),
        valid_recall=_rate(true_valid_pred_valid, n_valid),
        confusion=confusion,
    )
