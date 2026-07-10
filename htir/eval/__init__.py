"""
Offline evaluation utilities for AVG verifier experiments (avg.tex Sec. 4.4).

Two layers, both dependency-light and usable without an API key:

* ``weak_labels`` -- turn the trajectory-level supervision that recorded traces
  *do* carry (``reward in {0,1}``, per-step exit codes) into weak labels, and
  score a verifier's ``predicted_status`` against them (false-valid rate,
  resolved accuracy, abstention rate, ...). This is what the SA-1/SA-3
  packages report before the hand-labeled gold slice exists.
* ``datasets`` -- ingest the ``yoonholee/terminalbench-trajectories`` HF set
  (or any local JSON/JSONL in the same turn schema) and draw a balanced
  solved/unsolved sample.
"""

from __future__ import annotations

from htir.eval.weak_labels import (
    TraceLabel,
    VerifierMetrics,
    evaluate_predictions,
    extract_reward,
    label_from_reward,
    weak_step_labels,
)

__all__ = [
    "TraceLabel",
    "VerifierMetrics",
    "evaluate_predictions",
    "extract_reward",
    "label_from_reward",
    "weak_step_labels",
]
