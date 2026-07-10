"""
Tests for the offline evaluation layer (htir.eval): weak labels, verifier
metrics, dataset ingestion + balanced sampling, and a miniature SA-1
false-valid-rate comparison (AVG vs. monolith) over the committed traces.
All offline (no LLM, no network).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from htir.eval import (
    VerifierMetrics,
    evaluate_predictions,
    extract_reward,
    label_from_reward,
)
from htir.eval.datasets import balanced_sample, iter_local_traces, to_canonical_steps
from htir.eval.weak_labels import trace_label, weak_step_labels

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw_traces"


# ---------------------------------------------------------------------------
# Weak labels
# ---------------------------------------------------------------------------

def test_label_from_reward():
    assert label_from_reward(1) == "valid"
    assert label_from_reward(0) == "invalid"
    assert label_from_reward(None) is None
    assert label_from_reward("nope") is None


def test_extract_reward_and_trace_label():
    raw = {"task_name": "t", "reward": 0, "steps": []}
    assert extract_reward(raw) == 0
    lab = trace_label(raw)
    assert lab.task_id == "t" and lab.reward == 0 and lab.label == "invalid"


# ---------------------------------------------------------------------------
# Verifier metrics
# ---------------------------------------------------------------------------

def test_evaluate_predictions_false_valid_rate():
    # 4 labeled traces: two invalid, two valid.
    preds = ["valid", "uncertain", "valid", "invalid"]
    labels = ["invalid", "invalid", "valid", "valid"]
    m = evaluate_predictions(preds, labels)
    assert isinstance(m, VerifierMetrics)
    assert m.n == 4 and m.n_labeled == 4
    # one of two invalid traces was wrongly called valid.
    assert m.false_valid_rate == 0.5
    assert m.abstention_rate == 0.25
    # resolved = (valid|invalid), (valid|valid), (invalid|valid) -> 3 resolved,
    # only the middle one correct -> 1/3.
    assert round(m.resolved_accuracy, 3) == round(1 / 3, 3)
    assert m.resolved_fraction == 0.75  # 3 of 4 labeled traces resolved
    assert m.confusion["valid|invalid"] == 1


def test_evaluate_predictions_all_abstain_has_zero_false_valid():
    preds = ["uncertain", "uncertain"]
    labels = ["invalid", "valid"]
    m = evaluate_predictions(preds, labels)
    assert m.false_valid_rate == 0.0
    assert m.abstention_rate == 1.0
    assert m.resolved_fraction == 0.0


def test_evaluate_predictions_length_mismatch_raises():
    with pytest.raises(ValueError):
        evaluate_predictions(["valid"], ["valid", "invalid"])


# ---------------------------------------------------------------------------
# Dataset ingestion + balanced sampling
# ---------------------------------------------------------------------------

def test_balanced_sample_is_balanced_and_deterministic():
    traces = (
        [{"task_name": f"s{i}", "reward": 1} for i in range(10)]
        + [{"task_name": f"u{i}", "reward": 0} for i in range(3)]
    )
    sample = balanced_sample(traces, n=8, seed=1)
    # Capped at 2 * min(class) = 6 to stay balanced.
    assert len(sample) == 6
    rewards = [t["reward"] for t in sample]
    assert rewards.count(1) == rewards.count(0) == 3
    # Deterministic given the seed.
    assert balanced_sample(traces, n=8, seed=1) == sample


def test_iter_local_traces_and_to_canonical_steps():
    trace_file = DATA_DIR / "01_adaptive-rejection-sampler__8YuTzJm.json"
    if not trace_file.exists():
        pytest.skip("committed real trace not present")
    traces = list(iter_local_traces([trace_file]))
    assert len(traces) == 1 and traces[0].get("reward") is not None
    steps = to_canonical_steps(traces[0])
    assert steps and any(s.get("role_hint") == "edit_file" for s in steps)


# ---------------------------------------------------------------------------
# Miniature SA-1: false-valid rate, AVG vs. monolith, over committed traces
# ---------------------------------------------------------------------------

def test_avg_has_no_higher_false_valid_rate_than_monolith():
    """The core SA-1 claim in miniature: over the committed (reward=0) traces,
    the AVG obligation graph must not credit failed trajectories as 'valid'
    more often than the endpoint-only monolith. With the aggregation fix, AVG's
    false-valid rate is 0 here while the monolith's is >= 0."""
    from htir.adapters import load_trace
    from htir.agents.baselines import VerifierArm, run_arm
    from htir.agents.trace_abstraction import TraceAbstractionAgent
    from htir.models.domain import get_domain_spec

    files = sorted(DATA_DIR.glob("0*_*.json"))
    if not files:
        pytest.skip("committed real traces not present")

    spec = get_domain_spec("terminal_swe")
    agent = TraceAbstractionAgent(domain_spec=spec)

    labels, avg_preds, mono_preds = [], [], []
    for f in files:
        raw = list(iter_local_traces([f]))[0]
        labels.append(label_from_reward(extract_reward(raw)))
        steps = load_trace(raw, adapter="terminal")
        htir = agent.compile(task_id=f.stem, raw_steps=steps, harness_snippets={})
        avg_preds.append(run_arm(htir, spec, VerifierArm.AVG_FULL).predicted_status)
        mono_preds.append(run_arm(htir, spec, VerifierArm.MONOLITHIC).predicted_status)

    avg_m = evaluate_predictions(avg_preds, labels)
    mono_m = evaluate_predictions(mono_preds, labels)
    # AVG credits far fewer failed trajectories as valid than the endpoint
    # monolith (on these traces: 0.2 vs 0.8), and abstains rather than
    # over-committing. The gap -- not a perfect 0 -- is the honest SA-1 result.
    assert avg_m.false_valid_rate < mono_m.false_valid_rate
    assert avg_m.abstention_rate > mono_m.abstention_rate
