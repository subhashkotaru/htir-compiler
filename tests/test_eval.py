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


def test_evaluate_predictions_failure_flag_precision_recall():
    # 3 invalid, 2 valid. Failure flags (predicted 'invalid'): two, one correct.
    preds = ["invalid", "invalid", "uncertain", "valid", "uncertain"]
    labels = ["invalid", "valid", "invalid", "valid", "invalid"]
    m = evaluate_predictions(preds, labels)
    # precision: of 2 predicted-invalid, 1 truly invalid -> 0.5
    assert m.failure_flag_precision == 0.5
    # recall: of 3 truly invalid, 1 flagged -> 1/3
    assert round(m.failure_flag_recall, 3) == round(1 / 3, 3)


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


def test_normalize_steps_field_decodes_json_string():
    # The HF terminalbench dataset serializes `steps` as a JSON string; the
    # adapters need a decoded list. normalize_steps_field must decode it.
    import json as _json

    from htir.eval.datasets import normalize_steps_field

    steps = [{"src": "agent", "msg": "hi", "tools": [{"fn": "bash_command", "cmd": "ls"}], "obs": "x"}]
    raw = {"task_name": "t", "reward": 1, "steps": _json.dumps(steps)}
    fixed = normalize_steps_field(raw)
    assert isinstance(fixed["steps"], list) and fixed["steps"] == steps
    # An already-decoded list is returned untouched (same object identity ok).
    already = {"steps": steps}
    assert normalize_steps_field(already)["steps"] == steps
    # A bash_command tool must now type as a shell run, not 'other'.
    canonical = to_canonical_steps(raw)
    assert canonical and canonical[0]["role_hint"] == "run_command"


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


# ---------------------------------------------------------------------------
# SA-1 experiment harness (run_sa1)
# ---------------------------------------------------------------------------

def test_run_sa1_end_to_end_offline():
    """run_sa1 compiles + scores every arm offline and reports the Q1
    contrast: AVG's false-valid rate must not exceed the monolith's."""
    import json as _json

    from htir.agents.baselines import VerifierArm
    from htir.eval.experiment_sa1 import run_sa1

    files = sorted(DATA_DIR.glob("0*_*.json"))
    if not files:
        pytest.skip("committed real traces not present")

    # Feed traces with a JSON-string `steps` field to also exercise the HF-shape
    # decode path inside the harness.
    traces = []
    for f in files:
        raw = dict(list(iter_local_traces([f]))[0])
        raw["steps"] = _json.dumps(raw["steps"])
        traces.append(raw)

    result = run_sa1(traces, long_horizon_steps=1, progress_every=0, log=None)
    assert result.n_traces == len(files)
    assert result.n_labeled >= 1
    arms = {a.arm: a for a in result.arms}
    assert set(arms) == {a.value for a in VerifierArm}
    # exec_only never issues an LLM call; monolithic issues exactly one/trace.
    assert arms["exec_only"].total_llm_calls == 0
    assert arms["monolithic"].mean_llm_calls == 1.0
    # The Q1 headline holds in miniature.
    assert (
        arms["avg_full"].overall.false_valid_rate
        <= arms["monolithic"].overall.false_valid_rate
    )


# ---------------------------------------------------------------------------
# SA-2 experiment harness (run_sa2): universal-only vs. adapters (Q2)
# ---------------------------------------------------------------------------

def test_sa2_omega_bundle_loads_from_disk():
    """The committed terminal_swe Omega_d bundle (schema/policy/test) loads and
    carries the identifiers the obligation/checking layers key on."""
    from htir.models.domain import ArtifactKind, load_domain_artifacts

    bundle = load_domain_artifacts("terminal_swe")
    assert bundle is not None
    kinds = {a.artifact_kind for a in bundle.artifacts}
    assert {ArtifactKind.SCHEMA, ArtifactKind.POLICY, ArtifactKind.TEST} <= kinds
    # The schema artifact's identifier must match the hinted artifact type so
    # _inject_omega_schema_evidence can resolve it.
    assert bundle.get(ArtifactKind.SCHEMA, "test_report") is not None
    assert bundle.get(ArtifactKind.POLICY, "no-test-mutation") is not None


def test_sa2_adaptation_levels_are_monotone_in_budget():
    """build_adaptation_levels yields an ordered universal->adapter axis whose
    first level is universal-only and last carries the Omega_d bundle."""
    from htir.eval.experiment_sa2 import build_adaptation_levels

    levels = build_adaptation_levels()
    assert [l.name for l in levels][0] == "universal_only"
    assert levels[-1].name == "adapter_full_omega"
    assert levels[0].spec.domain_id == "default"
    assert levels[0].omega is None
    assert levels[-1].omega is not None
    budgets = [l.budget for l in levels]
    assert budgets == sorted(budgets)  # non-decreasing adaptation budget
    # Headline-only keeps exactly the three plan arms.
    head = build_adaptation_levels(include_intermediate=False)
    assert [l.name for l in head] == ["universal_only", "adapter_full", "adapter_full_omega"]


def test_run_sa2_end_to_end_offline():
    """run_sa2 recompiles each trace per adaptation level and reports the Q2
    contrast: the universal-only arm binds no dischargeable obligation on
    terminal traces (resolves nothing / abstains), while the adapter arm can
    resolve at least as much, at no higher false-valid rate."""
    import json as _json

    from htir.eval.experiment_sa2 import build_adaptation_levels, run_sa2

    files = sorted(DATA_DIR.glob("0*_*.json"))
    if not files:
        pytest.skip("committed real traces not present")

    traces = []
    for f in files:
        raw = dict(list(iter_local_traces([f]))[0])
        raw["steps"] = _json.dumps(raw["steps"])  # exercise the HF-shape decode path
        traces.append(raw)

    result = run_sa2(traces, long_horizon_steps=1, progress_every=0, log=None)
    assert result.n_traces == len(files)
    arms = {a.arm: a for a in result.arms}
    assert "universal_only" in arms and "adapter_full" in arms and "adapter_full_omega" in arms

    uni = arms["universal_only"].overall
    adapt = arms["adapter_full"].overall
    # Universal-only never commits on this domain (abstains everywhere).
    assert uni.resolved_fraction == 0.0
    assert uni.abstention_rate == 1.0
    # The adapter resolves at least as much, and never at a higher false-valid rate.
    assert adapt.resolved_fraction >= uni.resolved_fraction
    assert adapt.false_valid_rate <= arms["adapter_full"].overall.false_valid_rate + 1e-9
    # Omega_d adds semantic-routed obligation coverage over the bare adapter.
    assert (
        arms["adapter_full_omega"].mean_semantic_obligations
        >= arms["adapter_full"].mean_semantic_obligations
    )
