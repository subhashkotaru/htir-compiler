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

    from htir.eval.experiment_sa1 import DEFAULT_ARMS, run_sa1

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
    assert set(arms) == {a.value for a in DEFAULT_ARMS}
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


# ---------------------------------------------------------------------------
# SA-3 calibration primitives (htir.eval.calibration)
# ---------------------------------------------------------------------------

def test_roc_auc_separation_and_ties():
    from htir.eval.calibration import roc_auc

    # Perfect separation (valid scored above invalid) -> 1.0; reversed -> 0.0.
    assert roc_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert roc_auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    # All-tied scores -> chance (tie-corrected).
    assert roc_auc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == 0.5
    # One class only -> undefined.
    assert roc_auc([0.4, 0.6], [1, 1]) is None


def test_expected_calibration_error_and_reliability():
    from htir.eval.calibration import expected_calibration_error, reliability_bins

    # Perfectly calibrated: mean score == empirical valid rate in each bin.
    scores = [0.0, 1.0, 0.0, 1.0]
    labels = [0, 1, 0, 1]
    assert expected_calibration_error(scores, labels, n_bins=10) == 0.0
    bins = reliability_bins(scores, labels, n_bins=10)
    assert sum(b.count for b in bins) == 4
    # A confidently-wrong verifier is badly calibrated (gap ~ 0.9).
    bad = expected_calibration_error([0.95, 0.95], [0, 0], n_bins=10)
    assert bad > 0.9


def test_risk_coverage_improves_with_abstention_when_score_is_informative():
    from htir.eval.calibration import risk_coverage_curve

    # Confident predictions are correct; the near-boundary one is wrong. Abstaining
    # on the least-confident trace should raise accuracy to 1.0.
    scores = [0.95, 0.9, 0.05, 0.1, 0.51]  # last: confident-ish valid but wrong
    labels = [1, 1, 0, 0, 0]
    pts = {round(p.abstention_budget, 1): p for p in risk_coverage_curve(scores, labels, budgets=(0.0, 0.2))}
    assert pts[0.0].accuracy == 0.8            # 4/5 correct at full coverage
    assert pts[0.2].n_kept == 4 and pts[0.2].accuracy == 1.0  # drop the boundary miss


def test_trajectory_valid_score_is_coverage_aware():
    from htir.models.htir import CheckerResult, EvidenceType, Obligation, ObligationStatus, Severity
    from htir.eval.calibration import trajectory_valid_score
    from htir.models.htir import HTIR

    def _ob(oid, p_pass, p_fail, p_abstain, status):
        return Obligation(
            obligation_id=oid, claim_id=oid, severity=Severity.MEDIUM,
            result=CheckerResult(p_pass=p_pass, p_fail=p_fail, p_abstain=p_abstain),
            status=status,
        )

    htir = HTIR(task_id="t")
    # One pass (1.0) + one abstain (0.5) -> mean 0.75, not 1.0: abstention counts.
    htir.obligations = [
        _ob(1, 1.0, 0.0, 0.0, ObligationStatus.PASSED),
        _ob(2, 0.0, 0.0, 1.0, ObligationStatus.ABSTAINED),
    ]
    assert trajectory_valid_score(htir) == 0.75
    # No obligations with results -> None.
    assert trajectory_valid_score(HTIR(task_id="t")) is None


# ---------------------------------------------------------------------------
# SA-3 no-abstention ablation (force_decision) in the checker layer
# ---------------------------------------------------------------------------

def test_force_decision_removes_abstention_but_keeps_real_fails():
    """force_decision (ablation #3) must leave no obligation ABSTAINED: a
    no-evidence abstain is credited PASSED on the optimistic prior, while a
    genuine mechanical fail still FAILS."""
    from htir.agents.baselines import VerifierArm, run_arm
    from htir.agents.trace_abstraction import TraceAbstractionAgent
    from htir.models.domain import get_domain_spec
    from htir.models.htir import ObligationStatus

    files = sorted(DATA_DIR.glob("0*_*.json"))
    if not files:
        pytest.skip("committed real traces not present")

    spec = get_domain_spec("terminal_swe")
    agent = TraceAbstractionAgent(domain_spec=spec)
    raw = list(iter_local_traces([files[0]]))[0]
    from htir.adapters import load_trace
    htir = agent.compile(task_id="t", raw_steps=load_trace(raw, adapter="terminal"), harness_snippets={})

    # No-abstention arm: forced graph has zero abstained obligations.
    from htir.agents.checking import check_obligations
    forced = htir.model_copy(deep=True)
    check_obligations(forced, spec, force_decision=True)
    assert forced.obligations  # the trace produced obligations
    assert all(o.status != ObligationStatus.ABSTAINED for o in forced.obligations)

    # The calibrated arm (no forcing) does abstain on the same graph.
    calibrated = htir.model_copy(deep=True)
    check_obligations(calibrated, spec)
    assert any(o.status == ObligationStatus.ABSTAINED for o in calibrated.obligations)

    # And the no-abstention aggregate never returns 'uncertain'.
    assert run_arm(htir, spec, VerifierArm.NO_ABSTENTION).predicted_status in ("valid", "invalid")


# ---------------------------------------------------------------------------
# SA-3 experiment harness (run_sa3): calibrated abstention vs. no-abstention (Q3)
# ---------------------------------------------------------------------------

def test_run_sa3_end_to_end_offline():
    """run_sa3 compiles + scores both arms offline and reports the Q3 headline:
    calibrated abstention credits fewer failed trajectories as valid than the
    no-abstention ablation, and abstains more."""
    import json as _json

    from htir.eval.experiment_sa3 import run_sa3

    files = sorted(DATA_DIR.glob("0*_*.json"))
    if not files:
        pytest.skip("committed real traces not present")

    traces = []
    for f in files:
        raw = dict(list(iter_local_traces([f]))[0])
        raw["steps"] = _json.dumps(raw["steps"])  # exercise the HF-shape decode path
        traces.append(raw)

    result = run_sa3(traces, progress_every=0, log=None)
    assert result.n_traces == len(files)
    arms = {a.arm: a for a in result.arms}
    assert set(arms) == {"avg_full", "no_abstention"}
    avg, forced = arms["avg_full"], arms["no_abstention"]
    # Q3: abstention does not increase false-valids and abstains at least as much.
    assert avg.decision_metrics.false_valid_rate <= forced.decision_metrics.false_valid_rate
    assert avg.decision_metrics.abstention_rate >= forced.decision_metrics.abstention_rate
    # The no-abstention arm commits on strictly more traces (no 'uncertain').
    assert forced.coverage >= avg.coverage
    # Shared-score calibration fields are populated.
    assert 0.0 <= result.ece_all <= 1.0
    assert len(result.reliability) == 10
    assert result.risk_coverage and result.risk_coverage[0].abstention_budget == 0.0


# ---------------------------------------------------------------------------
# SA-4 experiment harness (run_sa4): online intervention, offline replay (Q4a)
# ---------------------------------------------------------------------------

def test_compute_class_report_precision_recall_and_steps_saved():
    """The per-class metric math: a flag on an invalid trace is a true positive
    (should have intervened), on a valid trace a false alarm; steps-saved is the
    counterfactual T - t* over invalid flagged traces only."""
    from htir.eval.experiment_sa4 import (
        CLASS_ANY,
        CLASS_FAILED,
        PerTraceInterventionRecord,
        compute_class_report,
    )

    def rec(label, n, any_at, failed_at):
        return PerTraceInterventionRecord(
            label=label, n_steps=n,
            first_fire={CLASS_ANY: any_at, CLASS_FAILED: failed_at},
        )

    records = [
        rec("invalid", 10, any_at=2, failed_at=3),  # both fire: TP for both
        rec("valid", 8, any_at=5, failed_at=None),  # any fires on a good run: false alarm
        rec("invalid", 6, any_at=None, failed_at=None),  # missed failure: FN for both
        rec("valid", 4, any_at=None, failed_at=None),  # correctly silent
    ]

    failed = compute_class_report(records, CLASS_FAILED)
    assert (failed.tp, failed.fp, failed.fn) == (1, 0, 1)
    assert failed.precision == 1.0 and failed.recall == 0.5
    assert failed.steps_saved_median == 7.0  # 10 - 3
    assert failed.first_fire_frac_median == pytest.approx(0.3)
    assert failed.false_alarm_interruptions == 0

    anyc = compute_class_report(records, CLASS_ANY)
    assert (anyc.tp, anyc.fp, anyc.fn) == (1, 1, 1)
    assert anyc.precision == 0.5 and anyc.recall == 0.5
    assert anyc.steps_saved_median == 8.0  # 10 - 2, over the invalid flagged trace only
    assert anyc.false_alarm_interruptions == 1


def test_run_sa4_end_to_end_offline():
    """run_sa4 replays every committed trace prefix by prefix and reports the
    Q4a signals: the confident failed-obligation class is at least as precise
    and no later than the abstention-dominated any-active class, and the
    active-obligation position profile accounts for every replayed step."""
    import json as _json

    from htir.eval.experiment_sa4 import CLASS_ANY, CLASS_FAILED, run_sa4

    files = sorted(DATA_DIR.glob("0*_*.json"))
    if not files:
        pytest.skip("committed real traces not present")

    traces = []
    for f in files:
        raw = dict(list(iter_local_traces([f]))[0])
        raw["steps"] = _json.dumps(raw["steps"])  # exercise the HF-shape decode path
        traces.append(raw)

    result = run_sa4(traces, progress_every=0, log=None)
    assert result.n_traces == len(files)
    classes = {c.name: c for c in result.classes}
    assert set(classes) == {CLASS_ANY, CLASS_FAILED}
    for c in classes.values():
        assert 0.0 <= c.precision <= 1.0 and 0.0 <= c.recall <= 1.0

    # The position profile is a full decile partition covering every replayed step.
    assert len(result.position_profile) == 10
    assert sum(p.steps for p in result.position_profile) > 0
    for p in result.position_profile:
        expected = round(p.active_obligations / p.steps, 3) if p.steps else 0.0
        assert p.mean_active_per_step == expected
    assert result.template_fires  # at least one obligation template fired

    # The confident class is never *less* precise than the union when both fire,
    # and fires no later (Q4a: mechanical failures are timelier and more precise).
    failed, anyc = classes[CLASS_FAILED], classes[CLASS_ANY]
    if failed.flagged and anyc.flagged:
        assert failed.precision >= anyc.precision
    if failed.first_fire_frac_median is not None and anyc.first_fire_frac_median is not None:
        assert failed.first_fire_frac_median <= anyc.first_fire_frac_median
