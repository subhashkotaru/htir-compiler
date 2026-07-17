"""
Offline regression tests for the SA-8 competitive-baseline arms (``prm`` and
``agent_judge``) added to :mod:`htir.agents.baselines`.

Both arms have a byte-deterministic offline realization that runs without an API
key: the ``prm`` arm is a step-heuristic process reward model, and the
``agent_judge`` arm degrades to a deterministic multi-hop step-outcome gather.
These tests build synthetic HTIRs by hand (no LLM calls) and assert the arm
verdicts, that the arms are wired into ``run_arm`` / ``DEFAULT_ARMS``, that the
offline path is deterministic, and that the paired-t significance helper behaves.
"""

from __future__ import annotations

from htir.agents.baselines import (
    VerifierArm,
    agent_judge_verdict,
    prm_verdict,
    run_arm,
)
from htir.eval.experiment_sa1 import DEFAULT_ARMS, _llm_calls_for_arm
from htir.eval.seeds import paired_t_test
from htir.models.htir import HTIR, ExecutionStatus, TraceStep


def _step(step_id: int, role: str, status: ExecutionStatus, response: str = "ok") -> TraceStep:
    return TraceStep(
        step_id=step_id, request_message="req", response_message=response,
        role=role, execution_status=status,
    )


def _htir(task_id: str, steps: list[TraceStep]) -> HTIR:
    return HTIR(task_id=task_id, steps=steps)


# ---------------------------------------------------------------------------
# PRM (process reward model) arm
# ---------------------------------------------------------------------------

def test_prm_credits_all_success_trace_valid():
    """A trajectory whose visible steps all succeed is credited valid -- the
    over-crediting failure mode of a step reward model on a plausible trace."""
    h = _htir("all-success", [
        _step(1, "edit_file", ExecutionStatus.SUCCESS),
        _step(2, "run_test", ExecutionStatus.SUCCESS, "3 passed"),
    ])
    assert prm_verdict(h).predicted_status == "valid"


def test_prm_flags_all_failure_trace_invalid():
    h = _htir("all-fail", [
        _step(1, "run_command", ExecutionStatus.FAILURE, "boom"),
        _step(2, "run_test", ExecutionStatus.FAILURE, "1 failed"),
    ])
    assert prm_verdict(h).predicted_status == "invalid"


def test_prm_commits_on_weak_label_steps():
    """A trace with no execution signal still gets a committed verdict (never
    abstains) -- the PRM must score every step, including weak-label ones."""
    h = _htir("unknown", [
        _step(1, "other", ExecutionStatus.UNKNOWN),
        _step(2, "other", ExecutionStatus.UNKNOWN),
    ])
    v = prm_verdict(h)
    assert v.predicted_status in ("valid", "invalid")  # committed, not 'uncertain'
    assert v.predicted_status == "valid"  # neutral 0.5 aggregate -> optimistic commit


def test_prm_error_marker_downgrades_unknown_step():
    h = _htir("errmark", [
        _step(1, "other", ExecutionStatus.UNKNOWN, "Traceback (most recent call last)"),
        _step(2, "other", ExecutionStatus.UNKNOWN, "Error: failed to build"),
    ])
    assert prm_verdict(h).predicted_status == "invalid"


def test_prm_abstains_only_on_empty_trace():
    assert prm_verdict(_htir("empty", [])).predicted_status == "uncertain"


def test_prm_min_aggregation_is_stricter():
    """Under min-aggregation a single failing step vetoes the trajectory."""
    h = _htir("mixed", [
        _step(1, "edit_file", ExecutionStatus.SUCCESS),
        _step(2, "edit_file", ExecutionStatus.SUCCESS),
        _step(3, "run_test", ExecutionStatus.FAILURE, "1 failed"),
    ])
    assert prm_verdict(h, aggregation="mean").predicted_status == "valid"
    assert prm_verdict(h, aggregation="min").predicted_status == "invalid"


def test_prm_deterministic():
    h = _htir("det", [
        _step(1, "edit_file", ExecutionStatus.SUCCESS),
        _step(2, "run_test", ExecutionStatus.SUCCESS, "1 passed"),
        _step(3, "run_command", ExecutionStatus.UNKNOWN),
    ])
    first = prm_verdict(h)
    for _ in range(5):
        again = prm_verdict(h)
        assert again.predicted_status == first.predicted_status
        assert again.uncertainty == first.uncertainty


# ---------------------------------------------------------------------------
# Agent-as-a-Judge arm (offline deterministic fallback)
# ---------------------------------------------------------------------------

def test_agent_judge_fooled_by_plausible_invalid_trace():
    """The offline evidence-gather still commits 'valid' when every visible step
    passes -- the plausible-but-invalid long trace it is fooled by."""
    h = _htir("plausible", [
        _step(1, "edit_file", ExecutionStatus.SUCCESS),
        _step(2, "run_command", ExecutionStatus.SUCCESS),
        _step(3, "run_test", ExecutionStatus.SUCCESS, "2 passed"),
    ])
    assert agent_judge_verdict(h).predicted_status == "valid"


def test_agent_judge_flags_unresolved_failure():
    """Multi-hop scan (beyond the endpoint): a failing step with no later
    success anywhere is flagged invalid even though a non-validation step is last."""
    h = _htir("unresolved", [
        _step(1, "run_test", ExecutionStatus.FAILURE, "1 failed"),
        _step(2, "read_info", ExecutionStatus.UNKNOWN),
    ])
    assert agent_judge_verdict(h).predicted_status == "invalid"


def test_agent_judge_abstains_without_observable_steps():
    h = _htir("noobs", [_step(1, "other", ExecutionStatus.UNKNOWN)])
    assert agent_judge_verdict(h).predicted_status == "uncertain"


def test_agent_judge_deterministic():
    h = _htir("det2", [
        _step(1, "edit_file", ExecutionStatus.SUCCESS),
        _step(2, "run_test", ExecutionStatus.FAILURE, "1 failed"),
        _step(3, "run_test", ExecutionStatus.SUCCESS, "1 passed"),
    ])
    first = agent_judge_verdict(h).predicted_status
    for _ in range(5):
        assert agent_judge_verdict(h).predicted_status == first


# ---------------------------------------------------------------------------
# Wiring into the arm framework
# ---------------------------------------------------------------------------

def test_run_arm_dispatches_new_arms_offline():
    h = _htir("dispatch", [
        _step(1, "edit_file", ExecutionStatus.SUCCESS),
        _step(2, "run_test", ExecutionStatus.SUCCESS, "1 passed"),
    ])
    # spec is unused by the PRM / agent_judge branches (both special-cased,
    # like MONOLITHIC), so None is a valid argument here.
    assert run_arm(h, None, VerifierArm.PRM).predicted_status == "valid"
    assert run_arm(h, None, VerifierArm.AGENT_JUDGE).predicted_status == "valid"


def test_new_arms_in_default_arm_list():
    names = {a.value for a in DEFAULT_ARMS}
    assert {"prm", "agent_judge"}.issubset(names)


def test_cost_proxy_for_new_arms():
    # PRM would issue one step-critic call per step; agent_judge one judge pass
    # (budget-matched to the monolith).
    assert _llm_calls_for_arm(VerifierArm.PRM, n_semantic=2, n_steps=7, use_llm=False) == 7
    assert _llm_calls_for_arm(VerifierArm.AGENT_JUDGE, n_semantic=2, n_steps=7, use_llm=False) == 1
    assert _llm_calls_for_arm(VerifierArm.MONOLITHIC, n_semantic=2, n_steps=7, use_llm=False) == 1


# ---------------------------------------------------------------------------
# Headline: PRM / agent_judge over-credit relative to full AVG (offline)
# ---------------------------------------------------------------------------

def test_prm_and_agent_judge_over_credit_vs_abstaining_avg():
    """On a batch of plausible-but-invalid traces the offline AVG arm (exec_only,
    abstain-aware) withholds credit while prm and agent_judge over-commit valid."""
    traces = [
        _htir(f"inv-{i}", [
            _step(1, "edit_file", ExecutionStatus.SUCCESS),
            _step(2, "run_test", ExecutionStatus.SUCCESS, "1 passed"),
        ])
        for i in range(5)
    ]
    prm_valid = sum(prm_verdict(h).predicted_status == "valid" for h in traces)
    judge_valid = sum(agent_judge_verdict(h).predicted_status == "valid" for h in traces)
    assert prm_valid == 5 and judge_valid == 5


# ---------------------------------------------------------------------------
# Significance helper
# ---------------------------------------------------------------------------

def test_paired_t_test_detects_consistent_gap():
    gap = paired_t_test([0.64, 0.65, 0.66], [0.15, 0.16, 0.16],
                        label="mono_vs_avg", a="monolithic", b="avg_full")
    assert gap.mean_diff > 0.4
    assert gap.n_seeds == 3 and gap.df == 2
    assert gap.p_value < 0.05


def test_paired_t_test_undefined_below_two_seeds():
    gap = paired_t_test([0.6], [0.1])
    assert gap.n_seeds == 1
    assert gap.p_value == 1.0
