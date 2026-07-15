"""
Tests for the τ-bench policy domain: the deterministic adapter, the reward /
loader plumbing, the domain spec + Ω_d, the multi-seed harness, the SA-6 policy
perturbations, and the cross-domain transfer cell. All offline (no LLM, no
network); the two corpus-backed checks skip gracefully if the cache is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from htir.adapters import detect_adapter, load_trace
from htir.agents.trace_abstraction import TraceAbstractionAgent
from htir.eval.datasets import (
    normalize_tau_record,
    load_tau_bench,
    to_canonical_steps,
)
from htir.eval.seeds import aggregate, mean_se, run_multiseed
from htir.eval.weak_labels import extract_reward, label_from_reward
from htir.models.domain import get_domain_spec, load_domain_artifacts
from htir.models.htir import CheckerType

TAU_CACHE = Path(__file__).resolve().parent.parent / "data" / "tau_cache" / "tau_all.jsonl"


# A compact synthetic τ-bench trace: authenticate (read), a failed read, a
# successful mutation, and a closing reply. Mirrors the AgentSuite schema.
TAU_TRACE = {
    "task_name": "retail",
    "eval_result": {"score": 1.0, "db_match": True},
    "meta": {"id": "retail_7", "is_correct": True},
    "messages": [
        {"role": "system", "content": "# Retail agent policy\nAuthenticate first."},
        {"role": "user", "content": "cancel my order"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "function": {"name": "find_user_id_by_email", "arguments": '{"email": "a@b.com"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "user_id=U1"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c2", "function": {"name": "get_order_details", "arguments": '{"order_id": "O1"}'}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "Error: order not found"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c3", "function": {"name": "cancel_pending_order", "arguments": '{"order_id": "O2"}'}}]},
        {"role": "tool", "tool_call_id": "c3", "content": "Order O2 cancelled."},
        {"role": "assistant", "content": "Your order has been cancelled. Anything else?"},
    ],
}


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

def test_tau_adapter_detected_over_openai():
    assert detect_adapter(TAU_TRACE["messages"]).name == "tau_bench"


def test_tau_adapter_roles_status_effects():
    steps = load_trace(TAU_TRACE["messages"], adapter="tau_bench")
    roles = [s["role_hint"] for s in steps]
    # find_user (read), get_order (read/failed), cancel (mutate), closing (final)
    assert roles[0] == "read_info"
    assert roles[2] == "mutate_state"
    assert roles[-1] == "final_submission"

    # The failed read is marked failure; the successful mutation success.
    read_fail = steps[1]
    assert read_fail["status_hint"] == "failure"
    mutate = steps[2]
    assert mutate["status_hint"] == "success"

    # The successful mutation records an artifact_change on its target order;
    # the failed read records no (retained) effect.
    effects = mutate.get("artifact_effects") or []
    assert any(e["effect_category"] == "artifact_change" and "O2" in e["affected_resource"] for e in effects)
    assert not (read_fail.get("artifact_effects") or [])


def test_tau_failed_mutation_drops_provenance():
    """A mutation whose tool result is an error must not manufacture provenance."""
    trace = {
        "messages": [
            {"role": "user", "content": "cancel it"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "cancel_pending_order", "arguments": '{"order_id": "O9"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "Error: order is not pending"},
        ]
    }
    steps = load_trace(trace["messages"], adapter="tau_bench")
    mutate = steps[0]
    assert mutate["role_hint"] == "mutate_state"
    assert mutate["status_hint"] == "failure"
    assert not (mutate.get("artifact_effects") or [])


# ---------------------------------------------------------------------------
# Reward + loader plumbing
# ---------------------------------------------------------------------------

def test_extract_reward_tau_fallbacks():
    assert extract_reward({"eval_result": {"score": 1.0}}) == 1
    assert extract_reward({"eval_result": {"score": 0.0}}) == 0
    assert extract_reward({"meta": {"is_correct": True}}) == 1
    assert extract_reward({"meta": {"is_correct": False}}) == 0
    # explicit top-level reward wins
    assert extract_reward({"reward": 0, "eval_result": {"score": 1.0}}) == 0


def test_normalize_tau_record():
    rec = normalize_tau_record(TAU_TRACE)
    assert rec["reward"] == 1
    assert rec["task_name"] == "retail_7"       # meta.id becomes the trace id
    assert rec["tau_domain"] == "retail"
    assert label_from_reward(rec["reward"]) == "valid"


def test_to_canonical_steps_autoroutes():
    # τ-shaped record -> tau_bench adapter
    tau_steps = to_canonical_steps(TAU_TRACE)
    assert any(s["role_hint"] == "mutate_state" for s in tau_steps)
    # terminal-shaped record -> terminal adapter (roles from the terminal vocab)
    term = {"steps": [{"src": "agent", "msg": "run", "tools": [{"fn": "bash", "cmd": "ls"}], "obs": "a\nb"}]}
    term_steps = to_canonical_steps(term)
    assert term_steps[0]["role_hint"] in {"run_command", "run_test", "read_file", "other"}


# ---------------------------------------------------------------------------
# Domain spec + Ω_d + compile
# ---------------------------------------------------------------------------

def test_tau_domain_spec_and_omega_load():
    spec = get_domain_spec("tau_bench")
    assert spec.domain_id == "tau_bench"
    assert "mutate_state" in spec.operation_type_names()
    # both SOP constraints govern mutations
    governed = {r for c in spec.constraints for r in c.applies_to_operations}
    assert "mutate_state" in governed
    omega = load_domain_artifacts("tau_bench")
    assert omega is not None
    kinds = {a.artifact_kind.value for a in omega.artifacts}
    assert "policy" in kinds and "schema" in kinds


def test_tau_compile_flags_unlinked_mutation():
    spec = get_domain_spec("tau_bench")
    omega = load_domain_artifacts("tau_bench")
    agent = TraceAbstractionAgent(domain_spec=spec, domain_artifacts=omega)
    steps = to_canonical_steps(TAU_TRACE)
    htir = agent.compile(task_id="t", raw_steps=steps, harness_snippets={},
                         generate_obligations=True, run_checks=False)
    assert htir.obligations
    # a consequential mutation with no policy link is withheld (policy-sensitive)
    rules = {w.rule_id for w in htir.wellformedness}
    assert "policy_action_unlinked" in rules
    # Ω_d policy seeds semantic policy-compliance obligations on the mutation
    assert any(o.checker == CheckerType.SEMANTIC for o in htir.obligations)


# ---------------------------------------------------------------------------
# Multi-seed harness (WP-0.1)
# ---------------------------------------------------------------------------

def test_mean_se():
    m = mean_se([0.1, 0.2, 0.3])
    assert m.mean == pytest.approx(0.2)
    assert m.se == pytest.approx(m.stdev / (3 ** 0.5))
    assert m.n == 3
    assert mean_se([]).n == 0
    assert mean_se([0.5]).se == 0.0  # SE undefined for n<2 -> 0


def test_run_multiseed_aggregate():
    data = list(range(10))
    summary, results = run_multiseed(
        sample_fn=lambda s: data[s:],
        run_fn=lambda sample: {"n": len(sample)},
        seeds=[0, 1, 2],
        extract=lambda r: {"n": float(r["n"])},
    )
    assert summary.seeds == [0, 1, 2]
    assert summary.n_per_seed == [10, 9, 8]
    assert summary.aggregate["n"].mean == pytest.approx(9.0)
    assert len(results) == 3


def test_aggregate_skips_missing_metric():
    agg = aggregate(
        [{"a": 1.0, "b": 2.0}, {"a": 3.0}],  # 'b' missing in 2nd
        extract=lambda r: {k: r.get(k) for k in ("a", "b")},
    )
    assert agg["a"].n == 2 and agg["b"].n == 1


# ---------------------------------------------------------------------------
# SA-6 policy perturbations + transfer cell
# ---------------------------------------------------------------------------

def test_sa6_perturbations_for_domain():
    from htir.eval.experiment_sa6 import perturbations_for, TAU_PERTURBATIONS, PERTURBATIONS
    assert perturbations_for("tau_bench") is TAU_PERTURBATIONS
    assert perturbations_for("terminal_swe") is PERTURBATIONS
    names = {n for n, _, _ in perturbations_for("tau_bench")}
    assert {"policy_drift", "large_tool_menu", "hidden_state_mismatch"} <= names


def test_sa6_policy_drift_caught_by_avg_not_monolith():
    from htir.eval.experiment_sa6 import run_sa6
    spec = get_domain_spec("tau_bench")
    omega = load_domain_artifacts("tau_bench")
    # one valid-labeled base trace -> perturbations append a reward-hack
    res = run_sa6([normalize_tau_record(TAU_TRACE)], spec=spec, domain_artifacts=omega,
                  progress_every=0, log=None)
    pol = next(p for p in res.perturbations if p.perturbation == "policy_drift")
    by_arm = {c.arm: c for c in pol.cells}
    # monolith credits the injected successful mutation; AVG withholds it
    assert by_arm["monolithic"].false_valid_rate == 1.0
    assert by_arm["avg_integrity"].false_valid_rate == 0.0


def test_transfer_offdiagonal_abstains():
    from htir.eval.experiment_sa2 import evaluate_spec_on_traces
    tau_spec = get_domain_spec("tau_bench")
    # terminal spec applied to a τ trace: roles collapse to 'other' -> abstain
    m = evaluate_spec_on_traces(get_domain_spec("terminal_swe"), [normalize_tau_record(TAU_TRACE)])
    assert m.resolved_fraction == 0.0
    # matched spec resolves *something* (or at least does not error)
    evaluate_spec_on_traces(tau_spec, [normalize_tau_record(TAU_TRACE)],
                            omega=load_domain_artifacts("tau_bench"))


# ---------------------------------------------------------------------------
# Corpus-backed smoke (skips if the cache is not present)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TAU_CACHE.exists(), reason="τ-bench cache not present")
def test_tau_cache_loads_and_labels():
    traces = load_tau_bench([str(TAU_CACHE)], limit=200)
    assert traces
    labels = [label_from_reward(t["reward"]) for t in traces]
    assert set(labels) <= {"valid", "invalid"}
    assert "valid" in labels and "invalid" in labels


# ---------------------------------------------------------------------------
# Dynamic escalation loop (offline no-op path)
# ---------------------------------------------------------------------------

def test_escalation_offline_is_static_noop():
    """With use_llm=False the escalation loop returns the static verdict, no rounds."""
    from htir.agents.escalation import verify_with_escalation
    from htir.agents.checking import check_obligations
    from htir.agents.witness import aggregate
    spec = get_domain_spec("tau_bench")
    omega = load_domain_artifacts("tau_bench")
    agent = TraceAbstractionAgent(domain_spec=spec, domain_artifacts=omega)

    steps = to_canonical_steps(TAU_TRACE)
    # reference: plain static check+aggregate
    h_ref = agent.compile(task_id="t", raw_steps=steps, harness_snippets={},
                          generate_obligations=True, run_checks=False, domain_artifacts=omega)
    check_obligations(h_ref, spec, use_semantic=False, domain_artifacts=omega)
    ref = aggregate(h_ref)

    h = agent.compile(task_id="t", raw_steps=steps, harness_snippets={},
                      generate_obligations=True, run_checks=False, domain_artifacts=omega)
    res = verify_with_escalation(h, spec, use_llm=False, domain_artifacts=omega)
    assert res.rounds == 0 and res.n_escalated == 0 and res.n_resolved == 0
    assert res.llm_calls == 0
    assert res.aggregate.predicted_status == ref.predicted_status
