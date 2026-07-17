"""
Tests for SA-10 -- SWE-Gym as the third domain + the 3x3 cross-domain transfer
matrix (Q2). All offline (``use_llm=False``, no network) and byte-deterministic:
the SWE-Gym fixture is a miniature OpenHands rollout, and the transfer fixture
exercises the transfer *safety property* -- a universal or off-family verifier
abstains, it never over-credits -- which is exact (0.0) at any sample size.
"""

from __future__ import annotations

import json
from collections import Counter

from htir.agents.trace_abstraction import TraceAbstractionAgent
from htir.eval.datasets import (
    _swe_gym_messages_to_turns,
    _swe_gym_tool_call,
    load_swe_gym,
    normalize_swe_gym_record,
    to_canonical_steps,
)
from htir.eval.experiment_sa10 import run_sa10
from htir.eval.weak_labels import extract_reward, label_from_reward
from htir.models.domain import get_domain_spec, load_domain_artifacts


# ---------------------------------------------------------------------------
# Fixtures: miniature OpenHands SWE-Gym rollouts (raw record schema)
# ---------------------------------------------------------------------------

def _swe_record(instance_id: str, *, resolved: bool, rich: bool) -> dict:
    """
    A minimal ``SWE-Gym/OpenHands-Sampled-Trajectories`` record. ``rich`` edits a
    source file then re-validates with a reproducer run (binds obligations);
    otherwise it is a short view-only rollout (binds nothing, the failed-rollout
    shape). ``resolved`` sets the boolean reward.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "<pr_description>KeyError on describe.</pr_description>"},
    ]
    if rich:
        messages += [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {
                    "name": "str_replace_editor",
                    "arguments": json.dumps({"command": "view", "path": "/workspace/pkg"})}}]},
            {"role": "tool", "name": "str_replace_editor", "tool_call_id": "c1",
             "content": "OBSERVATION:\n/workspace/pkg/mod.py"},
            {"role": "assistant", "content": "Reproduce the bug.", "tool_calls": [
                {"id": "c2", "type": "function", "function": {
                    "name": "execute_bash",
                    "arguments": json.dumps({"command": "python reproduce_error.py"})}}]},
            {"role": "tool", "name": "execute_bash", "tool_call_id": "c2",
             "content": "OBSERVATION:\nTraceback (most recent call last): KeyError"},
            {"role": "assistant", "content": "Fix the source.", "tool_calls": [
                {"id": "c3", "type": "function", "function": {
                    "name": "str_replace_editor",
                    "arguments": json.dumps({"command": "str_replace", "path": "/workspace/pkg/mod.py"})}}]},
            {"role": "tool", "name": "str_replace_editor", "tool_call_id": "c3",
             "content": "OBSERVATION:\nThe file /workspace/pkg/mod.py has been edited."},
            {"role": "assistant", "content": "Re-run the reproducer.", "tool_calls": [
                {"id": "c4", "type": "function", "function": {
                    "name": "execute_bash",
                    "arguments": json.dumps({"command": "python -m pytest -q"})}}]},
            {"role": "tool", "name": "execute_bash", "tool_call_id": "c4",
             "content": "OBSERVATION:\n3 passed\n<returncode>0</returncode>"},
        ]
    else:
        messages += [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "v1", "type": "function", "function": {
                    "name": "str_replace_editor",
                    "arguments": json.dumps({"command": "view", "path": "/workspace/pkg"})}}]},
            {"role": "tool", "name": "str_replace_editor", "tool_call_id": "v1",
             "content": "OBSERVATION:\n/workspace/pkg"},
        ]
    messages.append({"role": "assistant", "content": "I have finished.", "tool_calls": [
        {"id": "f1", "type": "function", "function": {"name": "finish", "arguments": "{}"}}]})
    return {"instance_id": instance_id, "run_id": "test-run", "resolved": resolved, "messages": messages}


def _swe_domain(n_each: int = 3) -> list[dict]:
    recs = []
    for i in range(n_each):
        recs.append(_swe_record(f"repo__pkg-{i}", resolved=True, rich=True))
        recs.append(_swe_record(f"repo__pkg-bad-{i}", resolved=False, rich=False))
    return [normalize_swe_gym_record(r) for r in recs]


def _terminal_trace(task_id: str, reward: int, ok: bool) -> dict:
    rc = "0" if ok else "1"
    return {
        "task_name": task_id, "reward": reward,
        "steps": [
            {"src": "user", "msg": "fix the bug", "tools": [], "obs": None},
            {"src": "agent", "msg": "edit", "tools": [{"fn": "edit_file", "cmd": "foo.py"}],
             "obs": f"<returncode>{rc}</returncode>"},
            {"src": "agent", "msg": "test", "tools": [{"fn": "bash", "cmd": "pytest -q"}],
             "obs": f"<returncode>{rc}</returncode>\n{'3 passed' if ok else '1 failed'}"},
        ],
    }


def _terminal_domain(n_each: int = 3) -> list[dict]:
    out = []
    for i in range(n_each):
        out.append(_terminal_trace(f"term-ok-{i}", 1, True))
        out.append(_terminal_trace(f"term-bad-{i}", 0, False))
    return out


def _tau_trace(task_id: str, reward: int, authed: bool) -> dict:
    messages = [
        {"role": "system", "content": "Authenticate before acting."},
        {"role": "user", "content": "cancel my order"},
    ]
    if authed:
        messages += [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "a", "function": {"name": "find_user_id_by_email",
                                         "arguments": '{"email": "a@b.com"}'}}]},
            {"role": "tool", "tool_call_id": "a", "content": "user_id=U1"},
        ]
    messages += [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c", "function": {"name": "cancel_pending_order",
                                     "arguments": '{"order_id": "O1"}'}}]},
        {"role": "tool", "tool_call_id": "c", "content": "Order O1 cancelled."},
        {"role": "assistant", "content": "Done."},
    ]
    return {"task_name": "retail", "reward": reward, "meta": {"id": task_id}, "messages": messages}


def _tau_domain(n_each: int = 3) -> list[dict]:
    out = []
    for i in range(n_each):
        out.append(_tau_trace(f"retail-ok-{i}", 1, True))
        out.append(_tau_trace(f"retail-bad-{i}", 0, False))
    return out


# ---------------------------------------------------------------------------
# Loader: normalize_swe_gym_record / load_swe_gym
# ---------------------------------------------------------------------------

def test_swe_gym_tool_call_mapping():
    # execute_bash -> a shell op carrying the command (so test/exit detection fires)
    assert _swe_gym_tool_call({"function": {"name": "execute_bash",
                                            "arguments": '{"command": "pytest"}'}}) == {"fn": "bash", "cmd": "pytest"}
    # str_replace_editor view -> read; str_replace -> edit; both carry the path
    assert _swe_gym_tool_call({"function": {"name": "str_replace_editor",
                                            "arguments": '{"command": "view", "path": "/w/x.py"}'}}) == {
        "fn": "view", "cmd": "/w/x.py"}
    assert _swe_gym_tool_call({"function": {"name": "str_replace_editor",
                                            "arguments": '{"command": "str_replace", "path": "/w/x.py"}'}}) == {
        "fn": "str_replace_editor", "cmd": "/w/x.py"}
    # finish -> final submission
    assert _swe_gym_tool_call({"function": {"name": "finish", "arguments": "{}"}})["fn"] == "finish"
    # malformed arguments never raise
    assert _swe_gym_tool_call({"function": {"name": "execute_bash", "arguments": "not json"}}) == {
        "fn": "bash", "cmd": ""}


def test_swe_gym_messages_to_turns_pairs_observations():
    rec = _swe_record("repo__pkg-1", resolved=True, rich=True)
    turns = _swe_gym_messages_to_turns(rec["messages"])
    # system dropped; one user request turn; the rest are agent turns
    assert turns[0]["src"] == "user"
    assert all(t["src"] in ("user", "agent") for t in turns)
    # the edit turn's observation is paired from the following tool message
    edit_turns = [t for t in turns if any(tl["fn"] == "str_replace_editor" for tl in t["tools"])]
    assert edit_turns and "edited" in (edit_turns[0]["obs"] or "")


def test_normalize_swe_gym_record_reward_and_taskname():
    rec = normalize_swe_gym_record(_swe_record("getmoto__moto-5321", resolved=True, rich=True))
    assert rec["reward"] == 1 and rec["resolved"] is True
    assert rec["task_name"] == "getmoto__moto-5321"
    assert isinstance(rec["steps"], list) and rec["steps"]
    assert label_from_reward(extract_reward(rec)) == "valid"
    assert normalize_swe_gym_record(_swe_record("x", resolved=False, rich=False))["reward"] == 0


def test_normalize_swe_gym_record_idempotent():
    once = normalize_swe_gym_record(_swe_record("repo__pkg-1", resolved=True, rich=True))
    twice = normalize_swe_gym_record(once)
    assert twice["steps"] == once["steps"]
    assert twice["reward"] == once["reward"] == 1


def test_swe_gym_routes_to_terminal_adapter_and_binds_roles():
    rec = normalize_swe_gym_record(_swe_record("repo__pkg-1", resolved=True, rich=True))
    steps = to_canonical_steps(rec)
    roles = Counter(s.get("role_hint") for s in steps)
    # the rich rollout binds real terminal operations, not just 'other'
    assert roles["edit_file"] >= 1
    assert roles["run_command"] >= 1 or roles["run_test"] >= 1
    # and the swe_gym spec generates obligations over them (the matched adapter binds)
    htir = TraceAbstractionAgent(domain_spec=get_domain_spec("swe_gym")).compile(
        task_id="repo__pkg-1", raw_steps=steps, harness_snippets={},
        generate_obligations=True, use_semantic_analysis=False, run_checks=False,
    )
    assert len(htir.obligations) > 0


def test_load_swe_gym_local_balances(tmp_path):
    path = tmp_path / "swe.jsonl"
    with path.open("w") as f:
        for i in range(3):
            f.write(json.dumps(_swe_record(f"r-{i}", resolved=True, rich=True)) + "\n")
            f.write(json.dumps(_swe_record(f"b-{i}", resolved=False, rich=False)) + "\n")
    traces = load_swe_gym([str(path)])
    assert len(traces) == 6
    labels = Counter(label_from_reward(extract_reward(t)) for t in traces)
    assert labels["valid"] == 3 and labels["invalid"] == 3


# ---------------------------------------------------------------------------
# 3x3 transfer: the safety property (offline, byte-deterministic)
# ---------------------------------------------------------------------------

def _domains() -> dict:
    return {
        "terminal_swe": {"spec": get_domain_spec("terminal_swe"),
                         "omega": load_domain_artifacts("terminal_swe"), "traces": _terminal_domain()},
        "tau_bench": {"spec": get_domain_spec("tau_bench"),
                      "omega": load_domain_artifacts("tau_bench"), "traces": _tau_domain()},
        "swe_gym": {"spec": get_domain_spec("swe_gym"),
                    "omega": load_domain_artifacts("swe_gym"), "traces": _swe_domain()},
    }


def test_sa10_universal_floor_and_cross_family_abstain():
    res = run_sa10(_domains(), seeds=[0, 1, 2], n=6, log=None)
    # universal_only binds nothing -> resolves nothing anywhere (the floor).
    for td in res.test_domains:
        c = res.cell("universal_only", td)
        assert c is not None and c.resolved_fraction.mean == 0.0
    assert res.headline["universal_only_resolved_fraction_max"] == 0.0

    # Cross-FAMILY safety: policy(tau) vs terminal-family cells abstain and never
    # over-credit -- resolved_fraction 0 and false_valid 0, both directions.
    for a, b in (("tau_bench", "terminal_swe"), ("tau_bench", "swe_gym"),
                 ("terminal_swe", "tau_bench"), ("swe_gym", "tau_bench")):
        c = res.cell(a, b)
        assert c is not None and c.cross_family
        assert c.resolved_fraction.mean == 0.0
        assert c.false_valid.mean == 0.0
    assert res.headline["cross_family_false_valid_max"] == 0.0


def test_sa10_three_seeds_and_significance_shape():
    res = run_sa10(_domains(), seeds=[0, 1, 2], n=6, log=None)
    assert res.seeds == [0, 1, 2]
    assert len(res.per_seed) == 3
    # one diagonal-vs-universal paired test per test domain, each with 3 seeds
    assert {g.n_seeds for g in res.significance} == {3}
    assert len(res.significance) == len(res.test_domains)


def test_sa10_byte_deterministic():
    a = run_sa10(_domains(), seeds=[0, 1, 2], n=6, log=None).model_dump_json(indent=2)
    b = run_sa10(_domains(), seeds=[0, 1, 2], n=6, log=None).model_dump_json(indent=2)
    assert a == b
