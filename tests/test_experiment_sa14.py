"""
Offline regression tests for SA-14 (Track M -- Meta-Harness vs. Base Agent on
Terminal-Bench 2.0).

No live calls: builds synthetic captured-trial records by hand, in the same
OpenAI-messages-with-tool_calls shape ``scripts/live_meta_harness_tb2.py`` is
expected to write, and exercises ``htir.eval.datasets.normalize_meta_harness_record``
/ ``load_meta_harness_tb2`` and ``htir.eval.experiment_sa14.run_sa14`` fully
offline (``use_llm=False``), asserting the headline contrast is byte-
deterministic: a harness whose visible steps all succeed but never runs a test
is credited ``false_valid`` by the endpoint monolith while AVG's mechanical
+abstention-aware arm withholds credit (abstains) instead.
"""

from __future__ import annotations

from htir.eval.datasets import normalize_meta_harness_record
from htir.eval.experiment_sa14 import HARNESS_BASE, HARNESS_META, run_sa14
from htir.models.domain import get_domain_spec


def _tool_call(call_id: str, name: str, **args) -> dict:
    return {"id": call_id, "function": {"name": name, "arguments": args}}


def _tb2_trial(task_id: str, *, harness: str, model: str, solved: bool) -> dict:
    """
    A minimal captured TB2 trial: an edit, optionally a validating test run
    (only on the solved path), and a final message. ``solved=False`` still
    produces an all-visible-steps-succeed transcript (the plausible-but-invalid
    trace AVG must abstain on rather than credit, and the monolith/PRM-style
    endpoint heuristic over-credits) -- it just skips the test-run step, so
    reward=0 despite a clean-looking trace.
    """
    messages: list[dict] = [
        {"role": "user", "content": f"Fix the bug for {task_id}."},
        {"role": "assistant", "content": None, "tool_calls": [
            _tool_call("e1", "str_replace_based_edit_tool", command="str_replace", path="solution.py"),
        ]},
        {"role": "tool", "tool_call_id": "e1", "content": "File solution.py edited."},
    ]
    if solved:
        messages += [
            {"role": "assistant", "content": None, "tool_calls": [
                _tool_call("t1", "bash", command="pytest -q"),
            ]},
            {"role": "tool", "tool_call_id": "t1", "content": "1 passed in 0.02s\n<returncode>0</returncode>"},
        ]
    messages.append({"role": "assistant", "content": "Done, the fix is complete."})

    return {
        "task_id": task_id,
        "harness": harness,
        "model": model,
        "reward": 1 if solved else 0,
        "messages": messages,
    }


def _fixture(harness: str, model: str = "openai/gpt-4o-mini") -> list[dict]:
    """4 tasks per harness: 3 solved (validated) + 1 unresolved (no test run, reward 0)."""
    trials = [_tb2_trial(f"tb2_{harness}_{i}", harness=harness, model=model, solved=True) for i in range(3)]
    trials.append(_tb2_trial(f"tb2_{harness}_3", harness=harness, model=model, solved=False))
    return trials


def _run(**kw):
    spec = get_domain_spec("terminal_swe")
    traces = [normalize_meta_harness_record(r) for r in (_fixture(HARNESS_META) + _fixture(HARNESS_BASE))]
    return run_sa14(traces, spec=spec, seeds=[0, 1], progress_every=0, log=None, **kw)


# ---------------------------------------------------------------------------
# normalize_meta_harness_record
# ---------------------------------------------------------------------------

def test_normalize_produces_turn_schema_steps():
    rec = normalize_meta_harness_record(_tb2_trial("t1", harness=HARNESS_META, model="openai/gpt-4o-mini", solved=True))
    assert isinstance(rec["steps"], list) and rec["steps"]
    assert rec["reward"] == 1
    assert rec["harness"] == HARNESS_META
    assert rec["task_name"] == "t1"


def test_normalize_accepts_reward_aliases():
    base = _tb2_trial("t2", harness=HARNESS_BASE, model="m", solved=False)
    del base["reward"]
    base["passed"] = True
    assert normalize_meta_harness_record(base)["reward"] == 1


def test_normalize_is_idempotent_on_turn_schema():
    rec = normalize_meta_harness_record(_tb2_trial("t3", harness=HARNESS_META, model="m", solved=True))
    again = normalize_meta_harness_record(rec)
    assert again["reward"] == rec["reward"]
    assert again["steps"] == rec["steps"]


# ---------------------------------------------------------------------------
# run_sa14
# ---------------------------------------------------------------------------

def test_both_harnesses_reported():
    result = _run()
    names = {h.harness for h in result.harnesses}
    assert names == {HARNESS_META, HARNESS_BASE}


def test_task_success_rate_matches_ground_truth():
    result = _run()
    for h in result.harnesses:
        # 3 of 4 tasks solved per harness, by construction.
        assert abs(h.task_success_rate.mean - 0.75) < 1e-9


def test_monolithic_over_credits_the_unvalidated_trace():
    """The endpoint monolith trusts the last successful step even with no test
    run -> credits the unresolved trace 'valid' -> false_valid_rate > 0."""
    result = _run()
    for h in result.harnesses:
        mono = next(a for a in h.arms if a.arm == "monolithic")
        assert mono.false_valid_rate.mean > 0.0


def test_avg_full_abstains_instead_of_crediting():
    """avg_full (offline: collapses onto exec_only) withholds credit on the same
    unvalidated trace rather than crediting it -- lower false_valid than monolith."""
    result = _run()
    for h in result.harnesses:
        avg = next(a for a in h.arms if a.arm == "avg_full")
        mono = next(a for a in h.arms if a.arm == "monolithic")
        assert avg.false_valid_rate.mean <= mono.false_valid_rate.mean


def test_deterministic_offline():
    first = _run()
    again = _run()
    assert first.model_dump_json() == again.model_dump_json()


def test_significance_gaps_populated_when_both_harnesses_present():
    result = _run()
    assert result.success_gap.n_seeds == 2
    assert result.false_valid_gap.n_seeds == 2


def test_cli_runs_from_written_cache(tmp_path):
    import json
    from htir.eval.experiment_sa14 import main

    meta_path = tmp_path / "meta_harness.jsonl"
    base_path = tmp_path / "base_agent.jsonl"
    meta_path.write_text("\n".join(json.dumps(r) for r in _fixture(HARNESS_META)), encoding="utf-8")
    base_path.write_text("\n".join(json.dumps(r) for r in _fixture(HARNESS_BASE)), encoding="utf-8")
    out_path = tmp_path / "sa14_results.json"

    rc = main([
        "--cache", str(meta_path), "--cache", str(base_path),
        "--seeds", "0,1", "--out", str(out_path),
    ])
    assert rc == 0
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert {h["harness"] for h in payload["harnesses"]} == {HARNESS_META, HARNESS_BASE}
