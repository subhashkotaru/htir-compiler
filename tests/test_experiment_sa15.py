"""
Offline regression tests for SA-15 (Track S -- SkillOpt vs. no-skill on
Terminal-Bench 2.0) and the SkillOpt TB2 env plugin's pure-Python surface.

No live calls: synthetic capture records exercise
``normalize_skillopt_record`` / ``run_sa15``, and the dataloader + dry-run
rollout path exercise ``scripts.skillopt_tb2`` without harbor/Docker.
"""

from __future__ import annotations

import json
from pathlib import Path

from htir.eval.datasets import normalize_skillopt_record
from htir.eval.experiment_sa15 import HARNESS_NO_SKILL, HARNESS_SKILLOPT, run_sa15
from htir.models.domain import get_domain_spec


def _tool_call(call_id: str, name: str, **args) -> dict:
    return {"id": call_id, "function": {"name": name, "arguments": args}}


def _tb2_trial(task_id: str, *, harness: str, model: str, solved: bool) -> dict:
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
    messages.append({"role": "assistant", "content": "Done."})
    return {
        "task_id": task_id,
        "harness": harness,
        "model": model,
        "reward": 1 if solved else 0,
        "messages": messages,
    }


def _fixture(harness: str, model: str = "openai/gpt-4o-mini") -> list[dict]:
    trials = [_tb2_trial(f"tb2_{harness}_{i}", harness=harness, model=model, solved=True) for i in range(3)]
    trials.append(_tb2_trial(f"tb2_{harness}_3", harness=harness, model=model, solved=False))
    return trials


def test_normalize_skillopt_preserves_harness():
    rec = normalize_skillopt_record(
        _tb2_trial("t1", harness=HARNESS_SKILLOPT, model="openai/gpt-4o-mini", solved=True)
    )
    assert rec["harness"] == HARNESS_SKILLOPT
    assert rec["reward"] == 1
    assert isinstance(rec["steps"], list) and rec["steps"]


def test_sa15_offline_skillopt_beats_no_skill_on_task_success():
    """
    Synthetic: skillopt solves 3/4, no_skill solves 1/4 -- task_success gap
    should favour skillopt. Verifier arms still run (offline, use_llm=False).
    """
    skill = _fixture(HARNESS_SKILLOPT)
    noskill = [
        _tb2_trial("tb2_no_skill_0", harness=HARNESS_NO_SKILL, model="openai/gpt-4o-mini", solved=True),
        _tb2_trial("tb2_no_skill_1", harness=HARNESS_NO_SKILL, model="openai/gpt-4o-mini", solved=False),
        _tb2_trial("tb2_no_skill_2", harness=HARNESS_NO_SKILL, model="openai/gpt-4o-mini", solved=False),
        _tb2_trial("tb2_no_skill_3", harness=HARNESS_NO_SKILL, model="openai/gpt-4o-mini", solved=False),
    ]
    traces = [normalize_skillopt_record(r) for r in (skill + noskill)]
    result = run_sa15(
        traces, spec=get_domain_spec("terminal_swe"), seeds=[0, 1], progress_every=0, log=None,
    )
    by = {h.harness: h for h in result.harnesses}
    assert HARNESS_SKILLOPT in by and HARNESS_NO_SKILL in by
    assert by[HARNESS_SKILLOPT].task_success_rate.mean > by[HARNESS_NO_SKILL].task_success_rate.mean
    assert result.success_gap.n_seeds == 2
    assert result.success_gap.mean_diff > 0


def test_terminal_bench_dataloader_ratio_split(tmp_path: Path):
    from scripts.skillopt_tb2.dataloader import TerminalBenchDataLoader

    tasks = tmp_path / "tasks.jsonl"
    with tasks.open("w", encoding="utf-8") as f:
        for i in range(10):
            f.write(json.dumps({"id": f"task-{i}", "task_name": f"task-{i}"}) + "\n")
    loader = TerminalBenchDataLoader(
        data_path=str(tasks),
        split_mode="ratio",
        split_ratio="5:2:3",
        split_seed=0,
        split_output_dir=str(tmp_path / "splits"),
        seed=0,
    )
    loader.setup({})
    assert len(loader.train_items) == 5
    assert len(loader.val_items) == 2
    assert len(loader.test_items) == 3
    assert all(it["task_type"] == "terminal_bench" for it in loader.train_items)


def test_terminal_bench_rollout_dry_run(tmp_path: Path):
    from scripts.skillopt_tb2.rollout import run_batch

    results = run_batch(
        [{"id": "adaptive-rejection-sampler", "task_name": "adaptive-rejection-sampler"}],
        out_root=str(tmp_path / "out"),
        skill_content="# dry",
        dry_run=True,
    )
    assert len(results) == 1
    assert results[0]["hard"] == 0
    assert results[0]["dry_run"] is True


def test_terminal_bench_adapter_rollout_dry_run(tmp_path: Path):
    from scripts.skillopt_tb2.adapter import TerminalBenchAdapter

    tasks = tmp_path / "tasks.jsonl"
    with tasks.open("w", encoding="utf-8") as f:
        for i in range(3):
            f.write(json.dumps({"id": f"t{i}", "task_name": f"t{i}"}) + "\n")
    adapter = TerminalBenchAdapter(
        data_path=str(tasks),
        split_mode="ratio",
        split_ratio="1:1:1",
        split_seed=0,
        split_output_dir=str(tmp_path / "splits"),
        dry_run=True,
        limit=3,
    )
    adapter.setup({"out_root": str(tmp_path / "run")})
    env = adapter.build_train_env(batch_size=1, seed=0)
    results = adapter.rollout(env, skill_content="hello", out_dir=str(tmp_path / "roll"))
    assert len(results) == 1
    assert results[0]["dry_run"] is True
