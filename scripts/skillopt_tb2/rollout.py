"""
TB2 scored rollout for SkillOpt: one ``harbor run`` trial per task item.

Each item's ``task_name`` is pinned via harbor's ``-i`` flag. The current
skill text is materialised as a temporary Agent Skill directory and passed
with ``--skill`` so Codex / Terminus2 can load it. The TB2 verifier reward
becomes SkillOpt's ``hard`` / ``soft`` score.

``dry_run=True`` (or ``HTIR_SKILLOPT_DRY_RUN=1``) skips harbor entirely and
returns zero-score stub results -- used by offline tests and by
``scripts/skillopt_train_tb2.py --dry-run``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _extract_reward(verifier_result: Any) -> float | None:
    if not isinstance(verifier_result, dict):
        return None
    rewards = verifier_result.get("rewards")
    if isinstance(rewards, dict):
        for key in ("reward", "passed", "resolved", "score", "success"):
            if key in rewards and rewards[key] is not None:
                try:
                    return float(rewards[key])
                except (TypeError, ValueError):
                    continue
    for key in ("reward", "passed", "resolved", "score"):
        if key in verifier_result and verifier_result[key] is not None:
            try:
                return float(verifier_result[key])
            except (TypeError, ValueError):
                continue
    return None


def _write_skill_dir(skill_content: str, parent: Path) -> Path:
    """
    Materialise ``skill_content`` as a harbor-loadable Agent Skill directory
    (``SKILL.md`` with YAML frontmatter). Empty content still writes a minimal
    skill so ``--skill`` is always a real path when the caller asked for one.
    """
    skill_dir = parent / "skillopt-tb2-candidate"
    skill_dir.mkdir(parents=True, exist_ok=True)
    body = (skill_content or "").strip()
    if not body.startswith("---"):
        body = (
            "---\n"
            "name: skillopt-tb2-candidate\n"
            "description: SkillOpt-candidate skill for Terminal-Bench 2.0 rollouts.\n"
            "---\n\n"
            + (body or "Complete Terminal-Bench tasks carefully and verify outcomes.")
        )
    (skill_dir / "SKILL.md").write_text(body + "\n", encoding="utf-8")
    return skill_dir


def _build_harbor_cmd(
    *,
    task_name: str,
    model: str,
    agent: str,
    jobs_dir: Path,
    skill_dir: Path | None,
    env: str,
    max_retries: int,
    agent_kwargs: list[str] | None,
) -> list[str]:
    cmd = [
        "harbor", "run",
        "-d", "terminal-bench@2.0",
        "-m", model,
        "-e", env,
        "-a", agent,
        "-n", "1",
        "--n-attempts", "1",
        "--max-retries", str(max_retries),
        "--jobs-dir", str(jobs_dir),
        "-i", task_name,
        "-y",
    ]
    if skill_dir is not None:
        cmd += ["--skill", str(skill_dir)]
    for kv in agent_kwargs or []:
        cmd += ["--ak", kv]
    return cmd


def _load_trial_result(jobs_dir: Path) -> tuple[float, dict[str, Any] | None, str | None]:
    """
    Return ``(reward, result_dict, trajectory_path)`` from the first usable
    per-trial ``result.json`` under ``jobs_dir``. Missing/unparseable trees
    score as 0.0 (failed rollout), matching SkillOpt's conservative convention.
    """
    for result_path in sorted(jobs_dir.rglob("result.json")):
        # Skip job-level summaries: trial dirs are nested one level deeper.
        if result_path.parent == jobs_dir:
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(result, dict):
            continue
        reward = _extract_reward(result.get("verifier_result"))
        if reward is None:
            continue
        traj = result_path.parent / "agent" / "trajectory.json"
        return reward, result, str(traj) if traj.exists() else None
    return 0.0, None, None


def _run_one(
    item: dict[str, Any],
    *,
    skill_content: str,
    out_dir: Path,
    model: str,
    agent: str,
    env: str,
    max_retries: int,
    agent_kwargs: list[str] | None,
    dry_run: bool,
    inject_skill: bool,
) -> dict[str, Any]:
    task_name = str(item.get("task_name") or item.get("id") or "").strip()
    task_id = str(item.get("id") or task_name)
    trial_root = out_dir / f"task__{task_id}"
    trial_root.mkdir(parents=True, exist_ok=True)

    if dry_run or not task_name:
        return {
            "id": task_id,
            "hard": 0,
            "soft": 0.0,
            "task_name": task_name,
            "task_type": item.get("task_type") or "terminal_bench",
            "predicted_answer": "",
            "fail_reason": "dry_run" if dry_run else "missing_task_name",
            "dry_run": bool(dry_run),
        }

    skill_dir: Path | None = None
    tmp_parent = tempfile.mkdtemp(prefix="skillopt_tb2_skill_")
    try:
        if inject_skill:
            skill_dir = _write_skill_dir(skill_content, Path(tmp_parent))
        jobs_dir = (trial_root / "jobs").resolve()
        jobs_dir.mkdir(parents=True, exist_ok=True)
        cmd = _build_harbor_cmd(
            task_name=task_name,
            model=model,
            agent=agent,
            jobs_dir=jobs_dir,
            skill_dir=skill_dir,
            env=env,
            max_retries=max_retries,
            agent_kwargs=agent_kwargs,
        )
        (trial_root / "harbor_cmd.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
        (trial_root / "harbor_stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
        (trial_root / "harbor_stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
        reward, result, traj_path = _load_trial_result(jobs_dir)
        hard = 1 if reward > 0 else 0
        fail_reason = ""
        if hard == 0:
            fail_reason = "tb2_verifier_reward_zero"
            if proc.returncode != 0:
                fail_reason = f"harbor_exit_{proc.returncode}"
        out = {
            "id": task_id,
            "hard": hard,
            "soft": float(reward),
            "task_name": task_name,
            "task_type": item.get("task_type") or "terminal_bench",
            "predicted_answer": f"reward={reward}",
            "fail_reason": fail_reason,
            "harbor_returncode": proc.returncode,
            "trajectory_path": traj_path,
            "result_summary": {
                "task_name": (result or {}).get("task_name"),
                "exception_type": ((result or {}).get("exception_info") or {}).get("exception_type")
                if isinstance(result, dict) else None,
            },
        }
        (trial_root / "rollout_result.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        return out
    finally:
        shutil.rmtree(tmp_parent, ignore_errors=True)


def run_batch(
    items: list[dict],
    *,
    out_root: str,
    skill_content: str,
    model: str = "openai/gpt-4o-mini",
    agent: str = "terminus-2",
    env: str = "docker",
    workers: int = 1,
    max_retries: int = 0,
    agent_kwargs: list[str] | None = None,
    dry_run: bool = False,
    inject_skill: bool = True,
    **_kwargs: Any,
) -> list[dict]:
    """
    Score a batch of TB2 tasks under the current skill.

    ``workers`` > 1 runs harbor trials in parallel (each trial is its own
    Docker job). Keep this small -- concurrent docker sandboxes are the
    expensive resource, not Python threads.
    """
    out_dir = Path(out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    env_dry = os.environ.get("HTIR_SKILLOPT_DRY_RUN", "").strip() in {"1", "true", "yes"}
    dry_run = bool(dry_run or env_dry)

    results: list[dict] = [None] * len(items)  # type: ignore[list-item]
    if workers <= 1 or dry_run:
        for i, item in enumerate(items):
            results[i] = _run_one(
                item,
                skill_content=skill_content,
                out_dir=out_dir,
                model=model,
                agent=agent,
                env=env,
                max_retries=max_retries,
                agent_kwargs=agent_kwargs,
                dry_run=dry_run,
                inject_skill=inject_skill,
            )
            print(
                f"[skillopt_tb2] {results[i]['id']} hard={results[i]['hard']} "
                f"soft={results[i]['soft']}",
                file=sys.stderr,
            )
        return results

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _run_one,
                item,
                skill_content=skill_content,
                out_dir=out_dir,
                model=model,
                agent=agent,
                env=env,
                max_retries=max_retries,
                agent_kwargs=agent_kwargs,
                dry_run=dry_run,
                inject_skill=inject_skill,
            ): i
            for i, item in enumerate(items)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            print(
                f"[skillopt_tb2] {results[i]['id']} hard={results[i]['hard']} "
                f"soft={results[i]['soft']}",
                file=sys.stderr,
            )
    return results
