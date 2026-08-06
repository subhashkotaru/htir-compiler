"""
GEPA adapter for Terminal-Bench 2.0 (Track G).

Mirrors the SkillOpt Track-S integration, but plugs the *same* harbor/TB2
rollout plumbing into GEPA's ``GEPAAdapter`` interface instead of SkillOpt's
``EnvAdapter``. The optimized artifact is a single skill/instruction document
(GEPA "component" named ``skill``), injected into the frozen ``terminus-2``
agent exactly as SkillOpt injects its skill -- so GEPA vs. SkillOpt is a clean
comparison of the *optimization algorithm* (Pareto reflective evolution vs.
single-best reflect-and-gate), harness held fixed.

The heavy lifting -- one ``harbor run`` trial per task, TB2 verifier reward as
the score -- is reused verbatim from ``scripts.skillopt_tb2.rollout.run_batch``
(zero new harbor code; the same plumbing validated over the 9.7h SkillOpt run).

GEPA contract (see ``gepa/core/adapter.py``):
- ``evaluate(batch, candidate, capture_traces)`` -> ``EvaluationBatch`` with
  per-task ``scores`` (TB2 reward, higher better) and, when
  ``capture_traces=True``, per-task ``trajectories`` for reflection.
- ``make_reflective_dataset(candidate, eval_batch, components_to_update)`` ->
  ``{"skill": [ {Inputs, Generated Outputs, Feedback}, ... ]}`` fed to the
  reflection LM.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

from gepa.core.adapter import EvaluationBatch, GEPAAdapter

from scripts.skillopt_tb2.rollout import run_batch

# The single optimized component. GEPA candidates are {SKILL_COMPONENT: text}.
SKILL_COMPONENT = "skill"

# Bound how much trajectory text is handed to the reflection LM per task, so a
# 600-message agent transcript can't blow up reflection token cost.
_TRAJ_TAIL_CHARS = 2000


def _read_trajectory_tail(traj_path: str | None) -> str:
    """Return a truncated, human-readable tail of a harbor trajectory.json.

    Best-effort: reflection still works (on reward + fail_reason alone) if the
    trajectory is missing or unparseable, so never raise here.
    """
    if not traj_path:
        return ""
    p = Path(traj_path)
    if not p.exists():
        return ""
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    steps = data.get("steps") if isinstance(data, dict) else None
    if not isinstance(steps, list):
        text = json.dumps(data)[-_TRAJ_TAIL_CHARS:]
        return text
    lines: list[str] = []
    for st in steps:
        if not isinstance(st, dict):
            continue
        src = st.get("source") or st.get("role") or "?"
        msg = st.get("content") or st.get("message") or st.get("text") or ""
        if isinstance(msg, (dict, list)):
            msg = json.dumps(msg)
        lines.append(f"[{src}] {str(msg)}")
    return "\n".join(lines)[-_TRAJ_TAIL_CHARS:]


class TerminalBenchGEPAAdapter(GEPAAdapter):
    """GEPA adapter that scores candidate skills via live TB2 harbor rollouts."""

    def __init__(
        self,
        *,
        out_root: str,
        harbor_model: str = "openai/gpt-4o-mini",
        harbor_agent: str = "terminus-2",
        harbor_env: str = "docker",
        harbor_max_retries: int = 0,
        workers: int = 1,
        dry_run: bool = False,
        inject_skill: bool = True,
    ) -> None:
        self.out_root = Path(out_root)
        self.harbor_model = harbor_model
        self.harbor_agent = harbor_agent
        self.harbor_env = harbor_env
        self.harbor_max_retries = harbor_max_retries
        self.workers = int(workers)
        self.dry_run = bool(dry_run)
        self.inject_skill = inject_skill
        # Monotonic counter so each evaluate() call routes its harbor jobs to a
        # distinct dir -- concurrent/re-evaluated candidates never collide.
        self._eval_counter = itertools.count()

    # -- GEPA: evaluate ----------------------------------------------------
    def evaluate(
        self,
        batch: list[dict[str, Any]],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        skill_text = candidate.get(SKILL_COMPONENT, "")
        eval_dir = self.out_root / "rollouts" / f"eval_{next(self._eval_counter):04d}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        # Record which candidate produced this batch (provenance for later audit).
        (eval_dir / "candidate_skill.md").write_text(skill_text or "", encoding="utf-8")

        results = run_batch(
            list(batch),
            out_root=str(eval_dir),
            skill_content=skill_text,
            model=self.harbor_model,
            agent=self.harbor_agent,
            env=self.harbor_env,
            workers=self.workers,
            max_retries=self.harbor_max_retries,
            dry_run=self.dry_run,
            inject_skill=self.inject_skill,
        )

        # TB2 verifier reward is the score (higher better); run_batch already
        # returns a per-task failure dict rather than raising, per GEPA's
        # "never raise for individual example failures" contract.
        scores = [float(r.get("soft") or 0.0) for r in results]
        outputs = results

        trajectories: list[dict[str, Any]] | None = None
        if capture_traces:
            trajectories = []
            for item, r in zip(batch, results):
                trajectories.append(
                    {
                        "task_name": r.get("task_name") or item.get("task_name"),
                        "question": item.get("question", ""),
                        "reward": float(r.get("soft") or 0.0),
                        "hard": int(r.get("hard") or 0),
                        "fail_reason": r.get("fail_reason") or "",
                        "trajectory_tail": _read_trajectory_tail(r.get("trajectory_path")),
                    }
                )

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    # -- GEPA: reflective dataset -----------------------------------------
    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        traces = eval_batch.trajectories or []
        records: list[dict[str, Any]] = []
        for tr in traces:
            passed = int(tr.get("hard") or 0) == 1
            reward = tr.get("reward")
            fail_reason = tr.get("fail_reason") or ""
            if passed:
                feedback = f"PASS (TB2 verifier reward={reward}). The skill guided a correct solution for this task."
            else:
                why = f" Harbor/verifier signal: {fail_reason}." if fail_reason else ""
                feedback = (
                    f"FAIL (TB2 verifier reward={reward}). The agent did not satisfy the task's "
                    f"verifier.{why} Revise the skill so a terminus-2 agent would avoid this failure "
                    f"mode -- add concrete, generalizable operating guidance (not task-specific hardcoding)."
                )
            records.append(
                {
                    "Inputs": {
                        "task_name": tr.get("task_name") or "",
                        "task": tr.get("question") or f"Solve Terminal-Bench 2.0 task '{tr.get('task_name')}'.",
                    },
                    "Generated Outputs": tr.get("trajectory_tail") or "(no trajectory captured)",
                    "Feedback": feedback,
                }
            )
        # Update whichever components GEPA asked for (only ever "skill" here).
        return {comp: records for comp in (components_to_update or [SKILL_COMPONENT])}
