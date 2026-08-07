#!/usr/bin/env python
"""
Track M capture driver: run Meta-Harness's Terminal-Bench 2.0 scaffold and/or a
plain baseline agent live via the ``harbor`` CLI, and normalize the results into
the local capture format ``htir.eval.datasets.load_meta_harness_tb2`` /
``htir.eval.experiment_sa14`` consume.

DELIBERATELY OUTSIDE THE ``htir`` PACKAGE. This script shells out to an
external CLI (``harbor``) and spends real API + sandbox budget; ``htir`` core
has no such dependency and must stay importable/offline everywhere else. This
is the *only* place in the repo that should ever invoke a live agent run.

Prerequisites:

    pip install --only-binary=:all: harbor   # plain `pip install harbor` needs a
                                              # working Rust toolchain for one of
                                              # litellm's transitive deps; the
                                              # binary-only resolve sidesteps it.
    export OPENAI_API_KEY=<your-key>          # for -m openai/gpt-4o-mini, or put
                                               # it in .env (see .env.example)
    # a local checkout of the Meta-Harness TB2 artifact, so -a agent:AgentHarness
    # resolves (this script cwd's into it for the meta_harness run):
    git clone https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact \
        vendor/meta-harness-tbench2-artifact
    # a running sandbox provider for -e (docker by default: Docker Desktop/daemon
    # must be up; `docker info` should succeed).

What this script does:

  1. Shells out to ``harbor run`` once per requested harness (``meta_harness``
     via ``-a agent:AgentHarness`` with cwd set to the artifact checkout,
     ``base_agent`` via a stock ``-a terminus``) against ``terminal-bench@2.0``,
     sampling ``--n-tasks`` distinct tasks at ``--n-attempts`` attempts (seeds)
     each.
  2. Reads each trial's ``result.json`` (reward, under
     ``verifier_result.rewards``) and sibling ``agent/trajectory.json`` (ATIF
     steps), and normalizes the pair into
     ``{task_id, harness, model, messages, reward}`` records --
     ``htir.eval.datasets.normalize_meta_harness_record``'s expected input shape.
  3. Writes one JSONL file per harness under ``data/live_traces/meta_harness_tb2/``,
     ready for ``python -m htir.eval.experiment_sa14 --cache <file> ...``.

Flags and the harbor/ATIF output schema (CLI flags, ``TrialResult``,
``Trajectory``/``Step``) were verified against an installed ``harbor==0.19.0``;
re-check ``harbor run --help`` if your installed version differs. Run once with
``--n-tasks 1 --n-attempts 1`` and inspect the printed "unparsed files" list
before trusting a larger capture. ``--dry-run`` prints the exact ``harbor``
command without spending anything.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
LIVE_TRACE_DIR = REPO_ROOT / "data" / "live_traces" / "meta_harness_tb2"

HARNESS_META = "meta_harness"
HARNESS_BASE = "base_agent"

# Reward-like keys checked (in order) on a per-trial result payload. Harbor's
# actual field name needs confirming against a real run -- see module docstring.
_REWARD_KEYS = ("reward", "passed", "resolved", "score", "success")


def _model_slug(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


# ---------------------------------------------------------------------------
# Step 1: invoke harbor
# ---------------------------------------------------------------------------

def build_harbor_command(
    *,
    harness: str,
    model: str,
    agent_import_path: str | None,
    baseline_agent: str,
    n_tasks: int,
    n_attempts: int,
    n_concurrent: int,
    env: str,
    jobs_dir: Path,
    include_task_names: list[str] | None = None,
    max_retries: int = 0,
    agent_kwargs: list[str] | None = None,
    skills: list[str] | None = None,
) -> list[str]:
    """
    Build the ``harbor run`` argv for one harness, verified against an
    installed ``harbor==0.19.0``'s ``harbor run --help``. ``meta_harness``
    passes the Meta-Harness artifact's import path directly as the ``-a``
    value (``agent:AgentHarness`` -- current harbor merged the older
    ``--agent-import-path`` flag from the artifact's own README into ``-a``,
    which now accepts either a built-in agent name or an import path);
    ``base_agent`` uses a stock ``-a terminus``. The import path is resolved
    relative to the subprocess's cwd, which the caller must set to the
    artifact checkout (see ``run_harbor``'s ``cwd`` arg).

    ``max_retries`` maps to harbor's own ``--max-retries``/``-r`` (job-level
    retry-on-exception count; harbor's own default is already 0 -- no
    automatic retries -- so pass 0 explicitly to make "no extra retries"
    intent visible rather than relying on an unstated default). ``agent_kwargs``
    maps to repeatable ``--ak key=value`` flags, e.g. ``["reasoning_effort=medium"]``
    for the ``codex``/``claude-code`` installed agents' own ``CLI_FLAGS``.
    """
    cmd = [
        "harbor", "run",
        "-d", "terminal-bench@2.0",
        "-m", model,
        "-e", env,
        "-n", str(n_concurrent),
        "--n-attempts", str(n_attempts),
        "--max-retries", str(max_retries),
        "--jobs-dir", str(jobs_dir),
        "-y",
    ]
    for kv in agent_kwargs or []:
        cmd += ["--ak", kv]
    for skill_path in skills or []:
        cmd += ["--skill", skill_path]
    if include_task_names:
        # Pin to specific tasks (e.g. to task-match against an already-captured
        # run of the other harness) instead of sampling the first --n-tasks.
        for name in include_task_names:
            cmd += ["-i", name]
    elif n_tasks > 0:
        cmd += ["--n-tasks", str(n_tasks)]
    if harness == HARNESS_META:
        if not agent_import_path:
            raise SystemExit("--agent-import-path is required for the meta_harness run")
        cmd += ["-a", agent_import_path]
    else:
        cmd += ["-a", baseline_agent]
    return cmd


def harness_label(harness: str, baseline_agent: str) -> str:
    """
    The label used for the JSONL ``harness`` field and output filename.

    ``meta_harness`` is always ``"meta_harness"``. For the base-agent side,
    the *default* ``terminus-2`` keeps the historical ``"base_agent"`` label
    (so old captures/docs stay valid), but any other ``--baseline-agent``
    (``codex``, ``claude-code``, ...) gets its own label -- otherwise two
    different agents both run with ``--harness base_agent`` would silently
    overwrite each other's file and be indistinguishable in
    ``experiment_sa14``'s per-harness grouping.
    """
    if harness == HARNESS_META:
        return HARNESS_META
    if baseline_agent in ("terminus-2", "", None):
        return HARNESS_BASE
    return baseline_agent.replace("-", "_")


def run_harbor(cmd: list[str], *, dry_run: bool, cwd: Path | None = None) -> int:
    print(f"[live] cwd={cwd or '.'} {' '.join(cmd)}", file=sys.stderr)
    if dry_run:
        print("[live] --dry-run: not executing.", file=sys.stderr)
        return 0
    env = None
    if cwd is not None:
        # harbor is an installed console-script entry point, so its own cwd is
        # NOT automatically on sys.path the way it would be for `python foo.py`
        # or `python -m foo` -- `import_class("agent:AgentHarness")` resolves
        # via plain `importlib.import_module("agent")`, which only finds the
        # checkout's agent.py if its directory is on PYTHONPATH. --cwd alone
        # (confirmed against a real run) is not sufficient; this is.
        env = {**os.environ, "PYTHONPATH": f"{cwd}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env)
    return proc.returncode


# ---------------------------------------------------------------------------
# Step 2: normalize harbor's output.
#
# Verified against an installed harbor==0.19.0 checkout (harbor/models/trial/*,
# harbor/models/trajectories/trajectory.py, harbor/agents/terminus_2/terminus_2.py):
# each trial directory under jobs_dir/<job>/trials/**/ holds
#   result.json          -- TrialResult: verifier_result.rewards (dict, e.g.
#                            {"reward": 1.0}), task_id.name / task_name,
#                            agent_info, NOT a message transcript.
#   agent/trajectory.json -- ATIF (Agent Trajectory Interchange Format):
#                            {schema_version, agent, steps: [Step, ...]}, each
#                            Step carrying step_id/source/message/tool_calls/
#                            observation -- NOT an OpenAI messages list.
# We convert each ATIF Step to an OpenAI-style turn (role=agent->assistant,
# role=user/system passthrough, tool_calls preserved) so the existing
# `htir.eval.datasets.normalize_meta_harness_record`'s `messages` contract
# (already tolerant of `_tb2_tool_call`'s permissive tool-call shape) keeps
# working unchanged; only this ATIF->messages step is new.
# ---------------------------------------------------------------------------

def _atif_step_to_message(step: dict[str, Any]) -> dict[str, Any] | None:
    source = step.get("source", "agent")
    role = {"agent": "assistant", "system": "system"}.get(source, "user")
    msg: dict[str, Any] = {"role": role, "content": step.get("message") or ""}
    tool_calls = step.get("tool_calls")
    if tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.get("tool_call_id", ""),
                "type": "function",
                "function": {
                    "name": tc.get("function_name", ""),
                    "arguments": json.dumps(tc.get("arguments", {})),
                },
            }
            for tc in tool_calls
        ]
    return msg


def _atif_step_observation_messages(step: dict[str, Any]) -> list[dict[str, Any]]:
    obs = step.get("observation")
    if not isinstance(obs, dict):
        return []
    out = []
    for result in obs.get("results", []):
        content = result.get("content")
        if content is None:
            continue
        call_id = result.get("source_call_id")
        if call_id:
            out.append({"role": "tool", "tool_call_id": call_id, "content": content})
        else:
            out.append({"role": "user", "content": content})
    return out


def _trajectory_to_messages(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for step in trajectory.get("steps", []):
        msg = _atif_step_to_message(step)
        if msg is not None:
            messages.append(msg)
        messages.extend(_atif_step_observation_messages(step))
    return messages


def _extract_reward(verifier_result: dict[str, Any] | None) -> int | None:
    if not verifier_result:
        return None
    rewards = verifier_result.get("rewards")
    if not isinstance(rewards, dict) or not rewards:
        return None
    for key in _REWARD_KEYS:
        if key in rewards and rewards[key] is not None:
            v = rewards[key]
            try:
                return 1 if float(v) > 0 else 0
            except (TypeError, ValueError):
                continue
    # Unrecognised reward key(s) -- fall back to "any positive value present".
    for v in rewards.values():
        try:
            return 1 if float(v) > 0 else 0
        except (TypeError, ValueError):
            continue
    return None


def _iter_result_files(root: Path) -> Iterator[Path]:
    yield from sorted(root.rglob("result.json"))


def _load_harbor_output(jobs_dir: Path, *, harness: str, model: str) -> tuple[list[dict[str, Any]], list[Path]]:
    """
    Scan a ``harbor run --jobs-dir`` tree for per-trial ``result.json`` files,
    pairing each with its sibling ``agent/trajectory.json``. Returns
    ``(records, unparsed)`` so the caller can print what needs a closer look.
    """
    records: list[dict[str, Any]] = []
    unparsed: list[Path] = []
    for result_path in _iter_result_files(jobs_dir):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            unparsed.append(result_path)
            continue
        if not isinstance(result, dict):
            unparsed.append(result_path)
            continue

        reward = _extract_reward(result.get("verifier_result"))
        if reward is None:
            unparsed.append(result_path)
            continue

        trial_dir = result_path.parent
        traj_path = trial_dir / "agent" / "trajectory.json"
        if not traj_path.exists():
            unparsed.append(traj_path)
            continue
        try:
            trajectory = json.loads(traj_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            unparsed.append(traj_path)
            continue
        messages = _trajectory_to_messages(trajectory)
        if not messages:
            unparsed.append(traj_path)
            continue
        # Trials that crash before the agent does anything (billing/quota/rate
        # limit errors like ApiUsageLimitError/RateLimitError -- see
        # ApiUsageLimitError seen live against gpt-5.4-mini) still produce a
        # verifier_result (harness auto-fails to reward=0.0) AND a
        # trajectory.json with a few steps -- but those steps are just the
        # echoed system prompt / task instruction (source in {"system",
        # "user"}), never a real model turn (source == "agent"). Recording
        # those as if they were genuine reward=0 task failures would silently
        # contaminate the capture with zero-signal rows. Exceptions that
        # happen *after* real agent activity (e.g. AgentTimeoutError once the
        # model has made real tool calls) are kept: the partial trajectory
        # plus reward=0 is a legitimate outcome, not an infra artifact.
        if not any(step.get("source") == "agent" for step in trajectory.get("steps", [])):
            unparsed.append(traj_path)
            continue

        task_id = str(
            result.get("task_name")
            or (result.get("task_id") or {}).get("path")
            or trial_dir.name
        )
        records.append({
            "task_id": task_id,
            "harness": harness,
            "model": model,
            "reward": reward,
            "messages": messages,
        })
    return records, unparsed


# ---------------------------------------------------------------------------
# Step 3: write the capture
# ---------------------------------------------------------------------------

def write_capture(records: list[dict[str, Any]], *, harness: str, model: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{harness}_{_model_slug(model)}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    try:
        # Populates os.environ (and therefore the harbor subprocess's inherited
        # env, e.g. OPENAI_API_KEY) from a local .env, if python-dotenv is
        # installed and a .env exists. No-op otherwise -- exporting the key in
        # your shell still works exactly as documented above.
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    p = argparse.ArgumentParser(
        description="Track M: capture Meta-Harness / base-agent Terminal-Bench 2.0 trials via harbor."
    )
    p.add_argument("--harness", choices=[HARNESS_META, HARNESS_BASE, "both"], default="both")
    p.add_argument("--model", type=str, default="openai/gpt-4o-mini")
    p.add_argument("--agent-import-path", type=str, default="agent:AgentHarness",
                    help="import path passed as -a for the meta_harness run, resolved from --agent-checkout-dir")
    p.add_argument("--agent-checkout-dir", type=str,
                    default=str(REPO_ROOT / "vendor" / "meta-harness-tbench2-artifact"),
                    help="local checkout of stanford-iris-lab/meta-harness-tbench2-artifact; "
                         "harbor is invoked with this as cwd so --agent-import-path resolves")
    p.add_argument("--baseline-agent", type=str, default="terminus-2",
                    help="stock harbor -a name for the base_agent comparison run. "
                         "terminus-2 (not plain terminus/terminus-1, which this harbor "
                         "version's AgentFactory doesn't have registered) is also the "
                         "fairer baseline: Meta-Harness's AgentHarness literally "
                         "subclasses harbor.agents.terminus_2.Terminus2, so this isolates "
                         "the env-bootstrapping delta rather than a whole different scaffold.")
    p.add_argument("--n-tasks", type=int, default=5,
                    help="distinct TB2 tasks to sample for the pilot (0 = full 89-task suite). "
                         "Ignored if --include-task-name is given.")
    p.add_argument("--include-task-name", action="append", default=None,
                    help="pin to this specific TB2 task name (repeatable) instead of sampling "
                         "the first --n-tasks -- e.g. to task-match against an already-captured "
                         "run of the other harness for a paired comparison.")
    p.add_argument("--n-attempts", type=int, default=1,
                    help="attempts (seeds) per task; keep at 1 for a broad-but-cheap pilot, "
                         "raise for variance estimates once the capture is validated")
    p.add_argument("--n-concurrent", type=int, default=4)
    p.add_argument("--max-retries", type=int, default=0,
                    help="harbor's own --max-retries (job-level retry-on-exception count). "
                         "Harbor's default is already 0 (no automatic retries); pass a value "
                         "explicitly to make the intent visible in the invocation.")
    p.add_argument("--agent-kwarg", action="append", default=None,
                    help="repeatable 'key=value' forwarded as --ak to harbor, for agent-specific "
                         "CLI_FLAGS -- e.g. --agent-kwarg reasoning_effort=medium for codex/claude-code.")
    p.add_argument("--skill", action="append", default=None, dest="skill",
                    help="repeatable path/git-source forwarded as --skill to harbor (Agent Skills spec). "
                         "Supported by both codex/claude-code (installed agents) and meta_harness/base_agent "
                         "(Terminus2's own skill-injection); see scripts/skills/ for local skill directories.")
    p.add_argument("--env", type=str, default="docker", help="harbor sandbox provider: docker | daytona | runloop | ...")
    p.add_argument("--jobs-root", type=str, default=str(REPO_ROOT / "data" / "live_traces" / "_harbor_jobs"))
    p.add_argument("--out-dir", type=str, default=str(LIVE_TRACE_DIR))
    p.add_argument("--dry-run", action="store_true", help="print the harbor command(s) without running anything")
    args = p.parse_args(argv)

    harnesses = [HARNESS_META, HARNESS_BASE] if args.harness == "both" else [args.harness]
    out_dir = Path(args.out_dir)
    exit_code = 0

    for harness in harnesses:
        label = harness_label(harness, args.baseline_agent)
        # Always absolute. meta_harness's `harbor` subprocess runs with cwd set
        # to the vendor checkout (see below), so a *relative* --jobs-root would
        # get silently resolved against that checkout dir by harbor itself, while
        # this same process's own _load_harbor_output call below (which never
        # chdirs) would look for it relative to the script's actual cwd instead
        # -- two different directories, so the post-run normalization pass would
        # find nothing and silently write an empty capture (seen live: a full
        # multi-hour/dollar run whose 25+ real trials were only recovered by
        # manually pointing a one-off script at the vendor-nested path). Forcing
        # this absolute up front makes both the harbor subprocess and this
        # process agree on the one real location regardless of either's cwd.
        jobs_dir = (Path(args.jobs_root) / label).resolve()
        cwd = Path(args.agent_checkout_dir) if harness == HARNESS_META else None
        cmd = build_harbor_command(
            harness=harness, model=args.model,
            agent_import_path=args.agent_import_path or None,
            baseline_agent=args.baseline_agent,
            n_tasks=args.n_tasks, n_attempts=args.n_attempts, n_concurrent=args.n_concurrent,
            env=args.env, jobs_dir=jobs_dir, include_task_names=args.include_task_name,
            max_retries=args.max_retries, agent_kwargs=args.agent_kwarg, skills=args.skill,
        )
        rc = run_harbor(cmd, dry_run=args.dry_run, cwd=cwd)
        if rc != 0:
            print(f"[live] harbor exited {rc} for {label}; skipping normalization.", file=sys.stderr)
            exit_code = exit_code or rc
            continue
        if args.dry_run:
            continue

        records, unparsed = _load_harbor_output(jobs_dir, harness=label, model=args.model)
        out_path = write_capture(records, harness=label, model=args.model, out_dir=out_dir)
        print(f"[live] {label}: wrote {len(records)} trials -> {out_path}", file=sys.stderr)
        if unparsed:
            print(
                f"[live] {label}: {len(unparsed)} file(s) under {jobs_dir} did not match the expected "
                "shape (see _load_harbor_output) -- inspect and adjust before trusting the capture:",
                file=sys.stderr,
            )
            for path in unparsed[:20]:
                print(f"    {path}", file=sys.stderr)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
