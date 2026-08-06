"""
Terminal-Bench trajectory ingestion (avg.tex Sec. 4.1 data; experiment-plan P0).

Loads the ``yoonholee/terminalbench-trajectories`` HF dataset -- 52,104 traces
over 89 Terminal-Bench tasks, each a ``{steps:[{src,msg,tools,obs}], reward,
agent, model, task_name}`` record (the same turn schema as
``data/raw_traces``) -- or any local JSON/JSONL in that schema, and draws a
balanced solved/unsolved sample.

The HF ``datasets`` dependency is imported lazily inside
``load_terminalbench`` so importing this module (and ``import htir``) never
requires it or network access. The balanced sampler and label plumbing are
pure Python and unit-testable offline.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable, Iterator

from htir.adapters import load_trace
from htir.eval.weak_labels import extract_reward, label_from_reward

HF_REPO = "yoonholee/terminalbench-trajectories"
# τ-bench trajectory corpus: per-model JSONL of full retail/airline runs with a
# DB-match ``eval_result.score`` reward (parallel to the terminalbench set).
TAU_HF_REPO = "AgentSuite/tau-bench-trajectories"
TAU_HF_BASE = f"https://huggingface.co/datasets/{TAU_HF_REPO}/resolve/main/"
# SWE-Gym trajectory corpus (the third, patch-based SWE domain -- SA-10): sampled
# OpenHands rollouts over SWE-Gym issue-resolution tasks, each an OpenAI-messages
# trajectory (``execute_bash`` / ``str_replace_editor`` / ``finish`` tool calls)
# with a ``resolved`` boolean reward. ``Sampled`` (not the SFT set) so it carries
# both resolved and unresolved rollouts -> a balanced solved/unsolved split.
SWE_GYM_HF_REPO = "SWE-Gym/OpenHands-Sampled-Trajectories"
SWE_GYM_HF_SPLIT = "train.raw"


def iter_local_traces(paths: Iterable[str | Path]) -> Iterator[dict[str, Any]]:
    """
    Yield raw trace dicts from local ``.json`` (one object, or an array/
    ``{traces:[...]}`` wrapper) or ``.jsonl`` files in the turn schema.
    """
    for path in paths:
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix == ".jsonl":
            for line in text.splitlines():
                if line.strip():
                    yield json.loads(line)
            continue
        data = json.loads(text)
        if isinstance(data, dict) and isinstance(data.get("traces"), list):
            yield from data["traces"]
        elif isinstance(data, list):
            yield from data
        else:
            yield data


def load_terminalbench(
    repo: str = HF_REPO,
    *,
    split: str = "train",
    limit: int | None = None,
    streaming: bool = True,
) -> list[dict[str, Any]]:
    """
    Load Terminal-Bench trajectories from the HF hub as raw trace dicts.

    Requires the optional ``datasets`` dependency (``pip install datasets``);
    raises an informative ``ImportError`` otherwise. ``streaming`` (default)
    avoids materializing all 52k traces; pass ``limit`` to cap how many are
    pulled.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "load_terminalbench requires the 'datasets' package. "
            "Install it with: pip install datasets"
        ) from exc

    ds = load_dataset(repo, split=split, streaming=streaming)
    out: list[dict[str, Any]] = []
    for i, rec in enumerate(ds):
        if limit is not None and i >= limit:
            break
        out.append(dict(rec))
    return out


def balanced_sample(
    traces: list[dict[str, Any]],
    n: int,
    *,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """
    Draw up to ``n`` traces with an equal split of solved (reward != 0) and
    unsolved (reward = 0), per avg.tex's matched-condition reporting. Traces
    with no recorded reward are excluded. If one class is smaller than ``n/2``,
    the sample is capped at ``2 * min(class sizes)`` so it stays balanced.
    Deterministic given ``seed``.
    """
    solved = [t for t in traces if label_from_reward(extract_reward(t)) == "valid"]
    unsolved = [t for t in traces if label_from_reward(extract_reward(t)) == "invalid"]

    rng = random.Random(seed)
    rng.shuffle(solved)
    rng.shuffle(unsolved)

    per_class = min(n // 2, len(solved), len(unsolved))
    sample = solved[:per_class] + unsolved[:per_class]
    rng.shuffle(sample)
    return sample


def normalize_steps_field(trace: dict[str, Any]) -> dict[str, Any]:
    """
    Return ``trace`` with its ``steps`` field guaranteed to be a list.

    The HF ``terminalbench-trajectories`` dataset (and JSONL caches of it)
    serialize the per-row ``steps`` as a *JSON string*, whereas the adapters
    expect a decoded ``list[dict]``. Decode it if needed; leave an already-list
    (this repo's ``data/raw_traces`` convention) untouched. A copy is returned
    only when a decode was necessary, so the caller's dict is never mutated.
    """
    steps = trace.get("steps")
    if isinstance(steps, str):
        try:
            decoded = json.loads(steps)
        except (json.JSONDecodeError, ValueError):
            decoded = []
        trace = {**trace, "steps": decoded if isinstance(decoded, list) else []}
    return trace


def _is_tau_shaped(trace: dict[str, Any]) -> bool:
    """A τ-bench record carries OpenAI ``messages`` and no ``steps`` turn list."""
    return (
        isinstance(trace, dict)
        and isinstance(trace.get("messages"), list)
        and not isinstance(trace.get("steps"), list)
    )


def to_canonical_steps(trace: dict[str, Any], adapter: str | None = None) -> list[dict[str, Any]]:
    """
    Canonical step dicts for one raw trace, ready for
    ``TraceAbstractionAgent.compile``. Thin convenience over ``load_trace`` so
    callers don't reach into the adapter layer.

    ``adapter`` selects the parser; when ``None`` (the default) it is chosen by
    the record's *shape* so a single call site serves both corpora:

    * a τ-bench record (an OpenAI ``messages`` list, no ``steps``) routes to the
      ``tau_bench`` adapter over its ``messages``;
    * a Terminal-Bench turn record (``steps`` list, JSON-string-decoded first)
      routes to the ``terminal`` adapter, exactly as before.
    """
    if adapter is None:
        adapter = "tau_bench" if _is_tau_shaped(trace) else "terminal"
    if adapter in ("tau_bench", "tau", "openai", "openai_messages"):
        payload = trace["messages"] if _is_tau_shaped(trace) else trace
        return load_trace(payload, adapter=adapter)
    return load_trace(normalize_steps_field(trace), adapter=adapter)


# ---------------------------------------------------------------------------
# τ-bench (Sierra retail/airline) trajectory ingestion
# ---------------------------------------------------------------------------

# The subset of AgentSuite/tau-bench-trajectories model files pulled by
# ``load_tau_bench(hf=True)`` when no explicit list is given. A spread of agent
# families keeps the solved/unsolved mix broad rather than one model's quirks.
TAU_DEFAULT_HF_FILES = (
    "claude-4-sonnet-thinking-off.jsonl",
    "gemini-2.5-flash-thinking-off.jsonl",
    "Qwen3-235B-A22B-FP8.jsonl",
    "DeepSeek-V3-0324.jsonl",
    "Kimi-K2-Instruct.jsonl",
)


def normalize_tau_record(rec: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize a raw ``AgentSuite/tau-bench-trajectories`` record into the fields
    the eval harness expects, without dropping the original payload:

    * ``reward``     -- int in {0, 1} from ``eval_result.score`` /
      ``meta.is_correct`` (so ``extract_reward`` / ``balanced_sample`` work).
    * ``task_name``  -- the per-task id (``meta.id``, e.g. ``retail_12``), which
      the experiment writers use as an opaque trace id.
    * ``tau_domain`` -- ``retail`` / ``airline`` (the original ``task_name``),
      kept for per-domain slicing / the policy-drift stress test.
    * ``messages``   -- the OpenAI-messages trajectory, passed through untouched
      so ``to_canonical_steps`` routes it to the ``tau_bench`` adapter.
    """
    meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
    reward = extract_reward(rec)
    task_id = str(meta.get("id") or rec.get("task_name") or "")
    return {
        **rec,
        "reward": reward,
        "task_name": task_id,
        "tau_domain": str(rec.get("task_name") or ""),
        "messages": rec.get("messages") or [],
    }


def load_tau_bench(
    paths: Iterable[str | Path] | None = None,
    *,
    hf: bool = False,
    hf_files: Iterable[str] | None = None,
    repo: str = TAU_HF_REPO,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """
    Load τ-bench trajectories as normalized raw trace dicts.

    ``paths`` reads local ``.jsonl`` caches of the corpus (the offline default;
    see ``data/tau_cache``). ``hf=True`` streams the per-model JSONL files from
    the HF hub over plain HTTP (no ``datasets`` dependency); ``hf_files`` picks
    which model files (defaults to :data:`TAU_DEFAULT_HF_FILES`). ``limit`` caps
    the number of records returned.
    """
    out: list[dict[str, Any]] = []
    if hf:
        import requests  # local, so importing this module needs no network

        for fname in (hf_files or TAU_DEFAULT_HF_FILES):
            resp = requests.get(TAU_HF_BASE + fname, timeout=120, stream=True)
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                out.append(normalize_tau_record(json.loads(line)))
                if limit is not None and len(out) >= limit:
                    return out
        return out

    if not paths:
        raise ValueError("load_tau_bench needs local paths= or hf=True")
    for rec in iter_local_traces(paths):
        out.append(normalize_tau_record(rec))
        if limit is not None and len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# SWE-Gym (patch-based issue resolution) trajectory ingestion -- SA-10
# ---------------------------------------------------------------------------
#
# The ``swe_gym`` domain (htir/domains/swe_gym.yaml) is *terminal-shaped* -- bash
# commands and file edits -- so once a SWE-Gym trajectory is expressed in this
# repo's ``{steps:[{src,msg,tools,obs}], reward, task_name}`` turn schema, the
# committed ``terminal`` adapter parses it directly (same code path as
# Terminal-Bench). What SWE-Gym ships instead is an OpenAI-messages transcript of
# OpenHands tool calls; ``normalize_swe_gym_record`` rewrites that transcript
# into the turn schema so no new adapter is needed. This is the single loader the
# SA-10 3x3 transfer matrix was gated on (avg.tex Sec. 4.2 / experiment plan).

# OpenHands multiplexes reads and edits through one ``str_replace_editor`` tool,
# discriminated by its ``command`` arg; ``execute_bash`` carries the shell
# command (so exit-code / test detection fires) and ``finish`` ends the run.
_SWE_GYM_EDIT_SUBCOMMANDS = frozenset({"create", "str_replace", "insert", "append", "write"})


def _swe_gym_tool_call(tc: dict[str, Any]) -> dict[str, str]:
    """
    Map one OpenHands tool call to the ``{fn, cmd}`` shape the ``terminal``
    adapter classifies. ``execute_bash`` -> a shell op carrying the command;
    ``str_replace_editor`` -> a read (``command=view``) or an edit (create /
    str_replace / insert) carrying the file path; ``finish`` -> final submission.
    An unrecognized tool keeps its name so it degrades to ``other`` rather than
    being silently dropped.
    """
    fn = str((tc.get("function") or {}).get("name") or tc.get("name") or "tool")
    raw_args = (tc.get("function") or {}).get("arguments")
    if raw_args is None:
        raw_args = tc.get("arguments")
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args) if raw_args.strip() else {}
        except (json.JSONDecodeError, ValueError):
            args = {}
    elif isinstance(raw_args, dict):
        args = raw_args
    else:
        args = {}
    command = args.get("command")
    path = args.get("path")

    if fn == "execute_bash":
        return {"fn": "bash", "cmd": str(command or "")}
    if "edit" in fn or "str_replace" in fn or fn == "editor":
        sub = str(command or "").strip().lower()
        role_fn = "str_replace_editor" if sub in _SWE_GYM_EDIT_SUBCOMMANDS else "view"
        return {"fn": role_fn, "cmd": str(path or "")}
    if fn == "finish":
        return {"fn": "finish", "cmd": str(command or "")}
    return {"fn": fn, "cmd": str(command or path or "")}


def _swe_gym_messages_to_turns(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert an OpenHands OpenAI-messages transcript into ``{src,msg,tools,obs}``
    turns. A ``user`` message becomes a request turn; an ``assistant`` message
    becomes an agent turn whose ``tools`` are its tool calls and whose ``obs`` is
    the concatenation of the ``tool`` messages that answer it; ``system`` messages
    are dropped (the terminal adapter has no role for them).
    """
    turns: list[dict[str, Any]] = []
    i = 0
    n = len(messages)
    while i < n:
        m = messages[i] if isinstance(messages[i], dict) else {}
        role = m.get("role")
        if role == "user":
            turns.append({"src": "user", "msg": str(m.get("content") or ""), "tools": [], "obs": None})
            i += 1
            continue
        if role == "assistant":
            tools = [_swe_gym_tool_call(t) for t in (m.get("tool_calls") or []) if isinstance(t, dict)]
            obs_parts: list[str] = []
            j = i + 1
            while j < n and isinstance(messages[j], dict) and messages[j].get("role") == "tool":
                obs_parts.append(str(messages[j].get("content") or ""))
                j += 1
            turns.append({
                "src": "agent",
                "msg": str(m.get("content") or ""),
                "tools": tools,
                "obs": "\n".join(p for p in obs_parts if p) or None,
            })
            i = j
            continue
        i += 1  # system / tool-without-a-preceding-assistant: skip
    return turns


def normalize_swe_gym_record(rec: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize a raw ``SWE-Gym/OpenHands-Sampled-Trajectories`` record into the
    turn-schema fields the eval harness expects:

    * ``steps``        -- ``{src,msg,tools,obs}`` turns from the OpenHands
      ``messages`` transcript (so ``to_canonical_steps`` routes to ``terminal``).
    * ``reward``       -- ``1`` if the rollout ``resolved`` the issue else ``0``
      (so ``extract_reward`` / ``label_from_reward`` / ``balanced_sample`` work).
    * ``task_name``    -- the SWE-Gym ``instance_id`` (e.g. ``getmoto__moto-5321``).
    * ``resolved`` / ``run_id`` -- passed through for slicing / provenance.

    Idempotent: a record already in turn schema (has a ``steps`` list) is returned
    with only its ``reward`` back-filled from ``resolved`` if missing, so a
    normalized local cache round-trips unchanged.
    """
    resolved = bool(rec.get("resolved"))
    if isinstance(rec.get("steps"), list):
        reward = extract_reward(rec)
        if reward is None:
            reward = 1 if resolved else 0
        return {**rec, "reward": reward, "resolved": resolved}
    messages = rec.get("messages") or []
    return {
        "steps": _swe_gym_messages_to_turns(messages if isinstance(messages, list) else []),
        "reward": 1 if resolved else 0,
        "task_name": str(rec.get("instance_id") or rec.get("task_name") or ""),
        "resolved": resolved,
        "run_id": rec.get("run_id"),
    }


def load_swe_gym(
    paths: Iterable[str | Path] | None = None,
    *,
    hf: bool = False,
    repo: str = SWE_GYM_HF_REPO,
    split: str = SWE_GYM_HF_SPLIT,
    limit: int | None = None,
    streaming: bool = True,
) -> list[dict[str, Any]]:
    """
    Load SWE-Gym trajectories as normalized turn-schema trace dicts (SA-10).

    ``paths`` reads local ``.json``/``.jsonl`` caches (the offline default; raw
    OpenHands records or an already-normalized cache both work -- see
    :func:`normalize_swe_gym_record`). ``hf=True`` streams the sampled-trajectory
    corpus from the HF hub via the optional ``datasets`` dependency (imported
    lazily, mirroring :func:`load_terminalbench`); ``limit`` caps the record count.
    """
    out: list[dict[str, Any]] = []
    if hf:
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "load_swe_gym(hf=True) requires the 'datasets' package. "
                "Install it with: pip install datasets"
            ) from exc
        ds = load_dataset(repo, split=split, streaming=streaming)
        for i, rec in enumerate(ds):
            if limit is not None and i >= limit:
                break
            out.append(normalize_swe_gym_record(dict(rec)))
        return out

    if not paths:
        raise ValueError("load_swe_gym needs local paths= or hf=True")
    for rec in iter_local_traces(paths):
        out.append(normalize_swe_gym_record(rec))
        if limit is not None and len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Meta-Harness / Terminal-Bench 2.0 live-captured trajectories (Track M, SA-14)
# ---------------------------------------------------------------------------
#
# Unlike terminalbench-trajectories / tau-bench-trajectories / SWE-Gym above,
# there is no pre-existing HF corpus here: these are trajectories *we* capture
# by actually executing an agent -- Meta-Harness's released Terminal-Bench 2.0
# scaffold (Lee et al. 2026, arXiv:2603.28052; artifact:
# stanford-iris-lab/meta-harness-tbench2-artifact), or a plain baseline agent --
# against live Terminal-Bench 2.0 tasks via the `harbor` CLI. See
# `scripts/live_meta_harness_tb2.py`, which is intentionally *outside* this
# package: htir core has no live-execution / subprocess / external-CLI
# dependency, only this loader for the already-captured local output.
#
# Each record is expected to carry an OpenAI-messages transcript (the common
# shape harbor's tool-calling agents -- Claude Code, Terminus, Codex-style --
# emit) plus the task id, the harness label (``meta_harness`` | ``base_agent``,
# the SA-14 grouping key), the executing model, and the real Terminal-Bench 2.0
# pass/fail outcome. ``_tb2_tool_call`` is deliberately more permissive than
# ``_swe_gym_tool_call`` above since the exact tool vocabulary depends on which
# agent produced the transcript; ``htir.adapters.terminal``'s tool-name sets
# already cover the common Claude-Code/Terminus/Codex names, so unrecognised
# tools degrade to ``other`` rather than being dropped.
_TB2_CMD_ARG_KEYS = ("command", "cmd", "file_text", "content")
_TB2_PATH_ARG_KEYS = ("path", "file_path", "target_file")


def _tb2_tool_call(tc: dict[str, Any]) -> dict[str, str]:
    """
    Map one tool call from a captured Terminal-Bench 2.0 transcript to the
    ``{fn, cmd}`` shape ``htir.adapters.terminal`` classifies. Best-effort: falls
    back to a JSON dump of the arguments when no recognised command/path key is
    present, so an unusual tool degrades to ``other`` rather than being dropped.
    """
    fn = str((tc.get("function") or {}).get("name") or tc.get("name") or "tool")
    raw_args = (tc.get("function") or {}).get("arguments")
    if raw_args is None:
        raw_args = tc.get("arguments")
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args) if raw_args.strip() else {}
        except (json.JSONDecodeError, ValueError):
            args = {}
    elif isinstance(raw_args, dict):
        args = raw_args
    else:
        args = {}

    path = next((args[k] for k in _TB2_PATH_ARG_KEYS if args.get(k)), "")
    cmd = next((args[k] for k in _TB2_CMD_ARG_KEYS if args.get(k)), "")
    return {"fn": fn, "cmd": str(cmd or path or (json.dumps(args) if args else ""))}


def _tb2_messages_to_turns(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert a captured TB2 OpenAI-messages transcript into ``{src,msg,tools,obs}``
    turns -- structurally identical to ``_swe_gym_messages_to_turns`` (see there
    for the shape), just paired with the more permissive ``_tb2_tool_call``.
    """
    turns: list[dict[str, Any]] = []
    i = 0
    n = len(messages)
    while i < n:
        m = messages[i] if isinstance(messages[i], dict) else {}
        role = m.get("role")
        if role == "user":
            turns.append({"src": "user", "msg": str(m.get("content") or ""), "tools": [], "obs": None})
            i += 1
            continue
        if role == "assistant":
            tools = [_tb2_tool_call(t) for t in (m.get("tool_calls") or []) if isinstance(t, dict)]
            obs_parts: list[str] = []
            j = i + 1
            while j < n and isinstance(messages[j], dict) and messages[j].get("role") == "tool":
                obs_parts.append(str(messages[j].get("content") or ""))
                j += 1
            turns.append({
                "src": "agent",
                "msg": str(m.get("content") or ""),
                "tools": tools,
                "obs": "\n".join(p for p in obs_parts if p) or None,
            })
            i = j
            continue
        i += 1  # system / tool-without-a-preceding-assistant: skip
    return turns


def normalize_meta_harness_record(rec: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize one captured Terminal-Bench 2.0 trial (the Meta-Harness scaffold
    or a plain baseline agent, run live via ``scripts/live_meta_harness_tb2.py``)
    into the turn-schema fields the eval harness expects:

    * ``steps``      -- ``{src,msg,tools,obs}`` turns from the captured
      OpenAI-messages transcript (so ``to_canonical_steps`` routes to
      ``terminal``).
    * ``reward``     -- ``1`` if Terminal-Bench 2.0 scored the trial solved else
      ``0``. Accepts ``reward``/``passed``/``resolved``/``score`` aliases (the
      capture script's exact field name depends on how harbor reports a task's
      outcome, which needs validating against a real run -- see the module-level
      docstring in ``scripts/live_meta_harness_tb2.py``).
    * ``task_name``  -- the TB2 task id.
    * ``harness``    -- ``'meta_harness'`` / ``'base_agent'`` (the SA-14
      grouping key); ``'unknown'`` if not recorded.
    * ``model``      -- the executing model string (e.g. ``openai/gpt-4o-mini``).

    Idempotent, like :func:`normalize_swe_gym_record`: a record already in turn
    schema round-trips with only its reward back-filled.
    """
    def _reward(r: dict[str, Any]) -> int:
        for key in ("reward", "passed", "resolved", "score"):
            if key in r and r[key] is not None:
                v = r[key]
                if isinstance(v, bool):
                    return 1 if v else 0
                try:
                    return 1 if float(v) > 0 else 0
                except (TypeError, ValueError):
                    continue
        return 0

    if isinstance(rec.get("steps"), list):
        return {**rec, "reward": _reward(rec)}
    messages = rec.get("messages") or []
    return {
        "steps": _tb2_messages_to_turns(messages if isinstance(messages, list) else []),
        "reward": _reward(rec),
        "task_name": str(rec.get("task_id") or rec.get("task_name") or ""),
        "harness": str(rec.get("harness") or "unknown"),
        "model": str(rec.get("model") or ""),
    }


def load_meta_harness_tb2(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """
    Load captured Terminal-Bench 2.0 trajectories (Track M, SA-14) from local
    JSON/JSONL caches under ``data/live_traces/meta_harness_tb2/``. There is no
    HF corpus for this -- it only exists once captured live via
    ``scripts/live_meta_harness_tb2.py``.
    """
    return [normalize_meta_harness_record(rec) for rec in iter_local_traces(paths)]


# ---------------------------------------------------------------------------
# SkillOpt / Terminal-Bench 2.0 live-captured trajectories (Track S, SA-15)
# ---------------------------------------------------------------------------
#
# Same OpenAI-messages + reward capture shape as Track M (SA-14). The harness
# field is ``skillopt`` (best_skill.md injected via harbor --skill) or
# ``no_skill`` (matched agent/model/tasks, no skill). Produced by the SkillOpt
# training / eval driver ``scripts/skillopt_train_tb2.py`` (and any later
# capture normaliser that reuses ``normalize_meta_harness_record``).


def normalize_skillopt_record(rec: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize one Track S capture record. Identical wire format to Track M, so
    this is a thin alias that keeps the SA-15 call site explicit and lets the
    harness vocabulary (``skillopt`` / ``no_skill``) stay documented here.
    """
    out = normalize_meta_harness_record(rec)
    harness = str(out.get("harness") or rec.get("harness") or "unknown")
    return {**out, "harness": harness}


def load_skillopt_tb2(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """
    Load captured SkillOpt / no-skill Terminal-Bench 2.0 trajectories (Track S,
    SA-15) from local JSON/JSONL caches under ``data/live_traces/skillopt_tb2/``.
    """
    return [normalize_skillopt_record(rec) for rec in iter_local_traces(paths)]
