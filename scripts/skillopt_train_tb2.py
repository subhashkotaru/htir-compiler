#!/usr/bin/env python
"""
Track S training driver: run SkillOpt's training loop against Terminal-Bench 2.0.

DELIBERATELY OUTSIDE THE ``htir`` PACKAGE. SkillOpt's built-in env registry
does not include Terminal-Bench; this script instantiates our local
``scripts.skillopt_tb2.TerminalBenchAdapter`` and hands it to
``skillopt.engine.trainer.ReflACTTrainer`` directly (no need to patch
SkillOpt's hard-coded registry).

Prerequisites:

    pip install skillopt
    pip install --only-binary=:all: harbor   # same as Track M
    export OPENAI_API_KEY=<key>             # or .env
    # Docker daemon up for live rollouts

Usage::

    # Offline smoke (no harbor / no API spend for rollouts):
    python scripts/skillopt_train_tb2.py \\
        --config configs/skillopt/terminal_bench_pilot.yaml --dry-run

    # Real (expensive) training -- decide later whether to run:
    python scripts/skillopt_train_tb2.py \\
        --config configs/skillopt/terminal_bench_pilot.yaml

The deployed artifact is ``<out_root>/best_skill.md``. Rollout trajectories
under ``<out_root>/`` can later be normalised into
``data/live_traces/skillopt_tb2/`` for SA-15 (skillopt vs. no_skill).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, required=True, help="Path to SkillOpt YAML config")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip harbor rollouts (stub zero scores). Still exercises SkillOpt wiring.",
    )
    p.add_argument(
        "--cfg-options",
        nargs="+",
        default=[],
        help="Override config keys: section.key=value (SkillOpt convention)",
    )
    args = p.parse_args(argv)

    from skillopt.config import flatten_config, load_config
    from skillopt.engine.trainer import ReflACTTrainer

    from scripts.skillopt_tb2 import TerminalBenchAdapter

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    cfg = flatten_config(load_config(str(cfg_path)))

    # Resolve relative paths against the repo root so the config stays portable.
    for key in ("data_path", "skill_init", "out_root", "split_dir", "split_output_dir"):
        val = cfg.get(key)
        if isinstance(val, str) and val and not Path(val).is_absolute():
            cfg[key] = str((REPO_ROOT / val).resolve())

    if args.dry_run:
        cfg["dry_run"] = True
        os.environ["HTIR_SKILLOPT_DRY_RUN"] = "1"

    # Apply --cfg-options (section.key=value or flat key=value).
    for opt in args.cfg_options:
        if "=" not in opt:
            raise SystemExit(f"bad --cfg-options entry (want key=value): {opt}")
        key, raw = opt.split("=", 1)
        key = key.strip()
        # Strip structured-section prefixes SkillOpt also accepts.
        if key.startswith("env."):
            key = key[4:]
        if key.startswith("train.") or key.startswith("model.") or key.startswith("evaluation."):
            # leave as-is for flatten_config-style keys already flattened;
            # also accept already-flat names.
            pass
        # Best-effort type coercion.
        val: object = raw
        if raw.lower() in {"true", "false"}:
            val = raw.lower() == "true"
        else:
            try:
                val = int(raw)
            except ValueError:
                try:
                    val = float(raw)
                except ValueError:
                    val = raw
        cfg[key] = val

    # Ensure SkillOpt sees our env name even though we bypass its registry.
    cfg.setdefault("env", "terminal_bench")
    cfg.setdefault("out_root", str((REPO_ROOT / "data" / "skillopt_runs" / "tb2_pilot").resolve()))

    adapter = TerminalBenchAdapter(
        split_dir=str(cfg.get("split_dir") or ""),
        data_path=str(cfg.get("data_path") or ""),
        split_mode=str(cfg.get("split_mode") or "ratio"),
        split_ratio=str(cfg.get("split_ratio") or "2:1:7"),
        split_seed=int(cfg.get("split_seed") or 42),
        split_output_dir=str(cfg.get("split_output_dir") or ""),
        workers=int(cfg.get("workers") or 1),
        analyst_workers=int(cfg.get("analyst_workers") or 4),
        failure_only=bool(cfg.get("failure_only") or False),
        minibatch_size=int(cfg.get("minibatch_size") or 4),
        edit_budget=int(cfg.get("edit_budget") or 4),
        seed=int(cfg.get("seed") or 42),
        limit=int(cfg.get("limit") or 0),
        harbor_model=str(cfg.get("harbor_model") or cfg.get("target_model") or "openai/gpt-4o-mini"),
        harbor_agent=str(cfg.get("harbor_agent") or "terminus-2"),
        harbor_env=str(cfg.get("harbor_env") or "docker"),
        harbor_max_retries=int(cfg.get("harbor_max_retries") or 0),
        dry_run=bool(cfg.get("dry_run") or False),
        inject_skill=bool(cfg.get("inject_skill", True)),
    )

    print(f"[skillopt_tb2] config={cfg_path}", file=sys.stderr)
    print(f"[skillopt_tb2] out_root={cfg['out_root']}", file=sys.stderr)
    print(f"[skillopt_tb2] dry_run={cfg.get('dry_run')}", file=sys.stderr)
    print(f"[skillopt_tb2] harbor_model={adapter.harbor_model} agent={adapter.harbor_agent}", file=sys.stderr)

    trainer = ReflACTTrainer(cfg, adapter)
    summary = trainer.train()
    print(f"[skillopt_tb2] done. summary keys={list(summary.keys()) if isinstance(summary, dict) else type(summary)}", file=sys.stderr)
    best = Path(cfg["out_root"]) / "best_skill.md"
    if best.exists():
        print(f"[skillopt_tb2] best_skill -> {best}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
