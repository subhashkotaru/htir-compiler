"""
Command-line interface for HTIR.

    htir compile TRACE [--adapter auto] [--domain default] [-o out.json]
    htir adapters          # list installed trace adapters (frameworks)
    htir domains           # list installed domain specs (S_d)
    htir version

``compile`` ingests a trace from any supported framework (auto-detected),
compiles it into an HTIR verification graph, runs the deterministic checkers,
and prints the verification witness. It runs fully offline (no API key) unless
you opt into ``--semantic`` or ``--harness-links``.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from htir import __version__
from htir.adapters import available_adapters, detect_adapter, get_adapter, read_source
from htir.adapters.base import load_trace
from htir.agents.trace_abstraction import TraceAbstractionAgent
from htir.models.domain import DOMAIN_SPECS, get_domain_spec, load_domain_artifacts


def _cmd_compile(args: argparse.Namespace) -> int:
    data = read_source(args.trace)
    adapter = detect_adapter(data) if args.adapter == "auto" else get_adapter(args.adapter)
    steps = adapter.parse(data)
    if not steps:
        print(f"error: adapter '{adapter.name}' produced no steps from {args.trace}", file=sys.stderr)
        return 2

    spec = get_domain_spec(args.domain)
    bundle = load_domain_artifacts(args.domain) if args.omega else None
    agent = TraceAbstractionAgent(model=args.model, domain_spec=spec, domain_artifacts=bundle)
    htir = agent.compile(
        task_id=args.task_id or _stem(args.trace),
        raw_steps=steps,
        harness_snippets={},
        outcome=args.outcome,
        generate_obligations=True,
        run_checks=not args.no_checks,
        use_semantic_analysis=args.semantic,
        infer_harness_links=args.harness_links,
    )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(htir.model_dump_json(indent=2))
        print(f"wrote HTIR graph -> {args.out}  (adapter={adapter.name}, domain={spec.domain_id})", file=sys.stderr)
    elif args.json:
        print(htir.model_dump_json(indent=2))

    if not args.quiet:
        _print_summary(htir, adapter.name, spec.domain_id, to=sys.stderr if (args.json and not args.out) else sys.stdout)
    return 0


def _print_summary(htir: Any, adapter_name: str, domain_id: str, to=sys.stdout) -> None:
    w = htir.witness
    agg = htir.aggregate
    n_tools = sum(len(s.tool_calls) for s in htir.steps)
    lines = [
        f"task:        {htir.task_id}",
        f"adapter:     {adapter_name}   domain: {domain_id}   schema: {htir.schema_version}",
        f"steps:       {len(htir.steps)}  ({n_tools} tool calls, {len(htir.artifacts)} artifacts)",
        f"obligations: {len(htir.obligations)}",
    ]
    if agg is not None:
        lines.append(
            f"status:      {agg.predicted_status.upper()}  "
            f"(coverage {agg.evidence_coverage:.0%}, uncertainty {agg.uncertainty:.0%})"
        )
    if w is not None:
        lines.append(
            f"witness:     {len(w.passed_obligation_ids)} passed / "
            f"{len(w.failed_obligation_ids)} failed / {len(w.abstained_obligation_ids)} abstained"
        )
        lines.append(f"review:      {w.review_recommendation}")
    print("\n".join(lines), file=to)


def _cmd_adapters(args: argparse.Namespace) -> int:
    print("Installed trace adapters (ingest --adapter <name>):")
    for name in available_adapters():
        a = get_adapter(name)
        aliases = ", ".join(x for x in a.aliases) or "-"
        print(f"  {name:<14} priority={a.priority:<3} aliases: {aliases}")
    return 0


def _cmd_domains(args: argparse.Namespace) -> int:
    print("Installed domain specs (--domain <id>):")
    for domain_id, spec in sorted(DOMAIN_SPECS.items()):
        print(
            f"  {domain_id:<16} ops={len(spec.operation_types)} "
            f"artifacts={len(spec.artifact_types)} constraints={len(spec.constraints)} "
            f"templates={len(spec.obligation_templates)}"
        )
    return 0


def _cmd_version(args: argparse.Namespace) -> int:
    print(f"htir {__version__}")
    return 0


def _stem(path: str) -> str:
    from pathlib import Path
    return Path(path).stem


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="htir", description="Verify black-box agent trajectories from any framework.")
    p.add_argument("--version", action="version", version=f"htir {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("compile", help="compile a trace into an HTIR verification witness")
    c.add_argument("trace", help="path to a trace file (.json / .jsonl)")
    c.add_argument("--adapter", default="auto", help="trace adapter/framework (default: auto-detect)")
    c.add_argument("--domain", default="default", help="domain spec S_d id (default: default)")
    c.add_argument("--task-id", default="", help="task id to label the graph (default: trace filename)")
    c.add_argument("--outcome", default="", help="externally-known outcome label, e.g. resolved/failed")
    c.add_argument("--no-checks", action="store_true", help="skip Step-5/6 checking + witness")
    c.add_argument("--semantic", action="store_true", help="enable LLM semantic checks/analysis (needs API key)")
    c.add_argument("--harness-links", action="store_true", help="infer HarnessFix step->step links (needs API key)")
    c.add_argument("--omega", action="store_true", help="load Omega_d domain artifacts for the domain, if present")
    c.add_argument("--model", default=None, help="model slug for any LLM passes")
    c.add_argument("-o", "--out", help="write the full HTIR graph JSON to this path")
    c.add_argument("--json", action="store_true", help="print the full HTIR graph JSON to stdout")
    c.add_argument("--quiet", action="store_true", help="suppress the human-readable witness summary")
    c.set_defaults(func=_cmd_compile)

    sub.add_parser("adapters", help="list installed trace adapters").set_defaults(func=_cmd_adapters)
    sub.add_parser("domains", help="list installed domain specs").set_defaults(func=_cmd_domains)
    sub.add_parser("version", help="print version").set_defaults(func=_cmd_version)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # --model defaults to the agent default when not given.
    if getattr(args, "model", None) is None and args.__dict__.get("func") is _cmd_compile:
        from htir.utils.llm import DEFAULT_MODEL
        args.model = DEFAULT_MODEL
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
