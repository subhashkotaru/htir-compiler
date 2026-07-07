"""
HTIR — Harness-aware Trace Intermediate Representation.

A framework-neutral verification layer for black-box agent trajectories. HTIR
compiles a raw agent trace (from any framework) into a typed verification
graph of operations, artifacts, claims, evidence, and obligations, then checks
those obligations and emits a compact *verification witness* describing what is
supported, what failed, and what remains unresolved.

HTIR is the reference implementation of Adaptive Verifier Graphs (AVG); see
``avg.tex`` and ``docs/avg-mapping.md``. The trace-abstraction layer originally
extended ideas from HarnessFix: From Failed Trajectories to Reliable LLM Agents
(arXiv:2606.06324), retained here only as an optional harness-attribution
extension (see the HarnessFix-marked classes in ``htir.models.htir``).

Quick start::

    from htir import TraceAbstractionAgent, load_trace

    steps = load_trace("trace.json")             # auto-detect the framework
    htir = TraceAbstractionAgent().compile(
        task_id="t1", raw_steps=steps, harness_snippets={}, run_checks=True,
    )
    print(htir.witness.review_recommendation)
"""

from htir.models.htir import HTIR, HTIR_SCHEMA_VERSION as SCHEMA_VERSION, ToolCall, TraceStep
from htir.agents.trace_abstraction import TraceAbstractionAgent
from htir.agents.checker_registry import CheckerContext, register_checker
from htir.adapters import (
    TraceAdapter,
    available_adapters,
    detect_adapter,
    get_adapter,
    load_trace,
    register_adapter,
)

__version__ = "0.1.0"

__all__ = [
    "HTIR",
    "TraceStep",
    "ToolCall",
    "TraceAbstractionAgent",
    # ingestion (any framework)
    "TraceAdapter",
    "load_trace",
    "get_adapter",
    "detect_adapter",
    "available_adapters",
    "register_adapter",
    # verification extension points
    "register_checker",
    "CheckerContext",
    "__version__",
    "SCHEMA_VERSION",
]
