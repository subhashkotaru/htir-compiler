"""
Harness-aware Trace Intermediate Representation (HTIR) compiler.

Converts raw agent execution traces into HTIR — a structured,
step-level representation that supports evidence tracing and
harness-layer attribution.

Based on HarnessFix: From Failed Trajectories to Reliable LLM Agents
(arxiv 2606.06324v1)
"""

from harnessfix.models.htir import HTIR, TraceStep
from harnessfix.agents.trace_abstraction import TraceAbstractionAgent

__version__ = "0.1.0"
__all__ = ["HTIR", "TraceStep", "TraceAbstractionAgent"]
