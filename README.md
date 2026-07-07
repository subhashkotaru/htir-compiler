# HTIR — Harness-aware Trace Intermediate Representation

[![CI](https://github.com/htir/htir/actions/workflows/ci.yml/badge.svg)](https://github.com/htir/htir/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

HTIR is a **framework-neutral verification layer for black-box agent
trajectories**. It compiles a raw agent trace — from *any* framework — into a
typed verification graph of operations, artifacts, claims, evidence, and
obligations, checks those obligations with the strongest available evidence,
and emits a compact **verification witness**: what is supported, what failed,
what remains unresolved, and which evidence backs each decision.

HTIR is the reference implementation of **Adaptive Verifier Graphs (AVG)**
(see [`avg.tex`](avg.tex) and [`docs/avg-mapping.md`](docs/avg-mapping.md) for
the full symbol-to-code mapping). The deterministic pipeline runs **offline —
no API key required**; LLM-backed semantic checks are opt-in.

---

## Why HTIR

An agent can reach a plausible endpoint while skipping validation, editing the
wrong artifact, violating a policy, or making unsupported claims. A single
pass/fail judge over a long trace is brittle, and full step-level labels are
rarely available. HTIR instead turns a messy trace into **local, evidence-
carrying verification obligations** and checks each with a mechanical checker
when possible, a narrow semantic judge when necessary, or an explicit
abstention when evidence is insufficient.

## Install

```bash
pip install htir            # deterministic core, no LLM dependency
pip install "htir[llm]"     # + optional semantic checks/analysis
```

## Quick start (works with any framework)

```python
from htir import load_trace, TraceAbstractionAgent

# Auto-detects OpenAI / Anthropic / LangChain / OpenInference-OTel / ... traces.
steps = load_trace("trace.json")

htir = TraceAbstractionAgent().compile(
    task_id="fix-parser",
    raw_steps=steps,
    harness_snippets={},
    run_checks=True,          # mechanical checkers + aggregation + witness
)

print(htir.aggregate.predicted_status)      # valid / invalid / uncertain
print(htir.witness.review_recommendation)   # e.g. "Status: uncertain. Inspect: obligation 1 ..."
```

### From the command line

```bash
htir compile trace.json                    # auto-detect framework, print witness
htir compile trace.json --domain terminal_swe -o graph.json
htir adapters                              # list supported frameworks
htir domains                               # list domain specs (S_d)
```

```
task:        trace
adapter:     openinference   domain: terminal_swe   schema: 1.0
steps:       3  (3 tool calls, 0 artifacts)
obligations: 4
status:      UNCERTAIN  (coverage 0%, uncertainty 100%)
witness:     0 passed / 0 failed / 4 abstained
review:      Status: uncertain. 2 unresolved high-severity obligation(s) abstained. Inspect: obligation 1 (swe-edit-then-validate).
```

## Supported frameworks (trace adapters)

Ingestion is the framework-neutral boundary. Each adapter maps a source trace
to a canonical step list with **structured tool calls** preserved (name,
arguments, result, status) — never string-concatenated away.

| Adapter | `--adapter` | Recognizes |
|---|---|---|
| OpenAI / ChatML | `openai` | chat-completion `messages` with `tool_calls` |
| Anthropic | `anthropic` | Messages API `tool_use` / `tool_result` blocks |
| LangChain / LangGraph | `langchain` | serialized `human`/`ai`/`tool` messages |
| OpenInference / OpenTelemetry | `openinference` | GenAI spans (Phoenix, OpenLLMetry, OTel) |
| Turns | `turns` | `{src, msg, tools, obs}` logs |
| Raw / passthrough | `raw` | `{request, response, tool_calls}` steps |

Adapters are **dependency-free** (they parse the serialized/exported form), so
you don't need the source framework installed to verify its traces.

## Extending HTIR without forking

HTIR is a plugin framework. Add support for a new framework or a new check by
**registering**, not by editing core — including from a separate package via
the `htir.adapters` / `htir.checkers` entry-point groups.

**A new framework adapter:**

```python
from htir.adapters import TraceAdapter, register_adapter, canonical_step

@register_adapter
class MyFrameworkAdapter(TraceAdapter):
    name = "myframework"
    def detect(self, data): ...
    def parse(self, data): return [canonical_step(request=..., response=...)]
```

**A new mechanical checker** (e.g. a JSON-Schema or SQL validator):

```python
from htir import register_checker, CheckerContext
from htir.models.htir import CheckerResult

@register_checker(claim_type="sql_result")
def check_sql(ctx: CheckerContext) -> CheckerResult:
    ...
```

**A new domain** is a small YAML spec (operations, artifact types,
constraints, obligation templates) in `htir/domains/`. See
[`htir/domains/terminal_swe.yaml`](htir/domains/terminal_swe.yaml).

## The pipeline (AVG Steps 1–8)

```
raw trace → typed events → verification graph Gτ → analysis modules
          → obligations → checkers → aggregation zτ → verification witness Wτ
```

| Stage | Module |
|---|---|
| Ingest (any framework) → canonical steps | `htir.adapters` |
| Graph construction (Steps 1–2) | `htir.agents.trace_abstraction` |
| Well-formedness + analysis modules (Step 3) | `htir.agents.analysis` |
| Obligation generation (Step 4) | `htir.agents.obligations` |
| Checking (Step 5) | `htir.agents.checking` + `htir.agents.checker_registry` |
| Aggregation + witness (Step 6) | `htir.agents.witness` |
| Online intervention (Step 7) | `htir.agents.intervention` |
| Offline harness improvement (Step 8) | `htir.agents.harness_improvement` |

The serialized graph carries a `schema_version` (interchange contract). Batch
many traces with `TraceAbstractionAgent().compile_many([...])`.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Contributions of new adapters, domain
specs, and checkers are especially welcome.

## License

Apache-2.0 — see [LICENSE](LICENSE).
