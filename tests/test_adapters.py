"""
Tests for the framework-neutral ingestion layer (htir.adapters), the checker
registry, structured tool calls, schema versioning, indexed lookups, batch
compile, and the CLI. All offline (no LLM / no API key).
"""

from __future__ import annotations

import json

import pytest

from htir import (
    HTIR,
    SCHEMA_VERSION,
    TraceAbstractionAgent,
    available_adapters,
    detect_adapter,
    load_trace,
)
from htir.adapters import TraceAdapter, canonical_step, get_adapter, register_adapter
from htir.agents.checker_registry import (
    CheckerContext,
    register_checker,
    registered_checkers,
    resolve_checker,
)
from htir.models.htir import CheckerResult, Obligation


# ---------------------------------------------------------------------------
# Adapter autodetection + structured tool calls
# ---------------------------------------------------------------------------

OPENAI_TRACE = {"messages": [
    {"role": "user", "content": "weather in Paris?"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
    ]},
    {"role": "tool", "tool_call_id": "c1", "content": "18C sunny"},
    {"role": "assistant", "content": "It is 18C and sunny."},
]}

ANTHROPIC_TRACE = [
    {"role": "user", "content": [{"type": "text", "text": "weather in Paris?"}]},
    {"role": "assistant", "content": [{"type": "tool_use", "id": "tu1", "name": "get_weather", "input": {"city": "Paris"}}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "18C sunny"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "18C and sunny."}]},
]

LANGCHAIN_TRACE = [
    {"type": "human", "content": "weather?"},
    {"type": "ai", "content": "", "tool_calls": [{"name": "get_weather", "args": {"city": "Paris"}, "id": "c1"}]},
    {"type": "tool", "tool_call_id": "c1", "content": "18C"},
]

OTEL_TRACE = [
    {"name": "llm", "attributes": {"openinference.span.kind": "LLM", "input.value": "weather?", "output.value": "calling tool"}, "status_code": "OK", "start_time": 1},
    {"name": "get_weather", "attributes": {"openinference.span.kind": "TOOL", "tool.name": "get_weather", "tool.parameters": '{"city": "Paris"}', "output.value": "18C"}, "status_code": "OK", "start_time": 2},
]

TURNS_TRACE = {"steps": [
    {"src": "user", "msg": "weather?"},
    {"src": "agent", "msg": "checking", "tools": [{"fn": "get_weather", "cmd": "Paris"}], "obs": "18C"},
]}


@pytest.mark.parametrize("data,expected", [
    (OPENAI_TRACE, "openai"),
    (ANTHROPIC_TRACE, "anthropic"),
    (LANGCHAIN_TRACE, "langchain"),
    (OTEL_TRACE, "openinference"),
    (TURNS_TRACE, "turns"),
])
def test_autodetection_routes_to_expected_adapter(data, expected):
    assert detect_adapter(data).name == expected


@pytest.mark.parametrize("data", [OPENAI_TRACE, ANTHROPIC_TRACE, LANGCHAIN_TRACE, OTEL_TRACE, TURNS_TRACE])
def test_adapters_preserve_structured_tool_calls(data):
    steps = load_trace(data)
    all_calls = [tc for s in steps for tc in s.get("tool_calls", [])]
    assert all_calls, "adapter should surface at least one structured tool call"
    names = {tc["name"] for tc in all_calls}
    assert "get_weather" in names


def test_openai_wires_tool_result_back_to_call():
    steps = load_trace(OPENAI_TRACE)
    call = next(tc for s in steps for tc in s.get("tool_calls", []))
    assert call["name"] == "get_weather"
    assert call["arguments"] == {"city": "Paris"}
    assert call["result"] == "18C sunny"
    assert call["status"] == "success"


def test_all_builtin_adapters_registered():
    assert set(available_adapters()) >= {"openai", "anthropic", "langchain", "openinference", "turns", "raw"}


def test_load_trace_from_jsonl_file(tmp_path):
    p = tmp_path / "trace.jsonl"
    p.write_text('{"request": "a", "response": "b"}\n{"request": "c", "response": "d"}\n', encoding="utf-8")
    steps = load_trace(str(p))
    assert len(steps) == 2
    assert steps[0]["request"] == "a"


def test_register_custom_adapter():
    @register_adapter
    class _MarkerAdapter(TraceAdapter):
        name = "unit_marker"
        priority = 999

        def detect(self, data):
            return isinstance(data, dict) and data.get("__marker__") is True

        def parse(self, data):
            return [canonical_step(request="x", response="y")]

    assert "unit_marker" in available_adapters()
    assert detect_adapter({"__marker__": True}).name == "unit_marker"
    assert load_trace({"__marker__": True})[0]["response"] == "y"


def test_unknown_adapter_raises():
    with pytest.raises(KeyError):
        get_adapter("does-not-exist")


# ---------------------------------------------------------------------------
# Structured tool calls survive into the compiled graph + drive domain roles
# ---------------------------------------------------------------------------

def test_compiled_graph_carries_tool_calls_and_schema_version():
    steps = load_trace(OTEL_TRACE)
    htir = TraceAbstractionAgent().compile(task_id="t", raw_steps=steps, harness_snippets={})
    assert htir.schema_version == SCHEMA_VERSION
    tool_names = [tc.name for s in htir.steps for tc in s.tool_calls]
    assert "get_weather" in tool_names


def test_tool_call_name_promotes_generic_role_to_domain_operation():
    from htir.models.domain import TERMINAL_DOMAIN_SPEC
    otel = [
        {"name": "run_test", "attributes": {"openinference.span.kind": "TOOL", "tool.name": "run_test", "output.value": "1 failed"}, "status_code": "ERROR", "start_time": 1},
    ]
    steps = load_trace(otel)
    htir = TraceAbstractionAgent(domain_spec=TERMINAL_DOMAIN_SPEC).compile(
        task_id="t", raw_steps=steps, harness_snippets={},
    )
    assert htir.steps[0].role == "run_test"  # promoted from the tool-call name


# ---------------------------------------------------------------------------
# Checker registry
# ---------------------------------------------------------------------------

def test_builtin_checkers_registered():
    keys = registered_checkers()
    assert "execution_status" in keys["claim_type"]
    assert "artifact_provenance" in keys["claim_type"]
    assert "schema" in keys["required_evidence"]


def test_register_custom_checker_is_resolved():
    @register_checker(claim_type="unit_custom_claim")
    def _check(ctx: CheckerContext) -> CheckerResult:
        return CheckerResult(p_pass=1.0)

    ob = Obligation(obligation_id=1, claim_id=1)
    from htir.models.htir import ClaimNode
    claim = ClaimNode(claim_id=1, statement="s", claim_type="unit_custom_claim")
    assert resolve_checker(ob, claim) is _check


# ---------------------------------------------------------------------------
# Indexed lookups + batch compile
# ---------------------------------------------------------------------------

def test_indexed_lookups_track_appends():
    from htir.models.htir import TraceStep
    h = HTIR(task_id="idx")
    assert h.get_step(1) is None                 # empty index
    h.steps.append(TraceStep(step_id=1, request_message="a", response_message="b"))
    assert h.get_step(1) is not None             # index refreshed on length change
    h.steps.append(TraceStep(step_id=2, request_message="c", response_message="d"))
    assert h.get_step(2).response_message == "d"


def test_compile_many_preserves_order():
    agent = TraceAbstractionAgent()
    traces = [
        {"task_id": "a", "raw_steps": load_trace([{"request": "1", "response": "x", "role_hint": "final_submission", "status_hint": "success"}])},
        {"task_id": "b", "raw_steps": load_trace([{"request": "2", "response": "y", "role_hint": "final_submission", "status_hint": "success"}])},
    ]
    results = agent.compile_many(traces, run_checks=True)
    assert [h.task_id for h in results] == ["a", "b"]
    assert all(h.witness is not None for h in results)


# ---------------------------------------------------------------------------
# CLI (offline)
# ---------------------------------------------------------------------------

def test_compile_degrades_gracefully_when_llm_unavailable(monkeypatch):
    """A trace with no role hints compiles (roles default) instead of crashing when no LLM."""
    agent = TraceAbstractionAgent()

    def _no_llm(*args, **kwargs):
        raise EnvironmentError("OPENROUTER_API_KEY is not set")

    monkeypatch.setattr(agent, "_annotate_step", _no_llm)
    with pytest.warns(RuntimeWarning):
        htir = agent.compile(
            task_id="offline",
            raw_steps=[{"request": "do a thing", "response": "did it"}],
            harness_snippets={},
        )
    assert len(htir.steps) == 1
    assert htir.steps[0].role == "other"


def test_cli_adapters_and_domains(capsys):
    from htir.cli import main
    assert main(["adapters"]) == 0
    out = capsys.readouterr().out
    assert "openinference" in out and "openai" in out
    assert main(["domains"]) == 0
    assert "terminal_swe" in capsys.readouterr().out


def test_cli_compile_offline_writes_witness(tmp_path, capsys):
    from htir.cli import main
    trace = tmp_path / "t.json"
    trace.write_text(json.dumps(OTEL_TRACE), encoding="utf-8")
    out = tmp_path / "graph.json"
    rc = main(["compile", str(trace), "--domain", "terminal_swe", "-o", str(out)])
    assert rc == 0
    graph = json.loads(out.read_text(encoding="utf-8"))
    assert graph["schema_version"] == SCHEMA_VERSION
    assert graph["witness"] is not None
