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
# Terminal adapter (deterministic Parse_{S_d}) + terminal_swe binding
# ---------------------------------------------------------------------------

# A small terminal trace in the {src,msg,tools,obs} turn format: read, edit,
# a passing command, a failing test, and a final answer.
TERMINAL_TRACE = {"steps": [
    {"src": "user", "msg": "fix the bug in solver.py"},
    {"src": "agent", "msg": "reading", "tools": [{"fn": "Read", "cmd": "/app/solver.py"}], "obs": "def solve(): ..."},
    {"src": "agent", "msg": "patching", "tools": [{"fn": "Edit", "cmd": "/app/solver.py"}], "obs": "Edited /app/solver.py"},
    {"src": "agent", "msg": "listing", "tools": [{"fn": "Bash", "cmd": "ls /app"}], "obs": "solver.py\nExit code 0"},
    {"src": "agent", "msg": "testing", "tools": [{"fn": "Bash", "cmd": "pytest -q"}], "obs": "1 failed\nExit code 1\n[error] tool reported failure"},
    {"src": "agent", "msg": "It should be fixed now."},
]}


def test_terminal_adapter_autodetected_over_turns():
    """A turn-format trace whose tools are terminal/file ops routes to the
    terminal adapter, not the generic turns adapter."""
    assert detect_adapter(TERMINAL_TRACE).name == "terminal"
    # ... but a non-terminal turn trace still goes to `turns`.
    assert detect_adapter(TURNS_TRACE).name == "turns"


def test_terminal_adapter_types_roles_and_parses_status():
    steps = load_trace(TERMINAL_TRACE)  # auto -> terminal
    roles = [s.get("role_hint") for s in steps]
    # Read -> read_file, Edit -> edit_file, `ls` -> run_command,
    # `pytest` -> run_test, trailing prose -> final_submission.
    assert roles == ["read_file", "edit_file", "run_command", "run_test", "final_submission"]
    statuses = [s.get("status_hint") for s in steps]
    assert statuses[1] == "success"          # Edit returned an observation
    assert statuses[2] == "success"          # `ls` exit code 0
    assert statuses[3] == "failure"          # pytest exit code 1 / [error]
    assert statuses[4] is None               # trailing prose: no observable outcome


def test_terminal_adapter_emits_file_artifact_effects():
    steps = load_trace(TERMINAL_TRACE)
    edit = steps[1]
    assert edit["artifact_effects"][0]["effect_category"] == "artifact_change"
    assert edit["artifact_effects"][0]["affected_resource"] == "/app/solver.py"
    read = steps[0]
    assert read["artifact_effects"][0]["effect_category"] == "read_only"
    # A command/test run produces no file artifact effect (verified by status).
    assert "artifact_effects" not in steps[2]


def test_terminal_adapter_returncode_tag_parsing():
    """The HF Terminal-Bench <returncode>N</returncode> convention is parsed."""
    trace = {"steps": [
        {"src": "agent", "msg": "run", "tools": [{"fn": "Bash", "cmd": "make build"}],
         "obs": "building...\n<returncode>0</returncode>"},
        {"src": "agent", "msg": "run", "tools": [{"fn": "Bash", "cmd": "make check"}],
         "obs": "boom\n<returncode>2</returncode>"},
    ]}
    steps = load_trace(trace)
    assert [s.get("status_hint") for s in steps] == ["success", "failure"]


def test_terminal_pipeline_binds_and_discharges_obligations():
    """End-to-end: with the terminal adapter + terminal_swe spec, obligations
    actually bind and discharge mechanically (some PASS, some FAIL) instead of
    universally abstaining, and a reward=0-style trace is NOT credited 'valid'."""
    from htir.models.domain import get_domain_spec
    from htir.models.htir import ObligationStatus

    steps = load_trace(TERMINAL_TRACE)
    htir = TraceAbstractionAgent(domain_spec=get_domain_spec("terminal_swe")).compile(
        task_id="term", raw_steps=steps, harness_snippets={}, run_checks=True,
    )
    statuses = [o.status for o in htir.obligations]
    assert htir.artifacts, "file edits should lift at least one artifact node"
    assert statuses.count(ObligationStatus.PASSED) > 0, "provenance/exec obligations should pass"
    assert statuses.count(ObligationStatus.ABSTAINED) < len(statuses), "not everything abstains"
    # Never the over-crediting bug: a failing test + broad abstention is not 'valid'.
    assert htir.aggregate.predicted_status in ("uncertain", "invalid")


def test_terminal_pipeline_on_committed_real_trace():
    """Regression on a committed real Terminal-Bench trace (previously produced
    56 all-abstained obligations -> falsely 'valid'). It must now type
    operations, bind artifacts, discharge obligations, and not read 'valid'."""
    from pathlib import Path
    from htir.models.domain import get_domain_spec
    from htir.models.htir import ObligationStatus

    data_dir = Path(__file__).resolve().parent.parent / "data" / "raw_traces"
    trace = data_dir / "01_adaptive-rejection-sampler__8YuTzJm.json"
    if not trace.exists():
        pytest.skip("committed real trace not present")

    assert detect_adapter(load_trace_source(trace)).name == "terminal"
    steps = load_trace(str(trace))
    htir = TraceAbstractionAgent(domain_spec=get_domain_spec("terminal_swe")).compile(
        task_id="ars", raw_steps=steps, harness_snippets={}, run_checks=True,
    )
    passed = sum(o.status == ObligationStatus.PASSED for o in htir.obligations)
    assert htir.artifacts, "at least one file artifact should be extracted"
    assert passed >= 5, "a real multi-edit trace should discharge several obligations"
    assert htir.aggregate.predicted_status != "valid"  # gt_reward == 0


def load_trace_source(path):
    from htir.adapters import read_source
    return read_source(path)


def test_swe_gym_second_domain_compiles_via_terminal_adapter():
    """The SWE-Gym spec is a distinct second domain (avg.tex Sec. 4.2 transfer)
    ingested by the same terminal adapter. A reproduce -> fix -> re-validate
    trace binds obligations and, crucially, the expected failing *reproducer*
    test does not spuriously veto the trajectory to 'invalid'."""
    from htir.models.domain import DOMAIN_SPECS, get_domain_spec
    from htir.models.htir import ObligationStatus

    assert "swe_gym" in DOMAIN_SPECS
    spec = get_domain_spec("swe_gym")
    assert "apply_patch" in spec.operation_type_names()  # distinct from terminal_swe

    trace = {"steps": [
        {"src": "user", "msg": "fix issue #12"},
        {"src": "agent", "msg": "repro", "tools": [{"fn": "Bash", "cmd": "pytest tests/test_bug.py"}],
         "obs": "1 failed\nExit code 1"},
        {"src": "agent", "msg": "fix", "tools": [{"fn": "Edit", "cmd": "/repo/src/mod.py"}],
         "obs": "Edited /repo/src/mod.py"},
        {"src": "agent", "msg": "verify", "tools": [{"fn": "Bash", "cmd": "pytest tests/test_bug.py"}],
         "obs": "1 passed\nExit code 0"},
        {"src": "agent", "msg": "done"},
    ]}
    htir = TraceAbstractionAgent(domain_spec=spec).compile(
        task_id="swe", raw_steps=load_trace(trace, adapter="terminal"),
        harness_snippets={}, run_checks=True,
    )
    assert any(o.status == ObligationStatus.PASSED for o in htir.obligations)
    # A failing reproducer must not veto: reproduce-then-fix is correct SWE.
    assert htir.aggregate.predicted_status != "invalid"


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
