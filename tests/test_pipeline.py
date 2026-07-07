"""
Smoke test for the deterministic (non-LLM) parts of the HTIR/AVG pipeline:
artifact extraction (E_prov), the Step-3 analysis layer (well-formedness +
analysis modules), and claim/obligation construction.

Builds synthetic HTIRs by hand (no OpenRouter/LLM calls) so this runs without
``OPENROUTER_API_KEY``, then exercises:

  * ``TraceAbstractionAgent._extract_artifacts`` -- artifact nodes +
    ``ArtifactProvenanceLink`` (E_prov).
  * ``harnessfix.agents.analysis.enrich`` -- well-formedness issues and the
    provenance / dependency / validation / state-transition / policy-linking
    / integrity analysis modules (avg.tex Sec. 3.4-3.5).
  * ``harnessfix.agents.obligations.build_claims_and_obligations`` -- claim /
    evidence / obligation nodes, the support edges, and obligations seeded
    from unresolved well-formedness issues.
  * a regression test for the ``role.value`` crash in
    ``TraceAbstractionAgent._infer_control_flow_batch`` (``role`` is a plain
    ``str``, not an enum).
"""

from __future__ import annotations

from harnessfix.agents.analysis import enrich
from harnessfix.agents.obligations import build_claims_and_obligations
from harnessfix.agents.trace_abstraction import TraceAbstractionAgent
from harnessfix.models.domain import DEFAULT_DOMAIN_SPEC, Constraint, DomainSpec
from harnessfix.models.htir import (
    HTIR,
    ArtifactEffect,
    ArtifactStateEvidence,
    CheckerType,
    ExecutionStatus,
    ProvenanceRelation,
    Severity,
    TraceStep,
)


def _build_synthetic_htir() -> HTIR:
    steps = [
        TraceStep(
            step_id=1,
            request_message="run pytest",
            response_message="1 failed, 0 passed",
            role="validation",
            execution_status=ExecutionStatus.FAILURE,
            artifact_state_effects=[
                ArtifactStateEvidence(
                    effect_category=ArtifactEffect.ARTIFACT_CHANGE,
                    affected_resource="test_report",
                    observed_change="test suite run, 1 failure",
                )
            ],
        ),
        TraceStep(
            step_id=2,
            request_message="edit parser.py to fix failing test",
            response_message="applied patch",
            role="artifact_editing",
            execution_status=ExecutionStatus.SUCCESS,
            artifact_state_effects=[
                ArtifactStateEvidence(
                    effect_category=ArtifactEffect.READ_ONLY,
                    affected_resource="test_report",
                    observed_change="inspected failure",
                ),
                ArtifactStateEvidence(
                    effect_category=ArtifactEffect.ARTIFACT_CHANGE,
                    affected_resource="parser.py",
                    observed_change="patched parsing logic",
                ),
            ],
        ),
        TraceStep(
            step_id=3,
            request_message="run pytest again",
            response_message="2 passed",
            role="validation",
            execution_status=ExecutionStatus.SUCCESS,
            artifact_state_effects=[
                ArtifactStateEvidence(
                    effect_category=ArtifactEffect.READ_ONLY,
                    affected_resource="parser.py",
                    observed_change="revalidated",
                ),
                ArtifactStateEvidence(
                    effect_category=ArtifactEffect.ARTIFACT_CHANGE,
                    affected_resource="test_report",
                    observed_change="test suite run, all passed",
                ),
            ],
        ),
        TraceStep(
            step_id=4,
            request_message="submit final answer",
            response_message="Fixed the parser bug; tests now pass.",
            role="final_submission",
            execution_status=ExecutionStatus.SUCCESS,
        ),
    ]
    return HTIR(
        task_id="synthetic-smoke-test",
        domain_id=DEFAULT_DOMAIN_SPEC.domain_id,
        steps=steps,
    )


def test_artifact_extraction_populates_provenance_links():
    htir = _build_synthetic_htir()
    agent = TraceAbstractionAgent(domain_spec=DEFAULT_DOMAIN_SPEC)

    agent._extract_artifacts(htir)

    assert len(htir.artifacts) == 2  # test_report, parser.py
    assert len(htir.provenance_links) == 5  # 3 effects on test_report + 2 on parser.py

    relations = {(lk.step_id, lk.artifact_id): lk.relation for lk in htir.provenance_links}
    test_report = next(a for a in htir.artifacts if a.identifier == "test_report")
    parser_py = next(a for a in htir.artifacts if a.identifier == "parser.py")

    assert relations[(1, test_report.artifact_id)] == ProvenanceRelation.CREATED
    assert relations[(2, test_report.artifact_id)] == ProvenanceRelation.READ
    assert relations[(3, test_report.artifact_id)] == ProvenanceRelation.MODIFIED
    assert relations[(2, parser_py.artifact_id)] == ProvenanceRelation.CREATED
    assert relations[(3, parser_py.artifact_id)] == ProvenanceRelation.READ


def test_claims_and_obligations_populate_dependency_links():
    htir = _build_synthetic_htir()
    agent = TraceAbstractionAgent(domain_spec=DEFAULT_DOMAIN_SPEC)
    agent._extract_artifacts(htir)

    # E_val / E_cons / E_causal now live in the Step-3 analysis layer, which
    # must run before obligation generation consumes it.
    enrich(htir, DEFAULT_DOMAIN_SPEC)
    build_claims_and_obligations(htir, DEFAULT_DOMAIN_SPEC)

    assert len(htir.claims) > 0
    assert len(htir.evidence) > 0
    assert len(htir.obligations) > 0
    assert len(htir.support_links) > 0
    assert len(htir.validation_links) > 0
    assert len(htir.constraint_links) > 0

    # E_causal (dependency): step 2 consumes test_report produced by step 1;
    # step 3 consumes parser.py produced by step 2.
    dep_pairs = {(lk.source_step_id, lk.target_step_id) for lk in htir.dependency_links}
    assert (2, 1) in dep_pairs
    assert (3, 2) in dep_pairs


def test_infer_control_flow_batch_role_is_plain_string(monkeypatch):
    """
    Regression test for the ``role.value`` crash: ``TraceStep.role`` was
    refactored from an enum to a plain ``str``, but
    ``_infer_control_flow_batch`` did ``s.role.value`` in its f-string,
    raising ``AttributeError: 'str' object has no attribute 'value'``.

    Stubs ``chat_json`` so the f-string is exercised (and the fix verified)
    without a network call.
    """
    steps = [
        TraceStep(
            step_id=1,
            request_message="run tool",
            response_message="tool output",
            role="tool_invocation",
            execution_status=ExecutionStatus.SUCCESS,
        ),
        TraceStep(
            step_id=2,
            request_message="run validation",
            response_message="1 failed",
            role="validation",
            execution_status=ExecutionStatus.FAILURE,
        ),
    ]

    def _fake_chat_json(messages, schema, model=None, max_tokens=None, **kwargs):
        return schema()  # empty _ControlFlowLinkList

    monkeypatch.setattr("harnessfix.agents.trace_abstraction.chat_json", _fake_chat_json)

    agent = TraceAbstractionAgent(domain_spec=DEFAULT_DOMAIN_SPEC)
    links = agent._infer_control_flow_batch(steps, harness_context="")

    assert links == []


def test_state_transition_pattern_matched_for_fail_edit_pass_trace():
    """State-transition analysis recognises failing-validation -> edit -> passing-validation."""
    htir = _build_synthetic_htir()
    agent = TraceAbstractionAgent(domain_spec=DEFAULT_DOMAIN_SPEC)
    agent._extract_artifacts(htir)

    enrich(htir, DEFAULT_DOMAIN_SPEC)

    matched = [p for p in htir.state_transitions if p.matched]
    assert len(matched) == 1
    assert matched[0].pattern_name == "failing_validation_edit_revalidation"
    assert matched[0].step_ids == [1, 2, 3]


def test_policy_sensitive_step_without_policy_artifact_emits_unresolved_obligation():
    """
    Well-formedness rule (e) + policy-linking analysis: a policy-sensitive
    step (governed by a domain constraint) with no linked policy artifact
    must emit an unresolved obligation, not a task failure.
    """
    spec = DomainSpec(
        domain_id="policy-test",
        constraints=[
            Constraint(
                constraint_id="policy-required",
                description="Final answers must cite an applicable policy.",
                severity=Severity.HIGH,
                applies_to_operations=["final_submission"],
            )
        ],
    )
    steps = [
        TraceStep(
            step_id=1,
            request_message="submit final answer",
            response_message="Done.",
            role="final_submission",
            execution_status=ExecutionStatus.SUCCESS,
        ),
    ]
    htir = HTIR(task_id="policy-smoke-test", domain_id=spec.domain_id, steps=steps)

    enrich(htir, spec)

    issues = [i for i in htir.wellformedness if i.rule_id == "policy_action_unlinked"]
    assert len(issues) == 1
    assert issues[0].offending_node_ids == [1]

    build_claims_and_obligations(htir, spec)
    seeded = [o for o in htir.obligations if o.template_id == "wellformedness:policy_action_unlinked"]
    assert len(seeded) == 1
    assert seeded[0].checker == CheckerType.ABSTENTION


def test_integrity_flags_direct_test_modification():
    """Integrity analysis flags a step that mutates a test artifact directly (tamper)."""
    steps = [
        TraceStep(
            step_id=1,
            request_message="edit test_report.py",
            response_message="patched test assertions directly",
            role="artifact_editing",
            execution_status=ExecutionStatus.SUCCESS,
            artifact_state_effects=[
                ArtifactStateEvidence(
                    effect_category=ArtifactEffect.ARTIFACT_CHANGE,
                    affected_resource="test_report.py",
                    observed_change="modified assertions to force a pass",
                )
            ],
        ),
    ]
    htir = HTIR(task_id="integrity-smoke-test", domain_id=DEFAULT_DOMAIN_SPEC.domain_id, steps=steps)
    agent = TraceAbstractionAgent(domain_spec=DEFAULT_DOMAIN_SPEC)
    agent._extract_artifacts(htir)

    enrich(htir, DEFAULT_DOMAIN_SPEC)

    issues = [i for i in htir.wellformedness if i.rule_id == "integrity_test_modified"]
    assert len(issues) == 1
    assert issues[0].offending_node_ids[0] == 1
