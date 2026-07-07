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
from harnessfix.agents.checking import check_obligations
from harnessfix.agents.obligations import _template_triggers, build_claims_and_obligations
from harnessfix.agents.trace_abstraction import TraceAbstractionAgent
from harnessfix.agents.witness import aggregate, build_witness
from harnessfix.models.domain import DEFAULT_DOMAIN_SPEC, Constraint, DomainSpec, ObligationTemplate
from harnessfix.models.htir import (
    HTIR,
    ArtifactEffect,
    ArtifactStateEvidence,
    CheckerResult,
    CheckerType,
    ClaimStatus,
    EvidenceType,
    ExecutionStatus,
    Obligation,
    ObligationStatus,
    ProvenanceRelation,
    Severity,
    SupportPolarity,
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


def _build_compiled_synthetic_htir():
    """``_build_synthetic_htir`` run through artifact extraction + enrich + obligations."""
    htir = _build_synthetic_htir()
    agent = TraceAbstractionAgent(domain_spec=DEFAULT_DOMAIN_SPEC)
    agent._extract_artifacts(htir)
    enrich(htir, DEFAULT_DOMAIN_SPEC)
    build_claims_and_obligations(htir, DEFAULT_DOMAIN_SPEC)
    return htir


def test_failing_step_evidence_refutes_its_execution_status_claim():
    """B1: a FAILURE execution-status claim is REFUTED, not SUPPORTED, by its evidence."""
    htir = _build_compiled_synthetic_htir()

    step1_claim = next(
        c for c in htir.claims if c.claim_type == "execution_status" and c.source_step_id == 1
    )
    links = [lk for lk in htir.support_links if lk.claim_id == step1_claim.claim_id]
    assert links, "expected at least one support link for the failing step's execution claim"
    assert all(lk.polarity == SupportPolarity.REFUTES for lk in links)


def test_obligation_candidate_evidence_matches_required_evidence_type():
    """B2: candidate_evidence_ids (E_i) only contain evidence of the required type r_i."""
    htir = _build_compiled_synthetic_htir()
    evidence_type_by_id = {e.evidence_id: e.evidence_type for e in htir.evidence}

    for ob in htir.obligations:
        for ev_id in ob.candidate_evidence_ids:
            assert evidence_type_by_id[ev_id] == ob.required_evidence


def test_validation_obligation_anchors_on_execution_status_not_artifact_provenance():
    """
    B3: step 1 and step 3 are both validations that also produce/mutate an
    artifact (test_report), so ``claim_ids[-1]`` (the old, positional logic)
    would incorrectly land on the artifact_provenance claim. The
    ``uni-validation-backed`` template declares ``target_claim_type:
    execution_status``, so its obligation must anchor on that step's
    execution_status claim instead.
    """
    htir = _build_compiled_synthetic_htir()

    validation_obligations = [o for o in htir.obligations if o.template_id == "uni-validation-backed"]
    assert validation_obligations

    claims_by_id = {c.claim_id: c for c in htir.claims}
    for ob in validation_obligations:
        claim = claims_by_id[ob.claim_id]
        assert claim.claim_type == "execution_status"


def test_explain_failure_obligation_has_routed_checker():
    """B4: trig-explain-failure (r_i = log) must be routed to a real checker, not UNASSIGNED."""
    htir = _build_compiled_synthetic_htir()

    explain_failure = [o for o in htir.obligations if o.template_id == "trig-explain-failure"]
    assert explain_failure
    assert all(o.checker != CheckerType.UNASSIGNED for o in explain_failure)


def test_build_claims_and_obligations_is_idempotent():
    """B5: calling build_claims_and_obligations twice must not duplicate nodes/edges."""
    htir = _build_synthetic_htir()
    agent = TraceAbstractionAgent(domain_spec=DEFAULT_DOMAIN_SPEC)
    agent._extract_artifacts(htir)
    enrich(htir, DEFAULT_DOMAIN_SPEC)

    build_claims_and_obligations(htir, DEFAULT_DOMAIN_SPEC)
    first_counts = (
        len(htir.claims), len(htir.evidence), len(htir.obligations), len(htir.support_links),
    )

    build_claims_and_obligations(htir, DEFAULT_DOMAIN_SPEC)
    second_counts = (
        len(htir.claims), len(htir.evidence), len(htir.obligations), len(htir.support_links),
    )

    assert first_counts == second_counts


def test_template_trigger_is_exact_not_substring():
    """B6: a 'decision' trigger must not fire on the 'orchestration_decision' operation type."""
    template = ObligationTemplate(
        template_id="t-decision",
        claim_template="unused",
        trigger="decision",
        required_evidence=EvidenceType.NONE,
    )
    step = TraceStep(
        step_id=1,
        request_message="plan next action",
        response_message="decided to run tests",
        role="orchestration_decision",
        execution_status=ExecutionStatus.SUCCESS,
    )
    assert _template_triggers(template, step) is False


def test_integrity_finding_escalates_to_veto():
    """B7: a HIGH-severity integrity finding seeds an obligation escalating to VETO."""
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
    htir = HTIR(task_id="integrity-escalation-test", domain_id=DEFAULT_DOMAIN_SPEC.domain_id, steps=steps)
    agent = TraceAbstractionAgent(domain_spec=DEFAULT_DOMAIN_SPEC)
    agent._extract_artifacts(htir)
    enrich(htir, DEFAULT_DOMAIN_SPEC)
    build_claims_and_obligations(htir, DEFAULT_DOMAIN_SPEC)

    from harnessfix.models.htir import EscalationRule

    seeded = [o for o in htir.obligations if o.template_id == "wellformedness:integrity_test_modified"]
    assert len(seeded) == 1
    assert seeded[0].escalation == EscalationRule.VETO


def test_obligations_deduped_by_claim_and_template():
    """B8: two templates with the same trigger firing on the same claim don't duplicate obligations."""
    spec = DomainSpec(
        domain_id="dedup-test",
        operation_types=DEFAULT_DOMAIN_SPEC.operation_types,
        obligation_templates=[
            ObligationTemplate(
                template_id="dup-a",
                claim_template="Final answer is supported.",
                trigger="final_submission",
                required_evidence=EvidenceType.NONE,
                target_claim_type="final_answer_support",
            ),
            ObligationTemplate(
                template_id="dup-a",
                claim_template="Final answer is supported (duplicate template id).",
                trigger="final_submission",
                required_evidence=EvidenceType.NONE,
                target_claim_type="final_answer_support",
            ),
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
    htir = HTIR(task_id="dedup-smoke-test", domain_id=spec.domain_id, steps=steps)
    enrich(htir, spec)
    build_claims_and_obligations(htir, spec)

    matching = [o for o in htir.obligations if o.template_id == "dup-a"]
    assert len(matching) == 1


# ---------------------------------------------------------------------------
# Step 5 -- checker execution (harnessfix.agents.checking)
# ---------------------------------------------------------------------------

def test_post_edit_validation_obligation_passes_mechanically():
    """
    Step 5: the trig-post-edit-validation obligation anchored on the edit
    step's (step 2, parser.py) artifact_provenance claim is PASSED
    mechanically, because the later validation step 3 revalidated it with
    ExecutionStatus.SUCCESS.
    """
    htir = _build_compiled_synthetic_htir()
    check_obligations(htir, DEFAULT_DOMAIN_SPEC)

    claims_by_id = {c.claim_id: c for c in htir.claims}
    post_edit = [
        o for o in htir.obligations
        if o.template_id == "trig-post-edit-validation"
        and claims_by_id[o.claim_id].source_step_id == 2
    ]
    assert len(post_edit) == 1
    ob = post_edit[0]
    assert ob.status == ObligationStatus.PASSED
    assert ob.result is not None
    assert ob.result.p_pass == 1.0


def test_failing_step_execution_claim_is_refuted_after_checking():
    """Step 5: the failing step-1 execution-status claim ends up REFUTED, never SUPPORTED."""
    htir = _build_compiled_synthetic_htir()
    check_obligations(htir, DEFAULT_DOMAIN_SPEC)

    step1_claim = next(
        c for c in htir.claims if c.claim_type == "execution_status" and c.source_step_id == 1
    )
    assert step1_claim.status == ClaimStatus.REFUTED


def test_wellformedness_seeded_obligation_abstains():
    """Step 5: a well-formedness-seeded (ABSTENTION-routed) obligation abstains with p_abstain==1.0."""
    htir = _build_compiled_synthetic_htir()
    check_obligations(htir, DEFAULT_DOMAIN_SPEC)

    seeded = [o for o in htir.obligations if o.checker == CheckerType.ABSTENTION]
    assert seeded, "expected at least one ABSTENTION-routed obligation in the synthetic trace"
    for ob in seeded:
        assert ob.status == ObligationStatus.ABSTAINED
        assert ob.result.p_abstain == 1.0


def test_semantic_obligations_abstain_without_llm_call(monkeypatch):
    """
    Step 5: with use_semantic=False (the default), every SEMANTIC obligation
    abstains and chat_json is never invoked.
    """
    htir = _build_compiled_synthetic_htir()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("chat_json must not be called when use_semantic=False")

    monkeypatch.setattr("harnessfix.agents.checking.chat_json", _fail_if_called)

    check_obligations(htir, DEFAULT_DOMAIN_SPEC, use_semantic=False)

    semantic_obligations = [o for o in htir.obligations if o.checker == CheckerType.SEMANTIC]
    assert semantic_obligations
    for ob in semantic_obligations:
        assert ob.status == ObligationStatus.ABSTAINED
        assert ob.result.p_abstain == 1.0


def test_check_obligations_is_idempotent():
    """Step 5: calling check_obligations twice yields identical results/statuses."""
    htir = _build_compiled_synthetic_htir()
    check_obligations(htir, DEFAULT_DOMAIN_SPEC)

    first = [(o.status, o.result.model_dump()) for o in htir.obligations]
    first_claim_statuses = [c.status for c in htir.claims]

    check_obligations(htir, DEFAULT_DOMAIN_SPEC)

    second = [(o.status, o.result.model_dump()) for o in htir.obligations]
    second_claim_statuses = [c.status for c in htir.claims]

    assert first == second
    assert first_claim_statuses == second_claim_statuses


# ---------------------------------------------------------------------------
# Step 6 -- aggregation z_tau + verification witness W_tau (harnessfix.agents.witness)
# ---------------------------------------------------------------------------

def _obligation(
    obligation_id: int, status: ObligationStatus, severity: Severity,
    *, p_pass: float = 0.0, p_fail: float = 0.0, p_abstain: float = 0.0,
    evidence_used: list[int] | None = None, candidate_evidence_ids: list[int] | None = None,
) -> Obligation:
    return Obligation(
        obligation_id=obligation_id,
        claim_id=obligation_id,
        severity=severity,
        status=status,
        candidate_evidence_ids=candidate_evidence_ids or [],
        result=CheckerResult(p_pass=p_pass, p_fail=p_fail, p_abstain=p_abstain, evidence_used=evidence_used or []),
    )


def test_aggregate_all_pass_trace_is_valid():
    """Step 6: a trace with only passed obligations aggregates to 'valid' with an empty O-."""
    htir = HTIR(task_id="agg-all-pass", obligations=[
        _obligation(1, ObligationStatus.PASSED, Severity.LOW, p_pass=1.0),
        _obligation(2, ObligationStatus.PASSED, Severity.HIGH, p_pass=1.0),
    ])

    z = aggregate(htir)
    assert z.predicted_status == "valid"

    w = build_witness(htir)
    assert w.failed_obligation_ids == []
    assert w.passed_obligation_ids == [1, 2]


def test_aggregate_forced_failed_critical_obligation_is_invalid():
    """Step 6: a failed CRITICAL obligation vetoes to 'invalid' even with many passes."""
    htir = HTIR(task_id="agg-veto", obligations=[
        _obligation(1, ObligationStatus.PASSED, Severity.LOW, p_pass=1.0),
        _obligation(2, ObligationStatus.PASSED, Severity.LOW, p_pass=1.0),
        _obligation(3, ObligationStatus.PASSED, Severity.LOW, p_pass=1.0),
        _obligation(4, ObligationStatus.FAILED, Severity.CRITICAL, p_fail=1.0, evidence_used=[10]),
    ])

    z = aggregate(htir)
    assert z.predicted_status == "invalid"

    w = build_witness(htir)
    assert w.failed_obligation_ids == [4]
    assert 10 in w.witness_evidence_ids


def test_aggregate_abstain_heavy_high_severity_is_uncertain():
    """Step 6: no failures but many abstained high-severity obligations => 'uncertain', not 'valid'."""
    htir = HTIR(task_id="agg-uncertain", obligations=[
        _obligation(1, ObligationStatus.ABSTAINED, Severity.HIGH, p_abstain=1.0),
        _obligation(2, ObligationStatus.ABSTAINED, Severity.HIGH, p_abstain=1.0),
        _obligation(3, ObligationStatus.PASSED, Severity.LOW, p_pass=1.0),
    ])

    z = aggregate(htir)
    assert z.predicted_status == "uncertain"


def test_build_witness_is_deterministic_and_idempotent():
    """Step 6: build_witness makes no LLM call and re-running it yields an identical witness."""
    htir = _build_compiled_synthetic_htir()
    check_obligations(htir, DEFAULT_DOMAIN_SPEC)
    aggregate(htir)

    first = build_witness(htir).model_dump()
    second = build_witness(htir).model_dump()

    assert first == second


def test_compile_run_checks_flag_populates_aggregate_and_witness(monkeypatch):
    """Step 6 wiring: TraceAbstractionAgent.compile(run_checks=True) fills HTIR.aggregate/witness."""
    def _fake_chat_json(messages, schema, model=None, max_tokens=None, **kwargs):
        return schema(role="final_submission", execution_status=ExecutionStatus.SUCCESS, artifact_effects=[])

    monkeypatch.setattr("harnessfix.agents.trace_abstraction.chat_json", _fake_chat_json)

    agent = TraceAbstractionAgent(domain_spec=DEFAULT_DOMAIN_SPEC)
    htir = agent.compile(
        task_id="compile-run-checks",
        raw_steps=[{"request": "submit", "response": "final answer"}],
        harness_snippets={},
        run_checks=True,
    )

    assert htir.aggregate is not None
    assert htir.witness is not None
    assert htir.aggregate.predicted_status in ("valid", "invalid", "uncertain")
    assert all(o.status != ObligationStatus.PENDING for o in htir.obligations)
