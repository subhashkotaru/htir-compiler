"""
Harness-aware Trace Intermediate Representation (HTIR)

Originally based on HarnessFix: From Failed Trajectories to Reliable LLM Agents
(arxiv 2606.06324v1), and extended to serve as the trace-abstraction layer of
Adaptive Verifier Graphs (AVG).

HTIR normalises heterogeneous agent execution traces into a common,
step-level graph. In AVG terms it realises the first two pipeline stages —
``raw trace -> typed events -> verification graph`` — over five node kinds
(operations, artifacts, claims, evidence, obligations) and the temporal,
provenance, causal, support, constraint, and validation edge families.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
#
# NOTE: operation *type* is no longer a fixed enum. AVG requires operation
# types to come from the active domain specification (S_d.P_d), so a step's
# ``role`` is a plain string validated against the domain spec's operation
# vocabulary. See harnessfix/models/domain.py.


class ExecutionStatus(str, Enum):
    """The outcome of a TraceStep's execution."""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


class ArtifactEffect(str, Enum):
    """The side-effect category of a TraceStep."""
    NONE = "none"
    READ_ONLY = "read_only"
    ARTIFACT_CHANGE = "artifact_change"
    STATE_CHANGE = "state_change"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ReuseRelation(str, Enum):
    """How a later request reuses content from an earlier step."""
    COPIED = "copied"
    SUMMARIZED = "summarized"
    TRANSFORMED = "transformed"
    SEMANTICALLY_REUSED = "semantically_reused"


class ControlFlowTrigger(str, Enum):
    """The harness logic that caused the next step to execute."""
    CONTINUE = "continue"
    RETRY = "retry"
    DELEGATE = "delegate"
    VALIDATE = "validate"
    FINALIZE = "finalize"
    TERMINATE = "terminate"
    OTHER = "other"


class HarnessLayer(str, Enum):
    """
    The seven ETCLOVG harness layers (HarnessFix diagnostic facet).

    HarnessFix extension, not part of AVG G_tau: this taxonomy and everything
    that attaches to it (``HarnessLayerFacet``, ``HarnessCodeRef``) are
    optional harness-attribution metadata layered on top of the AVG graph,
    not concepts defined by the AVG proposal.
    """
    EXECUTION = "Execution"
    TOOL_INTERFACE = "Tool Interface"
    CONTEXT_MEMORY = "Context/Memory"
    LIFECYCLE = "Lifecycle"
    OBSERVABILITY = "Observability"
    VERIFICATION = "Verification"
    GOVERNANCE = "Governance"


# ---- AVG verification enums -----------------------------------------------

class Severity(str, Enum):
    """Obligation severity (rho_i). Drives severity-aware aggregation."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EvidenceType(str, Enum):
    """Required/available evidence type (r_i)."""
    EXECUTABLE = "executable"      # exit codes, test runs, static analysis
    SCHEMA = "schema"              # schema / structure validation
    ARTIFACT = "artifact"         # produced/consumed object contents
    LOG = "log"                    # observability records
    POLICY = "policy"             # instructions, rules, SOPs
    SEMANTIC = "semantic"         # requires model judgement
    MANUAL = "manual"             # requires a human
    NONE = "none"                  # no evidence available


class CheckerType(str, Enum):
    """The checker class assigned to an obligation (q_i)."""
    MECHANICAL = "mechanical"      # deterministic / executable evidence
    SEMANTIC = "semantic"          # narrow claim-evidence model judge
    ABSTENTION = "abstention"      # insufficient evidence to decide
    UNASSIGNED = "unassigned"      # not yet routed to a checker


class EscalationRule(str, Enum):
    """
    What the harness may do when an obligation fails or abstains (alpha_i),
    part of the obligation tuple o_i = (c_i, r_i, E_i, q_i, rho_i, alpha_i).

    Intentionally shares its vocabulary with ``InterventionAction`` (the
    online intervention iota_t, avg.tex Sec. 3.11), but the two are distinct
    mechanisms: alpha_i is a static per-obligation escalation rule fixed at
    obligation-generation time, while iota_t is chosen online per step by
    monitoring active obligations. Do not conflate them when the online
    intervention loop is built.
    """
    ACCEPT = "accept"
    REQUEST_EVIDENCE = "request-evidence"
    RERANK = "rerank"
    VETO = "veto"
    REPAIR = "repair"
    CLARIFY = "clarify"
    ESCALATE = "escalate"


class InterventionAction(str, Enum):
    """
    Online intervention iota_t (avg.tex Sec. 3.11): the action the harness
    takes at step t in response to a failing/abstaining high-severity
    obligation in the partial graph G_{tau<=t}. Distinct from the per-
    obligation ``EscalationRule`` (alpha_i), which is fixed when the
    obligation is generated rather than chosen online. Not yet wired to any
    online loop; added for forward compatibility with Step-5 work.
    """
    ACCEPT = "accept"
    REQUEST_EVIDENCE = "request-evidence"
    RERANK = "rerank"
    VETO = "veto"
    REPAIR = "repair"
    CLARIFY = "clarify"
    ESCALATE = "escalate"


class ObligationScope(str, Enum):
    """Which template family produced an obligation (O_uni / O_dom / O_trig)."""
    UNIVERSAL = "universal"
    DOMAIN = "domain"
    TRAJECTORY_TRIGGERED = "trajectory_triggered"


class ClaimStatus(str, Enum):
    UNVERIFIED = "unverified"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    UNRESOLVED = "unresolved"


class ObligationStatus(str, Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    ABSTAINED = "abstained"


class SupportPolarity(str, Enum):
    SUPPORTS = "supports"
    REFUTES = "refutes"


class ValidationKind(str, Enum):
    """Granularity of a validation, per AVG well-formedness rules."""
    FULL = "full"
    TARGETED = "targeted"
    MANUAL = "manual"


# ---------------------------------------------------------------------------
# Harness-code reference
# ---------------------------------------------------------------------------

class HarnessCodeRef(BaseModel):
    """
    Points to the harness source/prompt artifact responsible for a behaviour.

    HarnessFix extension, not part of AVG G_tau.
    """
    file_path: str = Field(..., description="Repository-relative path to the file")
    start_line: Optional[int] = Field(None, description="Start line (1-indexed)")
    end_line: Optional[int] = Field(None, description="End line (inclusive)")
    description: str = Field("", description="Human-readable note about what this reference does")


# ---------------------------------------------------------------------------
# Edge types
# ---------------------------------------------------------------------------

class TemporalLink(BaseModel):
    """E_temp: preserves the original execution order between two steps."""
    source_id: int
    target_id: int


class InputReuseLink(BaseModel):
    """
    HarnessFix input-reuse relation (NOT AVG provenance): shows how the target
    step's request was assembled from an earlier step's content (or from
    harness logic that constructs it). Kept as a HarnessFix diagnostic
    extension; AVG's artifact-centric provenance edge is
    ``ArtifactProvenanceLink`` (E_prov) below.
    """
    source_id: int = Field(..., description="ID of the earlier TraceStep")
    target_id: int = Field(..., description="ID of the later TraceStep")
    source_span: str = Field("", description="Slice of the earlier request/response used")
    target_span: str = Field("", description="Slice of the current request derived from source")
    reuse_relation: ReuseRelation = ReuseRelation.SEMANTICALLY_REUSED
    harness_code_refs: list[HarnessCodeRef] = Field(
        default_factory=list,
        description="Harness artifacts that constructed this request content",
    )


class ProvenanceRelation(str, Enum):
    """How an operation relates to an artifact for E_prov purposes."""
    CREATED = "created"
    READ = "read"
    MODIFIED = "modified"


class ArtifactProvenanceLink(BaseModel):
    """
    E_prov: links an artifact to the operation that created, read, or modified
    it (avg.tex Sec. 3, Provenance analysis). This is the AVG provenance edge;
    it is artifact-centric, unlike the HarnessFix step->step ``InputReuseLink``.
    """
    step_id: int = Field(..., description="Operation (TraceStep) id")
    artifact_id: int = Field(..., description="Artifact node id")
    relation: ProvenanceRelation


class ControlFlowLink(BaseModel):
    """
    HarnessFix control-flow transition (harness extension, not AVG E_causal):
    shows why the harness executed a particular step (the controller
    transition that produced it, e.g. retry/delegate/finalize). AVG's actual
    causal/dependency edge is ``DependencyLink`` (E_causal) below.
    """
    source_id: int
    target_id: int
    triggering_logic: ControlFlowTrigger = ControlFlowTrigger.CONTINUE
    triggering_condition: str = Field("", description="Condition evaluated by the harness controller")
    execution_status: Optional[ExecutionStatus] = None
    harness_code_refs: list[HarnessCodeRef] = Field(default_factory=list)


class DependencyLink(BaseModel):
    """
    E_causal: an operation depends on an earlier artifact or step (avg.tex
    Sec. 3, Dependency analysis), e.g. an edit that depends on a failing test.
    Distinct from the HarnessFix ``ControlFlowLink`` controller transition.
    """
    source_step_id: int = Field(..., description="The dependent operation")
    target_step_id: Optional[int] = Field(None, description="Step this operation depends on, if any")
    target_artifact_id: Optional[int] = Field(None, description="Artifact this operation depends on, if any")
    reason: str = Field("", description="Why the dependency holds, e.g. 'edit follows failing test'")


class SupportLink(BaseModel):
    """E_sup: an evidence node supports or refutes a claim node."""
    evidence_id: int
    claim_id: int
    polarity: SupportPolarity = SupportPolarity.SUPPORTS
    weight: float = Field(1.0, description="Strength of support/refutation in [0, 1]")
    rationale: str = ""


class ConstraintLink(BaseModel):
    """E_cons: links a domain constraint (S_d.K_d) to the step/artifact it governs."""
    constraint_id: str = Field(..., description="Constraint id from the domain spec")
    step_id: Optional[int] = None
    artifact_id: Optional[int] = None
    satisfied: Optional[bool] = Field(None, description="Known outcome, or None if unresolved")
    note: str = ""


class ValidationLink(BaseModel):
    """E_val: connects a validation operation to the step/artifact it validates."""
    source_id: int = Field(..., description="TraceStep id of the validation operation")
    target_step_id: Optional[int] = None
    target_artifact_id: Optional[int] = None
    validation_kind: ValidationKind = ValidationKind.TARGETED
    outcome: Optional[ExecutionStatus] = None
    note: str = ""


# ---------------------------------------------------------------------------
# Artifact / state effect annotation
# ---------------------------------------------------------------------------

class ArtifactStateEvidence(BaseModel):
    """Records externally observable consequences of a TraceStep."""
    effect_category: ArtifactEffect = ArtifactEffect.UNKNOWN
    affected_resource: str = Field("", description="Identifier of the resource or object affected")
    observed_change: str = Field("", description="Description of the change, or its absence")
    supporting_evidence: str = Field("", description="Tool return values, state-diff snapshots, etc.")


# ---------------------------------------------------------------------------
# Harness layer responsibility facet
# ---------------------------------------------------------------------------

class HarnessLayerFacet(BaseModel):
    """
    Maps a TraceStep to the ETCLOVG layers implicated by its local evidence.

    HarnessFix extension, not part of AVG G_tau.
    """
    implicated_layers: list[HarnessLayer] = Field(default_factory=list)
    rationale: str = Field("", description="Why these layers are implicated")


# ---------------------------------------------------------------------------
# Node-local diagnostic evidence
# ---------------------------------------------------------------------------

class NodeLocalEvidence(BaseModel):
    """
    Organises the evidence attached to a single TraceStep for failure
    attribution.  Three evidence types mirror the three link categories.
    """
    input_reuse_evidence: list[InputReuseLink] = Field(default_factory=list)
    control_flow_evidence: list[ControlFlowLink] = Field(default_factory=list)
    artifact_state_evidence: list[ArtifactStateEvidence] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

class TraceStep(BaseModel):
    """
    Operation node: a single recoverable execution step in an agent trajectory
    (a model call, a tool-mediated action, a validation step, or a final
    submission decision).
    """
    step_id: int = Field(..., description="Sequential position in execution order (1-indexed)")
    request_message: str = Field(..., description="Full message sent to the model, tool, or environment")
    response_message: str = Field(..., description="Full message returned by the model, tool, or environment")

    # Operation type drawn from the active domain spec's vocabulary (S_d.P_d).
    role: str = Field("other", description="Domain operation type name; 'other' if unmatched")
    execution_status: ExecutionStatus = ExecutionStatus.UNKNOWN
    artifact_state_effects: list[ArtifactStateEvidence] = Field(default_factory=list)

    # First-class artifact-node references (provenance edges to ArtifactNode ids)
    consumed_artifact_ids: list[int] = Field(default_factory=list)
    produced_artifact_ids: list[int] = Field(default_factory=list)

    # Node-local diagnostic evidence (assembled after link creation)
    node_local_evidence: Optional[NodeLocalEvidence] = None

    # Harness layer facet (filled after node-local evidence is assembled)
    harness_layer_facet: Optional[HarnessLayerFacet] = None

    # Raw metadata preserved from the original trace
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactNode(BaseModel):
    """
    Artifact node: a produced or consumed object (file, patch, table, command
    output, log, test report, policy, database state, ...). First-class so the
    same object can be tracked across the operations that read and mutate it.
    """
    artifact_id: int = Field(..., description="Unique artifact identifier within the graph")
    artifact_type: str = Field("state", description="Artifact type name from S_d.R_d")
    identifier: str = Field(..., description="Human/tool identifier, e.g. a file path or table name")
    description: str = ""
    content_summary: str = ""
    produced_by_step_id: Optional[int] = Field(None, description="Step that first produced this artifact")
    version: int = Field(0, description="Bumped each time the artifact is mutated")
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


class ClaimNode(BaseModel):
    """
    Claim node: a checkable statement induced by the trajectory (e.g. 'the
    command exited with code 0' or 'the final report follows the policy').
    Claims are untrusted until discharged by an obligation.
    """
    claim_id: int
    statement: str
    claim_type: str = Field("", description="e.g. execution_status, artifact_provenance, final_answer_support")
    source_step_id: Optional[int] = None
    artifact_ids: list[int] = Field(default_factory=list)
    status: ClaimStatus = ClaimStatus.UNVERIFIED


class EvidenceNode(BaseModel):
    """
    Evidence node: points to artifacts or graph neighbourhoods that may support
    or refute claims.
    """
    evidence_id: int
    evidence_type: EvidenceType = EvidenceType.NONE
    description: str = ""
    content: str = ""
    artifact_ids: list[int] = Field(default_factory=list)
    step_ids: list[int] = Field(default_factory=list)


class CheckerResult(BaseModel):
    """
    Output of a checker q_i(o_i, G_tau) = (p+, p-, p_abstain, s, eta).
    Left unset until the (future) checking stage runs.
    """
    p_pass: float = 0.0
    p_fail: float = 0.0
    p_abstain: float = 1.0
    score: float = 0.0
    evidence_used: list[int] = Field(default_factory=list, description="Evidence node ids used (eta_i)")


class Obligation(BaseModel):
    """
    Verification obligation o_i = (c_i, r_i, E_i, q_i, rho_i, alpha_i): the
    central unit of AVG verification. Localises verification to a specific
    claim discharged by specific evidence.
    """
    obligation_id: int
    claim_id: int = Field(..., description="Claim to verify (c_i)")
    required_evidence: EvidenceType = Field(EvidenceType.NONE, description="Required evidence type (r_i)")
    candidate_evidence_ids: list[int] = Field(default_factory=list, description="Candidate evidence nodes (E_i)")
    checker: CheckerType = Field(CheckerType.UNASSIGNED, description="Assigned checker class (q_i)")
    severity: Severity = Field(Severity.MEDIUM, description="Severity (rho_i)")
    escalation: EscalationRule = Field(EscalationRule.REQUEST_EVIDENCE, description="Escalation rule (alpha_i)")
    scope: ObligationScope = ObligationScope.UNIVERSAL
    template_id: Optional[str] = None
    description: str = ""
    result: Optional[CheckerResult] = None
    status: ObligationStatus = ObligationStatus.PENDING


# ---------------------------------------------------------------------------
# Analysis-module outputs (avg.tex Sec. 3.4-3.5)
# ---------------------------------------------------------------------------

class WellFormednessIssue(BaseModel):
    """
    A domain-independent structural well-formedness failure (avg.tex Sec.
    3.4, "Well-Formedness Checks"). A well-formedness failure does not mean
    the task failed -- it means the trace is missing evidence needed for
    confident verification, so it seeds an unresolved obligation instead of
    an assigned pass/fail.
    """
    rule_id: str = Field(..., description="Which well-formedness/analysis rule failed")
    severity: Severity = Severity.MEDIUM
    offending_node_ids: list[int] = Field(
        default_factory=list,
        description="step_id/artifact_id values implicated (which kind depends on rule_id)",
    )
    message: str = ""


class StateTransitionPattern(BaseModel):
    """
    A recognised (or attempted-but-broken) expected-pattern instance from
    State-transition analysis (avg.tex Sec. 3.5), e.g. failing-validation ->
    relevant-edit -> post-edit-validation -> passing-validation.
    """
    pattern_name: str
    step_ids: list[int] = Field(default_factory=list, description="Ordered step ids forming the (partial) pattern")
    matched: bool = Field(False, description="Whether the full expected pattern was observed")
    note: str = ""


class CoverageReport(BaseModel):
    """
    Evidence coverage by obligation type x evidence type (avg.tex Sec. 3.5,
    Coverage analysis). Populated after obligation generation, since it
    reports over ``HTIR.obligations``.
    """
    by_obligation_type: dict[str, int] = Field(default_factory=dict)
    by_evidence_type: dict[str, int] = Field(default_factory=dict)
    covered_obligations: int = 0
    total_obligations: int = 0


# ---------------------------------------------------------------------------
# Aggregation + verification witness (avg.tex Sec. 3.9-3.10, AVG Step 6)
# ---------------------------------------------------------------------------

class AggregateResult(BaseModel):
    """
    z_tau = (y_hat, u_hat, c_hat, eta_hat): the trajectory-level status
    aggregated from all checked obligations (avg.tex Sec. 3.9). Populated by
    ``harnessfix.agents.witness.aggregate``, which must run after
    ``harnessfix.agents.checking.check_obligations``.
    """
    predicted_status: str = Field("uncertain", description="y_hat: 'valid' / 'invalid' / 'uncertain'")
    uncertainty: float = Field(0.0, description="u_hat: severity-weighted abstention mass in [0, 1]")
    evidence_coverage: float = Field(0.0, description="c_hat: rolled up from CoverageReport")
    aggregated_evidence_ids: list[int] = Field(
        default_factory=list, description="eta_hat: union of evidence used across all checked obligations"
    )


class VerificationWitness(BaseModel):
    """
    W_tau = (O+, O-, O-empty, E_W, R_W): the stated output of AVG (avg.tex
    Sec. 3.10). Populated by ``harnessfix.agents.witness.build_witness``,
    which must run after ``aggregate``.
    """
    passed_obligation_ids: list[int] = Field(default_factory=list, description="O+")
    failed_obligation_ids: list[int] = Field(default_factory=list, description="O-")
    abstained_obligation_ids: list[int] = Field(default_factory=list, description="O-empty")
    witness_evidence_ids: list[int] = Field(default_factory=list, description="E_W")
    review_recommendation: str = Field("", description="R_W: a short deterministic review summary")


# ---------------------------------------------------------------------------
# Online intervention log (avg.tex Sec. 3.11, AVG Step 7)
# ---------------------------------------------------------------------------

class InterventionLogEntry(BaseModel):
    """
    One recorded intervention decision: at step ``step_id``, over the partial
    graph G_{tau<=step_id}, ``obligation_id`` was active (high-severity,
    failing/abstaining) and the harness chose ``action`` (iota_t). Purely a
    recommendation trace (``harnessfix.agents.intervention``); does not drive
    an agent.
    """
    step_id: int
    obligation_id: int
    action: InterventionAction
    rationale: str = ""


# ---------------------------------------------------------------------------
# HTIR graph
# ---------------------------------------------------------------------------

class HTIR(BaseModel):
    """
    The complete Harness-aware Trace Intermediate Representation / AVG
    verification graph for one agent execution run.
    """
    task_id: str = Field(..., description="Identifier for the task this trace belongs to")
    agent_name: str = Field("", description="Name of the agent harness that produced this trace")
    outcome: str = Field("", description="Externally evaluated result (e.g. 'failed', 'resolved')")
    domain_id: str = Field("default", description="Domain specification (S_d) used to compile this graph")

    # Nodes
    steps: list[TraceStep] = Field(default_factory=list)
    artifacts: list[ArtifactNode] = Field(default_factory=list)
    claims: list[ClaimNode] = Field(default_factory=list)
    evidence: list[EvidenceNode] = Field(default_factory=list)
    obligations: list[Obligation] = Field(default_factory=list)

    # Edges
    temporal_links: list[TemporalLink] = Field(default_factory=list)
    input_reuse_links: list[InputReuseLink] = Field(default_factory=list)
    control_flow_links: list[ControlFlowLink] = Field(default_factory=list)
    support_links: list[SupportLink] = Field(default_factory=list)
    constraint_links: list[ConstraintLink] = Field(default_factory=list)
    validation_links: list[ValidationLink] = Field(default_factory=list)
    provenance_links: list[ArtifactProvenanceLink] = Field(default_factory=list)
    dependency_links: list[DependencyLink] = Field(default_factory=list)

    # Analysis-module outputs (avg.tex Sec. 3.4-3.5). Populated by
    # harnessfix.agents.analysis.enrich() (well-formedness / most modules) and
    # harnessfix.agents.analysis.compute_coverage() (coverage, after
    # obligation generation). default_factory so existing serialized outputs
    # (data/htir_outputs/*.json) stay backward compatible.
    wellformedness: list[WellFormednessIssue] = Field(default_factory=list)
    state_transitions: list[StateTransitionPattern] = Field(default_factory=list)
    coverage: CoverageReport = Field(default_factory=CoverageReport)

    # AVG Step 6 outputs (avg.tex Sec. 3.9-3.10). Populated by
    # harnessfix.agents.witness.aggregate() / build_witness(), which run
    # after harnessfix.agents.checking.check_obligations(). Optional and
    # default None so existing serialized outputs (data/htir_outputs/*.json)
    # stay backward compatible.
    aggregate: Optional[AggregateResult] = None
    witness: Optional[VerificationWitness] = None

    # AVG Step 7 output (avg.tex Sec. 3.11). Populated by
    # harnessfix.agents.intervention.run_intervention_loop() over a *prefix*
    # replay of a recorded trace; empty by default so existing serialized
    # outputs (data/htir_outputs/*.json) stay backward compatible.
    intervention_log: list[InterventionLogEntry] = Field(default_factory=list)

    # Path to the harness code under analysis
    harness_root: str = Field("", description="Root directory of the harness codebase")

    # -- accessors ----------------------------------------------------------

    def get_step(self, step_id: int) -> Optional[TraceStep]:
        for s in self.steps:
            if s.step_id == step_id:
                return s
        return None

    def get_artifact(self, artifact_id: int) -> Optional[ArtifactNode]:
        for a in self.artifacts:
            if a.artifact_id == artifact_id:
                return a
        return None

    def get_claim(self, claim_id: int) -> Optional[ClaimNode]:
        for c in self.claims:
            if c.claim_id == claim_id:
                return c
        return None

    def get_obligation(self, obligation_id: int) -> Optional[Obligation]:
        for o in self.obligations:
            if o.obligation_id == obligation_id:
                return o
        return None

    def steps_in_order(self) -> list[TraceStep]:
        return sorted(self.steps, key=lambda s: s.step_id)
