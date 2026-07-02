"""
Harness-aware Trace Intermediate Representation (HTIR)

Based on HarnessFix: From Failed Trajectories to Reliable LLM Agents
(arxiv 2606.06324v1)

HTIR normalises heterogeneous agent execution traces into a common,
step-level representation that supports evidence tracing, failure
attribution, and harness-layer diagnosis.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class StepRole(str, Enum):
    """The function a TraceStep plays in the execution."""
    INFORMATION_ACQUISITION = "information_acquisition"
    TOOL_INVOCATION = "tool_invocation"
    ARTIFACT_EDITING = "artifact_editing"
    VALIDATION = "validation"
    ORCHESTRATION_DECISION = "orchestration_decision"
    FINAL_SUBMISSION = "final_submission"
    OTHER = "other"


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
    """The seven ETCLOVG harness layers."""
    EXECUTION = "Execution"
    TOOL_INTERFACE = "Tool Interface"
    CONTEXT_MEMORY = "Context/Memory"
    LIFECYCLE = "Lifecycle"
    OBSERVABILITY = "Observability"
    VERIFICATION = "Verification"
    GOVERNANCE = "Governance"


# ---------------------------------------------------------------------------
# Harness-code reference
# ---------------------------------------------------------------------------

class HarnessCodeRef(BaseModel):
    """Points to the harness source/prompt artifact responsible for a behaviour."""
    file_path: str = Field(..., description="Repository-relative path to the file")
    start_line: Optional[int] = Field(None, description="Start line (1-indexed)")
    end_line: Optional[int] = Field(None, description="End line (inclusive)")
    description: str = Field("", description="Human-readable note about what this reference does")


# ---------------------------------------------------------------------------
# Link types
# ---------------------------------------------------------------------------

class TemporalLink(BaseModel):
    """Preserves the original execution order between two steps."""
    source_id: int
    target_id: int


class InputProvenanceLink(BaseModel):
    """
    Shows how the target step's request was assembled from an earlier step's
    content (or from harness logic that constructs the request).
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


class ControlFlowLink(BaseModel):
    """
    Shows why the harness executed a particular step (i.e. the controller
    transition that produced it).
    """
    source_id: int
    target_id: int
    triggering_logic: ControlFlowTrigger = ControlFlowTrigger.CONTINUE
    triggering_condition: str = Field("", description="Condition evaluated by the harness controller")
    execution_status: Optional[ExecutionStatus] = None
    harness_code_refs: list[HarnessCodeRef] = Field(default_factory=list)


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
    """Maps a TraceStep to the ETCLOVG layers implicated by its local evidence."""
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
    input_provenance_evidence: list[InputProvenanceLink] = Field(default_factory=list)
    control_flow_evidence: list[ControlFlowLink] = Field(default_factory=list)
    artifact_state_evidence: list[ArtifactStateEvidence] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Core node: TraceStep
# ---------------------------------------------------------------------------

class TraceStep(BaseModel):
    """
    A single recoverable execution step in an agent trajectory:
    a model call, a tool-mediated action, a validation step, or a final
    submission decision.
    """
    step_id: int = Field(..., description="Sequential position in execution order (1-indexed)")
    request_message: str = Field(..., description="Full message sent to the model, tool, or environment")
    response_message: str = Field(..., description="Full message returned by the model, tool, or environment")

    # Derived annotations (filled by the trace abstraction agent)
    role: StepRole = StepRole.OTHER
    execution_status: ExecutionStatus = ExecutionStatus.UNKNOWN
    artifact_state_effects: list[ArtifactStateEvidence] = Field(default_factory=list)

    # Node-local diagnostic evidence (assembled after link creation)
    node_local_evidence: Optional[NodeLocalEvidence] = None

    # Harness layer facet (filled after node-local evidence is assembled)
    harness_layer_facet: Optional[HarnessLayerFacet] = None

    # Raw metadata preserved from the original trace
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# HTIR graph
# ---------------------------------------------------------------------------

class HTIR(BaseModel):
    """
    The complete Harness-aware Trace Intermediate Representation for one
    agent execution run.
    """
    task_id: str = Field(..., description="Identifier for the task this trace belongs to")
    agent_name: str = Field("", description="Name of the agent harness that produced this trace")
    outcome: str = Field("", description="Externally evaluated result (e.g. 'failed', 'resolved')")

    steps: list[TraceStep] = Field(default_factory=list)
    temporal_links: list[TemporalLink] = Field(default_factory=list)
    input_provenance_links: list[InputProvenanceLink] = Field(default_factory=list)
    control_flow_links: list[ControlFlowLink] = Field(default_factory=list)

    # Path to the harness code under analysis
    harness_root: str = Field("", description="Root directory of the harness codebase")

    def get_step(self, step_id: int) -> Optional[TraceStep]:
        for s in self.steps:
            if s.step_id == step_id:
                return s
        return None

    def steps_in_order(self) -> list[TraceStep]:
        return sorted(self.steps, key=lambda s: s.step_id)
