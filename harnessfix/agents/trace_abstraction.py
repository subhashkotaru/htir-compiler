"""
Trace Abstraction Agent  (Section III-A of the paper)

Compiles raw agent execution traces and harness code into HTIR by:
  1. Normalising each step into a TraceStep with role, execution status,
     and artifact/state effect annotations.
  2. Inferring input provenance links between steps.
  3. Inferring control flow links between steps.
  4. Assembling node-local diagnostic evidence per step.
  5. Attaching the harness layer responsibility facet per step.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from harnessfix.models.htir import (
    HTIR,
    ArtifactEffect,
    ArtifactStateEvidence,
    ControlFlowLink,
    ControlFlowTrigger,
    ExecutionStatus,
    HarnessCodeRef,
    HarnessLayer,
    HarnessLayerFacet,
    InputProvenanceLink,
    NodeLocalEvidence,
    ReuseRelation,
    StepRole,
    TemporalLink,
    TraceStep,
)
from harnessfix.utils.llm import chat_json, system, user, DEFAULT_MODEL
from harnessfix.utils.io import truncate


# ---------------------------------------------------------------------------
# LLM response schemas
# ---------------------------------------------------------------------------

class _StepAnnotation(BaseModel):
    role: StepRole = StepRole.OTHER
    execution_status: ExecutionStatus = ExecutionStatus.UNKNOWN
    artifact_effects: list[dict] = Field(default_factory=list)


class _ProvenanceLink(BaseModel):
    source_id: int
    target_id: int
    source_span: str = ""
    target_span: str = ""
    reuse_relation: str = "semantically_reused"
    harness_code_refs: list[dict] = Field(default_factory=list)


class _ProvenanceLinkList(BaseModel):
    links: list[_ProvenanceLink] = Field(default_factory=list)


class _ControlFlowLink(BaseModel):
    source_id: int
    target_id: int
    triggering_logic: str = "continue"
    triggering_condition: str = ""
    execution_status: str = "unknown"
    harness_code_refs: list[dict] = Field(default_factory=list)


class _ControlFlowLinkList(BaseModel):
    links: list[_ControlFlowLink] = Field(default_factory=list)


class _LayerFacet(BaseModel):
    implicated_layers: list[str] = Field(default_factory=list)
    rationale: str = ""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

LINK_BATCH_SIZE = 20   # max steps per provenance / CF inference call

LAYER_DESCRIPTIONS = {
    HarnessLayer.EXECUTION: "Provides safe, isolated, reproducible environments",
    HarnessLayer.TOOL_INTERFACE: "Governs tool discovery, description, invocation, and error feedback",
    HarnessLayer.CONTEXT_MEMORY: "Determines what the model sees: context window, session state, memory",
    HarnessLayer.LIFECYCLE: "Controls execution flow: loops, retries, multi-agent coordination, termination",
    HarnessLayer.OBSERVABILITY: "Records traces, logs, tool calls, errors, cost information",
    HarnessLayer.VERIFICATION: "Connects tasks to feedback through validation, evaluation, regression testing",
    HarnessLayer.GOVERNANCE: "Defines permissions, policies, approvals, audit trails",
}


class TraceAbstractionAgent:
    """
    Converts raw heterogeneous agent traces into HTIR.

    Usage::

        agent = TraceAbstractionAgent(model="openai/gpt-4o")
        htir = agent.compile(
            task_id="task_001",
            raw_steps=[{"request": "...", "response": "..."}, ...],
            harness_snippets={"agent.py": "..."},
            outcome="failed",
        )
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compile(
        self,
        task_id: str,
        raw_steps: list[dict[str, Any]],
        harness_snippets: dict[str, str],
        outcome: str = "failed",
        agent_name: str = "",
        harness_root: str = "",
    ) -> HTIR:
        """Compile a raw trace into HTIR."""
        steps = self._build_steps(raw_steps)
        htir = HTIR(
            task_id=task_id,
            agent_name=agent_name,
            outcome=outcome,
            steps=steps,
            harness_root=harness_root,
        )

        # Temporal links (trivially sequential)
        for i in range(len(steps) - 1):
            htir.temporal_links.append(
                TemporalLink(source_id=steps[i].step_id, target_id=steps[i + 1].step_id)
            )

        harness_context = self._summarise_harness(harness_snippets)

        # Input provenance links
        prov_links = self._infer_provenance(steps, harness_context)
        htir.input_provenance_links = prov_links

        # Control flow links
        cf_links = self._infer_control_flow(steps, harness_context)
        htir.control_flow_links = cf_links

        # Node-local evidence + layer facets
        self._attach_node_local_evidence(htir)
        self._attach_layer_facets(htir, harness_context)

        return htir

    # ------------------------------------------------------------------
    # Step construction
    # ------------------------------------------------------------------

    def _build_steps(self, raw_steps: list[dict[str, Any]]) -> list[TraceStep]:
        steps: list[TraceStep] = []
        for i, raw in enumerate(raw_steps, start=1):
            req = str(raw.get("request", raw.get("prompt", raw.get("input", ""))))
            resp = str(raw.get("response", raw.get("output", raw.get("result", ""))))

            annotation = self._annotate_step(i, req, resp)

            effects = [
                ArtifactStateEvidence(
                    effect_category=ArtifactEffect(e.get("effect_category") or "unknown"),
                    affected_resource=str(e.get("affected_resource") or ""),
                    observed_change=str(e.get("observed_change") or ""),
                    supporting_evidence=str(e.get("supporting_evidence") or ""),
                )
                for e in annotation.artifact_effects
            ]

            steps.append(
                TraceStep(
                    step_id=i,
                    request_message=req,
                    response_message=resp,
                    role=annotation.role,
                    execution_status=annotation.execution_status,
                    artifact_state_effects=effects,
                    raw_metadata={k: v for k, v in raw.items() if k not in ("request", "response", "prompt", "output")},
                )
            )
        return steps

    def _annotate_step(self, step_id: int, req: str, resp: str) -> _StepAnnotation:
        msgs = [
            system(
                "You are an expert agent-trace analyst. "
                "Given a single agent step (request + response), infer its role, "
                "execution status, and artifact/state effects. "
                "Use ONLY the provided enum values."
            ),
            user(
                f"Step {step_id}:\n"
                f"REQUEST:\n{truncate(req, 1500)}\n\n"
                f"RESPONSE:\n{truncate(resp, 1500)}\n\n"
                "Valid roles: information_acquisition, tool_invocation, artifact_editing, "
                "validation, orchestration_decision, final_submission, other\n"
                "Valid execution_status: success, failure, timeout, blocked, unknown\n"
                "artifact_effects is a list of objects with keys: "
                "effect_category (none/read_only/artifact_change/state_change/mixed/unknown), "
                "affected_resource, observed_change, supporting_evidence."
            ),
        ]
        return chat_json(msgs, _StepAnnotation, model=self.model)

    # ------------------------------------------------------------------
    # Input provenance links
    # ------------------------------------------------------------------

    def _infer_provenance(
        self, steps: list[TraceStep], harness_context: str
    ) -> list[InputProvenanceLink]:
        if len(steps) < 2:
            return []

        all_links: list[InputProvenanceLink] = []
        # Process in overlapping windows so each batch has context from prev steps
        for start in range(0, len(steps), LINK_BATCH_SIZE):
            batch = steps[max(0, start - 3): start + LINK_BATCH_SIZE]  # 3-step lookback overlap
            all_links.extend(self._infer_provenance_batch(batch, harness_context))
        return all_links

    def _infer_provenance_batch(
        self, steps: list[TraceStep], harness_context: str
    ) -> list[InputProvenanceLink]:
        if len(steps) < 2:
            return []

        steps_summary = "\n".join(
            f"S{s.step_id}: req={truncate(s.request_message, 200)} | "
            f"resp={truncate(s.response_message, 200)}"
            for s in steps
        )

        msgs = [
            system(
                "You are an expert agent-trace analyst specialising in input provenance. "
                "For each step, identify which earlier steps' content was reused in its request "
                "(explicitly copied, summarised, transformed, or semantically reused)."
            ),
            user(
                f"Steps summary:\n{steps_summary}\n\n"
                f"Harness context:\n{truncate(harness_context, 800)}\n\n"
                "Produce a list of provenance links. "
                "Each link: source_id, target_id, source_span, target_span, reuse_relation "
                "(copied/summarized/transformed/semantically_reused), harness_code_refs (list of "
                "{file_path, start_line, end_line, description})."
            ),
        ]
        result = chat_json(msgs, _ProvenanceLinkList, model=self.model, max_tokens=2048)

        links: list[InputProvenanceLink] = []
        for lk in result.links:
            try:
                rr = ReuseRelation(lk.reuse_relation)
            except ValueError:
                rr = ReuseRelation.SEMANTICALLY_REUSED
            code_refs = [HarnessCodeRef(**r) for r in lk.harness_code_refs if isinstance(r, dict) and "file_path" in r]
            links.append(
                InputProvenanceLink(
                    source_id=lk.source_id,
                    target_id=lk.target_id,
                    source_span=lk.source_span,
                    target_span=lk.target_span,
                    reuse_relation=rr,
                    harness_code_refs=code_refs,
                )
            )
        return links

    # ------------------------------------------------------------------
    # Control flow links
    # ------------------------------------------------------------------

    def _infer_control_flow(
        self, steps: list[TraceStep], harness_context: str
    ) -> list[ControlFlowLink]:
        if len(steps) < 2:
            return []

        all_links: list[ControlFlowLink] = []
        for start in range(0, len(steps), LINK_BATCH_SIZE):
            batch = steps[max(0, start - 2): start + LINK_BATCH_SIZE]
            all_links.extend(self._infer_control_flow_batch(batch, harness_context))
        return all_links

    def _infer_control_flow_batch(
        self, steps: list[TraceStep], harness_context: str
    ) -> list[ControlFlowLink]:
        if len(steps) < 2:
            return []

        steps_summary = "\n".join(
            f"S{s.step_id}: role={s.role.value} status={s.execution_status.value} "
            f"req={truncate(s.request_message, 150)}"
            for s in steps
        )

        msgs = [
            system(
                "You are an expert agent-trace analyst specialising in control flow. "
                "For each consecutive pair of steps, identify the harness controller logic "
                "that caused the transition."
            ),
            user(
                f"Steps:\n{steps_summary}\n\n"
                f"Harness context:\n{truncate(harness_context, 600)}\n\n"
                "Produce a list of control-flow links. "
                "Each: source_id, target_id, triggering_logic "
                "(continue/retry/delegate/validate/finalize/terminate/other), "
                "triggering_condition, execution_status, harness_code_refs "
                "(list of {file_path, start_line, end_line, description})."
            ),
        ]
        result = chat_json(msgs, _ControlFlowLinkList, model=self.model, max_tokens=2048)

        links: list[ControlFlowLink] = []
        for lk in result.links:
            try:
                tl = ControlFlowTrigger(lk.triggering_logic)
            except ValueError:
                tl = ControlFlowTrigger.OTHER
            try:
                es = ExecutionStatus(lk.execution_status)
            except ValueError:
                es = ExecutionStatus.UNKNOWN
            code_refs = [HarnessCodeRef(**r) for r in lk.harness_code_refs if isinstance(r, dict) and "file_path" in r]
            links.append(
                ControlFlowLink(
                    source_id=lk.source_id,
                    target_id=lk.target_id,
                    triggering_logic=tl,
                    triggering_condition=lk.triggering_condition,
                    execution_status=es,
                    harness_code_refs=code_refs,
                )
            )
        return links

    # ------------------------------------------------------------------
    # Node-local evidence
    # ------------------------------------------------------------------

    def _attach_node_local_evidence(self, htir: HTIR) -> None:
        prov_by_target: dict[int, list[InputProvenanceLink]] = {}
        for lk in htir.input_provenance_links:
            prov_by_target.setdefault(lk.target_id, []).append(lk)

        cf_by_target: dict[int, list[ControlFlowLink]] = {}
        for lk in htir.control_flow_links:
            cf_by_target.setdefault(lk.target_id, []).append(lk)

        for step in htir.steps:
            step.node_local_evidence = NodeLocalEvidence(
                input_provenance_evidence=prov_by_target.get(step.step_id, []),
                control_flow_evidence=cf_by_target.get(step.step_id, []),
                artifact_state_evidence=step.artifact_state_effects,
            )

    # ------------------------------------------------------------------
    # Harness layer facets
    # ------------------------------------------------------------------

    def _attach_layer_facets(self, htir: HTIR, harness_context: str) -> None:
        layer_names = [l.value for l in HarnessLayer]
        layer_descriptions = "\n".join(
            f"- {l.value}: {desc}" for l, desc in LAYER_DESCRIPTIONS.items()
        )

        for step in htir.steps:
            if step.node_local_evidence is None:
                continue
            evidence_text = json.dumps(
                {
                    "provenance": [lk.model_dump() for lk in step.node_local_evidence.input_provenance_evidence],
                    "control_flow": [lk.model_dump() for lk in step.node_local_evidence.control_flow_evidence],
                    "artifact_state": [e.model_dump() for e in step.node_local_evidence.artifact_state_evidence],
                },
                default=str,
            )

            msgs = [
                system(
                    "You are an expert in agent harness engineering. "
                    "Given node-local evidence for a trace step, identify which ETCLOVG "
                    "harness layers are implicated.\n\n"
                    f"Harness layers and their responsibilities:\n{layer_descriptions}"
                ),
                user(
                    f"Step {step.step_id} evidence:\n{truncate(evidence_text, 2000)}\n\n"
                    f"Valid layers: {layer_names}\n"
                    "Produce: implicated_layers (list of layer names), rationale (string)."
                ),
            ]
            result = chat_json(msgs, _LayerFacet, model=self.model)

            valid_layers: list[HarnessLayer] = []
            for lname in result.implicated_layers:
                try:
                    valid_layers.append(HarnessLayer(lname))
                except ValueError:
                    pass

            step.harness_layer_facet = HarnessLayerFacet(
                implicated_layers=valid_layers,
                rationale=result.rationale,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _summarise_harness(snippets: dict[str, str]) -> str:
        if not snippets:
            return "(no harness code provided)"
        parts = []
        for path, content in list(snippets.items())[:10]:  # cap at 10 files
            parts.append(f"=== {path} ===\n{truncate(content, 500)}")
        return "\n\n".join(parts)
