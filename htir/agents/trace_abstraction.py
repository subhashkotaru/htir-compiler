"""
Trace Abstraction Agent  (AVG Steps 1-2: raw trace -> typed events ->
verification graph)

Compiles raw agent execution traces and harness code into HTIR by:
  1. Normalising each step into a TraceStep with role, execution status,
     and artifact/state effect annotations.
  2. Extracting first-class artifact nodes and the AVG artifact-provenance
     edges (E_prov) between artifacts and the operations that created, read,
     or modified them.
  3. Inferring input-reuse links between steps (HarnessFix extension, NOT
     AVG provenance).
  4. Inferring control-flow links between steps (HarnessFix extension, NOT
     AVG E_causal).
  5. Assembling node-local diagnostic evidence per step.
  6. Optionally attaching the ETCLOVG harness layer responsibility facet per
     step (HarnessFix extension, off by default on the AVG path -- see
     ``attach_harness_layers``).
  7. Running the Step-3 analysis layer (``htir.agents.analysis.enrich``):
     well-formedness checks and the provenance / dependency / validation /
     state-transition / policy-linking / integrity analysis modules
     (avg.tex Sec. 3.4-3.5).
  8. Generating claims, evidence, and obligations
     (``htir.agents.obligations.build_claims_and_obligations``), then
     coverage analysis (``htir.agents.analysis.compute_coverage``).
"""

from __future__ import annotations

import json
import warnings
from typing import Any

from pydantic import BaseModel, Field

from htir.models.domain import DEFAULT_DOMAIN_SPEC, DomainArtifactBundle, DomainSpec
from htir.models.htir import (
    HTIR,
    ArtifactEffect,
    ArtifactNode,
    ArtifactProvenanceLink,
    ArtifactStateEvidence,
    ControlFlowLink,
    ControlFlowTrigger,
    ExecutionStatus,
    HarnessCodeRef,
    HarnessLayer,
    HarnessLayerFacet,
    InputReuseLink,
    NodeLocalEvidence,
    ProvenanceRelation,
    ReuseRelation,
    TemporalLink,
    ToolCall,
    TraceStep,
)
from htir.agents.analysis import compute_coverage, enrich
from htir.agents.checking import check_obligations
from htir.agents.obligations import build_claims_and_obligations
from htir.agents.witness import aggregate, build_witness
from htir.utils.llm import chat_json, system, user, DEFAULT_MODEL
from htir.utils.io import truncate


# ---------------------------------------------------------------------------
# LLM response schemas
# ---------------------------------------------------------------------------

class _StepAnnotation(BaseModel):
    role: str = "other"
    execution_status: ExecutionStatus = ExecutionStatus.UNKNOWN
    artifact_effects: list[dict] = Field(default_factory=list)


class _InputReuseLink(BaseModel):
    """LLM schema for a HarnessFix input-reuse link (NOT AVG E_prov)."""
    source_id: int
    target_id: int
    source_span: str = ""
    target_span: str = ""
    reuse_relation: str = "semantically_reused"
    harness_code_refs: list[dict] = Field(default_factory=list)


class _InputReuseLinkList(BaseModel):
    """LLM schema wrapper for a batch of ``_InputReuseLink`` results."""
    links: list[_InputReuseLink] = Field(default_factory=list)


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


def _coerce_artifact_effect(value: Any) -> ArtifactEffect:
    """
    Coerce an LLM-supplied ``effect_category`` string into ``ArtifactEffect``,
    falling back to ``UNKNOWN`` instead of raising when the model returns a
    value outside the enum (mirrors the ``role`` fallback to ``"other"``).
    """
    try:
        return ArtifactEffect(value or "unknown")
    except ValueError:
        return ArtifactEffect.UNKNOWN


def _coerce_execution_status(value: Any) -> ExecutionStatus:
    """Coerce an adapter-supplied status hint into ``ExecutionStatus``."""
    try:
        return ExecutionStatus(value or "unknown")
    except ValueError:
        return ExecutionStatus.UNKNOWN


# Canonical step-dict keys (see ``htir.adapters.base``); everything else on a
# raw step dict is preserved as ``TraceStep.raw_metadata``.
_CANONICAL_STEP_KEYS = frozenset({
    "request", "response", "prompt", "output", "input", "result", "completion",
    "tool_calls", "role_hint", "status_hint", "artifact_effects", "metadata",
})


def _parse_tool_calls(raw_calls: list[dict[str, Any]]) -> list[ToolCall]:
    """Build ``ToolCall`` nodes from canonical tool-call dicts (see adapters)."""
    calls: list[ToolCall] = []
    for tc in raw_calls:
        if not isinstance(tc, dict):
            continue
        calls.append(
            ToolCall(
                name=str(tc.get("name") or "tool"),
                arguments=tc.get("arguments") if isinstance(tc.get("arguments"), dict) else {},
                arguments_text=str(tc.get("arguments_text") or ""),
                result=str(tc.get("result") or ""),
                status=_coerce_execution_status(tc.get("status")),
                tool_call_id=str(tc.get("tool_call_id") or ""),
                raw=tc.get("raw") if isinstance(tc.get("raw"), dict) else {},
            )
        )
    return calls


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

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        domain_spec: DomainSpec = DEFAULT_DOMAIN_SPEC,
        domain_artifacts: DomainArtifactBundle | None = None,
    ):
        self.model = model
        self.domain_spec = domain_spec
        # Omega_d (avg.tex Sec. 2, work item A): optional weak-supervision
        # artifacts consulted by the analysis/obligation/checking layers.
        # None (the default) leaves every downstream pass's behavior
        # identical to before Omega_d was introduced.
        self.domain_artifacts = domain_artifacts

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
        generate_obligations: bool = True,
        attach_harness_layers: bool = False,
        infer_harness_links: bool = False,
        use_semantic_analysis: bool = False,
        run_checks: bool = False,
        domain_artifacts: DomainArtifactBundle | None = None,
    ) -> HTIR:
        """
        Compile a raw trace into HTIR.

        The deterministic AVG pipeline runs with **no LLM / no API key**:
        artifact extraction, analysis modules, obligation generation, and
        (with ``run_checks``) the mechanical checkers, aggregation, and
        witness. The LLM is used only for the opt-in passes below and for
        annotating steps whose adapter did not supply a ``role_hint``.

        ``attach_harness_layers`` controls the HarnessFix ETCLOVG layer facet
        (see ``_attach_layer_facets``), which is a per-step LLM call outside
        the AVG proposal. It defaults to off on the AVG path; pass ``True`` to
        opt into the HarnessFix harness-code-attribution extension.

        ``infer_harness_links`` gates the two HarnessFix step->step link passes
        (input-reuse and control-flow inference), each an LLM call and neither
        part of the AVG graph. Off by default so the core path is offline and
        reproducible; pass ``True`` to populate ``input_reuse_links`` /
        ``control_flow_links`` for harness-attribution diagnostics.

        ``use_semantic_analysis`` gates the LLM-backed passes inside the
        Step-3 analysis layer (``htir.agents.analysis.enrich``), e.g.
        free-text final-answer provenance and soft policy-relevance linking.
        Deterministic analysis passes always run regardless of this flag. The
        same flag also gates the SEMANTIC checker class in
        ``htir.agents.checking.check_obligations`` when ``run_checks``
        is on.

        ``run_checks`` (AVG Steps 5-6, off by default so Step-1..4 output is
        unchanged when it's off) extends the pipeline tail to
        ``check_obligations -> aggregate -> build_witness`` after obligation
        generation, filling ``Obligation.result``/``status``,
        ``ClaimNode.status``, ``HTIR.aggregate``, and ``HTIR.witness``.
        Requires ``generate_obligations=True`` (there is nothing to check
        otherwise). ``domain_artifacts`` is an optional Omega_d bundle
        (avg.tex Sec. 2) threaded through the analysis (policy-linking),
        obligation-generation (schema/policy evidence), and checking layers;
        defaults to ``self.domain_artifacts`` when not given explicitly, and
        absent entirely by default, in which case those checkers abstain
        rather than fake a pass.
        """
        domain_artifacts = domain_artifacts if domain_artifacts is not None else self.domain_artifacts
        steps = self._build_steps(raw_steps)
        htir = HTIR(
            task_id=task_id,
            agent_name=agent_name,
            outcome=outcome,
            domain_id=self.domain_spec.domain_id,
            steps=steps,
            harness_root=harness_root,
        )

        # First-class artifact nodes + AVG artifact-provenance edges (E_prov)
        self._extract_artifacts(htir)

        # Temporal links (trivially sequential)
        for i in range(len(steps) - 1):
            htir.temporal_links.append(
                TemporalLink(source_id=steps[i].step_id, target_id=steps[i + 1].step_id)
            )

        # HarnessFix step->step link inference (input-reuse + control-flow) is
        # an opt-in LLM extension, off by default so the core path is offline.
        harness_context = ""
        if infer_harness_links or attach_harness_layers:
            harness_context = self._summarise_harness(harness_snippets)
        if infer_harness_links:
            # Input-reuse links (HarnessFix extension, NOT AVG provenance)
            htir.input_reuse_links = self._infer_input_reuse(steps, harness_context)
            # Control flow links (HarnessFix extension, NOT AVG E_causal)
            htir.control_flow_links = self._infer_control_flow(steps, harness_context)

        # Node-local evidence + (optional) layer facets
        self._attach_node_local_evidence(htir)
        if attach_harness_layers:
            self._attach_layer_facets(htir, harness_context)

        # Step-3 analysis layer: well-formedness checks + provenance /
        # dependency / validation / state-transition / policy-linking /
        # integrity analysis modules (avg.tex Sec. 3.4-3.5). Runs after graph
        # construction and before obligation generation, which consumes it.
        enrich(htir, self.domain_spec, use_semantic=use_semantic_analysis, domain_artifacts=domain_artifacts)

        # Claims, obligations, and support edges; obligations are also seeded
        # from unresolved well-formedness/analysis-module issues above.
        if generate_obligations:
            build_claims_and_obligations(htir, self.domain_spec, domain_artifacts=domain_artifacts)
            # Coverage analysis (avg.tex Sec. 3.5) depends on obligations, so
            # it runs last rather than inside ``enrich``.
            compute_coverage(htir)

            # AVG Steps 5-6 (avg.tex Sec. 3.8-3.10): check each obligation,
            # aggregate into a trajectory-level status, and build the
            # verification witness. Off by default; requires obligations.
            if run_checks:
                check_obligations(
                    htir, self.domain_spec,
                    use_semantic=use_semantic_analysis,
                    domain_artifacts=domain_artifacts,
                    model=self.model,
                )
                aggregate(htir)
                build_witness(htir)

        return htir

    def compile_prefix(
        self,
        task_id: str,
        raw_steps: list[dict[str, Any]],
        harness_snippets: dict[str, str],
        prefix_len: int,
        **compile_kwargs: Any,
    ) -> HTIR:
        """
        Compile only the first ``prefix_len`` steps of ``raw_steps`` into an
        HTIR -- the partial graph G_{tau<=t} online intervention (AVG Step 7,
        avg.tex Sec. 3.11) monitors. This is the "simplest" incremental-
        compilation option noted in the Step-7 handoff: it re-runs the whole
        deterministic pipeline over a prefix rather than incrementally
        updating an existing ``HTIR``, which is fine for offline replay of a
        *recorded* trace (not a live agent).

        ``generate_obligations`` and ``run_checks`` default to ``True`` here
        (unlike ``compile``) since intervention needs checked obligations;
        pass them explicitly in ``compile_kwargs`` to override.
        """
        compile_kwargs.setdefault("generate_obligations", True)
        compile_kwargs.setdefault("run_checks", True)
        return self.compile(
            task_id=task_id,
            raw_steps=raw_steps[:prefix_len],
            harness_snippets=harness_snippets,
            **compile_kwargs,
        )

    def compile_many(
        self,
        traces: list[dict[str, Any]],
        *,
        max_workers: int | None = None,
        **compile_kwargs: Any,
    ) -> list[HTIR]:
        """
        Compile many traces (enterprise batch verification). Each item in
        ``traces`` is a dict of keyword arguments for :meth:`compile` (at least
        ``task_id`` and ``raw_steps``; ``harness_snippets`` defaults to ``{}``).
        Results preserve input order.

        ``max_workers`` runs compiles concurrently in a thread pool. Threads
        help when the pipeline is LLM/IO-bound (``run_checks``/
        ``use_semantic``); the default deterministic path is CPU-bound, so
        leave it ``None`` (sequential) unless you are network-bound.
        """
        def _one(spec: dict[str, Any]) -> HTIR:
            kwargs = {"harness_snippets": {}, **compile_kwargs, **spec}
            return self.compile(**kwargs)

        if max_workers and len(traces) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                return list(pool.map(_one, traces))
        return [_one(spec) for spec in traces]

    # ------------------------------------------------------------------
    # Step construction
    # ------------------------------------------------------------------

    def _build_steps(self, raw_steps: list[dict[str, Any]]) -> list[TraceStep]:
        steps: list[TraceStep] = []
        valid_roles = set(self.domain_spec.operation_type_names())
        llm_available = True  # flips off (once) if no API key / no openai extra
        for i, raw in enumerate(raw_steps, start=1):
            req = str(raw.get("request", raw.get("prompt", raw.get("input", ""))))
            resp = str(raw.get("response", raw.get("output", raw.get("result", ""))))
            tool_calls = _parse_tool_calls(raw.get("tool_calls") or [])

            # Adapter-provided annotation (role_hint / status_hint /
            # artifact_effects) short-circuits the LLM annotation pass, so a
            # trace whose adapter already knows the operation type compiles
            # fully offline (no API key). Otherwise use the LLM -- and if the
            # LLM is unavailable (offline / no key), degrade gracefully to a
            # default annotation rather than crashing, so a graph is always
            # produced (roles fall back to "other"/tool-name promotion).
            if any(k in raw for k in ("role_hint", "status_hint", "artifact_effects")):
                role_raw = str(raw.get("role_hint") or "other")
                status = _coerce_execution_status(raw.get("status_hint"))
                effects_src = raw.get("artifact_effects") or []
            else:
                annotation = _StepAnnotation()
                if llm_available:
                    try:
                        annotation = self._annotate_step(i, req, resp)
                    except (EnvironmentError, ImportError) as exc:
                        # No API key / openai extra: annotate deterministically
                        # for the rest of the trace instead of failing.
                        llm_available = False
                        warnings.warn(
                            f"LLM step annotation unavailable ({exc}); compiling with "
                            "default roles. Supply adapter role hints or set an API key "
                            "for richer operation-type inference.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                role_raw = annotation.role
                status = annotation.execution_status
                effects_src = annotation.artifact_effects

            role = role_raw if role_raw in valid_roles else "other"
            # Let structured tool calls drive the domain role: if the role is
            # still generic, promote it from a tool-call name that matches a
            # domain operation type (e.g. an OTel/openai `run_test` tool call
            # -> the `run_test` operation), else fall back to `tool_invocation`.
            if role in ("other", "tool_invocation") and tool_calls:
                named = next((tc.name for tc in tool_calls if tc.name in valid_roles), None)
                if named is not None:
                    role = named
                elif role == "other" and "tool_invocation" in valid_roles:
                    role = "tool_invocation"

            effects = [
                ArtifactStateEvidence(
                    effect_category=_coerce_artifact_effect(e.get("effect_category")),
                    affected_resource=str(e.get("affected_resource") or ""),
                    observed_change=str(e.get("observed_change") or ""),
                    supporting_evidence=str(e.get("supporting_evidence") or ""),
                )
                for e in effects_src
            ]

            metadata = dict(raw.get("metadata") or {})
            metadata.update({
                k: v for k, v in raw.items()
                if k not in _CANONICAL_STEP_KEYS
            })

            steps.append(
                TraceStep(
                    step_id=i,
                    request_message=req,
                    response_message=resp,
                    role=role,
                    execution_status=status,
                    artifact_state_effects=effects,
                    tool_calls=tool_calls,
                    raw_metadata=metadata,
                )
            )
        return steps

    def _annotate_step(self, step_id: int, req: str, resp: str) -> _StepAnnotation:
        op_lines = "\n".join(
            f"  - {op.name}: {op.description}" for op in self.domain_spec.operation_types
        )
        valid_roles = ", ".join(self.domain_spec.operation_type_names())
        msgs = [
            system(
                "You are an expert agent-trace analyst. "
                "Given a single agent step (request + response), infer its role "
                "(operation type), execution status, and artifact/state effects. "
                "Use ONLY the provided operation-type and enum values."
            ),
            user(
                f"Step {step_id}:\n"
                f"REQUEST:\n{truncate(req, 1500)}\n\n"
                f"RESPONSE:\n{truncate(resp, 1500)}\n\n"
                f"Valid roles (operation types) for domain "
                f"'{self.domain_spec.domain_id}':\n{op_lines}\n"
                f"Return role as exactly one of: {valid_roles}\n"
                "Valid execution_status: success, failure, timeout, blocked, unknown\n"
                "artifact_effects is a list of objects with keys: "
                "effect_category (none/read_only/artifact_change/state_change/mixed/unknown), "
                "affected_resource, observed_change, supporting_evidence."
            ),
        ]
        return chat_json(msgs, _StepAnnotation, model=self.model)

    # ------------------------------------------------------------------
    # Artifact node extraction
    # ------------------------------------------------------------------

    def _extract_artifacts(self, htir: HTIR) -> None:
        """
        Lift the per-step artifact_state_effects into first-class ArtifactNodes,
        deduplicating by identifier and versioning on repeated mutation. Wires
        each step's consumed/produced artifact references and the AVG
        artifact-provenance edges (E_prov): an ``ArtifactProvenanceLink`` per
        (step, artifact) recording whether the step created, modified, or
        read the artifact.
        """
        artifact_type_names = set(self.domain_spec.artifact_type_names())
        index: dict[str, ArtifactNode] = {}
        next_id = 1

        for step in htir.steps_in_order():
            for eff in step.artifact_state_effects:
                ident = (eff.affected_resource or "").strip()
                if not ident:
                    continue

                is_change = eff.effect_category in (
                    ArtifactEffect.ARTIFACT_CHANGE,
                    ArtifactEffect.STATE_CHANGE,
                    ArtifactEffect.MIXED,
                )
                is_read = eff.effect_category == ArtifactEffect.READ_ONLY

                artifact = index.get(ident)
                is_new_artifact = artifact is None
                if artifact is None:
                    artifact = ArtifactNode(
                        artifact_id=next_id,
                        artifact_type=self._infer_artifact_type(ident, artifact_type_names),
                        identifier=ident,
                        description=eff.observed_change,
                        produced_by_step_id=step.step_id if is_change else None,
                        version=0,
                    )
                    next_id += 1
                    index[ident] = artifact
                    htir.artifacts.append(artifact)
                elif is_change:
                    artifact.version += 1
                    if artifact.produced_by_step_id is None:
                        artifact.produced_by_step_id = step.step_id

                if is_change and artifact.artifact_id not in step.produced_artifact_ids:
                    step.produced_artifact_ids.append(artifact.artifact_id)
                elif is_read and artifact.artifact_id not in step.consumed_artifact_ids:
                    step.consumed_artifact_ids.append(artifact.artifact_id)

                # AVG artifact-provenance edge (E_prov).
                if is_change:
                    relation = ProvenanceRelation.CREATED if is_new_artifact else ProvenanceRelation.MODIFIED
                    htir.provenance_links.append(
                        ArtifactProvenanceLink(
                            step_id=step.step_id, artifact_id=artifact.artifact_id, relation=relation,
                        )
                    )
                elif is_read:
                    htir.provenance_links.append(
                        ArtifactProvenanceLink(
                            step_id=step.step_id, artifact_id=artifact.artifact_id,
                            relation=ProvenanceRelation.READ,
                        )
                    )

    @staticmethod
    def _infer_artifact_type(identifier: str, known_types: set[str]) -> str:
        """Best-effort mapping of an identifier to a domain artifact type."""
        ident = identifier.lower()
        if ("." in ident and "/" in ident) or ident.endswith(
            (".py", ".r", ".js", ".ts", ".json", ".yaml", ".csv", ".txt", ".md")
        ):
            candidate = "source_file" if "source_file" in known_types else "file"
        elif "tool" in ident:
            candidate = "tool_result"
        elif "test" in ident or "report" in ident:
            candidate = "test_report"
        elif "log" in ident:
            candidate = "log"
        elif "plan" in ident or "todo" in ident:
            candidate = "task_plan"
        else:
            candidate = "state"
        return candidate if candidate in known_types else (
            "state" if "state" in known_types else candidate
        )

    # ------------------------------------------------------------------
    # Input-reuse links (HarnessFix extension, NOT AVG provenance)
    # ------------------------------------------------------------------

    def _infer_input_reuse(
        self, steps: list[TraceStep], harness_context: str
    ) -> list[InputReuseLink]:
        if len(steps) < 2:
            return []

        all_links: list[InputReuseLink] = []
        # Process in overlapping windows so each batch has context from prev steps
        for start in range(0, len(steps), LINK_BATCH_SIZE):
            batch = steps[max(0, start - 3): start + LINK_BATCH_SIZE]  # 3-step lookback overlap
            all_links.extend(self._infer_input_reuse_batch(batch, harness_context))
        return all_links

    def _infer_input_reuse_batch(
        self, steps: list[TraceStep], harness_context: str
    ) -> list[InputReuseLink]:
        if len(steps) < 2:
            return []

        steps_summary = "\n".join(
            f"S{s.step_id}: req={truncate(s.request_message, 200)} | "
            f"resp={truncate(s.response_message, 200)}"
            for s in steps
        )

        msgs = [
            system(
                "You are an expert agent-trace analyst specialising in input reuse. "
                "For each step, identify which earlier steps' content was reused in its request "
                "(explicitly copied, summarised, transformed, or semantically reused)."
            ),
            user(
                f"Steps summary:\n{steps_summary}\n\n"
                f"Harness context:\n{truncate(harness_context, 800)}\n\n"
                "Produce a list of input-reuse links. "
                "Each link: source_id, target_id, source_span, target_span, reuse_relation "
                "(copied/summarized/transformed/semantically_reused), harness_code_refs (list of "
                "{file_path, start_line, end_line, description})."
            ),
        ]
        result = chat_json(msgs, _InputReuseLinkList, model=self.model, max_tokens=2048)

        links: list[InputReuseLink] = []
        for lk in result.links:
            try:
                rr = ReuseRelation(lk.reuse_relation)
            except ValueError:
                rr = ReuseRelation.SEMANTICALLY_REUSED
            code_refs = [HarnessCodeRef(**r) for r in lk.harness_code_refs if isinstance(r, dict) and "file_path" in r]
            links.append(
                InputReuseLink(
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
    # Control flow links (HarnessFix extension, NOT AVG E_causal)
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
            f"S{s.step_id}: role={s.role} status={s.execution_status.value} "
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
        reuse_by_target: dict[int, list[InputReuseLink]] = {}
        for lk in htir.input_reuse_links:
            reuse_by_target.setdefault(lk.target_id, []).append(lk)

        cf_by_target: dict[int, list[ControlFlowLink]] = {}
        for lk in htir.control_flow_links:
            cf_by_target.setdefault(lk.target_id, []).append(lk)

        for step in htir.steps:
            step.node_local_evidence = NodeLocalEvidence(
                input_reuse_evidence=reuse_by_target.get(step.step_id, []),
                control_flow_evidence=cf_by_target.get(step.step_id, []),
                artifact_state_evidence=step.artifact_state_effects,
            )

    # ------------------------------------------------------------------
    # Harness layer facets (HarnessFix extension, not part of AVG G_tau;
    # only invoked when ``compile(attach_harness_layers=True)``)
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
                    "input_reuse": [lk.model_dump() for lk in step.node_local_evidence.input_reuse_evidence],
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
