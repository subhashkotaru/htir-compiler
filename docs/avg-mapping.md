# AVG -> HTIR mapping

This file tracks, for every symbol in the AVG proposal (`avg.tex`), the
concrete class/field in this codebase that realises it, and any known
deviation. HarnessFix concepts that are *not* part of AVG are called out
explicitly so they are not mistaken for proposal constructs.

Scope: this repo currently implements AVG Steps 1-4: `raw trace -> typed
events -> verification graph G_tau` (Steps 1-2), well-formedness checks and
the analysis modules (Step 3, `harnessfix.agents.analysis`), and claim /
evidence / obligation generation (Step 4,
`harnessfix.agents.obligations.build_claims_and_obligations`). Steps 5-6
(checker execution, aggregation, verification witness) and the online
intervention loop are unbuilt; see "Out of scope" at the bottom.

## Node kinds

| AVG concept | Class | Notes |
|---|---|---|
| Operation node | `TraceStep` (harnessfix/models/htir.py) | One recoverable execution step. `role` is drawn from the active domain spec's operation vocabulary (`S_d.P_d`), not a fixed enum. |
| Artifact node | `ArtifactNode` | First-class produced/consumed object. `artifact_type` drawn from `S_d.R_d`. |
| Claim node | `ClaimNode` | Checkable statement induced from the trajectory; untrusted until discharged by an obligation. |
| Evidence node | `EvidenceNode` | Points to artifacts/graph neighbourhoods that may support or refute claims. |
| Obligation | `Obligation` | See obligation tuple below. |

## Edge families

| AVG edge | Class | Field on `HTIR` | Notes |
|---|---|---|---|
| `E_temp` (temporal) | `TemporalLink` | `temporal_links` | Preserves original execution order. |
| `E_prov` (provenance) | `ArtifactProvenanceLink` | `provenance_links` | Artifact-centric: links each artifact to the operation that `created`/`read`/`modified` it (avg.tex Sec. 3, Provenance analysis). Populated deterministically in `TraceAbstractionAgent._extract_artifacts`. |
| `E_causal` (dependency) | `DependencyLink` | `dependency_links` | "Which operations depend on earlier artifacts" (avg.tex Sec. 3, Dependency analysis). Populated by the Step-3 analysis module `harnessfix/agents/analysis.py::link_dependencies`: consumer step -> producer step of a consumed artifact, edit -> most recent failing validation, and final answer -> policy artifact. |
| `E_sup` (support) | `SupportLink` | `support_links` | Evidence supports/refutes a claim. |
| `E_cons` (constraint) | `ConstraintLink` | `constraint_links` | Domain constraint (`S_d.K_d`) governing a step/artifact. |
| `E_val` (validation) | `ValidationLink` | `validation_links` | Validation operation -> the step/artifact it validates. |

## HarnessFix extensions (NOT AVG constructs)

These are carried over from the original HarnessFix paper for harness-code
attribution / diagnostics. They are additive, optional, and must not be
confused with the AVG edges/nodes above:

| Class / field | Purpose | AVG equivalent |
|---|---|---|
| `InputReuseLink` (`input_reuse_links`) | Step->step: how a later request reused an earlier step's content. | None. Previously mislabelled `E_prov`; renamed to avoid confusion with the real, artifact-centric `ArtifactProvenanceLink`. |
| `ControlFlowLink` (`control_flow_links`) | Step->step: harness controller transition (retry/delegate/finalize/...) that produced the next step. | None. Previously mislabelled `E_causal`; renamed/annotated to avoid confusion with `DependencyLink`. |
| `HarnessLayer`, `HarnessLayerFacet` (`TraceStep.harness_layer_facet`), `HarnessCodeRef` | The seven ETCLOVG harness layers implicated by a step's local evidence, and pointers to the harness source responsible. | None. Off by default (`TraceAbstractionAgent.compile(attach_harness_layers=False)`); costs one LLM call per step when enabled. |

## Obligation tuple `o_i = (c_i, r_i, E_i, q_i, rho_i, alpha_i)`

| Symbol | Field on `Obligation` | Type |
|---|---|---|
| `c_i` (claim) | `claim_id` | int (-> `ClaimNode`) |
| `r_i` (required evidence) | `required_evidence` | `EvidenceType` |
| `E_i` (candidate evidence) | `candidate_evidence_ids` | list[int] (-> `EvidenceNode`) |
| `q_i` (checker) | `checker` | `CheckerType` |
| `rho_i` (severity) | `severity` | `Severity` |
| `alpha_i` (escalation) | `escalation` | `EscalationRule` |

`EscalationRule` (alpha_i) intentionally shares its action vocabulary
(`accept`/`request-evidence`/`rerank`/`veto`/`repair`/`clarify`/`escalate`)
with `InterventionAction`, the online intervention `iota_t` from avg.tex Sec.
3.11. They are distinct mechanisms: `alpha_i` is fixed per-obligation at
generation time; `iota_t` is chosen online per step from the active
obligation set. `InterventionAction` is defined now for forward
compatibility but not yet wired to any online loop.

## Domain spec `S_d = (P_d, R_d, K_d, B_d)`

| Symbol | Class | Notes |
|---|---|---|
| `P_d` (operation types) | `OperationType`, `DomainSpec.operation_types` | |
| `R_d` (artifact types) | `ArtifactTypeSpec`, `DomainSpec.artifact_types` | |
| `K_d` (constraints) | `Constraint`, `DomainSpec.constraints` | |
| `B_d` (obligation templates) | `ObligationTemplate`, `DomainSpec.obligation_templates` | Split into universal / domain / trajectory-triggered via `ObligationScope`. |

Domain specs are YAML-driven (`harnessfix/domains/*.yaml`), loaded via
`harnessfix.models.domain.load_domain_spec`.

## Out of scope (Steps 5-6, unbuilt, not inconsistencies)

Left as `TODO` seams; Steps 1-4 above are prerequisites for them:

- `Ω_d` domain artifacts beyond `S_d` (if the proposal introduces more).
- Checker execution (Step 5) filling `CheckerResult` (`p_pass`/`p_fail`/
  `p_abstain`/`score`/`evidence_used`) — obligations are emitted `PENDING`
  with a routed `checker` class only.
- Aggregation `z_tau` and severity-aware veto (Step 6).
- Verification witness `W_tau` (Step 6).
- Online intervention loop (`InterventionAction` / `iota_t`).
