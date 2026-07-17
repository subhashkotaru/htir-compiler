# AVG Experimentation Plan

Maps the proposed experiments in `avg.tex` (Sec. 4) onto concrete, runnable
sub-agent work packages against this repo (`htir/`). **One experiment set per
sub-agent.** Each package is self-contained: goal, arms, data, metrics,
deliverable, and feasibility.

This plan is grounded in the current implementation state (full AVG pipeline,
Steps 1–8, deterministic path runs with no API key) and the first terminalbench
run. Read the two blocking sections (Inconsistencies + P0) before assigning any
experiment sub-agent — they gate every downstream result.

@avg.tex  has my proposal and experimentation plan. This repo has implementation for that.

---

## 0. Inconsistencies to resolve first (avg.tex vs. implementation)

Ranked by how much each distorts an experiment.

1. **Over-crediting: zero-evidence trajectories marked `valid` (real bug).**
   `htir/agents/witness.py::aggregate` (line ~98) falls to `STATUS_VALID`
   whenever no HIGH/CRITICAL obligation *failed* and not-too-many HIGH
   abstained. If a trace binds **no** high-severity obligations at all, the
   `elif` is skipped → `valid`, regardless of coverage. `scratch_results.json`
   shows 56 obligations, all abstained, `coverage 0.0`, `uncertainty 1.0` →
   `"valid"`. Contradicts avg.tex Sec. 3.4 ("emits unresolved obligations
   instead of assigning unsupported credit") and Sec. 3.8 ("...treated as
   uncertain rather than successful"). **Root cause of the prior 100%
   false-valid rate.** Fix: `coverage == 0` / all-abstained → `uncertain`;
   broad abstention (not only HIGH) must raise uncertainty.

2. **`Parse_{S_d}` for terminal free-text not upstreamed.** The `turns`/`raw`
   adapters extract 0 artifacts from terminal stdout/stderr, so obligations
   never bind → everything abstains → feeds bug (1). The deterministic terminal
   extractor that fixed this lives only in the scratchpad. Every terminal
   experiment depends on committing it as an adapter.

3. **Sec. 4.1 benchmark suite exceeds runnable domains.** Only `default` and
   `terminal_swe` specs exist, with data for terminal only. MCPVerse / DAComp /
   JourneyBench / R2E-Gym have no adapters, specs, or data. Cross-domain
   *transfer* (Q2) needs at least one real second domain.

4. **Named baselines in Sec. 4.3 don't exist as runnable arms.** No monolithic
   single-scalar judge, no execution-only / execution-free split, no SkillOpt /
   Meta-Harness / Life-Harness comparators. These are scaffolding to build.

5. **Sec. 4.4 verifier metrics assume labels not present.** Local
   classification accuracy, abstention-calibrated AUROC, ECE, intervention
   precision/recall, evidence-localization quality all need step/obligation
   level gold labels. Datasets carry only trajectory-level `reward ∈ {0,1}`.

**Scoped limitations (documented, not contradictions):** intervention &
harness-improvement are recommendation-only over replayed recorded traces (no
live agent); harness edits apply only to `DOMAIN_SPEC`; `R_W` is deterministic.
All experiments are therefore **offline replay**.

---

## P0 — Prerequisites (blocking; not a research experiment)

Assign this first. Everything downstream depends on it.

- Fix aggregation over-crediting in `htir/agents/witness.py::aggregate`:
  `coverage == 0` / all-abstained → `uncertain`; add a regression test.
- Upstream the deterministic terminal `Parse_{S_d}` extractor as a committed
  adapter (`htir/adapters/terminal.py`) + regression test reproducing the
  ~7.8 artifacts / 31 obligations-per-trace behavior.
- Weak-label harness: map trajectory `reward` + `<returncode>N</returncode>`
  tags to per-trace labels; hand-label a **100-trace gold slice** for the
  verifier metrics that need step-level truth.

**Dataset for all terminal packages:** `yoonholee/terminalbench-trajectories`
(HF) — 52,104 traces, 89 Terminal-Bench tasks, trajectory `reward` labels, same
`{steps:[{src,msg,tools,obs}], reward, agent, model, task_name}` schema as
`data/raw_traces/`. Use a balanced sample (equal solved / unsolved).

---

## Experiment packages (one per sub-agent)

| ID | Research question / ablation | Runnable when |
|----|------------------------------|---------------|
| SA-1 | Q1 — Graph vs. Monolith (+ exec-only / exec-free) | After P0 |
| SA-2 | Q2 — Universal-only vs. Universal+Adapters, transfer | After P0 + 2nd domain |
| SA-3 | Q3 — Calibrated abstention | After P0 + gold slice |
| SA-4 | Q4a — Online intervention (offline replay) | After P0 |
| SA-5 | Q4b — Offline harness improvement loop | After P0 (scratch prototype exists) |
| SA-6 | Stress tests + integrity ablation | Partial after P0; full needs 2nd domain |

---

### SA-1 — Q1: Graph vs. Monolith (verifier factorization)

- **Claim (avg.tex Sec. 4, Q1):** the obligation graph beats a single scalar
  judge, especially on long-horizon traces.
- **Arms:**
  - (a) **AVG** full pipeline (`compile(run_checks=True)`).
  - (b) **Monolithic** baseline — one LLM judge over the whole trace → pass/fail.
  - (c) **Execution-only** — mechanical checkers only, semantic disabled.
  - (d) **Execution-free** — semantic/artifact checkers only, no executable evidence.
  - (build b–d; c/d are ablations of the AVG checker router.)
- **Data:** balanced 3k terminalbench sample.
- **Metrics:** false-valid rate, resolved accuracy vs. 50% base rate,
  failure-flag precision/recall, abstention rate. Maps to Ablation #1.
- **Deliverable:** results table + cost-normalized performance curve (Sec. 4.7).
- **Feasibility:** fully runnable post-P0; (a),(c),(d) offline, (b) needs a key.

---

### SA-2 — Q2: Universal-only vs. Universal + Adapters (transfer & sample efficiency)

- **Claim (Q2):** domain adapters + `Ω_d` improve few-artifact adaptation while
  preserving cross-domain transfer.
- **Arms:** `default` spec only → `default` + `terminal_swe` → + `Ω_d` bundle
  (`load_domain_artifacts`: schemas / policies / tests).
- **Protocols (Sec. 4.2):** zero-shot transfer, few-artifact adaptation
  (sweep artifact budget), online-trace adaptation.
- **Second domain (required for transfer):** add a **SWE-Gym / R2E-Gym**
  adapter + spec (executable, labeled, closest to `terminal_swe`). Without a
  real second domain, run only universal-vs-adapters *within* terminal and mark
  transfer as future work.
- **Metrics:** sample-efficiency curve (perf vs. #artifacts), transfer delta,
  negative-transfer on unrelated tasks. Maps to Ablations #2, #7 (template-free).
- **Deliverable:** transfer matrix (train-domains × test-domains).
- **Feasibility:** universal-vs-adapters runnable post-P0; transfer needs the
  second-domain adapter.

---

### SA-3 — Q3: Calibrated Abstention

- **Claim (Q3):** abstention reduces harmful false positives and improves
  intervention precision.
- **Arms:** calibrated abstention (default) vs. **no-abstention** (force every
  checker to emit pass/fail — Ablation #3).
- **Data:** terminalbench sample + P0 100-trace gold slice for calibration truth.
- **Metrics:** abstention-calibrated AUROC, expected calibration error (ECE),
  false-valid reduction, precision at fixed abstention budgets.
- **Deliverable:** reliability diagram + AUROC/ECE table.
- **Feasibility:** runnable after P0 gold slice.

---

### SA-4 — Q4a: Online Intervention (offline replay)

- **Claim (Q4):** monitoring obligations on the partial graph enables timely,
  precise intervention.
- **Method:** `htir.agents.intervention.run_intervention_loop` over
  `TraceAbstractionAgent.compile_prefix` replays (partial graph `G_{τ≤t}`).
  Counterfactual analysis: "discharging obligation X at step t would have
  avoided the failure / broken the loop." Scale up the `chess-best-move` case
  (21 wasted `echo` repeats caught at step 4).
- **Metrics:** intervention precision/recall, steps-saved distribution,
  per-step active-obligation set.
- **Deliverable:** intervention-timing plot + per-step obligation walkthrough.
- **Feasibility:** runnable now (recommendation-only; no live agent driven).

---

### SA-5 — Q4b: Offline Harness Improvement Loop

- **Claim (Q4):** witnesses drive `S_d` edits that generalize without changing
  the base model.
- **Method:** formalize & scale the prior 3-iteration run —
  `mine_recurring_failures → score_config → accept_edit →
  apply_domain_spec_edit` over a recorded witness corpus. 3×N experience
  batches + a held-out future set.
- **Metrics:** Ĵ trajectory, edit-acceptance rate, held-out generalization,
  **negative-transfer rate** on unrelated tasks. Maps to Ablations #5 (no
  offline loop), #6 (no online loop).
- **Deliverable:** learning-curve + spec-growth table (obligation-set growth on
  a held-out trace across spec versions).
- **Feasibility:** largely runnable — scratch prototype exists; commit + scale.

---

### SA-6 — Stress Tests + Integrity (Sec. 4.6, 4.7)

- **Claim:** AVG improves reliability under enterprise-relevant failure modes,
  not just average benchmark score.
- **Method:** synthetic perturbations of terminal traces —
  - shortcut opportunity (hidden-test pass / grader fail — 41% observed),
  - artifact inconsistency (logs/tables contradict),
  - policy drift (SOP changes mid-run),
  - large tool menus (action space expands at test time),
  - hidden-state mismatch (visible output correct, latent state missing).
- **Ablation #4:** remove the integrity verifier and measure shortcut-catch drop.
- **Metrics:** shortcut-catch rate, false-valid rate under each perturbation.
- **Deliverable:** stress-test grid (perturbation × arm).
- **Feasibility:** shortcut + artifact-inconsistency runnable post-P0;
  policy-drift needs a policy-bearing domain (defer with SA-2's second domain).

---

## Cross-cutting reporting (Sec. 4.7)

For every package: mean ± standard error over seeds / task subsamples; matched
model, tool, and budget conditions; cost-normalized curves in addition to
best-score snapshots; disclose runtime config, intervention budgets, and any
`Ω_d` artifacts used.

## Out-of-reach without new data (frame as future work in the paper)

MCPVerse (large tool ecosystems), DAComp (enterprise data workflows),
JourneyBench (policy-heavy SOPs), and the enterprise case study require new
adapters + domain specs + sourced traces. Recommend scoping the paper's
empirical claims to Terminal-Bench + one SWE domain, and presenting the rest as
the framework's extensibility surface (`htir/adapters/`, `htir/domains/*.yaml`).
