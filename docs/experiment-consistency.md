# Experiment consistency cross-check: `avg.tex` ↔ plan ↔ implementation

Audit of the implemented experiments (SA-1…SA-6 + the τ-bench campaign) against
`avg.tex` Sec. 3–4 and `docs/paper-completion-plan.md`. **Read-only analysis.**
Ordered by severity. **Date:** 2026-07-14.

---

## 1. Systematic section-number desync (avg.tex ↔ codebase/docs) — ✅ RESOLVED (2026-07-14)

**Fixed:** 127 `Sec. 3.x` / `Sec. 4.x` citations across `htir/`, `tests/`, and
the docs were re-synced to `avg.tex`'s current numbering (topic-verified, not a
blind shift — the `Python < 3.8` version string and this audit doc were left
untouched, and the Ablations↔Stress pair was *swapped*, not shifted). The table
below is the mapping that was applied; kept for the record.

`avg.tex` was reordered/renumbered after the code and plan were written; the
`Sec. 3.x` / `Sec. 4.x` citations throughout `htir/` and the docs no longer
pointed at the right subsections.

**Sec. 3 (Methodology) — code refs are +1 from "Verification Witness" on:**

| Subsection | avg.tex actual | codebase cites |
|---|---|---|
| Checking Obligations | 3.7 | (implied 3.8) |
| Aggregating | 3.8 | 3.9 (in prose) |
| Verification Witness | 3.9 | **3.10** (`witness.py`) |
| Online Intervention | 3.10 | **3.11** (`intervention.py`, `trace_abstraction.py`) |
| Offline Harness Improvement | 3.11 | **3.12** (`harness_improvement.py`) |

Also, "Abstention and escalation" is a *paragraph inside* Checking Obligations
(3.7), not a standalone 3.8.

**Sec. 4 (Experiments) — Ablations ↔ Stress swapped, Stats shifted:**

| Subsection | avg.tex actual | codebase cites |
|---|---|---|
| Core Ablations | **4.5** | 4.6 (`baselines.py`, `checking.py`, `sa3`, `analysis.py`) |
| Stress Tests | **4.6** | 4.5 (`sa6`) |
| Statistical Reporting | **4.7** | 4.8 (`seeds.py`, `sa1`) |
| Expected Outcomes | 4.8 | — |

The τ-bench docs added this week inherited the old convention. **Fix:** decide
whether `avg.tex`'s current ordering is final, then re-sync every `Sec. 3.x` /
`Sec. 4.x` reference in one mechanical sweep (I left `escalation.py` citing by
name to avoid adding a 21st wrong number). Nothing about the *experiments* is
wrong here — only the pointers.

## 2. Implemented domain suite ≠ `avg.tex` Sec. 4.1 — HIGH

Sec. 4.1 names six benchmarks: Terminal-Bench, MCPVerse, DAComp, **JourneyBench**,
**R2E-Gym/SWE**, and an enterprise case study. The implementation runs
**Terminal-Bench + τ-bench**. But:

- **τ-bench is not in the Sec. 4.1 list** (only mentioned in the intro/related
  work). The actual second domain is unlisted.
- The paper's *policy* domain is **JourneyBench**; τ-bench stands in for it
  (plan open-decision #2 chose τ-bench, but `avg.tex` still says JourneyBench).
- The paper's *SWE* domain (**R2E-Gym/SWE**) has a spec (`swe_gym.yaml`) but **no
  loader/data** (`datasets.py` has no `load_swe_gym`), so it is unimplemented.

**Fix:** add τ-bench to Sec. 4.1 (note it fills the policy slot in place of
JourneyBench); mark MCPVerse / DAComp / R2E-Gym-SWE / enterprise as the
extensibility surface, matching the plan's own framing.

## 3. Task metrics & operating modes presuppose a live agent — HIGH

Sec. 4.4 *task metrics* (success rate, partial completion, policy-violation
rate, rollback rate, cost, wall-clock, tool budget) and Sec. 4.2's three
*operating modes* (`Base` / `Base+Online AVG` / `+Offline`) all require running
the agent and measuring an outcome delta. The implementation **replays recorded
traces**: SA-4 (intervention) and SA-5 (harness loop) are recommendation-only
(`intervention.py`: *"purely a recommendation trace … does not drive an agent"*).
So no task-success delta exists and those metrics/modes cannot be populated as
written.

**Fix:** scope 4.2/4.4 explicitly as the offline-replay analog (the plan's WP-2.3
stance), or add the live micro-study (plan WP-3.4).

## 4. Q4 heterogeneity — ✅ ADDRESSED (2026-07-14)

Q4 = online control + offline optimization *"across heterogeneous domains."*
SA-4 and SA-5 now run on **τ-bench** as well as Terminal-Bench
(`data/sa4_tau_results.json`, `data/sa5_tau_results.json`; both `run_sa*` accept
`spec=`). Findings on τ:
- **SA-4 (online intervention):** confident `failed_obligation` interventions
  are precise (0.649 vs. 0.50 base) and timely (median 4 steps saved) — actually
  *stronger* than terminal, since failed *mutation* tool calls give a real
  mechanical signal.
- **SA-5 (offline harness loop):** the loop mines/gates/accepts an edit and
  `J_loop` improves on training, but held-out false-valid was already low (0.037)
  so the (terminal-flavored) mined template gives 0% held-out reduction with zero
  negative transfer — an honest null on the policy domain (τ-specific remediation
  templates would be needed for a held-out gain).

Q4 is thus demonstrated across two domains, offline. A *live* loop still needs
the plan's WP-3.4 micro-study.

## 5. Missing baselines (4.3) and ablations (4.5) — MEDIUM

- **Baselines missing:** no-verifier base agent (undefined offline — we never
  run the agent), **SkillOpt / prompt-skill**, **Meta-Harness / Life-Harness**.
  Only monolithic (now LLM ✓), exec-only ✓, exec-free ✓ exist.
- **Ablations missing as clean arms:** #6 no-online-loop, #7 template-free
  adaptation. The plan's "solid" bar (complete #1–#7 table) is not met (#1–#5
  present).

**Fix:** add the `*-style` comparators (plan WP-3.1) and #6/#7 arms.

## 6. Stress-test perturbations are single-step proxies, not the definitions — MEDIUM

`avg.tex` Stress Tests define:
- *policy drift* = "the SOP/compliance policy **changes mid-evaluation**";
- *large tool menus* = "action spaces **expand substantially** at test time";
- *hidden-state mismatch* = "visible outputs correct while **latent required
  state is missing**".

The τ-bench SA-6 implementations approximate these with a single appended step
(one unconfirmed mutation; one exotic-tool call; one unsupported final claim).
They exercise the *right verifier pathway* but are not the literal stressors —
`policy_drift` in particular is an unconfirmed action, not a mid-run SOP change.

**Fix:** either strengthen the perturbations (e.g. swap the Ω_d policy mid-trace
for real policy drift; expand the tool vocabulary wholesale) or relabel/scope
them honestly in the write-up.

## 7. "Online trace adaptation" (4.2) is still unimplemented — MEDIUM

Sec. 4.2's third protocol is the verifier **updating its adapters incrementally
from newly observed failures**. The new dynamic escalation loop
(`htir/agents/escalation.py`) is *within-trajectory* re-verification
(`request-evidence`), not cross-trace adapter adaptation. SA-5's offline harness
loop is the nearest analog but is offline, not online. Don't conflate the three.

## 8. Gold-dependent metrics not produced (4.4) — MEDIUM

Sec. 4.4 lists **evidence-localization quality** and gold-calibrated
AUROC/ECE/intervention-precision. Implemented AUROC/ECE use **weak reward
labels**; localization is not measured at all. Needs the ~150-trace gold slice
(enabler E3), still unimplemented.

## 9. Cost-normalized curves (Statistical Reporting) not produced — LOW

The subsection asks for cost-normalized performance curves. A per-arm LLM-call
**cost proxy** exists (and the LLM slice now has real call counts), but no
cost-normalized curve/figure is generated yet.

---

## What is consistent (for the record)

- Q1 (SA-1), Q3 (SA-3), integrity ablation #4 (SA-6) are faithfully implemented
  and now run on **two** domains with mean±SE — matching the plan's Phase-1 bar.
- The Q2 transfer matrix exists and behaves exactly as the "adapters are
  necessary" thesis predicts (diagonal-only resolution).
- The monolithic-**LLM** baseline (Sec. 4.3) now runs — the plan's top gap.
- Abstention-as-feature reporting (false-valid + resolved-fraction, not accuracy
  alone) is consistent across every experiment, per Sec. 4.4 and the plan's
  honesty principle.

## Priority for a clean submission

1. Re-sync section numbers (**#1**) — pure hygiene, but pervasive and cheap.
2. List τ-bench in Sec. 4.1 and reconcile JourneyBench/SWE (**#2**).
3. Honestly scope offline-replay for modes/task-metrics/Q4 (**#3, #4**).
4. Add #6/#7 ablations and the SkillOpt/Meta-Harness baselines (**#5**).
5. Strengthen or relabel the stress perturbations (**#6**).
