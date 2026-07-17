# AVG — Gap-Filling Plan for a Solid ICLR Paper

A sequenced plan to close every gap in `docs/proposal-gap-analysis.md` and turn
the current single-domain / offline / no-LLM results into a defensible empirical
section for `avg.tex`.

**Hard constraint (from the request):** *fill gaps and experiments only — do not
change the methodology.* Every work package below adds **data, LLM enablement,
baselines, labels, seeds, domains, or perturbations that the proposal already
names**. None adds/removes an AVG node, edge, checker class, obligation
template mechanism, or aggregation rule. Where a package could be mistaken for a
method change, it is flagged **[method-preserving]** with why.

**Audit inputs:** `docs/proposal-gap-analysis.md`, `docs/experiment-plan.md`,
`docs/avg-mapping.md`. **Date:** 2026-07-13.

---

## 1. What "solid" requires here

The paper's thesis is **factorized, evidence-local, abstention-aware
verification beats a monolithic judge and transfers across domains.** Right now
three load-bearing claims are *structurally untested* (not merely weak):

- **Q1 rests on a strawman** — the monolithic baseline never ran its LLM judge;
  offline, `avg_full` collapses to `exec_only` and `exec_free` is a null arm.
- **Q2 has no transfer** — one domain, so zero-shot transfer and the transfer
  matrix don't exist; `Ω_d` adaptation is offline-inert.
- **Verifier metrics ride weak labels** — AUROC/ECE/localization use
  reward-derived labels, no human gold.

A submission is "solid" when: (a) the Q1 win holds against an **LLM** monolith
across **≥2 domains** with **mean±SE over seeds**; (b) there is a real
**transfer matrix**; (c) calibration/localization are reported on a **gold
slice**; (d) the ablation table is complete (#1–#7); (e) stress tests cover the
enterprise failure modes. Everything below serves those five.

---

## 2. Guiding principles

1. **Method-preserving.** Only fill gaps the proposal already specifies.
2. **Matched conditions.** Same model, tool, and budget across arms
   (avg.tex Sec. 4.7); every LLM arm shares one judge model per run.
3. **Honest abstention.** Keep reporting false-valid + resolved-fraction, not
   accuracy alone — abstention is a feature, not hidden failure.
4. **Reproducibility.** Every run pins seed, model slug, artifact bundle, and
   projected/actual LLM-call cost (the cost fields already exist in results).
5. **Graceful degradation.** Phases 1–2 alone = a *minimum solid submission*;
   Phases 3–4 strengthen it. If time is short, cut from the bottom.

---

## 3. The four enablers and the dependency graph

Almost every gap reduces to four foundational enablers:

- **E1 — LLM on.** `use_llm=True` end-to-end (semantic checkers, `exec_free`
  arm, monolithic-LLM judge). *Already wired* (`htir/utils/llm.py`, OpenRouter,
  `--model`/`use_semantic` flags) — this is a run + budget task, not new code.
- **E2 — Second domain.** SWE-Gym/R2E-Gym trace loader. *Spec already exists*
  (`htir/domains/swe_gym.yaml`, terminal-adapter-compatible); only the HF
  loader in `htir/eval/datasets.py` + sourced traces are missing.
- **E3 — Gold slice.** ~150 hand-labeled traces with step/obligation/evidence
  annotations for trustworthy AUROC/ECE/localization/intervention-precision.
- **E4 — Multi-seed + SE.** A thin sweep-and-aggregate wrapper (the `--seed`
  knob already exists; just loop and report mean±SE).

```
                 E1 (LLM on) ──► SA-1*, SA-3* (Q1/Q3 honest)      ┐
                     │                                            │
E4 (seeds+SE) ──► every table/figure                             ├─► solid paper
                     │                                            │
E2 (2nd domain) ─► SA-2 transfer matrix, protocols, #7 ─► E2+policy(3rd) ─► stress:policy-drift
                     │
E3 (gold slice) ─► SA-3 calib, SA-4 intervention-precision, localization metric
```

Critical path: **E1 ∥ E3 ∥ E4** first (parallelizable), then **E2**, then the
third (policy) domain + baselines.

---

## 4. Phased work packages

Effort in person-days (pd) assumes one researcher. `$` = LLM spend (see §5).

### Phase 0 — Foundations (unblocks everything) · ~4–6 pd

| WP | Objective | Key tasks | Files | Effort |
|----|-----------|-----------|-------|--------|
| **0.1** | **E4** multi-seed + SE harness | `--seeds 0,1,2,3,4` loop over `balanced_sample`; aggregate to mean±SE; add to every `experiment_sa*` writer | `htir/eval/*` (shared helper) | 1.5 pd |
| **0.2** | **E1** LLM smoke + cost calibration | Run SA-1 on 50 traces `--use-llm`; measure real tokens/call & per-arm calls; pick judge model(s); confirm `chat_json` structured output is stable | `htir/eval/experiment_sa1.py` | 1 pd + ~$1 |
| **0.3** | **E3** gold-slice protocol + tooling | Labeling schema (trajectory ✓reward; per-obligation pass/fail/NA; evidence-node correct y/n); small CLI to dump a trace's obligations+evidence for annotation; label **150 balanced traces** (Terminal-Bench first) | new `htir/eval/gold.py`, `data/gold/` | 2–3 pd (human) |

Start 0.3 immediately — it's the long human-latency item.

### Phase 1 — Make Q1 & Q3 honest with the LLM · ~4 pd · $6–95/run

| WP | Objective | Method-preserving? | Deliverable |
|----|-----------|--------------------|-------------|
| **1.1** | **SA-1 with LLM.** `avg_full` vs **monolithic-LLM** vs `exec_only` vs `exec_free`, on the gold slice + a 500-trace balanced subsample, then scale to 3k if budget ok. | Yes — just `use_llm=True` on existing arms | **Table 1** (Q1, per-domain, mean±SE) + **cost-normalized curve** (accuracy vs LLM-call budget; the axis already exists) |
| **1.2** | **SA-3 gold-calibrated.** AUROC/ECE of the aggregate score vs **gold** (and vs weak labels, side by side); reliability diagram; `no_abstention` ablation (#3) with LLM. | Yes — `calibration.py` already has `roc_auc`/ECE; feed gold labels | **Reliability diagram** + AUROC/ECE table (gold vs weak) |

Honest-result guardrail: if the LLM monolith is *stronger* than expected,
report it — the thesis predicts AVG still wins on **long-horizon** traces
(`n_long_horizon=1548/3000` slice already tracked), so lead with the
long-horizon cut.

### Phase 2 — Second domain → Q2 transfer + protocols · ~7 pd · $20–150

| WP | Objective | Method-preserving? | Deliverable |
|----|-----------|--------------------|-------------|
| **2.1** | **E2 SWE-Gym/R2E-Gym loader.** Source traces (HF `SWE-Gym`/`R2E-Gym`), add a schema-specific `load_*`/canonicalizer in `datasets.py`; the `terminal` adapter + `swe_gym.yaml` already consume terminal-shaped steps. Add a reward/label extractor. | Yes — new data loader only | Runnable 2nd domain |
| **2.2** | **SA-2 transfer matrix.** `universal_only` → `adapter_full` → `+Ω_d`, **with LLM** (so `Ω_d`/policy obligations resolve instead of abstaining). Protocols: zero-shot transfer (train-spec on domain A, test on B), few-artifact sweep, template-free ablation (**#7**: replace artifact-grounded templates with a direct end-to-end LLM adapter). | Yes — data + existing flags; #7 is an ablation the proposal names | **Table 2** transfer matrix (train×test) + **sample-efficiency curve** + #7 row |
| **2.3** | **Three operating modes** (`Base` / `Base+Online AVG` / `+Offline`) framed as the offline-replay analog, scoped honestly (no live agent). | Yes — reporting/packaging of SA-4+SA-5 | Modes table |

### Phase 3 — Baselines, stress, third (policy) domain · ~10–12 pd · $30–200

| WP | Objective | Method-preserving? | Deliverable |
|----|-----------|--------------------|-------------|
| **3.1** | **Missing baselines (Sec. 4.3).** Representative **SkillOpt-style** prompt/skill optimizer (LLM proposes harness-prompt edits scored on held-out) and a **Meta-Harness-style** outer-loop search, as comparators to SA-5's witness-driven loop. Framed as "*-style*" reimplementations, not the exact systems. | Yes — external baselines, AVG untouched | SA-5 baseline rows |
| **3.2** | **Stress tests (Sec. 4.6) now-feasible two.** `large_tool_menus` (expand the action/tool vocabulary at test time) and `hidden_state_mismatch` (visible output correct, latent state absent) as synthetic perturbations, added to SA-6's grid. | Yes — SA-6 already has the perturbation harness | Extends **stress grid** |
| **3.3** | **Third domain = policy** (τ-bench or JourneyBench) loader + `S_d` spec. Unlocks **policy-linking**, **policy-drift** stress, and a 3-domain transfer row. | Yes — data + YAML spec + adapter | Policy domain + policy-drift row + 3-domain transfer |
| **3.4** | **[stretch] Live-agent intervention micro-study (Q4a).** Drive **10–30 live tasks** through `run_intervention_loop` against a real harness to convert SA-4's offline counterfactual into a real demonstration. | **[method-preserving]** — Sec. 3.10 already specifies live intervention; this builds the loop the proposal describes, it does not change it | Small live case study; keep offline-replay as main result |

### Phase 4 — Consolidation & write-up · ~4 pd

- Consolidated **ablation table (#1–#7)** in one place.
- Statistical-reporting pass (Sec. 4.7): mean±SE everywhere, cost-normalized
  curves, full config disclosure (seed, model slug, `Ω_d` bundle, budgets).
- Reproducibility appendix: exact commands per table/figure.
- Frame out-of-reach domains (MCPVerse, DAComp, enterprise case study) as the
  framework's **extensibility surface**, matching the proposal's own scoping.

---

## 5. Compute & cost budget (grounded)

From the existing SA-1 cost model (`data/sa1_results.json`): mean **1.29
semantic calls/trace** (`avg_full`, `exec_free`) and **1.0/trace** (monolith).
So a full 3k × all-arms LLM SA-1 run ≈ **~10.8k LLM calls**. Order-of-magnitude
token accounting (~1.5k in/0.2k out per semantic; ~6k in/0.1k out per monolith):

| Run | Calls | `gpt-4o-mini`-tier | `gpt-4o`-tier |
|-----|------:|-------------------:|--------------:|
| SA-1, 500-trace subsample, all arms | ~1.8k | ~$1 | ~$16 |
| SA-1, 3k, all arms, 1 seed | ~10.8k | ~$6 | ~$95 |
| **Whole campaign** (SA-1/2/3 × ~3 seeds × 2 domains) | ~90–120k | **~$50–80** | **~$600–900** |

Practical policy: **cheap-tier judge for the bulk grid; strong-tier for the
gold-slice validation and one headline 3k run** to show robustness to judge
choice. The judge model is a free knob (`--model`, `DEFAULT_MODEL`), so a
judge-model sensitivity row is cheap and strengthens the paper.

---

## 6. Statistical reporting standard (applied to every table)

- Mean ± standard error over **≥3 seeds** (5 preferred) via WP-0.1.
- Report on the **full labeled set** and the **long-horizon slice** separately
  (the slice is where the thesis predicts the biggest Q1 gap).
- Every LLM table pins the judge model and shows the **cost-normalized** variant.
- Gold-slice metrics report **n** and, if multiple annotators, inter-annotator
  agreement on a shared subset.

---

## 7. Risk register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| LLM monolith beats AVG on short traces | Med | Lead with long-horizon slice; report both; the abstention/false-valid axis still favors AVG even if accuracy ties |
| SWE-Gym trace schema differs from expected | Med | `swe_gym.yaml` is schema-independent; isolate risk to one loader function; fall back to R2E-Gym if SWE-Gym traces are thin |
| Gold labeling too slow | Med | Annotate only **bound** obligations, not every step; 150 traces is enough for AUROC CIs; start Phase 0 |
| `Ω_d` still moves few offline verdicts even with LLM | Low | Expected — report obligation-coverage gain explicitly (already done in SA-2), not just verdict flips |
| "*-style" baselines seen as unfaithful | Low | Name them explicitly as representative reimplementations; cite originals; hold model/budget matched |
| Policy (3rd) domain slips | Med | It only gates policy-drift + a 3-domain row; paper is solid with 2 domains — mark policy as the strengthening tier |
| LLM cost overrun | Low | Cheap-tier default; subsample-first; costs above are bounded and small |

---

## 8. Paper narrative → experiment → artifact map

| avg.tex section | Claim | Experiment (post-plan) | Artifact |
|-----------------|-------|------------------------|----------|
| Sec. 4 Q1 / Ablation #1 | Graph > monolith | SA-1 **+LLM**, 2 domains, seeds | **Table 1** + cost curve |
| Sec. 4 Q2 / #2, #7 | Adapters transfer & sample-efficient | SA-2 transfer matrix + few-artifact + template-free | **Table 2** + efficiency curve |
| Sec. 4 Q3 / #3 | Calibrated abstention | SA-3 gold AUROC/ECE + no-abstention | Reliability diagram + calib table |
| Sec. 4 Q4a | Timely online intervention | SA-4 (offline) + **[stretch]** live micro-study | Intervention-timing plot |
| Sec. 4 Q4b / #5, #6 | Offline harness loop generalizes | SA-5 + SkillOpt/Meta-Harness baselines | Learning curve + spec-growth table |
| Sec. 4.6 / #4 | Reliability under enterprise failures | SA-6 + large-tool-menu + hidden-state + policy-drift | Stress grid |
| Sec. 4.5 | Factorization/adaptation/abstention each matter | All ablations #1–#7 | Consolidated ablation table |
| Sec. 4.7 | Rigorous reporting | WP-0.1, Phase 4 | mean±SE + cost-normalized everywhere |

Target empirical section: **~4 tables + ~4 figures** across 2–3 domains.

---

## 9. Definition of Done (per proposal section)

> **τ-bench update (2026-07-14):** the policy domain (τ-bench) was chosen as the
> second real domain instead of SWE-Gym and is now runnable end-to-end with
> mean±SE over 3 seeds. See `docs/tau-bench-results.md`. The **LLM slice ran**
> (`gpt-4o-mini`, n=120): the monolithic-LLM judge that "never ran" now does —
> AVG false-valid 0.050 vs LLM-monolith 0.733 (15×). Remaining gaps are the
> human gold slice and the SkillOpt/Meta-Harness baselines.

- [x] **4.1 Benchmarks** — ≥2 real domains runnable (Terminal-Bench ✓,
      **τ-bench ✓** as the policy domain); rest framed as extensibility.
- [~] **4.2 Protocols** — zero-shot transfer matrix ✓ (terminal ↔ τ-bench,
      diagonal-only); few-artifact curve ✓ (SA-2 within-domain); template-free
      (#7) and operating-modes still to add.
- [~] **4.3 Baselines** — exec-only ✓; **monolithic-LLM ran** ✓ (gpt-4o-mini,
      AVG 0.050 vs 0.733) + exec-free real ✓; SkillOpt/Meta-Harness comparators TODO.
- [~] **4.4 Metrics** — AUROC/ECE on the τ domain ✓ (weak labels, mean±SE);
      gold slice + localization still out (needs human labels).
- [~] **4.5 Ablations** — #1 (SA-1) ✓, #3 no-abstention (SA-3) ✓, #4 integrity
      (SA-6) ✓ on τ; #6/#7 still to add.
- [x] **4.6 Stress** — shortcut ✓, artifact-inconsistency ✓, **large-tool-menu ✓,
      hidden-state ✓, policy-drift ✓** (all three active on τ-bench, SA-6).
- [x] **4.7 Reporting** — mean±SE over 3 seeds ✓ (WP-0.1 `htir/eval/seeds.py`);
      full config disclosure ✓ (result JSONs pin seeds, n, domain); cost-normalized
      curve needs the LLM slice.

---

## 10. Open decisions (need a call before Phase 2/3)

1. **Second domain source:** SWE-Gym vs R2E-Gym (which has cleaner, more
   available traces with rewards?). Default: whichever HF split has trajectory
   rewards ready.
2. **Third (policy) domain:** τ-bench (more available) vs JourneyBench (named in
   proposal). Default: τ-bench, cite JourneyBench as the target class.
3. **Gold-slice size / annotators:** 150 single-annotator (fast) vs 150 + a
   50-trace double-annotated subset for agreement (more rigorous). Default: the
   latter if a second labeler is available.
4. **Judge model:** cheap-tier bulk + strong-tier headline (default), or
   single-model throughout for simplicity.
5. **Live-agent micro-study (WP-3.4):** attempt or defer to future work? Default:
   defer unless Phases 1–2 finish with time to spare.
