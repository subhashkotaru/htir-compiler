# AVG Build Roadmap (Section-0 fixes → experiment-ready)

Companion to `docs/experiment-plan.md`. That doc plans the **research layer**
(SA-1..SA-6). This doc plans the **engineering layer**: the build order,
effort, and dependency DAG for the Section-0 inconsistencies and the
scaffolding they imply. Nothing in `experiment-plan.md`'s SA packages is
runnable-for-real until the items here land.

Effort key: **S** < 0.5 day · **M** 0.5–2 days · **L** 2–5 days · **XL** > 1 week
(or gated on external data / human labeling time).

---

## Status (implemented 2026-07-08)

All six build items landed; full test suite **82 passed** (`~/.venv`, `pytest`).

| item | status | where |
|------|--------|-------|
| **B1** fix `aggregate()` over-crediting | ✅ done | `htir/agents/witness.py` (+3 regressions) |
| **B2** terminal `Parse_Sd` adapter + spec | ✅ done | `htir/adapters/terminal.py`, `htir/domains/terminal_swe.yaml` (+6 tests) |
| **B4** verifier arms + monolith | ✅ done | `htir/agents/baselines.py`, `checking.py` `disable_mechanical` flag (+4 tests) |
| **B3** weak labels + verifier metrics | ✅ done | `htir/eval/weak_labels.py` (+tests) |
| **B6** terminalbench ingestion + balanced sampler | ✅ done | `htir/eval/datasets.py` (+tests) |
| **B5** SWE-Gym second-domain **spec** | ✅ done | `htir/domains/swe_gym.yaml` (reuses terminal adapter, +test) |

**Verified end-to-end on the committed real trace** (`01_...ars`):
`terminal` adapter autodetected → 1 artifact, 109 obligations, **31 passed / 3
failed / 75 abstained**, status **uncertain** (was 56/56 abstained → falsely
**valid**). Miniature SA-1 over the 5 committed reward-0 traces:
**AVG false-valid rate 0.2 vs monolith 0.8**.

### Remaining (gated on external data / human time — not code)
- **B3 gold slice**: hand-label 100 traces at step/obligation level for the
  metrics that need true labels (AUROC, ECE, evidence-localization). Weak
  reward-based metrics run now; these do not.
- **B5 SWE data adapter**: the `swe_gym` *spec* is done and runs over any
  terminal-shaped SWE trace via the `terminal` adapter, but a dataset-specific
  loader awaits the confirmed SWE-Gym/R2E-Gym HF trace schema
  (`htir/eval/datasets.py` is where it plugs in).
- **B6 at scale**: `load_terminalbench` needs `pip install datasets` + the 52k
  pull; the balanced sampler and label plumbing are tested offline on synthetic
  records.

---

## Corrections vs. the Section-0 notes (verified in-repo, 2026-07-08)

Two facts changed the effort picture since those notes were written:

1. **The terminal extractor is gone, not "in the scratchpad."** The scratchpad
   is session-scoped and is empty in a fresh session. There is no
   `htir/adapters/terminal.py`. Item I2 is a **rebuild from spec**, not an
   upstream/move → size **M–L**, not S.

2. **Why terminal traces bind 0 obligations is structural.**
   `trace_abstraction.py::_extract_artifacts` (~L502) lifts first-class
   artifacts from each step's `artifact_state_effects`. For terminal free-text
   those annotations are empty unless (a) the LLM annotation pass runs, or
   (b) a deterministic pre-annotator fills `artifact_effects` +
   `<returncode>` before compile. The fix is a deterministic annotator, so the
   whole pipeline stays runnable with no API key.

3. **Local terminal data is 5 traces of one task** (`data/raw_traces/`, the
   `adaptive-rejection-sampler` task). Scale experiments need the external HF
   set `yoonholee/terminalbench-trajectories` (52k). Ingestion is its own item.

Section 0 has **5** numbered inconsistencies (I1–I5) + scoped limitations —
not 8. The "Steps 5–8" in memory refer to AVG *pipeline* stages, not to
section-0 items.

---

## Dependency DAG

```
                    ┌──────────────────────────────┐
                    │ B1  Fix aggregate() (I1) [S]  │  true bug fix
                    └───────────────┬──────────────┘
                                    │ (trustworthy status on every arm)
                    ┌───────────────▼──────────────┐
                    │ B2  Terminal Parse_Sd (I2)[M-L]│  rebuild; obligations bind
                    └───┬───────────┬───────────┬───┘
                        │           │           │
          ┌─────────────▼──┐  ┌─────▼──────┐  ┌─▼──────────────┐
          │ B3 Weak-label +│  │ B4 Baseline│  │ B6 52k HF      │
          │ gold slice(I5) │  │ arms  (I4) │  │ ingestion  [M] │
          │       [M + L]  │  │   [M]      │  └─┬──────────────┘
          └───────┬────────┘  └─────┬──────┘    │
                  │                 │           │
                  ▼                 ▼           ▼
             SA-3, verifier    SA-1 (a–d)   SA-1/3/6 at scale
             metrics (4.4)

   ┌──────────────────────────────┐
   │ B5 Second domain (I3) [L-XL] │  parallel track, needs B1; reuses B2 pattern
   └───────────────┬──────────────┘
                   ▼
              SA-2 transfer, SA-6 policy-drift
```

Critical path to a *first real result*: **B1 → B2 → B4 → SA-1**.
Everything else hangs off B2.

---

## Build items

### B1 — Fix `aggregate()` over-crediting (I1) — **S**, no deps · TRUE BUG
- **File:** `htir/agents/witness.py::aggregate` (L91–99).
- **Defect:** when a trace binds **no** HIGH/CRITICAL obligation, the `elif
  high_severity and (...)` guard is false and control falls to `else →
  STATUS_VALID`, regardless of coverage. `scratch_results.json` real_traces:
  56 obligations, all abstained, coverage 0.0 → `"valid"`.
- **Fix:** a trajectory with `evidence_coverage == 0` (or all obligations
  abstained) → `uncertain`; broad abstention across **all** severities (not
  only HIGH) must raise uncertainty, per avg.tex §3.4 / §3.8.
- **Test:** regression from `scratch_results.json` real_traces[0] and
  controls[*] (controls already resolve to `uncertain` and must stay green).
- **Unblocks:** trustworthy `predicted_status` on *every* arm and every SA
  package. Nothing downstream is measurable until this is correct.

### B2 — Terminal `Parse_{S_d}` extractor as committed adapter (I2) — **M–L**, deps: (pairs with B1) · REBUILD
- **New:** `htir/adapters/terminal.py` + a deterministic annotator that emits
  `artifact_effects` (command_output / source_file / patch / test_report) and
  parses `<returncode>N</returncode>` from stdout/stderr.
- **Target behavior:** reproduce the prior ~7.8 artifacts and ~31 obligations
  per trace (regression test with a fixed fixture).
- **Wire-in:** register via `register_adapter`; bind to `terminal_swe.yaml`
  operation/artifact types.
- **Unblocks:** SA-1, SA-3, SA-4, SA-6 — every terminal experiment. This is
  the linchpin.

### B3 — Weak-label harness + 100-trace gold slice (I5) — **M** (harness) **+ L** (labeling) · deps: B2
- **Harness (M):** map trajectory `reward ∈ {0,1}` + `<returncode>` tags to
  per-trace weak labels; align to bound obligations.
- **Gold slice (L, human time):** hand-label a 100-trace slice at
  step/obligation level for the metrics that need ground truth (local
  classification accuracy, AUROC, ECE, intervention P/R, evidence
  localization). This is the only item gated on *human* effort.
- **Unblocks:** SA-3 calibration, all §4.4 verifier metrics, SA-1
  resolved-accuracy vs. base rate.

### B4 — Baseline arms (I4) — **M** · deps: B1, B2
- **Exec-only / exec-free (S–M):** flags on the checker router
  (`htir/agents/checking.py`) — disable semantic / disable mechanical.
  These are ablations of AVG, cheap.
- **Monolithic judge (M):** one LLM call over the whole trace → pass/fail.
  Separate arm; needs an API key.
- **Defer to future work (XL):** SkillOpt / Meta-Harness / Life-Harness — no
  runnable comparators exist; frame as related-work, not arms.
- **Unblocks:** SA-1 (a–d), Ablation #1.

### B5 — Second domain adapter + spec (I3) — **L–XL** · deps: B1, reuses B2 · PARALLEL
- **Pick:** SWE-Gym / R2E-Gym (executable, labeled, closest to `terminal_swe`).
- **Build:** adapter + `htir/domains/<swe>.yaml` + sourced traces.
- **Unblocks:** SA-2 transfer matrix, SA-6 policy-drift (partial). Without a
  real second domain, SA-2 runs only universal-vs-adapters *within* terminal
  and transfer is future work.

### B6 — 52k HF dataset ingestion — **M** · deps: B2
- Ingest `yoonholee/terminalbench-trajectories` through the B2 adapter; build
  a balanced solved/unsolved sample loader.
- **Unblocks:** scale for SA-1 / SA-3 / SA-6 (the 5 local traces are a smoke
  test only).

---

## Recommended sequencing

1. **B1** (half day, unblocks correctness) →
2. **B2** (the linchpin; everything terminal waits on it) →
3. In parallel once B2 lands: **B4** (→ ship SA-1, the headline Q1 result),
   **B6** (scale), and start **B3**'s labeling (long human lead time).
4. **B5** on its own track whenever a second-domain owner is free; it only
   gates SA-2.

## Out of reach without new data (frame as extensibility, not results)
MCPVerse, DAComp, JourneyBench, enterprise case study → new adapters + specs +
sourced traces each. Scope the paper's empirical claims to **Terminal-Bench +
one SWE domain**; present the rest as the framework's extension surface
(`htir/adapters/`, `htir/domains/*.yaml`).
