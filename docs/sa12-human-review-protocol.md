# SA-12 — Human-review efficiency of the verification witness (protocol)

**Claim (intro promise: "the witness compresses the trace for a reviewer").**
Given a trajectory to audit, a human reviewer reaches a **correct verdict faster
and/or more accurately** from the verification witness `W_τ` (avg.tex Sec. 3.9)
than from the raw trace.

This is a small controlled study. The code (`htir/eval/experiment_sa12.py`) is a
**turnkey harness**: it generates the study instruments deterministically now, and
becomes a finished measurement the moment rater CSVs are returned — a "fill in the
CSV" workflow. Until real raters run, a documented, deterministic *simulated
rater* exercises the whole pipeline so every number is reproducible and the
harness is regression-tested; those numbers are flagged `simulated=true` and are a
**pipeline self-check, not a human result**.

## Design

- **Corpus.** ~50 traces, balanced valid/invalid (`balanced_sample`), from any
  domain (τ-bench used for the committed dry-run).
- **Conditions (within-trace, 2 levels).** Each trace is rendered two ways:
  - `raw` — the full trajectory, step by step (`render_raw_trace`).
  - `witness` — `W_τ`: passed/failed/abstained obligations, the evidence to
    inspect (`E_W`), and the one-line review recommendation `R_W`
    (`render_witness`).
- **Counterbalancing.** Raters are split into two groups by parity. For trace
  index `i` and group `g ∈ {0,1}` the condition is `CONDITIONS[(i+g) % 2]`. This
  guarantees: (i) every trace is reviewed in **both** conditions across the pool;
  (ii) **no rater sees the same trace twice**; (iii) each rater's condition mix is
  balanced ~50/50. Condition is therefore not confounded with trace or with rater.
- **Measures (per rating).** `verdict ∈ {valid, invalid}` and `seconds`
  (time-to-verdict). The verdict is scored against the ground-truth reward label
  (`trace_label`); the answer key is held in `items.json` and never shown to
  raters.

## Metrics (`SA12Result`)

- Per condition: verdict **accuracy**, **median / mean** time-to-verdict, the
  reviewer **false-valid rate** (credited `valid` on a truly-invalid trace — the
  human analogue of the monolith's headline failure), and **inter-rater
  agreement** (mean within-trace pairwise agreement).
- **Headline contrasts (paired over traces).** For each trace, per-condition mean
  accuracy and median time; the witness−raw gap is reported with a paired t-test
  (`accuracy_gap`, `time_gap`). The trace is the paired unit, so the contrast is
  robust to trace difficulty.

## Running the study

1. **Export packets** for raters (deterministic):

   ```
   python -m htir.eval.experiment_sa12 --mode export --domain tau_bench \
     --cache data/tau_cache/tau_all.jsonl --n 50 --raters 6 \
     --packet-dir data/sa12_packets
   ```

   Writes, per rater, `packet_R##.html` (the review material, no answer key) and a
   blank `responses_R##.csv` template; plus the experimenter-only `items.json`
   (answer key) and `assignments.json` (the counterbalanced design).

2. **Raters** read each `packet_R##.html` and fill `verdict` + `seconds` in their
   `responses_R##.csv`. Keep it honest about `n` — 2–3 raters × ~50 traces lands
   the claim only if the effect size is clear.

3. **Score** the returned CSVs into the results JSON:

   ```
   python -m htir.eval.experiment_sa12 --mode score \
     --items data/sa12_packets/items.json \
     --responses 'data/sa12_packets/responses_*.csv' \
     --out data/sa12_results.json
   ```

## Simulated dry-run (default `--mode dryrun`)

The default mode fills the scoring pipeline with a deterministic two-condition
rater model (`SimRaterParams`), so `data/sa12_results.json` is reproducible before
any human runs. The model is explicit: **time** is a reading-cost model (raw pays
per trajectory step, witness per surfaced obligation — so the witness is faster
exactly when it compresses the trace); **accuracy** encodes that a raw-trace
reviewer skims and is fooled by reward-hacks like the endpoint monolith, while a
witness reviewer follows the localized recommendation. It is **not** a claim about
humans — it validates the harness and sets the expected direction the real study
will confirm or refute. Replace it verbatim with `--mode score` on real CSVs.
