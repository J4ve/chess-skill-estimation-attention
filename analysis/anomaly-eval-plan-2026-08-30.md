# Plan: run the anomaly-detection validation (Objective 3) to a decision point

## Context

While the deeper-CNN rating arm trains (~3 days, self-healing), the highest-value
work is Objective 3: does the attention-weighted per-move deviation signal
localise engine-substituted moves in the synthetic Maia corpus. This is what
Omori called "the more novel part," and it needs no GPU contention with the
rating arm and never touches the held-out rating test set.

**This is not a greenfield build.** Exploration found that most of the pipeline
already exists and a first evaluation already ran and *failed*:

- **90,000-game synthetic corpus** is complete on HPC at `data/anomaly_corpus/`
  (9 Maia bands 1100-1900 x 2 engines stockfish16/lc0 x 5 substitution rates
  0/5/15/30/60% x 1000 games). Each game pickle carries `Positions`, `Moves`,
  per-ply `move_is_substituted` ground truth, `suspect_color`, `maia_band`,
  `WhiteElo`/`BlackElo` (= band). All cases marked `.done`.
- **Two eval scripts on HPC**: `~/<workdir>/eval_anomaly.py` (nominal-band
  baseline) and `eval_anomaly_v2.py` (empirical clean-game baseline). Both load
  a trained attention checkpoint, run `return_attention=True`, feed
  `per_move_preds` + `attention_weights` + baseline into the non-parametric
  `prototype/src/anomaly.py::AnomalyDetector`, and report game-level ROC-AUC +
  move-level substituted-vs-clean deviation ratio.
- **v2 was run 2026-08-26** on a 2,700-game subset:
  **game-level ROC-AUC = 0.3438** (worse than chance),
  substituted/clean per-ply deviation ratio = **1.05x** (no signal).
  Empirical clean baseline came out far above nominal: Maia-1100 read as
  ~1,817-1,922, Maia-1900 as ~2,710. The human-trained rating model does not
  see Maia's play as its nominal band.
- **`analysis/scripts/synthesize_anomaly_clocks.py`** (v1.0.0, 47 KB, 9 self-test
  groups) is a finished, label-blind clock synthesizer: parametric-share /
  empirical-share / constant methods, real 2-countdown clock arithmetic with
  increment, seeded from (band, game index) only so clocks cannot confound the
  across-rate comparison, `--mode sidecar` retrofit that never mutates the
  corpus. **Not deployed to HPC.**
- **`analysis/anomaly-clock-synthesis-plan.md`** defines a 5-gate validation
  protocol (sections 8-9 carry the exact commands).

### Leading hypotheses for the AUC 0.34 failure
- **H1 clocks (fixable):** the corpus `Clocks` field is engine compute time
  (Maia runs at `nodes=1`, ~instant), not a remaining-time countdown. Every
  clock z-scores to ~ (0-273)/380 = -0.72, a near-constant far outside the
  training distribution. The synthesizer fixes exactly this.
- **H2 domain gap (maybe fundamental):** Maia at `nodes=1` is not a human of its
  band; the model reads Maia-1100 as ~1900, so a Stockfish substitution on top
  is a small relative jump and `R_baseline = band` is meaningless.
- **H3 length confound:** substitution rate correlates with game length
  (~0.05 sd on the clock feature, label-correlated). Gate 5.
- **H4 baseline choice:** nominal vs empirical-clean vs early-plies. v2 already
  tried empirical; still 0.34.

The goal of this plan is to **fix the clocks, run the 5 gates, re-evaluate, and
reach a decision point**: did the clock fix rescue detection, or is synthetic-
Maia validation of the deviation detector a documented Chapter 4/5 limitation
(the closed-account real-world path is already declined, so this is the only
anomaly evidence).

## Confirmed scope (captain, 2026-08-30)

- **Fix clocks and re-run now.** Do not message Omori first; the clock bug is
  well diagnosed and cheap to fix. Omori gets involved only once there is a real
  post-fix number.
- **Stop at the decision point (Step 5).** Fix clocks, run the gates, evaluate on
  a small stratified subset, then STOP and report whether detection now works.
  The full 90k run and the Chapter 4 write-up are decided together after seeing
  that number - not part of this pass.

## Approach

All work on HPC in the `ratingnet2` conda env. Scoring GPU: one of the idle
GPUs 0/2/3 (GPU 1 has the deeper-CNN arm). Nothing in `data/anomaly_corpus/` is
mutated (sidecar mode). Nothing here touches the rating test set.

### Step 0 - Deploy and self-test
- Copy `analysis/scripts/synthesize_anomaly_clocks.py` to
  `~/<workdir>/analysis/scripts/` on HPC. Confirm it is the
  post-`anomaly-clock-fixes.md` version (constant method sits flat; 9 self-test
  groups).
- `python3 synthesize_anomaly_clocks.py --self-test` must pass.
- Scoring checkpoint: **tuned attention arm**
  `models/preflight_check_2m/best_model.pth` (val MAE 172.4, has attention, so
  `alpha_t` is available for `S_att`). NOT `model_55.pth` (no attention).

### Step 1 - Gate 3 + Gate 1 (clock-model realism)
- **Gate 3:** measure the real corpus per-ply clock mean/std from the training
  pickles; compare to 273/380. Resolves whether the constants are simply
  mis-centred for everyone (fine) or the model is wrong.
- **Gate 1:** `--fit-from <one month of real pickles> --fit-out clock_params.json`
  then `--compare-real <a different month> --report-json clock_ks_report.json`.
  Accept if KS D < 0.10 for blitz/bullet/rapid; if bullet cannot clear 0.10 due
  to 1-second granularity, record the number rather than move the bar. If the
  empirical fit fails broadly, fall back to `--method parametric-share` and note
  the limitation.

### Step 2 - Retrofit the corpus (sidecar, both arms)
- **Treatment arm (label-blind):**
  `synthesize_anomaly_clocks.py --corpus data/anomaly_corpus --mode sidecar
  --out data/anomaly_clocks --params clock_params.json`
- **Control arm (Gate 4):** same with `--method constant --out
  data/anomaly_clocks_constant`
- **Sensitivity arm (optional):** `--cheater-timing --out
  data/anomaly_clocks_labelled` - a deliberately label-leaking clock, to
  quantify how much a naive clock model would have inflated AUC (a defensive
  Chapter 3/4 result).
- Sidecar = one parallel clock artifact per game; corpus pickles untouched.
  Cost is ~21 s of synthesis + one read pass per arm.

### Step 3 - Gate 5 (ply counts)
- Count plies per game per (band, engine, rate) directly on HPC (never done; the
  length gradient is only storage-inferred). Report mean/median ply count per
  arm.

### Step 4 - Re-evaluate (Gate 4 + full Chapter 3 metrics)
Write `eval_anomaly_v3.py` (extends v2), or extend v2 in place, to:
- read clocks from a sidecar dir instead of the pickle `Clocks`;
- compute all three Chapter 3 aggregations per game: `S_att` (primary,
  attention-weighted), `S_max`, `S_mean` (formulas: `chapter3.tex` lines
  129-148; `d_t = |R_hat_t - R_baseline|`);
- game-level ROC-AUC of the suspect-side score vs the clean/cheat label, broken
  down per band, per engine, per substitution rate, and pooled;
- precision @ 1% and 5% FPR, threshold calibrated on a held-out split of clean
  (r00) games (Chapter 3 "calibrate on the validation set" - here, held-out
  clean games), seed 42;
- move-level localisation: top-k by `alpha_t . d_t` hit-rate against
  `move_is_substituted`; substituted-vs-clean per-ply deviation ratio;
- ROC-AUC within ply-count strata (Gate 5);
- run across three clock arms (treatment / constant control / labelled) and two
  baselines (nominal band, empirical-clean-half) and print them side by side;
- **regression anchor:** pointed at the OLD pickle clocks + empirical baseline it
  must reproduce v2's 0.3438.
- Start on a stratified subset (~50-100 games/case; ~4,500-9,000 games; minutes
  on an idle GPU), then scale to the full 90k only if the subset shows signal.

### Step 5 - Decision point (checkpoint with the captain)
- **Treatment AUC strong and beats the constant control:** clocks were the
  problem. Scale to full 90k, freeze numbers, draft the Chapter 4 anomaly
  subsection.
- **AUC improves but stays weak:** report honestly with the confound analysis;
  synthetic validation is partial evidence; Chapter 5 limitation. Raise with
  Omori.
- **AUC near chance even with good clocks (H2):** document that the human-trained
  deviation signal does not transfer to Maia's move distribution; the
  `AnomalyDetector` design is sound but this synthetic validation is limited.
  Consider a narrow existence-proof figure (high bands + r060 only, largest
  substitution jump). Raise with Omori before any further corpus work.

### Step 6 - Document
- `analysis/anomaly-eval-results.md` (new): results tables, all 5 gate outcomes,
  the decision.
- `chapter3-4-process.md`: new dated status subsection.
- `contexts/consultation_log.md`: append to Round 6 open follow-ups (clock
  synthesis validated; detection result to raise with Omori).
- Chapter 4 anomaly subsection stub only if Step 5 supports it.

## Critical files

- `prototype/src/anomaly.py` - `AnomalyDetector`, non-parametric, reused as-is.
- `prototype/src/chess_rating_net.py` - `forward(..., return_attention=True)`
  returns `per_move_preds` + `attention_weights` (query-averaged `alpha_t`).
- `~/<workdir>/eval_anomaly_v2.py` - basis for v3; already wires model +
  detector + ROC-AUC + move-level split.
- `analysis/scripts/synthesize_anomaly_clocks.py` - clock synthesizer, deploy
  and run; do not modify.
- `analysis/anomaly-clock-synthesis-plan.md` sections 8-9 - gate definitions and
  exact commands.
- `CCS Thesis - Integrated/chapters/chapter3.tex` lines 124-155, 264-274 -
  anomaly formulas, metrics (AUC, precision @ 1%/5% FPR),
  operating-point calibration.

## Verification

- `synthesize_anomaly_clocks.py --self-test` passes (9 groups).
- `--simulate 5000 --report-json`: synthesized clock mean/std near real 273/380;
  per-ply drawdown monotone (increment games may rise).
- Gate 1 KS D reported per bucket; treatment accepted or the gap quantified.
- `eval_anomaly_v3.py` on OLD clocks + empirical baseline reproduces 0.3438
  (regression anchor), then reports the new numbers on the fixed clocks.
- Control sanity: on the constant-clock arm, mean `S_att` for r060 games >
  r00 games; if not, the detector has no signal regardless of clocks.
- Gate 5: counted ply gradient reported next to every AUC.

## Not in scope this pass

- Scaling to the full 90k and the Chapter 4 write-up (gated on Step 5).
- Time-control subgroup analysis of the rating model (separate task).
- Prototype UI / live Lichess streaming (separate task, good to parallelise).
- Any test-set rating evaluation (architecture not frozen, per Omori 2026-08-29).
