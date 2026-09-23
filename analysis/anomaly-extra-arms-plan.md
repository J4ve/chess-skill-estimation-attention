# Anomaly detection: extra arms, pre-registered plan

Written and committed on branch `fm/anomaly-extra-arms` on 2026-09-15, BEFORE any arm below
produced a result. Approved by the captain on 2026-09-15 ("do more anomaly trainings
simultaneously ... run it then we can prob add it too"). Nothing in this file may be changed after
results are seen, except by appending a dated "Deviation" entry to section 9 that gives the reason.
The results file (`analysis/anomaly-extra-arms-results.md`) reports every arm listed here, whatever
its outcome.

## 1. Why this plan exists

Methods in the order they were tried on the synthetic corpus:

1. The parameter-free computed suspicion score S_att: ROC-AUC 0.555 [0.527, 0.584] on the v1
   corpus and about 0.50 on v2, i.e. poor.
2. A LightGBM detector on per-game summary features of the v2 corpus: 0.807 on seen conditions,
   0.685 on withheld Maia bands 1300 and 1700, 0.685 on the withheld 10 percent rate against clean
   games, and 0.629 on both (`analysis/anomaly-trained-detector-v2-results.json`). Its known
   weakness: at 2 to 5 percent substitution a game has 1 or 2 engine moves and per-game averages
   bury them (withheld-band per-rate AUC 0.530 at 2 percent, 0.558 at 5 percent).
3. This batch: arms A0 to A4 below.

The title defense committed to no numeric anomaly target. Every additional method invites the
objection "you kept trying until something worked". The defense is this file: arms, splits,
metrics, settings and reporting rules fixed in advance, and every arm reported.

## 2. Data (fixed)

- Corpus: `~/<workdir>/data/anomaly_corpus_v2/`, 134 cells, 268,000 games. Never modified.
- Features: `~/<workdir>/analysis/v2_features/rating_cheap.npz` (frozen rating model
  `models/preflight_check_2m/best_model.pth` per-ply outputs plus engine-free board and clock
  features). This cache already holds per-move sequences, so A3 needs no re-extraction.
- Game label: `game_label` = at least one substituted ply (same as the v2 LightGBM run).
- Pre-existing corpus property, found while reading the generator before any result
  (`prototype/src/generate_anomaly_corpus.py:557-616`): the side, opening, Maia move and clock RNG
  streams are seeded by (band, game index) only. Consequences, checked on the feature cache:
  1. `{band}_lc0_r00` and `{band}_stockfish16_r00` are the same 2,000 games (2,000 of 2,000 feature
     signatures identical in bands 1100 and 1500).
  2. Game i of every rate cell in a band is a counterfactual twin of game i in the clean cell: same
     opening, suspect side and Maia stream, diverging only at the first substitution.
  3. The v2 split shuffles each cell independently, so twins (including exact clean duplicates)
     cross the train/test boundary on seen conditions. Withheld bands are twin-free with respect to
     training (the whole band is out). The withheld rate is not: its twins at other rates of the
     same band are in training.
  This does not change the primary splits (section 3, required for comparability with 0.807/0.685),
  but it motivates the grouped-split sensitivity analysis A0g (section 5) and the deduplication in
  A1.

## 3. Splits (fixed)

Identical to the v2 LightGBM run: `make_split` in `analysis/scripts/train_anomaly_detector.py` (the
HPC copy, which is commit 7f21613 with `HELDOUT_RATE = 10`), seed 42.

- `heldout_band`: Maia bands 1300 and 1700, all rates except 10 (plus their hard-negative cells).
- `heldout_rate`: rate 10 in the seven other bands.
- `heldout_both`: rate 10 in bands 1300 and 1700.
- Remaining cells (including the hard-negative cells of seen bands): 70/15/15 train/val/test,
  stratified by (band, engine, rate).

Withheld games never touch training, early stopping, model selection, normalization statistics
(except the one stated A2 assumption), or threshold choice. Early stopping and every threshold use
`val` only. Counts: train 126,000, val 27,000, test 27,000, heldout_band 52,000, heldout_rate
28,000, heldout_both 8,000.

Sensitivity split `grouped` (A0g only, plus A2 and A3 if time allows): same withheld definitions,
but the 70/15/15 assignment is by (band, game index) with seed 42, so no twin crosses
train/val/test within seen conditions.

## 4. Metrics (fixed)

### 4.1 Headline conditions (definitions reproduce the four published numbers)

| Name | Games | Positives / negatives |
|---|---|---|
| `seen` | `test` split | `game_label` |
| `withheld_band` | `heldout_band` | `game_label` |
| `withheld_rate_vs_clean` | rate 10 games (`heldout_rate` and `heldout_both`) plus rate 0 games from `test` and `heldout_band` | `game_label` |
| `withheld_both` | `heldout_both` | `game_label` (88.7 percent positive; directional only) |

Secondary conditions: `withheld_rate` raw (`heldout_rate`, `game_label`), `withheld_both_vs_clean`
(rate 10 games of bands 1300/1700 plus rate 0 games of those bands), and per rate on withheld bands:
rate r games plus rate 0 games of `heldout_band`, r in {2, 5, 20, 40, 60}.

### 4.2 Statistics

- Game-level ROC-AUC, 95 percent percentile bootstrap CI, 1,000 replicates, resampling games with
  replacement stratified by label, seed 20260915.
- Paired bootstrap of (arm AUC minus A0 AUC) on the same games and replicates, for each headline
  condition, for every arm that scores the same games. Reported descriptively as the difference
  with its CI.
- Precision at 1 and 5 percent FPR: threshold = the (1 - FPR) quantile of the arm's scores on `val`
  games with `game_label == 0`. On each condition report precision, TPR, realized FPR and number
  flagged. Precision depends on the corpus's high positive prevalence; it is reported, not
  interpreted as a deployment figure.
- Strength-confound check (all game-level arms): AUC of hard-negative games (clean, suspect plays at
  a stronger Maia band) against rate 0 games, on `test` plus `heldout_band`. An AUC well above 0.5
  means the arm flags strong clean play, i.e. partly detects strength rather than substitution.

### 4.3 Comparison points, no numeric target

The title defense committed to no numeric anomaly target, so no AUC value is a target or a
pass/fail bar. Every arm is reported descriptively (AUC with CIs, per-rate AUC, precision,
localization) in one side-by-side table next to A0 and S_att. 0.80 appears in that table only as a
reference line labelled "reference level", never as a criterion.

### 4.4 Move-level localization (A3, A4, and S_att's per-move alpha times d as a reference)

- Games: `game_label == 1` games in each condition.
- Candidate plies: suspect-side plies with ply index 16 or more (the generator's substitution
  eligibility, `generate_anomaly_corpus.py:20`). Arms rank only these.
- Metrics: hit@1 (top-ranked ply is substituted), precision@3 and precision@5 (share of the top k
  that are substituted, k capped at the number of candidates), and pooled ply-level ROC-AUC of the
  per-move score for `move_is_substituted` over candidate plies.
- Random baseline, per game and averaged the same way: hit@1 = s/n, precision@k = s/n, where s is
  substituted and n candidate plies; ply-level AUC 0.5.
- Engine-proximity check (A3, A4): among non-substituted analysed plies in `engine.npz` (Stockfish
  depth 20 matched clean sample), ROC-AUC of the per-move score for `top1_match`. Near 0.5 means the
  model does not flag human-looking moves merely for matching the engine; well above 0.5 means it
  has learned an engine-agreement signal from boards. This uses engine output only as an
  evaluation probe, never as a model input.

## 5. Arms (fixed)

All arms write `~/<workdir>/analysis/extra_arms/<arm>/results.json`, per-game scores
(`scores.npz`), a `DONE` marker, or a `FAILED` marker with the traceback.

### A0 anchor (CPU)

Re-run the v2 LightGBM "full" detector unchanged: the same `pool_game` features (97, sorted keys),
`make_split` seed 42, `LGBMClassifier(n_estimators=2000, learning_rate=0.02, num_leaves=31,
subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, min_child_samples=30,
scale_pos_weight=neg/pos, random_state=42, n_jobs=4)`, early stopping 100 on val AUC. The
engine-feature model is not re-run (leaked). Adds the section 4 metrics.

**Gate:** A0 must reproduce `seen` 0.807 and `withheld_band` 0.685 within 0.005. If not, every
other arm stops and the lane reports `needs-decision`. The launcher enforces this with an
`A0_GATE_PASS` or `A0_GATE_FAIL` marker that A1 to A4 require.

**A0g** (sensitivity): the same model on the `grouped` split. Reported next to A0. If A0g's `seen`
AUC differs from A0's by more than 0.02, every arm's `seen` number carries a twin-leakage caveat in
the results.

### A1 player-level aggregation (CPU, no training)

A different question: "is this player cheating", not "is this game substituted". Labelled as such
wherever reported.

- Scores: S_att (higher = more suspicious, as in the published evaluation) and A0 probabilities.
- A player = k games drawn without replacement from one cell's evaluation pool, k in {1, 5, 10,
  20}. Cheater players: one (band, engine, rate) cell, rate > 0; player label 1 whatever the
  individual games' substitution counts. Clean players: k games from the rate 0 games of one band,
  deduplicated by (band, game index) because the lc0 and stockfish16 rate 0 cells are identical.
  Hard-negative players reported separately against clean players as a strength check.
- Aggregation: mean and max of the k per-game scores.
- Conditions and pools: `seen` (test split, seen bands, rates 2/5/20/40/60 vs seen-band clean),
  `withheld_band` (bands 1300/1700, same rates, vs their clean), `withheld_rate` (rate 10 of seen
  bands vs seen-band test clean), `withheld_both` (rate 10 of 1300/1700 vs their clean). Per-rate
  player AUC is also reported.
- 500 players per cheater cell, 1,000 clean players per band, seed 20260915 (NumPy
  `default_rng`).
- CI: hierarchical bootstrap, 200 replicates: resample games with replacement within each pool,
  redraw all players, recompute AUC. With 300 test games per seen cell, players overlap heavily at
  k = 20; the hierarchical CI is the honest interval, the point estimate alone is not.

### A2 band-normalized features (CPU)

Same LightGBM and settings as A0. The rating-model-derived features (prefixes `d_t_`, `alphad_`,
`r_hat_suspect_`, `run_std_`, `abs_first_diff_`, `abs_second_diff_`, plus `r_hat_final`,
`run_std_final`, `d_t_lastq_mean`) are replaced by z-scores against clean-game (rate 0) statistics
of the same nominal band: z = (x - mean_band) / std_band. Engine-free board and clock features stay
raw.

- Training bands: mean and std from rate 0 games in the `train` split of that band. Applied to
  train, val and test.
- Withheld bands, variant `A2_own` (primary): statistics from that band's own rate 0 games. Stated
  assumption: a platform knows the clean population at each rating. To avoid a game being
  normalized with its own values or its twin's, statistics are cross-fitted by game-index parity:
  games with an even index use statistics from odd-index clean games, and vice versa.
- Withheld bands, variant `A2_interp`: mean and std are the average of the two neighbouring training
  bands' statistics (1300 from 1200 and 1400; 1700 from 1600 and 1800). No withheld-band data used.
- One model is trained; the two variants differ only in how withheld-band features are transformed.
  Rate 10 games of seen bands use training-band statistics in both.
- Target: the withheld-band drop (A0 0.807 seen vs 0.685 withheld).

### A3 sequence detector on per-move outputs (GPU)

- Model: bidirectional GRU. Chosen over a small transformer because sequences are at most 100 plies,
  the inputs are 16 low-dimensional per-move features, it needs no positional encoding or warmup
  tuning, and bidirectional recurrence matches the rating model's own BiLSTM reading of a game.
- Input per ply (all from `rating_cheap.npz`, all engine-free and Maia-likelihood-free):
  `r_hat_suspect`, `r_hat_other`, signed deviation `r_hat_suspect - maia_band`, `d_t`, `alpha`
  times `n_plies`, `run_mean`, `run_std`, `first_diff`, `second_diff`, `is_capture`, `is_check`,
  `material_balance`, `clock_remaining`, `clock_delta` (NaN to 0 plus a missing indicator),
  `is_suspect_move`, `ply_idx / 100`. Continuous inputs standardized with `train` ply statistics.
  Excluded generator bookkeeping: `move_is_substituted`, `engine_differs`, `is_eligible`,
  `maia_argmax_present`, and every per-game count.
- Architecture: linear 16 to H, 2-layer BiGRU hidden H, dropout p; per-ply instance logit
  s_t = w^T h_t; gated attention pooling (Ilse et al. 2018) a_t over suspect-side plies; game logit
  = sum_t a_t s_t plus bias. Per-move localization score: s_t (primary), a_t (secondary).
- Training: AdamW lr 1e-3, weight decay 1e-4, batch 512, BCE with pos_weight = neg/pos on train, at
  most 40 epochs, early stopping on val AUC with patience 5, gradient clip 1.0.
- Validation-only tuning, fully logged: grid H in {64, 128} x p in {0.1, 0.3} with seed 0; the
  config with the best val AUC is chosen; only that config is then scored on withheld conditions.
  Seed 1 re-trains the chosen config; both seeds are reported, seed 0 is the headline.

### A4 end-to-end CNN-BiLSTM detector (GPU, may finish after 2026-09-18)

- Initialized from the frozen rating model (CNN, 3-layer BiLSTM, causal Bahdanau attention).
- Choice: **frozen CNN**, fine-tuned BiLSTM and attention (lr 1e-4), new anomaly head as in A3
  (gated attention pooling over suspect plies, per-ply instance logits; lr 1e-3). Reasons: the CNN
  is a per-board encoder trained on 2.55M human games; freezing it keeps the representation human-
  rating-oriented, cuts overfitting risk on 126k synthetic games, and lets CNN embeddings be
  computed once (float16 cache) so training fits the timeline. Full fine-tuning is not run.
- Inputs: board tensors and standardized clocks from the corpus pickles (read-only), exactly the
  rating model's input path (`ChessGamesDataset` conventions, 100-ply cap, clock mean 273 std 380).
- Training: AdamW, batch 256, at most 15 epochs, early stopping on val AUC patience 3, seed 0,
  BCE with pos_weight. No tuning grid.
- **Flag:** a board-level model can learn to recognize engine-style moves from consecutive boards.
  That is a legitimate detector in principle, but on this corpus substituted moves are exactly the
  generating engine's choices, so it may approach the leaked engine-agreement signal (0.9997 to
  1.0). A4 is reported in its own row, never pooled with A0 to A3, together with: withheld-band
  AUC (twin-free), per-rate AUC at 2 and 5 percent on withheld bands, the hard-negative strength
  check, and the engine-proximity check (section 4.4). The corpus has only two engines, both in
  every training band, so no withheld-engine check is possible. A near-perfect A4 on withheld bands
  at 2 percent is read as likely engine-move recognition, not as a solved problem.

## 6. Exclusions (fixed)

No engine-agreement features (centipawn loss, top-k match, evaluation swing, anything computed by
an engine on the played moves) and no Maia-likelihood features (Maia policy probability of the
played move, Maia argmax agreement). Both leak by construction: substituted moves are engine moves
and clean moves are Maia samples. Engine output appears only in the A3/A4 engine-proximity probe.

## 7. Honesty rules (fixed)

1. Arms are compared side by side against A0 and S_att, with 0.80 shown only as a reference
   level (section 4.3). No arm is described as passing or failing a numeric target.
2. Every arm in section 5 is reported whatever its outcome, including failures and unfinished runs
   (reported as such).
3. No arm is re-run with new settings after withheld-condition results are seen. Tuning happens on
   `val` only and is logged in `<arm>/tuning_log.json`.
4. Smoke tests run on tiny subsets (a few percent of games, 1 to 2 epochs) to catch crashes. Their
   metrics are written to `extra_arms_smoke/` and are not read, reported or used for any decision.
5. A bug fix that changes no setting (a crash, a wrong path) may re-run an arm; it is logged as a
   deviation in section 9 and in the results file.
6. Results that are not ready by the Chapter 4 draft (2026-09-21) are reported as pending, not
   dropped.

## 8. Compute placement and schedule

- CPU: A0, A0g, A1, A2 (32 cores; LightGBM `n_jobs=4` as in the anchor).
- GPU: A3 on one idle GPU, A4 on a different idle GPU, chosen by `nvidia-smi` at launch; other
  users' processes are never touched. If no GPU is idle, A3/A4 are reported `paused`.
- Order: A0 first (gate). After `A0_GATE_PASS`: A1 and A2 (CPU), A3 and A4 (GPU) in parallel.
  A4's board-embedding cache is not a result and may start while A0 runs.
- Target: A0 to A3 finished by 2026-09-18; A4 may run longer.

## 9. Deviations

None at the time of the first commit. Append dated entries here only.

- 2026-09-15, before any arm ran or any result existed: section 1 and the section 4.3 / rule 1
  wording were reworded at the captain's request (no numeric target; methods described in the
  order tried, 0.80 shown only as a reference level). Splits, metrics, arms and reporting rules
  are unchanged.
- 2026-09-15, smoke tests: while checking that the smoke outputs and the aggregator table had the
  right structure, the smoke-subset tables (about 3 percent of games, 50 trees, 1 to 2 epochs) were
  displayed. Nothing was decided from them and no setting was changed afterwards; the real runs
  launched at 13:40 (HPC clock) with the settings committed above.
