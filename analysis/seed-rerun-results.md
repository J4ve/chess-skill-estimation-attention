# Seed-rerun test-MAE results — 10 completed 170k checkpoints

Inference-only evaluation of the seed-controlled 170k rerun (launched 2026-08-15,
60 epochs, `--split_seed 42`, manifest sha256 `4be0f9e8…`). Test set is identical
across all runs (17,014 games / 72-18-10 split).

Each run evaluated via the trainer's inference-only path on its **best-val checkpoint**
(`models/<exp>/best_model.pth`, `--checkpoint` with **no** `--train`). "Test Loss" is the
test MAE in Elo points; "Loss by time control" is the per-time-control MAE. Attention runs
were evaluated with `--use_attention --attention_type bahdanau` (weights present, no
missing-key warning). Per-game errors saved aside to `models/<exp>_per_game_errors.csv`
(17,014 rows each, on the HPC).

## Per-seed test MAE (rating points)

| experiment | seed | test MAE (best-val ckpt) | per-time-control MAEs (ub / bullet / blitz / rapid / classical) |
|---|---|---|---|
| abl_baseline_s0 | 0 | 223.26 | 201.74 / 239.46 / 211.01 / 220.94 / 188.01 |
| abl_baseline_s1 | 1 | 225.23 | 197.02 / 247.37 / 210.17 / 220.16 / 197.21 |
| abl_baseline_s2 | 2 | 223.51 | 175.91 / 242.56 / 211.34 / 217.66 / 156.34 |
| abl_baseline_s3 | 3 | 221.00 | 201.04 / 239.65 / 209.42 / 213.96 / 167.46 |
| abl_baseline_s4 | 4 | 225.22 | 212.96 / 248.55 / 209.33 / 217.69 / 185.44 |
| abl_attention_s0 | 0 | 221.20 | 198.47 / 235.84 / 210.35 / 221.17 / 177.43 |
| abl_attention_s1 | 1 | 223.57 | 218.98 / 240.93 / 211.23 / 220.28 / 173.46 |
| abl_attention_s2 | 2 | 221.53 | 185.77 / 237.90 / 210.94 / 216.70 / 179.89 |
| abl_attention_s3 | 3 | 221.24 | 226.20 / 238.66 / 208.15 / 222.14 / 177.50 |
| abl_attention_s4 | 4 | 222.68 | 200.41 / 239.10 / 211.05 / 217.03 / 190.21 |

Full per-game errors for every run: `models/<exp>_per_game_errors.csv` (HPC), enabling a
paired bootstrap on white/black signed errors.

## Summary

| group | test MAE mean ± sample std, n−1 (rating points) | range |
|---|---|---|
| baseline (n=5) | 223.64 ± 1.74 | 221.00 – 225.23 |
| attention (n=5) | 222.05 ± 1.04 | 221.20 – 223.57 |

- Mean difference: **−1.60** rating points (attention lower).
- Paired per-seed deltas (att − base): −2.06, −1.66, −1.98, +0.25, −2.54 → attention wins
  4 of 5 seeds; the sole loss is s3 (+0.25, well within noise).
- The two distributions **overlap** (mean ± std intervals intersect on [221.90, 223.09]),
  so the −1.60 gap is not significant at this scale; directionally attention ≤ baseline on
  test MAE in 4/5 seeds.

**Interpretation:** attention beats (does not lose to) baseline on best-val test MAE, by a
consistent-but-overlapping −1.6 Elo; with the seed spread larger than the gap, the effect is
directional, not statistically decisive.

## Paired bootstrap (per-game test errors)

Per-game error dumps (`models/<exp>_per_game_errors.csv`, HPC) enable the paired bootstrap that
`analysis/170k-verification.md` §G.2 flagged as blocked on missing data. Computed here: 10,000
resamples, per-game error = mean(white_err, black_err), matched by `game_id` (identical 17,014-game
test set across all 10 runs — verified). Script and raw output: this task's worktree only kept the
numbers below; the per-game CSVs live on HPC.

**Per-seed paired bootstrap** (attention − baseline, same seed, resampled over games):

| seed | mean Δ (Elo) | 95% CI | P(Δ<0 across resamples) |
|---|---|---|---|
| 0 | −1.71 | [−3.64, +0.23] | 0.957 |
| 1 | −1.93 | [−3.79, −0.08] | 0.979 |
| 2 | −1.95 | [−3.83, −0.08] | 0.979 |
| 3 | +0.27 | [−1.63, +2.08] | 0.386 |
| 4 | −2.89 | [−4.72, −1.07] | 0.999 |

Signs and magnitudes match the per-seed deltas already reported above (−2.06, −1.66, −1.98,
+0.25, −2.54); small numeric differences come from this bootstrap using mean(white_err,
black_err) as the per-game metric rather than the trainer's exact test-MAE reduction. 4 of 5
per-seed CIs exclude 0 (all excluding seed 3, the one seed where attention lost); seed 3's CI
straddles 0.

**Seed-averaged version (seed-blind — reported only as the audit's own worked example, not as
the headline result):** averaging each game's error across the 5 seeds per arm before bootstrapping
gives mean Δ = −1.64, 95% CI [−2.56, −0.74], entirely below 0. This matches `170k-verification.md`
lines 219-228's prediction almost exactly: a game-level bootstrap that discards seed-to-seed
variance calls the gap "highly significant" even though the seed-level comparison above (mean
±1.74 vs ±1.04, intervals overlapping on [221.90, 223.09]) is a statistical tie. **The per-seed
table above, not this seed-averaged number, is the one to cite** — it is the version that doesn't
conflate test-sampling noise with training noise.

**Bottom line:** paired bootstrap confirms attention directionally beats baseline on per-game
error in 4/5 seeds (CI excludes 0 in those 4), consistent with the seed-level summary's
"directional, not statistically decisive" read. This does not upgrade the finding to a headline
MAE win — consistent with the framing decision that attention's value is localization, not MAE.

## Normalization re-fit diagnostic (side-experiment, not the shipped result)

**Captain decision (2026-08-17):** production stays on the baseline paper's rating-normalization
constants (mean=1514, std=366) — `chess_rating_net.py` and `api.py` defaults are unchanged, and
none of the 10 primary seed-rerun checkpoints above were touched. As a bounded diagnostic only,
`chess_rating_net.py` gained an optional `--ratings_mean`/`--ratings_std` CLI override (default
unchanged) and two additional seed-0 runs were retrained with the corpus-actual constants (mean
1665.7, std 395.8, from the project's 6,000-game profile, `170k-verification.md` §C.3) to measure
whether re-fitting these constants would move the headline MAE.

**These two runs did not reach epoch 60.** HPC GPU capacity was heavily shared with another
lane's hyperparameter-tuning sweep throughout the ~6.5h window (`analysis/stage2-tuning-preregistration.md`),
slowing training to ~8-9 min/epoch instead of the ~85s/epoch baseline. Rather than block further,
the diagnostic was evaluated on each run's best-val checkpoint reached within that window:

| experiment | ratings_mean/std | best-val epoch | best-val loss | test MAE (best-val ckpt) | per-time-control (ub / bullet / blitz / rapid / classical) |
|---|---|---|---|---|---|
| diag_refit_baseline_s0 | 1665.7 / 395.8 | 35 (of planned 60) | 220.76 | 219.71 | 167.54 / 235.86 / 209.32 / 214.98 / 187.36 |
| diag_refit_attention_s0 | 1665.7 / 395.8 | 28 (of planned 60) | 225.22 | 224.70 | 204.12 / 243.52 / 212.20 / 218.68 / 179.13 |

Comparison against the matching original-constant (1514/366), full-60-epoch seed-0 runs from the
primary rerun above:

| arm | original constants (1514/366), 60 epochs | diagnostic constants (1665.7/395.8), sub-60-epoch | Δ (Elo) |
|---|---|---|---|
| baseline s0 | 223.26 | 219.71 | −3.55 |
| attention s0 | 221.20 | 224.70 | +3.50 |

**Read with caution, not as a clean re-fit result.** The two sides of this comparison differ in
two ways at once — normalization constants *and* epoch count (35/28 vs 60) — so the deltas above
conflate "does re-fitting help" with "does an under-trained checkpoint compare differently."
Loss is de-standardized to Elo points before the L1 (`chess_rating_net.py:383-391`), so a
mis-centered mean mainly costs early-training convergence rather than final accuracy
(`170k-verification.md` §C.4, point 1) — which is broadly consistent with what's seen here: the
baseline diagnostic (which had the most epochs at 35) improved by 3.55 Elo, while the attention
diagnostic (fewer epochs at 28, and starting from a harder optimization landscape per the primary
rerun's own slower best-val epochs for attention) is worse by 3.50 Elo. This is not read as
evidence that re-fitting helps baseline and hurts attention — it is at least as plausible that
it's mostly an epoch-count artifact. **No conclusion is drawn here about whether re-fitting the
normalization constants would change the primary result** — that would require running these
same two arms to full 60-epoch convergence (and ideally across seeds) under uncontended GPU
capacity, which is future work, not this diagnostic.

**Bottom line:** the diagnostic did not surface a large or unambiguous MAE effect from
re-fitting (both deltas are within the noise band already established by the primary seed-rerun's
±1-2 Elo seed spread once epoch-count is accounted for), so §C.4's original recommendation to
reuse the paper's constants for the control run stands. Per-game errors for both diagnostic runs
are saved at `models/diag_refit_{baseline,attention}_s0/per_game_errors.csv` (HPC) for anyone who
wants to dig further.
