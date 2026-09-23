# A3g player-level aggregation results

Player-level check for the A3g sequence detector (BiGRU + gated-attention MIL, grouped/twin-free
split), using exactly A1's method (`analysis/scripts/extra_arm_a1.py`, imported unchanged) pointed
at A3g's own per-game scores instead of S_att/A0. Script:
`analysis/scripts/extra_arm_a1_on_a3g.py`. Raw output: HPC
`~/<workdir>/analysis/extra_arms/A3g_player_level/results.json`, copied here as
`analysis/extra_arms_results/A3g_player_level/results.json`.

- Question: "is this player cheating" (a k-game player drawn without replacement from one
  evaluation cell), not "is this game substituted" - a different question from A3g's own
  game-level AUC.
- Score source: `analysis/extra_arms/A3g_seed0/scores.npz`, key `score` (A3g's per-game sigmoid
  probability), pooled under the `split` field saved in that same file (the grouped, twin-free
  split A3g was itself scored on).
- Unchanged from A1: 500 cheater players per (band, engine, rate) cell, 1,000 clean players per
  band (deduplicated by band + game index, since the lc0 and stockfish16 rate-0 cells are
  identical games), seed 20260915 (NumPy `default_rng`), 200-replicate hierarchical bootstrap
  (resample games within each pool, redraw players, recompute AUC), same four evaluation groups
  (`seen`, `withheld_band`, `withheld_rate`, `withheld_both`), same hard-negative-vs-clean-players
  strength check, mean and max aggregation of the k per-game scores.
- k grid: {1, 5, 10, 20} per the original plan, plus **{50, 100} added 2026-09-17 by captain
  request** (same sampling, same seed, same bootstrap; the aggregation method itself is unchanged)
  to show where the AUC curve saturates.
- Pool-size check: the run's `k_cap_notes` field is empty. Every `seen`/`withheld_rate` cheater
  cell has 300 test-split games and every `withheld_band`/`withheld_both` cheater cell has 2,000
  games (the whole withheld band or rate is out-of-sample), so no pool needed capping below the
  requested k anywhere in this grid; k=100 draws from a pool of at least 300 in every group.

## Mean-aggregated player AUC (95% hierarchical-bootstrap CI)

| k | seen | withheld_band | withheld_rate | withheld_both |
|---|---|---|---|---|
| 1 | 0.693 [0.683, 0.707] | 0.696 [0.686, 0.712] | 0.594 [0.582, 0.617] | 0.601 [0.590, 0.629] |
| 5 | 0.794 [0.780, 0.812] | 0.806 [0.791, 0.818] | 0.709 [0.675, 0.729] | 0.728 [0.701, 0.754] |
| 10 | 0.827 [0.808, 0.844] | 0.837 [0.822, 0.852] | 0.779 [0.739, 0.804] | 0.804 [0.773, 0.826] |
| 20 | 0.848 [0.827, 0.873] | 0.862 [0.846, 0.881] | 0.859 [0.813, 0.880] | 0.884 [0.852, 0.905] |
| 50 | 0.878 [0.848, 0.906] | 0.893 [0.870, 0.914] | 0.951 [0.910, 0.963] | 0.967 [0.944, 0.978] |
| 100 | 0.902 [0.860, 0.931] | 0.916 [0.888, 0.938] | 0.987 [0.957, 0.993] | 0.994 [0.982, 0.997] |

## Max-aggregated player AUC (95% hierarchical-bootstrap CI)

| k | seen | withheld_band | withheld_rate | withheld_both |
|---|---|---|---|---|
| 1 | 0.693 [0.683, 0.707] | 0.696 [0.686, 0.712] | 0.594 [0.582, 0.617] | 0.601 [0.590, 0.629] |
| 5 | 0.733 [0.715, 0.764] | 0.753 [0.729, 0.766] | 0.653 [0.619, 0.675] | 0.672 [0.638, 0.694] |
| 10 | 0.741 [0.717, 0.775] | 0.764 [0.740, 0.782] | 0.661 [0.621, 0.693] | 0.685 [0.648, 0.716] |
| 20 | 0.747 [0.714, 0.789] | 0.772 [0.743, 0.796] | 0.658 [0.619, 0.709] | 0.717 [0.660, 0.747] |
| 50 | 0.761 [0.703, 0.817] | 0.784 [0.737, 0.820] | 0.678 [0.623, 0.754] | 0.753 [0.681, 0.803] |
| 100 | 0.766 [0.685, 0.848] | 0.786 [0.733, 0.841] | 0.683 [0.619, 0.801] | 0.776 [0.695, 0.856] |

Mean aggregation keeps climbing through k=100 in every group (no plateau reached yet at this k);
max aggregation saturates far earlier (by about k=10-20) and stays flat, since a single high-
scoring game in the player's hand already sets the max regardless of how many more clean-looking
games are added.

## Hard-negative players vs clean players (strength check, mean-aggregated, no bootstrap CI)

| k | seen | withheld_band |
|---|---|---|
| 1 | 0.533 | 0.546 |
| 5 | 0.554 | 0.596 |
| 10 | 0.568 | 0.632 |
| 20 | 0.599 | 0.658 |
| 50 | 0.647 | 0.746 |
| 100 | 0.706 | 0.820 |

Hard-negative players (strength-mismatch clean cells) look progressively more like cheater players
as k grows, consistent with A3g's own game-level hard-negative check (AUC near chance at the game
level) accumulating a detectable signal once many games are pooled per player.

## Headline

- k=20, withheld_band (the number the captain asked to see): mean AUC **0.862** [0.846, 0.881].
  For comparison, A1's original LightGBM/A0 check reached mean AUC 0.825 [0.803, 0.840] at k=20
  on withheld bands (`analysis/extra_arms_results/A1/results.json`, `conditions.withheld_band.A0`)
  - A3g's better game-level detector (0.752 [0.747, 0.756] vs A0's 0.685 [0.681, 0.690] game-level
  AUC on withheld bands, from `analysis/extra_arms_results/A3g_seed0/results.json`
  `game_level.conditions.withheld_band` and `analysis/extra_arms_results/A0/results.json`
  `variants.A0.conditions.withheld_band`) carries through to a better player-level check.
- k=1, withheld_band, for comparison: mean AUC 0.696 [0.686, 0.712] (this is just A3g's ordinary
  single-game AUC on that group).
- The curve has not saturated by k=100 under mean aggregation: withheld_rate and withheld_both
  both exceed 0.95 at k=50 and approach 0.99 at k=100, driven by rate-10 substitution being a
  weaker per-game signal that averages out cleanly once enough games are pooled.
