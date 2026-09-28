# Mixed-cheater player-level pooling: post hoc check

**Post hoc check on synthetic players, not a pre-registered result.** A1's player-level
analysis (chapter3.tex "Player-level aggregation"; `analysis/extra_arm_a1.py` and
`analysis/extra_arm_a1_on_a3g.py`, results in `analysis/a3-player-level-results.md`)
was pre-registered and simulated a cheating player as *k* games drawn from one nonzero-
substitution cell, so every game in that player's hand was a cheating game. This script asks
a question A1 never tested: what happens when only a fraction *f* of a player's games are
actually cheating games, the rest being that player's own clean games? It was written and run
after A1's and A3g's numbers were already known, using no new training, only the two
detectors' already-saved per-game scores. Treat everything below as exploratory, not as a
claim the study committed to in advance.

Script: `analysis/mixed_cheater_pooling.py`. Chart: `analysis/plot_mixed_cheater_pooling.py`.
Raw output: `analysis/mixed_cheater_pooling/results.json`, `analysis/mixed_cheater_pooling/auc_vs_games.png`.

## Method

Reuses A1's own `pools()`, `draw()`, and `summarize()` by import, unchanged
(`analysis/extra_arm_a1.py`); the only new code is `simulate_mixed()`, which builds a
player's hand of *k* games as a mix of two draws instead of one: `m` games from the cheating
cell (band, engine, one nonzero rate) via `A1.draw`, and `k - m` games from that same band's
clean pool, also via `A1.draw`. Clean players are A1's own clean-player draw, unchanged, so a
"mixed cheater" at *f* = 0 would be statistically identical to a clean player, and every *f*
curve is compared against the same clean-player pool. Both draws share A1's kmax-then-prefix
trick (one draw at kmax = 100, every smaller *k* is a prefix of it), so different *k* values
for the same player are not independent redraws, and the same cheat/clean game pools, seed
(20260915), and player counts (500 per cheating cell, 1,000 per clean band) as A1 are used
throughout.

Score sources, exactly A1's and A3g's own:
- **A0**: the v2 LightGBM detector's per-game probability, and **S_att** (the computed
  attention-times-deviation score), both from `A0/scores.npz`, under the game-level
  (non-grouped) split.
- **A3g**: the BiGRU + gated-attention MIL sequence detector's per-game score, from
  `A3g_seed0/scores.npz`, under its own grouped, twin-free split.

Four evaluation groups, A1's own (`seen`, `withheld_band`, `withheld_rate`, `withheld_both`);
**withheld_band is the headline**, per rate breakdown (2/5/20/40/60 percent) is reported for
it, and both mean and max aggregation are reported throughout. Bootstrap: 200-replicate
hierarchical resampling (games resampled within each pool, players redrawn), same as A1; the
95 percent CI on the per-rate breakdown is new (A1's own bootstrap loop only attaches a CI to
the overall AUC).

**Rounding.** `m = max(1, floor(f*k + 0.5))`, never `round()`/`np.round()` (banker's rounding
would silently zero some cells: `round(0.5) == 0`). A cheating player who cheats "some" of the
time cheats at least once, so `m` is never 0 when `f > 0`. This means small *f* at small *k*
is not what it says: at k=1 every *f* rounds up to a fully-cheated hand (`f_eff = 1.0`), and at
k=5, f=0.1 rounds up to `f_eff = 0.2`, not 0.1. The realized `f_eff = m/k` for every (f, k)
pair is in `results.json`'s `mix_grid`, and the tables below use *f* to mean "the intended
value passed to the sweep", not the fraction actually realized at small k.

## f = 1.0 reproduces the published numbers

At f = 1.0 every game is a cheating game, i.e. A1's own player, so this is the sanity check
the task asked for. This script's own `A0`/`A3g` mixed-grid draw uses kmax = 100 uniformly (so
every *f* and *k* share one draw), while A1's own published run used kmax = 20 for A0 and
kmax = 100 for A3g; `np.argpartition`'s first-*k* columns are guaranteed to be the *k* smallest
keys but are not internally sorted, so a kmax = 100 draw's first-20 columns are not
guaranteed to be A1's own kmax = 20 draw's players. Bit-exact agreement is therefore not
expected below k = 20 for A0; the check is whether f = 1.0 falls inside A1's/A3g's own
published CI, not exact equality.

Withheld band, k = 20, mean aggregation:

| detector | this script's f=1.0 | published (A1 / A3g_player_level) | inside published CI? |
|---|---|---|---|
| A0 | 0.817 [0.800, 0.840] | 0.825 [0.803, 0.840] | yes |
| A3g | 0.854 [0.846, 0.879] | 0.862 [0.846, 0.881] | yes |

Every other shared (k, agg) cell (k = 1, 5, 10 for A0; k = 1, 5, 10, 50, 100 for A3g; mean and
max both) agreed to within 0.02 AUC; full table in `results.json`'s `f1_check_vs_published`.

## Headline: withheld-band mean-aggregation AUC, by cheat fraction and games pooled

| f | k=1 | k=5 | k=10 | k=20 | k=50 | k=100 |
|---|---|---|---|---|---|---|
| **A0 (LightGBM)** | | | | | | |
| 0.10 | 0.660 [0.632, 0.663] | 0.568 [0.542, 0.593] | 0.552 [0.514, 0.577] | 0.566 [0.520, 0.601] | 0.601 [0.530, 0.647] | 0.628 [0.542, 0.680] |
| 0.25 | 0.660 [0.632, 0.663] | 0.568 [0.542, 0.593] | 0.639 [0.606, 0.663] | 0.652 [0.614, 0.684] | 0.716 [0.661, 0.750] | 0.745 [0.683, 0.783] |
| 0.50 | 0.660 [0.632, 0.663] | 0.679 [0.660, 0.703] | 0.705 [0.678, 0.726] | 0.746 [0.720, 0.771] | 0.796 [0.759, 0.823] | 0.818 [0.780, 0.847] |
| 0.75 | 0.660 [0.632, 0.663] | 0.719 [0.699, 0.739] | 0.765 [0.746, 0.785] | 0.793 [0.770, 0.815] | 0.830 [0.801, 0.855] | 0.847 [0.815, 0.874] |
| 1.00 | 0.660 [0.632, 0.663] | 0.749 [0.731, 0.767] | 0.791 [0.773, 0.808] | 0.817 [0.800, 0.840] | 0.849 [0.824, 0.871] | 0.865 [0.836, 0.888] |
| **A3g (sequence)** | | | | | | |
| 0.10 | 0.709 [0.685, 0.712] | 0.593 [0.574, 0.622] | 0.570 [0.535, 0.599] | 0.584 [0.557, 0.630] | 0.634 [0.584, 0.688] | 0.679 [0.607, 0.737] |
| 0.25 | 0.709 [0.685, 0.712] | 0.593 [0.574, 0.622] | 0.687 [0.661, 0.713] | 0.699 [0.678, 0.737] | 0.770 [0.734, 0.809] | 0.804 [0.757, 0.839] |
| 0.50 | 0.709 [0.685, 0.712] | 0.735 [0.723, 0.758] | 0.759 [0.744, 0.783] | 0.793 [0.782, 0.823] | 0.836 [0.814, 0.865] | 0.860 [0.832, 0.888] |
| 0.75 | 0.709 [0.685, 0.712] | 0.773 [0.765, 0.794] | 0.811 [0.802, 0.832] | 0.831 [0.822, 0.859] | 0.867 [0.849, 0.893] | 0.890 [0.863, 0.916] |
| 1.00 | 0.709 [0.685, 0.712] | 0.800 [0.791, 0.818] | 0.832 [0.823, 0.851] | 0.854 [0.846, 0.879] | 0.888 [0.871, 0.913] | 0.914 [0.887, 0.936] |

At k = 1 every row is identical within a detector (the rounding rule above: any f > 0 gives a
fully-cheated single game). At k = 5, f = 0.1 (`f_eff = 0.2`) actually reads *worse* than k = 1
for both detectors (A0: 0.568 vs 0.660; A3g: 0.593 vs 0.709): with only one cheat game and four
clean games in the hand, mean-pooling has diluted the one cheating game's signal further than
it helps to have a second data point. The curve only turns around once the absolute number of
cheat games in the hand grows past that dilution, around k = 10-20 for f = 0.1-0.25. This
dip-then-recover shape is a real property of mean pooling under a small, fixed cheat fraction,
not noise: it appears in both detectors and both aggregation methods at low f (max-aggregation
table below shows the same shape, milder).

**Max aggregation** (same headline group), for comparison:

| f | k=1 | k=5 | k=10 | k=20 | k=50 | k=100 |
|---|---|---|---|---|---|---|
| **A0** 0.10 | 0.660 | 0.570 | 0.553 | 0.550 | 0.559 | 0.566 |
| **A0** 0.25 | 0.660 | 0.570 | 0.617 | 0.608 | 0.619 | 0.623 |
| **A0** 0.50 | 0.660 | 0.655 | 0.661 | 0.661 | 0.671 | 0.674 |
| **A0** 0.75 | 0.660 | 0.681 | 0.700 | 0.698 | 0.704 | 0.704 |
| **A0** 1.00 | 0.660 | 0.700 | 0.719 | 0.721 | 0.726 | 0.726 |
| **A3g** 0.10 | 0.709 | 0.604 | 0.582 | 0.589 | 0.626 | 0.634 |
| **A3g** 0.25 | 0.709 | 0.604 | 0.664 | 0.662 | 0.700 | 0.700 |
| **A3g** 0.50 | 0.709 | 0.699 | 0.708 | 0.722 | 0.746 | 0.744 |
| **A3g** 0.75 | 0.709 | 0.724 | 0.744 | 0.752 | 0.773 | 0.772 |
| **A3g** 1.00 | 0.709 | 0.745 | 0.762 | 0.770 | 0.784 | 0.789 |

As A1's own note on max-aggregation already found, max saturates quickly (by k = 10-20) and
sits well below mean at every fraction: a single strong game already sets the max, so adding
more clean games to the mix costs mean pooling far more than it costs max. For a mixed cheater
specifically, this means max is the more robust aggregation of the two whenever f is small and
k is not tiny, exactly the regime where mean pooling's dip is worst.

## Per-rate breakdown: the headline number hides a floor near chance

The pooled withheld-band AUC mixes five substitution rates (2/5/20/40/60 percent) whose
per-game signal is very different, and this gap does not close just because more games are
pooled. At f = 0.5, k = 100 (mean aggregation, withheld band):

| detector | rate 2% | rate 5% | rate 20% | rate 40% | rate 60% |
|---|---|---|---|---|---|
| A0 | 0.545 [0.464, 0.627] | 0.626 [0.551, 0.692] | 0.928 [0.875, 0.946] | 0.994 [0.988, 0.997] | 0.999 [0.998, 1.000] |
| A3g | 0.587 [0.510, 0.664] | 0.724 [0.652, 0.803] | 0.991 [0.980, 0.995] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |

Even with half the player's games truly substituted and 100 games pooled, a player who only
ever cheats at 2 percent substitution is barely distinguishable from a clean player (AUC close
to a coin flip); the pooled 0.82-0.86 headline number is driven almost entirely by the
40-60 percent cells. The same pattern holds at f = 0.75, k = 50 (full table in `results.json`).
This mirrors the per-game weakness the LightGBM detector was already known to have at low
substitution rates (`analysis/anomaly-extra-arms-plan.md`: "at 2 to 5 percent substitution a
game has 1 or 2 engine moves and per-game averages bury them"): mixing in clean games on top
of an already-weak per-game signal compounds the problem rather than being independent of it.

## Which evaluation groups share games with training

A cheat or clean game's evaluation pool can still be leaked into if a "twin" game (same band
and game index, differing only in engine/rate/substitution) sits in the training partition;
this is the twin-leakage issue documented for the v2 corpus (CLAUDE.md, "v2 corpus twin
games"). Fraction of each pool's games with a same-(band, game_index) twin in the detector's
own training split:

| condition | A0 (game-level split) | A3g (grouped, twin-free split) |
|---|---|---|
| seen | 1.00 (cheat and clean) | 0.00 (cheat and clean) |
| **withheld_band (headline)** | **0.00 (cheat and clean)** | **0.00 (cheat and clean)** |
| withheld_rate | 1.00 (cheat and clean) | 0.70 (cheat), 0.00 (clean) |
| withheld_both | 0.00 (cheat and clean) | 0.00 (cheat and clean) |

This exactly matches what chapter3.tex already states in prose for the game-level scores
("only the two withheld-band groups are free of related games") and for A3g ("the seen-band
group is also free of related games in that run, whereas the withheld-rate group is not").
**withheld_band, the headline group used above, is twin-free for both detectors**, so the
mixed-cheater numbers above are not inflated by training leakage. withheld_rate's cheat games
are NOT twin-free for A3g despite its split being otherwise grouped: a rate-10 game's twin at
the same (band, index) in a different rate cell of the same (non-heldout) band still lands
mostly in train under that band's 70/15/15 grouped split, so withheld_rate results (reported
in `results.json` for completeness, not used as headline) should be read with that caveat.

One further, smaller optimism source specific to the mixed design: a mixed player's hand can
contain a cheat-cell game and its own r00 clean twin (since the filler pool and the cheat cell
draw from the same band independently). At f = 0.5, k = 100 with 2,000-game withheld-band
pools this affects roughly one pair per player in expectation; not corrected for, and small
enough not to change any conclusion here.

## Does matching the all-cheating separation need about 1/f² as many games?

The rough intuition: if mean-pooling shifts a player's score by an amount proportional to the
cheat fraction, and averaging k games shrinks per-game noise by √k, then matching the
separation an all-cheating player reaches at k=20 should need roughly k = 20/f² games at
fraction f. This splits into two separate claims, tested against `f_eff` (the realized
fraction from the rounding rule above) at withheld band, mean aggregation:

**Claim A (fixed k, does the separation scale with f_eff?)** Using d' = √2·Φ⁻¹(AUC) (the
equal-variance Gaussian separation implied by an AUC) as a stand-in for the score-space
separation, d'(f, k) / d'(1, k) at fixed k should equal f_eff, because mean pooling is linear
in the per-game scores. This holds well:

| f | k=5 | k=10 | k=20 | k=50 | k=100 |
|---|---|---|---|---|---|
| 0.50 (A0) | ratio 0.69 vs predicted 0.60 | 0.66 vs 0.50 | 0.73 vs 0.50 | 0.80 vs 0.50 | 0.82 vs 0.50 |
| 0.75 (A0) | ratio 0.86 vs predicted 0.80 | 0.89 vs 0.80 | 0.90 vs 0.75 | 0.93 vs 0.76 | 0.93 vs 0.75 |

The observed ratio consistently runs a bit *above* the naive f_eff prediction rather than
matching it exactly (full table, both detectors, in `results.json`'s `scaling_check`), which
is the direction you would expect if the clean-game filler score is not quite as clean, on
average, as the reference clean-player pool (see the twin overlap note above) or if the
cheat-cell score distribution has a heavier tail than Gaussian. The claim is directionally
right but not a precise match.

**Claim B (games needed to match the f=1, k=20 target).** This additionally assumes the
target's own separation keeps growing like √k out to whatever k the prediction calls for. It
does not hold as cleanly:

| f | naive k = 20/f_eff² | smallest k in {1..100} that actually reaches the target | reached in grid? |
|---|---|---|---|
| 0.10 | 2,000 | -- | no |
| 0.25 | 320 | -- | no |
| 0.50 | 80 | 100 | yes, but later than predicted |
| 0.75 | 36 | 50 | yes, but later than predicted |
| 1.00 | 20 | 20 | trivially, by construction |

(Same pattern for both A0 and A3g.) At f = 0.5 and f = 0.75 the naive rule under-predicts the
games actually needed by about 25-40 percent: the target curve (f = 1.0) is already flattening
by k = 20-50 rather than continuing to climb like √k (visible in the chart and in
`analysis/a3-player-level-results.md`'s own note that "mean aggregation keeps climbing through
k=100... no plateau reached yet"), so a smaller-f curve chasing that target needs to close a
gap against a moving, slowing goalpost. At f = 0.1 and f = 0.25 the naive target (2,000 and
320 games) is so far outside the tested range, and so dominated by the near-chance 2/5 percent
rate cells (see above), that no exercise inside k ≤ 100 can confirm or refute it; take those
two rows as "clearly not reached by k=100", not as a tested prediction.

**Where the rule holds:** the direction (more of a player's games truly cheating needs
quadratically fewer games to reach the same separation) and the fixed-k linear-in-f_eff shape
of Claim A. **Where it fails:** the games-needed magnitude in Claim B, because the reference
curve itself saturates rather than growing indefinitely as √k, and because pooling anything
across substitution rates means the "average" separation is set largely by the weak 2-5
percent cells, not by the strong 40-60 percent ones the 1/f² intuition implicitly assumes are
representative.

## Key takeaway, in plain language

A synthetic player who cheats in every game reaches AUC 0.82-0.86 (LightGBM / sequence
detector) at 20 pooled games on withheld Maia bands (the pre-registered A1/A3g result). A
synthetic player who cheats in only **half** their games needs roughly **100 games**, not the
naively expected 80, to reach that same separation, because the games where they don't cheat
dilute the average, and because the curve for a fully-cheating player itself is already
flattening out by then. A player who cheats in only a **quarter** of their games does not reach
that separation anywhere in the tested range (up to 100 games); a player cheating at even lower
rates within their cheating games (2 or 5 percent substitution) stays close to a coin flip even
when half their games are substituted and 100 games are pooled. Player-level pooling, as
tested here, is a strong tool against a player who cheats consistently and heavily, and a weak
one against a player who cheats occasionally or lightly.

## Reproducing

```
cd analysis
python3 mixed_cheater_pooling.py \
  --rating-cheap ~/Bacsain/thesis2/analysis/v2_features/rating_cheap.npz \
  --a0-scores ~/Bacsain/thesis2/analysis/extra_arms/A0/scores.npz \
  --a3g-scores ~/Bacsain/thesis2/analysis/extra_arms/A3g_seed0/scores.npz
python3 plot_mixed_cheater_pooling.py
```

`np.load` on an `.npz` is lazy per array, so pointing this at the full HPC files (run from
`<user>@<hpc-host>`, read-only) reads only the handful of per-game arrays it needs and never
touches the 19.7-million-row per-ply arrays those same files also carry; no GPU, training run,
or write access to the HPC's own `analysis/` tree is involved. Runtime on a laptop CPU: about
42 minutes for the full sweep (both detectors, 4 evaluation groups, 5 cheat fractions, 6 k
values, 200-replicate bootstrap).
