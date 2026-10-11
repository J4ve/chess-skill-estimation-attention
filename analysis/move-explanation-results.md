# Move-level explanation of the rating estimate

Written 2026-10-11 for the pre-oral defense on 2026-10-15, at the thesis consultant's
request: why the model's rating estimate rises or falls at particular moves,
interpreted through engine evaluation, position, and clock. Method, sample, and
numbers below; case-study figures and a small per-ply JSON for each are in
`analysis/figures/` and `analysis/move-explanation-data/`.

**The model never sees an engine.** It is trained only on board positions, moves,
and clock times. Every Stockfish number in this note, in the figures, and in the
JSON files is an external yardstick used here only to interpret the model's
estimate after the fact. The model cannot and does not use it.

## Method

**"Live" means the prefix-recompute estimate the prototype actually shows**, not
the offline bidirectional per-ply output. The served model's BiLSTM is not
incremental (see `src/live.py`): scoring ply *t* live means re-running the whole
model on plies `1..t` and keeping only that final ply's output. This note recomputes
that quantity directly, with the same forward-pass code path (`analysis/move_explanation_curves.py`),
one call per prefix length, against the `src/` at commit `8c7f23f` (the HEAD this
analysis branched from; a later commit, `e1f8fec`, added a `model` selector to
`src/api.py` that lets a request choose `mirror_arm_a_gapweight` instead of the
default checkpoint, but does not change either model's own forward pass). Before
trusting the recompute loop at scale, the script reconstructed PGN prefixes for 3
sampled games at several ply checkpoints and called the prototype's own
`src.api._run_inference` (the default-model code path; both checkpoints share the
same `_run_inference` logic, only the loaded weights differ) on each one: the two
agreed to within 0.005 rating points at every checkpoint tested
(`analysis/move-explanation-data/curves-equivalence-check.json`), which is
floating-point rounding, not a different computation. The live curve's value at the
final ply is, by construction, the same forward pass as the offline full-game pass,
so the case-study figures' attention weight (a property of one forward pass) is read
from that pass without ambiguity.

**Two checkpoints.** The served model (`attn_tuned`, sha256 `ff6370e4...6124275a0`,
matches the hash published in the top-level README) has nearly identical White and
Black estimates within a game (median final-ply gap 0.72 rating points across the
3,000 sampled games here, consistent with the project's documented 0.71-point figure
on the full 255,000-game test set). Per the task brief, `mirror_arm_a_gapweight`
(same architecture, gap-weighted training loss, sha256 `97ffa6e7...ebcfd77c0481`,
passed the pre-registered criterion (a) accuracy check in
`analysis/mirror-retrain-results.md`, and is now also selectable in the served
prototype as of `e1f8fec`, alongside, not instead of, the default checkpoint) is run
for comparison, since its sides do separate (median final-ply gap 16.2 rating points
here). Both checkpoints were loaded by building `ChessEloPredictor` from each
checkpoint's own stored `params` (matching `src/api.py`'s own loading logic) and
asserting zero missing state-dict keys outside the parameter-free
`anomaly_detector` module; neither checkpoint's weights were modified.

**Engine evaluation.** Stockfish 16, depth 17, 1.0-second-per-position cap, single
thread, deterministic hash size, one `analyse` call per position (not per move):
evaluating positions `0..T` of a `T`-ply game and taking consecutive differences
gives both the before/after evaluation and the centipawn loss of each move,
without a second engine call
(`analysis/move_explanation_engine.py`). Mate scores are capped at 10,000 cp before
differencing; centipawn loss is capped at 2,000 cp per move. The depth actually
reached (recorded per position for exactly this reason) hit the full depth 17 at
188,054 of 188,059 scored positions (99.997%); the remaining 5 stopped one ply
short at depth 16 under the 1.0-second cap, so "depth 17" in every figure caption
is accurate for essentially the whole sample.

## Sample

A fixed, seeded sample of 3,000 games from the frozen 255,000-game TEST partition
(`sample_seed=20261011`, drawn with `analysis/sample_test_games.py` against
`data/split_manifest_2p55M_seed42.json`, `manifest_sha256=dee7d117...bd9ed2b3b`,
matching the value recorded in `analysis/mirror-retrain-preregistration.md`).
Games are truncated at 100 plies, the same cap the served prototype itself uses
(`api.MAX_PLIES`). 2,992 of the 3,000 sampled games yielded at least 3 plies of
usable data (the rest were excluded only by that minimum-length filter, not by any
engine or model failure: the Stockfish pass completed all 3,000 games with zero
errors). After dropping each game's first ply (no earlier estimate to diff
against), the per-ply dataset is 185,058 rows; 182,066 of those also have a usable
time-spent figure (see Limitations). Full reproduction commands are in
`analysis/run_move_explanation.sh`; raw per-game curves and the per-ply CSV used
for the regression below are not committed (same convention as this repository's
other large per-game CSVs), but are reproducible from the committed scripts, the
committed sample, and the HPC run log quoted in
`analysis/move-explanation-data/engine-run-meta.json`.

## What does the live estimate's change actually track?

For every ply after the first, the change in the live estimate from the previous
ply was regressed on: the ply band (six bands, 1-10 through 81-100), the game's
time-control bucket, who moved, the square-root of that move's Stockfish centipawn
loss, the signed White-POV evaluation change caused by the move, and the capped
time spent on the move. **All three nested models (ply/time-control/mover only;
plus move quality; plus time spent) are fit on the same 182,066-row subset**, the
one with a usable time-spent figure (see Limitations), because R-squared is only
comparable across nested models fit on identical rows: fitting the time-spent
model on a larger row set than the other two would make its R-squared partly
reflect the different sample rather than the added predictor, up to and including
producing a non-monotonic sequence where adding a predictor appears to lower
R-squared. The 185,058-row full-scored sample
(2,992 extra rows, exactly the dropped ply-2 of each game) is used only for a
move-quality-only check reported in `analysis/move-explanation-data/regression-results.json`'s
`scored_sample` block, which agrees with the headline below. Confidence intervals
are a 1,000-resample **cluster bootstrap over games**, not over plies: plies
within a game are not independent, so a per-ply bootstrap or per-ply OLS standard
error would be far too narrow given roughly 62 plies per game. Four response
variables are reported per checkpoint: the mover's own estimate change, the
opponent's, the mean of both, and the own-minus-opponent difference (the one
response where a per-side effect, if present, would show up most directly).

**Headline: the relationship is real but small, and time spent tracks the live
estimate somewhat more than engine-measured move quality does.** Adding
centipawn-loss and signed-eval-change to the ply/time-control/mover baseline moves
R-squared by 0.0002 to 0.0012 across every response and both checkpoints; adding
time spent on top moves it by a further 0.0066 to 0.0096 for the own/opponent/mean
responses, roughly 8 to 12 times the move-quality contribution on the same rows.
None of this explains much of the live estimate's ply-to-ply movement in absolute
terms (R-squared stays under 0.025 everywhere except the one response discussed
below), which is itself the honest finding: most of what moves the live curve from
one ply to the next is not captured by engine-measured move quality, time spent,
or coarse game phase.

| Checkpoint | Response | R2, ply+TC+mover only | R2, + move quality | R2, + time spent |
| --- | --- | --- | --- | --- |
| served | own | 0.0054 | 0.0064 | 0.0148 |
| served | opponent | 0.0054 | 0.0063 | 0.0148 |
| served | mean | 0.0054 | 0.0064 | 0.0148 |
| served | own minus opponent | 0.0042 | 0.0044 | 0.0045 |
| mirror_arm_a | own | 0.0123 | 0.0135 | 0.0231 |
| mirror_arm_a | opponent | 0.0140 | 0.0145 | 0.0210 |
| mirror_arm_a | mean | 0.0051 | 0.0060 | 0.0145 |
| mirror_arm_a | own minus opponent | 0.1372 | 0.1377 | 0.1389 |

All on the same 182,066-row subset; full coefficients and CIs for every term,
including this table's, are in
`analysis/move-explanation-data/regression-results.json`
(`by_arm.<checkpoint>.<response>.timed_sample_nested`).

**Costlier moves do move the estimate down, by a small but statistically
non-zero amount, and for the served model that movement is shared almost equally
by both sides.** Holding ply band, time control, and mover fixed, the
`sqrt(centipawn loss)` coefficient is about -0.188 (95% CI roughly [-0.21, -0.17])
in the served model's own, opponent, and mean estimate changes alike (the three
agree to three decimal places): a concrete 400-centipawn blunder
(`sqrt(400) = 20`) is associated with a fall of about 3.8 rating points, against a
per-ply standard deviation of about 55 points for the same response. The
own-minus-opponent version of the same coefficient is -0.00004 (95% CI
[-0.00013, 0.00007], crossing zero): centipawn loss has **no detectable per-side
effect at all** for the served model on this response. A costly move moves the
whole game-level estimate, not one player's estimate
relative to the other's, exactly what the served model's near-equal White/Black
estimates would predict.

**`mirror_arm_a_gapweight`, whose sides do separate, shows a real per-side
effect that the served model does not.** Its own-minus-opponent coefficient on
`sqrt(centipawn loss)` is -0.075 (95% CI [-0.11, -0.05], clearing zero): the same
400-centipawn blunder is associated with the mover's own estimate falling by about
1.5 points relative to the opponent's, in the expected direction, where the served
model's own-minus-opponent response shows nothing.

**Time spent on a move tracks the live estimate somewhat more than move quality
does, in the opposite direction from what intuition about "careful play" might
suggest, and unlike Stockfish, the clock is a real model input.** Holding the
other covariates fixed, more time spent (capped at 120 seconds) on a move is
associated with a *fall* in the own, opponent, and mean live estimate for both
checkpoints, roughly -0.64 to -0.77 rating points per additional capped second
(every CI excludes zero); ten extra seconds is about -6.6 to -7.7 points, almost
twice the 400-centipawn-blunder comparison above. Because the model's clock input
(`clockTime`) is the one yardstick in this note it can actually see, unlike
Stockfish's evaluation, this is the more direct candidate explanation of the
three, though still only correlational: one plausible reading, not tested further
here, is that harder or more unclear positions both take longer to think about and
coincide with lower estimates for reasons unrelated to the thinking time itself.

**Robustness: the two headline effects hold, and slightly strengthen, when
decided positions are excluded, and when the opening is excluded entirely.**
Restricting to the 163,886 rows with |evaluation before the move| under 700
centipawns (88.6% of the scored rows), the served model's mean-response
`sqrt(centipawn loss)` coefficient moves from -0.107 [-0.129, -0.085] to -0.137
[-0.167, -0.100] (on the full 185,058-row scored sample, not the 182,066-row
timed one, so not directly comparable to the -0.188 above), and `mirror_arm_a`'s
own-minus-opponent coefficient moves from -0.083 [-0.114, -0.052] to -0.125
[-0.156, -0.091]. Full numbers in
`analysis/move-explanation-data/robustness-decided-subset.json`. Separately,
restricting to ply 11 and later (158,266 rows, excluding the opening entirely
rather than only controlling for it) lowers every nuisance-only R-squared, most
sharply the served model's (0.0054 to 0.0010 for own/opponent/mean), but leaves
move quality's own small marginal contribution essentially unchanged (delta-R2
0.0002 to 0.0008 across responses and checkpoints): the opening's volatility
inflates the ply-band baseline's own explanatory power, not move quality's
contribution on top of it.
(`analysis/move-explanation-data/regression-results.json`'s `ply_gt_10_only`
block.)

**Game phase**, used here as the ply band plus a material-based opening/
middlegame/endgame split in the engine pass, is folded into the baseline model
above rather than treated as a free-standing predictor: it is confounded with ply
by construction, and the baseline model's own small R-squared values (0.004 to
0.014 across responses, before move quality or time are added) already show that
knowing only the ply band, time control, and who moved explains little of the
estimate's ply-to-ply movement on its own.

**Why `mirror_arm_a`'s own-minus-opponent nuisance R-squared (0.137) is so much
larger than every other cell in the table** is itself worth reporting rather than
leaving as an unexplained outlier: two covariates in that nuisance-only model
carry essentially all of it. The `tc_bucket=ultrabullet` coefficient is +100.8
(95% CI [85.1, 115.0]), a very large shift confined to a very small, unusual
sample (1,666 of 185,058 scored rows, 34 of 2,992 games, mean own-minus-opponent
change about +105 per row and consistent across those 34 games, not one outlier
game). The `mover=white` coefficient is a smaller but still clearly non-zero
-1.33 (95% CI [-1.40, -1.26]): arm A's own-minus-opponent change averages about
1.3 points lower on White's moves than on Black's, holding ply and time control
fixed, a genuine but small mover-parity effect. Both sit in the nuisance model
the move-quality and time-spent terms are added on top of, which is why they do
not change the headline comparisons above, but they are the honest answer to
"why is this one cell so much bigger."

## A caveat the figures below depend on: attention is partly front-loaded by construction

The served model's attention weight, averaged over 3,000 games, is 0.1486 at ply 1
and 0.00016 by ply 90, a 909-fold gap (`src/attention.py`'s `forward_causal`:
`weights[b, t, :]` sums to 1 over valid keys `s <= t`, confirmed by reading the
code directly for this note). `api.py`'s `attention_weights = ...mean(dim=1)` then
averages each key's received weight over **all** `T` queries, including the
causally-masked zero entries for queries before that key, so an early ply is a
valid attention key for nearly every later query while a late ply is valid for
almost none, regardless of what the model actually learned. This mechanical
dilution alone, simulated as a uniform softmax over each query's valid keys at the
sampled games' actual lengths, already produces a 93-fold ply-1-to-ply-90 gap
(0.0925 vs 0.00099). **The observed 909-fold gap is about 10 times steeper than
dilution alone predicts, so the learned weights are not just mechanically
front-loaded, they are also genuinely more concentrated on the first couple of
plies than a uniform policy would be** (ply 1 observed 0.1486 vs 0.0925 uniform;
ply 2 observed 0.1419 vs 0.0685 uniform) **and correspondingly less attentive in
the middle and late game than dilution alone would predict** (ply 90 observed
0.00016 vs 0.00099 uniform, about 6-fold lower). Put plainly: part of the
front-loading is a construction artifact of the mean-over-all-queries formula,
and part of it is what the model actually learned; this note cannot cleanly
separate the two from the attention weights alone, so it treats both as reasons
not to read the ply 1-9 region as a "this move mattered" signal.
`src/critical_moves.py` already works around the same concern by excluding plies
before `DEFAULT_MIN_PLY` (10) from its attention-ranked critical-move list; the
case-study figures below shade that same region and the written interpretation
does not read anything into the shaded ply 1-9 attention values. It does read the
unshaded region (ply 10 and later), where the observed weights sit close to, and
by ply 20 onward below, the dilution-only baseline rather than far above it, so a
local peak there is more likely to reflect something the model actually weighted,
though still not proof of it.

## Case studies

Two TEST games, picked by a rule fixed before plotting (not cherry-picked
afterward), among the 2,408 sampled games with at least 40 plies:

- **Most accurate**: the smallest final-ply mean |live estimate - true rating|
  across both sides.
- **Largest swing**: the largest (max - min) of the mean-of-both-sides live
  estimate, restricted to plies after 15 so opening volatility (see the
  attention caveat above) cannot win this by construction alone.

Both figures show, top to bottom, sharing a ply axis: the live (prefix-only)
rating estimate for both sides against the true pre-game rating; the Stockfish
evaluation track in White's point of view (the model never sees this, only shown
as a yardstick); and the model's own attention weight, with the partly
mechanically front-loaded ply 1-9 region shaded (see the caveat above). Markers
on the top panel mark the five highest centipawn-loss moves at ply 10 or later;
the three largest are labeled. Per-ply JSON for both games (rating, evaluation,
centipawn loss, phase, attention, clock remaining, and time spent, the model's
own clock input) is at `analysis/move-explanation-data/case-accurate.json` and
`case-swing.json`.

### Most accurate: `2021-06_game_0026363.pkl` (White 1248, Black 1244)

![Most accurate case study](figures/move-explanation-accurate.png)

Final-ply mean absolute error 1.14 rating points across both sides, over 100
plies. White and Black's live curves track each other almost exactly throughout
(consistent with the served model's near-zero median White/Black gap), swinging
widely in the first 20 plies (1540 down to 1145) before settling into a
tighter band around the true rating from about ply 35 onward. The two
largest-centipawn-loss moves in the game, `Ke6` and `Rb3` (both capped at the
2,000 cp ceiling, i.e. a lost/winning position thrown into something close to the
opposite), sit inside that already-settled late-game region and produce no
visible jump in the live curve: consistent with the aggregate finding that
engine-measured move quality is a weak driver of the live estimate's movement.
What does coincide with the curve's movement is ply: the largest swings are in
the first 10 to 20 plies, where attention (bottom panel) is also at its highest,
and where the dilution-only baseline from the caveat above predicts most of that
height. Time spent over plies 1-20 stays in a narrow 1-13 second band (this is a 5-minute,
no-increment blitz time control) with nothing that stands out as coinciding with
the largest jumps; the first move of each side (ply 1, ply 2) has no earlier clock to
diff against and is excluded from this reading. Nothing in this game's early
volatility is clearly attributable to move quality or time spent by this note's
own predictors; it reads as the recompute mechanism's own early-game noise.

### Largest swing: `2025-08_game_0017598.pkl` (White 2246, Black 2267)

![Largest swing case study](figures/move-explanation-swing.png)

The live estimate falls from about 1710 to about 1290 over the first 18 plies,
then climbs sharply to above 2300 by around ply 28 and stays near the true rating
(2246 to 2267) for the rest of the game: a swing of just over 1,000 rating points,
the largest after ply 15 among the 2,408 eligible (40-ply-or-longer) sampled
games. The single largest centipawn-loss move in the game is `Kxh2` at ply 61
(White's king recaptures into what turns out to be a drawn-by-repetition queen
check sequence, confirmed by the game's own recorded result, 1/2-1/2, inside this
note's 100-ply window; throwing away a position Stockfish had at +657 centipawns
for White down to a dead-equal 0): a real, sizeable blunder by the engine's own
measure, in exactly the kind of position this note's regression says centipawn
loss barely moves the live estimate, and the live curve agrees: there is no
visible drop in the rating estimate at ply 61.

The estimate's one large, sustained move happens earlier, between roughly ply 18
and ply 28. The attention panel's one clear peak outside the shaded ply 1-9
region, 0.17 at ply 28, sits exactly there too, and lines up with the two
largest-centipawn-loss moves in that stretch, `Qd2` (189 cp) and `Kh7` (183 cp) at
plies 27 and 28: unlike the shaded opening region, this peak is not a construction
artifact by the caveat above (past ply 20, observed attention runs at or below the
dilution-only baseline on the 3,000-game average), so it is a plausible, though
not proven, case of attention actually tracking a real blunder exchange in this
one game. That is a different claim from "centipawn loss moves the rating
estimate," which the aggregate regression's small effect sizes still do not
support, and the two 189/183-point losses at plies 27-28 do not produce any
unusual drop in the live curve themselves (the curve keeps climbing through them).
Time spent across plies 17-31 (a 3-minute-plus-2-second blitz time control) stays
in an ordinary 1-15 second range with no value that stands out as explaining the
climb; the largest single jumps in this window (such as +197 points at ply 18 and
+274 points at ply 20) occur on plies with modest time spent (2 and 4 seconds, at
or below this window's median of 4) and modest centipawn loss (28 and 14), not on
the slowest or costliest moves in the stretch. As with the accurate-game case study, no single predictor
in this note cleanly explains the swing; it is better read as the recompute
mechanism's own volatility than as a response to any one measured quantity. Both
case studies illustrate the same pattern the aggregate regression finds: engine
move quality and time spent are both associated with the live estimate, but only
weakly, and most of its ply-to-ply movement, including the large early-game
swings in both games here, is not explained by either.

## Limitations

- **This is a correlational analysis of one trained checkpoint pair**, not a
  causal claim about what the model "uses." The regression's R-squared values are
  small throughout; most of the live estimate's ply-to-ply movement is not
  explained by any predictor used here.
- **Time-spent is unreliable for each side's first move** (ply 1 and ply 2):
  `compute_time_spent` falls back to the `TimeControl` base allotment there, which
  berserk (a Lichess feature that halves a player's own clock) can make
  meaningless; those two plies per game are dropped from every time-spent figure,
  and negative values (observed occasionally, likely from increment or lag
  edge cases) are clipped to zero rather than guessed at further.
- **Centipawn loss is capped at 2,000** and the underlying evaluation at 10,000
  (mate scores), so a handful of the largest blunders in the sample are recorded
  as tied rather than ranked against each other; this affects labeling in the case
  studies (see `Ke6`/`Rb3` above) more than the regression, where the centipawn-loss
  cap binds for 2.1% of scored plies and the evaluation (mate) cap for 4.4%.
- **Stockfish search depth is a race against a 1-second-per-position cap**, not a
  guaranteed fixed depth, though it reached depth 17 at 99.997% of scored
  positions in this sample (see Method); this note does not further stratify by
  the 5 positions that stopped one ply short.
- **Attention weight, read only past ply 10 per the caveat above**, is still a
  property of one trained checkpoint's one attention mechanism, not a general
  claim about what "matters" in a position, and the aggregate regression above
  does not test whether attention tracks centipawn loss, only whether the rating
  estimate does.
- **The regression's move-quality and time-spent effect sizes are both small
  relative to the live estimate's overall ply-to-ply volatility** (standard
  deviation around 55 rating points for the served model's own-estimate change);
  the case studies above are illustrations of the aggregate pattern, not
  independent evidence for it.
- **`mirror_arm_a`'s own-minus-opponent nuisance R-squared is driven
  substantially by a 34-game ultrabullet subsample** (1,666 of 185,058 scored
  rows) with an unusually large and consistent effect; see the regression
  section above for the coefficient. A pattern this concentrated in one small,
  unusual time-control bucket warrants more games before treating it as a
  general property of fast time controls.
- **`mirror_arm_a_gapweight` is a gap-weighted training loss on the same shared
  architecture**, not a separate-heads model (that is the different, and
  pre-registration-failing, `mirror_arm_b_diffhead` arm); its larger
  own-minus-opponent separation comes from training emphasis on wide-gap games,
  not an architectural change. It is not the default checkpoint the prototype
  serves, though as of `e1f8fec` a request can select it instead.

## Reproduction

HPC work ran under `~/thesis2/scratch/move_explain/` (not `~` itself), on GPU 2
(GPUs 0 and 1 left untouched), in tmux session `fm_holder` so the training
watchdog's `--experiment` substring match could not mistake this analysis run for
a training run. This repository's `src/` (at commit `8c7f23f`) was copied to
`~/thesis2/scratch/move_explain/src_snapshot/src`, and
`move_explanation_curves.py`, `move_explanation_engine.py` and
`sample_test_games.py` to `~/thesis2/scratch/move_explain/scripts/`. Commands,
run from `~/thesis2`:

```
python3 scratch/move_explain/scripts/sample_test_games.py \
  --manifest data/split_manifest_2p55M_seed42.json \
  --n 3000 --seed 20261011 --out scratch/move_explain/out/sample_games_3000.json

bash scratch/move_explain/scripts/run_move_explanation.sh
  # launches both the GPU curve job and the CPU Stockfish job (concurrently,
  # since they don't contend), writes .DONE/.FAILED markers and logs under
  # scratch/move_explain/out/ and scratch/move_explain/logs/
```

Then, after copying `out/curves_3000.jsonl` and `out/engine_3000.jsonl` back into
a local `hpc_out/` directory: `analysis/move_explanation_merge.py` (run from the
directory containing `hpc_out/`; builds `merged_per_ply.csv`, the per-ply table
used below, not committed, same convention as this repository's other large
per-game CSVs), `analysis/move_explanation_regression.py` (reads
`merged_per_ply.csv` from the same directory; the bootstrap regression above),
`analysis/move_explanation_case_studies.py` (reads `hpc_out/` directly; selects
the two games and renders the figures and per-ply JSON in this note).
