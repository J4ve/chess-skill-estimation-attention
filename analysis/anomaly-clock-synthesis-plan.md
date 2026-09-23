# Synthetic remaining-time clock for the 90,000-game anomaly corpus

Date: 2026-08-17. Scope: Part B of `CLAUDE_TASK_2.md`.
Revised 2026-08-18 (`analysis/anomaly-clock-fixes.md`) and 2026-08-28
(`analysis/anomaly-clock-recalibrate.md`: first-order pacing recalibrated to
Sigman's 40th-move budget, and Maia-band dependence threaded into the
think-time model).

Status: **designed and implemented.** The implementation is a new standalone
script, `analysis/scripts/synthesize_anomaly_clocks.py`. It is pure standard
library, it runs and self-tests on a machine with no corpus and no torch, and
it has a built-in validation harness. Nothing under `prototype/src/` was
touched. The one thing that is **not** done, and cannot be done from this
clone, is the comparison against real Lichess clock traces. That is spelled
out in section 8 as a hard gate before Chapter 4 uses any of this.

---

## 1. What is actually wrong with the corpus today

Two separate defects, not one.

**Defect 1: `Clocks` holds think-time, not remaining time.**
`prototype/src/generate_anomaly_corpus.py:105-118` times a single engine call
with `time.monotonic()` and writes that duration through `seconds_to_clock`
(`:63-67`) into the per-ply `Clocks` list. Every engine here is deliberately
fast: Maia is `chess.engine.Limit(nodes=1)` (`:98`), the Stockfish substitute
is `Limit(time=0.05)` (`:163`), the Lc0 substitute is `Limit(nodes=100)`
(`:166`). So the recorded values are near zero and they are a *duration*.
The trained model's clock feature is the opposite quantity: it is harvested
from Lichess `[%clk ...]` comments, which are the mover's *remaining* time
(`prototype/src/format_data.py:67-69`), converted by `time_to_seconds`
(`:33-36`) and z-scored with mean 273 and std 380
(`prototype/src/chess_rating_net.py:63-64`, applied at `:79-81`). This is the
gap already recorded in `analysis/anomaly-corpus-generated.md:80-89` and
`analysis/anomaly-corpus-pilot.md:141-150`.

**Defect 2: the corpus has no `Time` key at all.** This one fires first, before
the clock feature is ever read. `ChessGamesDataset.__getitem__` does
`initial_time, increment = map(int, game_info["Time"].split("+"))`
(`chess_rating_net.py:93`) on every game. The anomaly generator writes
`Positions, Moves, Clocks, Result, move_is_substituted`
(`generate_anomaly_corpus.py:122-128`) plus `WhiteElo, BlackElo,
suspect_color, engine, substitution_rate, maia_band` (`:177-182`) and never
writes `Time`. Loading a corpus game through the existing Dataset therefore
raises `KeyError: 'Time'` before the clock feature is even reached. Any fix
for defect 1 has to supply `Time` too, because a synthesized countdown is
meaningless without the base and increment it counts down from. This is also
the reason the choice of time control cannot be dodged.

This second defect was found independently in the same session by the Part E
schema audit and is written up at `analysis/schema-completeness-check.md:260-276`,
which reaches the same conclusion: both defects are fixed in one place, because
the countdown needs a nominal base and increment anyway and that is exactly what
`Time` should hold. Two independent derivations, same fix. That document's
Finding 5 (`:277-305`), that the generated corpus does not actually contain
matched clean-versus-substituted *pairs*, is a separate problem that this script
does not address and that Chapter 4's evaluation design has to settle on its own.
It does interact with the regenerate question, and section 5 handles that.

---

## 2. Reality check on the "sample from the real corpus" direction

The task text supposes the real 1.2M-game Lichess corpus "is either complete
or well underway" and can supply an empirical think-time distribution.
**Verified against the repo: it cannot, not today.**

- There is no game data of any kind in this clone. A search for `*.pkl`,
  `*.pgn*`, `*.zst` and `*.csv` under the repo root returns nothing, and there
  is no `data/` directory.
- The 1.2M pipeline was **relaunched today**, 2026-08-17 at 13:53
  (`CLAUDE.md:32`), after the `has_full_clocks()` scan fix. Before that
  relaunch, `analysis/corpus-pipeline-fix-plan.md:209` states plainly: "No
  month in the 2021-04..2024-01 window is complete." The same document's
  throughput table (`:112-119`) puts the remaining 34 months at roughly 80 to
  105 days on the HPC at 4 workers, or about 2 to 3 months with sustained
  local help. `CLAUDE_TASK.md:46-47` independently describes the pipeline as
  "currently running unattended and will keep running for roughly 2-3 weeks".
  Nothing in those numbers produces a usable empirical clock distribution this
  week.
- What does exist is the earlier ~170k-game subset, 2024-02 to 2024-07
  (`CLAUDE.md:25`), which does carry real `[%clk]` traces and real
  `TimeControl` headers. It lives on the HPC at
  `~/<workdir>/data/processed_games/` (`analysis/hpc-snapshot.txt:10-23`)
  and is unreachable from here. It is more than enough to fit a think-time
  model. It is simply not available to this session.

So the empirical direction is right, and it is the direction the method below
is built to accept, but writing a plan whose central step is "sample from the
1.2M corpus" would be writing a plan that cannot be executed. The method
below is therefore two-layered: a first-principles parametric model that runs
today, and a **well-specified plug-in point** where an empirical fit replaces
it the moment any real preprocessed games are reachable. Both layers are
implemented. The fitter is implemented and tested, it simply has no real input
yet.

---

## 3. The method

### 3.1 Clock arithmetic, per player

Two independent countdowns that interleave into one flat per-ply list. Ply `i`
(0-based) belongs to White when `i` is even, because the generator starts from
`chess.Board()` and never skips a turn (`generate_anomaly_corpus.py:96-120`),
so `Moves[0]` is always White's. For the mover on ply `i`, with `k = i // 2`
being that player's own 0-based move index:

```
before   = c[player]                        # that player's own clock
d        = think time for this move         # section 3.2
d        = min(max(d, min_move_s), kappa * before)
after    = max(floor_s, before - d)         # floored at the flag
c[player] = after + increment               # increment credited after the move
Clocks[i] = seconds_to_clock(c[player])
```

Four properties fall out of this and all four are checked by the self-test:

- **Per player, not per ply.** White's series is `Clocks[0::2]` and Black's is
  `Clocks[1::2]`. Each is separately non-increasing modulo its increment. A
  naive single monotone countdown would fail this immediately, and would be
  visibly wrong to anyone who has read a Lichess PGN.
- **Not merely decreasing.** With an increment, a player who moves faster than
  the increment gains time. Real 15+10 games do this constantly. A sample of
  the first 24 plies of a synthesized 900+10 game, White's own series:
  `0:14:55  0:14:59  0:15:07  0:15:06  0:15:13  0:15:14  0:15:15  0:15:12
  0:15:11  0:14:06  0:14:00  0:12:13`. It rises through the opening and turns
  over when the middlegame thinks start. That is the shape a panel expects.
- **Floored, never flagged.** `floor_s` keeps the clock strictly positive.
  This is deliberate: the corpus games ended by checkmate, by draw, or by the
  100-ply cap (`generate_anomaly_corpus.py:60,101,122-128`), never on time. A
  synthesized clock that hits zero would contradict the `Result` string stored
  in the same file. Not flagging is a consistency requirement, not a
  convenience.
- **Integer seconds only.** `format_data.time_to_seconds` does `int(parts[2])`
  and raises on a fractional-second clock, a low-severity issue already noted
  in `analysis/170k-verification.md:447-449` where it is also observed that it
  "has not fired" on the real corpus. Emitting `H:MM:SS` integers via the same
  `seconds_to_clock` the generator already uses keeps the retrofitted corpus
  inside the range the existing loader survives.

One convention is an open assumption: whether the `[%clk]` value printed for a
move already includes the increment credited for that move. The script exposes
`report_after_increment` (default true) and the difference is a constant offset
of at most one increment, which for a typical 3-second blitz increment is
3/380 = 0.008 in normalized units. Confirm it with one grep against any real
Lichess PGN when one is at hand, and flip the flag if needed.

### 3.2 Think-time model

Per move, the mover spends a *share* of their own remaining time:

```
d_k = phi_k * (c / n_k) * X ,    X ~ LogNormal(mu = -sigma^2/2, sigma)
```

- `c / n_k` is the classic time-budget rule: remaining time divided by the
  number of moves still expected. `n_k = max(n_floor, n_expected - k)` with
  `n_expected = 40` and `n_floor = 12`. Forty moves per player is also the
  figure the pipeline itself already assumes when it buckets a time control by
  `base + 40 * increment` (`chess_rating_net.py:93-95`,
  `format_data.py:98-109`), so the corpus and the model agree on it, and it is
  independently the "around 40 moves" figure Sigman et al. use (their `:87`,
  `:92`).
- `phi_k` is a game-phase profile, piecewise linear over the player's own move
  index, with knots (0, 0.10), (3, 0.30), (7, 0.68), (20, 0.75), (35, 0.65),
  (50, 0.58). Fast opening, slowest early middlegame, gradual speed-up after.
  These knot amplitudes were recalibrated on 2026-08-28 (down by about 0.68
  from the original (0.15, 0.45, 1.00, 1.10, 0.95, 0.85)) so the model consumes
  Sigman's published 74.7 to 77.9 percent of a 180 s budget by White's 40th
  move at 180+0 rather than about 88 percent; see
  `analysis/anomaly-clock-recalibrate.md` deliverable 1 and section 7.3 below.
  The knots also carry a small, monotone, band-dependent see-saw, section 3.8.
- `X` is a unit-mean multiplicative factor standing in for "this position was
  hard or easy". `mu = -sigma^2/2` fixes `E[X] = 1`, so the *mean* share is
  exactly `phi_k / n_k` and the median is a little lower. `sigma` is band
  dependent (section 3.8), about 0.78 to 1.02 across the bands with 0.91 at the
  1500 centre, clipped to [0.08, 8].
- `kappa = 0.45` caps any single move at 45 percent of the remaining clock,
  and `min_move_s = 0.1` sets a floor.

Why this shape rather than something invented. The qualitative pattern is one
of the reported statistical regularities of Sigman, Etchemendy, Fernandez
Slezak and Cecchi (2010), "Response Time Distributions in Rapid Chess: A
Large-Scale Decision Making Experiment", *Frontiers in Neuroscience*, drawn
from a FICS database of more than 2.8 million lightning and blitz games and
filtered for that analysis to 3-minute games without increment (91,340 such
games listed as analyzed): response times are "rapid during the first moves
(the opening of the game) and the last moves (the endgame) and significantly
longer during the middle game", and the late speed-up "is not dominated by the
complexity of the position, being most likely determined by time pressure".
The paper calls that trend "expected" in the same sentence that reports it, and
its own headline results are different: RT distributions that are long-tailed
and qualitatively distinct at different stages of the game, and strong serial
correlation between successive moves. The pacing pattern is real, is in the
paper, and is a fair thing to build a phase profile on, but it is not what the
paper leads with and should not be described as its main finding.
The `phi_k` profile encodes the first half of that and the `c / n_k` form
encodes the second: as the clock drains, the absolute time spent per move
drains with it, without needing a separate time-pressure term.

Two honest caveats on that citation, both of which belong in the thesis text
rather than being quietly skipped:

1. Sigman et al. report **power-law tails** for the fast (opening and endgame)
   moves, not a lognormal. A lognormal `X` has a lighter tail than that. The
   parametric layer is therefore a deliberate simplification, and it will
   under-represent the rare very long think. It matters less than it sounds,
   because the model consumes *remaining* time rather than think time, and
   `kappa` bounds how far one draw can move the countdown. It is still a real
   approximation, and it is exactly what the empirical layer replaces.
2. Their corpus is 3-minute games without increment on a different server.
   Their finding is used here for *shape*, never for magnitudes.

### 3.3 The empirical plug-in (specified, implemented, not yet fed)

`--fit-from <dir of real preprocessed .pkl games>` estimates, per time-control
bucket and per move-index bin, the median share `d / c_before` and the spread
of `log(share / median)`, and writes them to a JSON file. `--params <file>`
then switches synthesis to `method = empirical-share`, where the fitted median
and spread replace `phi_k / n_k` and `sigma` entirely. Bins are on the player's
own move index with lower edges 0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32. Bins with
fewer than 30 observations are filled from the nearest estimated neighbour and
flagged by their spread.

The fitter recovers think times from remaining times by inverting the same
arithmetic as section 3.1, which is the correct direction: real games publish
the countdown, not the think time. This is also the ablation that
`analysis/170k-verification.md:443-445` suggests separately (time-spent is
derivable from consecutive remaining values plus the increment).

Any of these inputs works, in order of preference: the completed 1.2M corpus;
the existing ~170k subset at `~/<workdir>/data/processed_games/`; a
single month of it; a single flattened directory such as
`/tmp/ratingnet_data_flat`. It needs perhaps 20,000 games to fit the blitz,
bullet and rapid buckets well. It cannot fit ultrabullet from a
mix-proportional sample, for reasons measured in section 7.

### 3.4 Time-control assignment

Each game draws a concrete `base+increment` from a mix whose bucket weights
reproduce the training corpus's realized mix as reported in
`analysis/170k-verification.md:683`: roughly 8,320 blitz, 6,343 bullet, 2,183
rapid, 125 classical and 40 ultrabullet in a 17,014-game test set, so 48.9
percent blitz, 37.3 percent bullet, 12.8 percent rapid, 0.73 percent classical,
0.24 percent ultrabullet. Note that document tags the figure `[derived]`,
scaled from a 6,000-game profile rather than counted directly, so it is the
best available estimate and should be re-measured from the real corpus's
`.sampling_summary.json` files (`CLAUDE.md:26`) when they exist.

The split of each bucket into concrete controls (60+0 and 120+1 for bullet;
180+0, 180+2, 300+0, 300+3 for blitz; 600+0, 600+5, 900+10 for rapid; 1800+0
and 1800+20 for classical; 20+0 for ultrabullet) is this script's own choice of
common Lichess controls. It is an assumption. It is also easy to replace,
since the mix is a parameter. The self-test asserts that every entry lands in
the bucket the comment claims under the pipeline's own
`categorize_time_control` thresholds, which is the kind of thing that silently
goes wrong (30+0, for instance, is *bullet* under those thresholds, not
ultrabullet, because 30 is not less than 29).

`--fixed-tc 300+0` collapses the mix to one control, for a single-control
robustness arm.

### 3.5 Seeding, and why it protects the Chapter 4 comparison

Both random streams are seeded from `(global_seed, band, game_index)` and
**deliberately not** from engine or substitution rate. Consequence: game
`00042` of `1500_stockfish16_r060` gets the same time control and the same
sequence of think-time draws as game `00042` of `1500_stockfish16_r00`. Since
every case directory holds the same 1,000 indices, the *empirical* distribution
of time controls is not merely equal in expectation across the ten cases of a
band, it is exactly identical, case by case.

This matters more than it looks. Chapter 4 reports ROC-AUC broken down by
substitution rate (`CCS Thesis - Integrated/chapters/chapter4.tex:107`). If the
clock channel varied across rate arms, a panelist could argue the detector was
reading a clock artifact rather than move quality. Index-matched assignment
removes the time control as a between-arm confound by construction. The
self-test asserts this directly on a fixture corpus.

The band-dependent pacing added in section 3.8 does not weaken this. The phase
profile and dispersion vary with the Maia band, but the band is a function of
the case directory's band prefix alone, identical across the five
substitution-rate arms of a band. So game `00042` of `1500_..._r060` and game
`00042` of `1500_..._r00` still draw from exactly the same pacing distribution.
The band varies across the nine band groups, which is deliberate: that is the
axis Chapter 4 breaks ROC-AUC down by, and the axis Sigman published
band-varying coefficients for.

One thing this does **not** do, and must not be read as doing: it does not pair
the underlying games. `analysis/schema-completeness-check.md:277-300` shows that
game `00042` of the clean case is not the same game as game `00042` of a
substituted case, and reading the generator confirms it. `use_cheater` is
`suspect_to_move and cheater_engine is not None and rng.random() < rate`
(`generate_anomaly_corpus.py:103`), and Python's `and` short-circuits, so a
clean case (`--engine none`, `cheater_engine is None`) never consumes
`rng.random()` inside the game loop while a substituted case consumes one draw
per suspect ply. The two streams diverge after the first game even though both
seed at 42 (`:146,154`). So the clock assignment is index-matched; the games
themselves are not. Any claim about "matched pairs" is a separate open question
belonging to that document's Finding 5, not to this one.

#### Game length is a real between-arm confound, and seed matching does not fix it

An earlier draft of this document claimed the seed matching meant the clock
"cannot confound the headline comparison". That is false and has been corrected
in section 4. Index matching fixes the *time control*. It does not fix *game
length*, and game length is not random with respect to the label.

`analysis/anomaly-corpus-generated.md:54-56` states the mechanism directly:
higher-substitution-rate games are shorter, because the Maia-versus-Maia clean
control more often runs out the 100-ply cap. The per-rate storage table at
`:46-52` is monotone over identical 18,000-game counts: 5.10, 4.71, 4.54, 4.44
and 4.16 GiB for the 0, 5, 15, 30 and 60 percent arms, an 18 percent decline.

Because the countdown decays with ply index, a shorter game leaves the clock
systematically **higher**. Taking storage as proportional to ply count (each
game stores per-ply position tensors, so this is a reasonable but unverified
proxy) and anchoring the 0 percent arm at the 99-ply cap, the implied gradient
in the model's own normalized units, 6,000 games per cell:

| rate | GiB | rel. length | implied mean plies | per-game mean z | delta vs 0% |
|---|---|---|---|---|---|
| 0%  | 5.10 | 1.0000 | 99 | -0.4061 | 0 |
| 5%  | 4.71 | 0.9235 | 91 | -0.3820 | +0.0240 |
| 15% | 4.54 | 0.8902 | 88 | -0.3742 | +0.0319 |
| 30% | 4.44 | 0.8706 | 86 | -0.3723 | +0.0338 |
| 60% | 4.16 | 0.8157 | 81 | -0.3536 | **+0.0525** |

The scale to judge that against is the residual stochastic sd of the same
feature, 0.0581 under this seed. The between-arm length artifact is therefore
about **90 percent of the entire stochastic component of the clock feature**,
and it is monotone in substitution rate, which is exactly the axis Chapter 4
breaks ROC-AUC down by (`CCS Thesis - Integrated/chapters/chapter4.tex:107`).
Unlike the label-independent clock nuisance discussed in section 3.6, this
artifact is label-*correlated*, which is the kind that can move an AUC.

**Why this is not fixed in the synthesis code.** Length matching was considered
and rejected. The length difference is produced by
`generate_anomaly_corpus.py`, in how the games end; the synthesis only retrofits
clocks onto plies that already exist and has no freedom to change how long a
game ran. Matching by truncation would discard roughly 18 percent of the plies
in the clean arm, and those are endgame plies, which is where both the clock
signal and the move-quality evidence the detector actually uses are richest:
that trades a 0.05 sd clock artifact for damage to the primary evidence.
Matching by padding would fabricate plies that were never played. The confound
is also not clock-specific, since mean game length differs between arms whether
or not clocks exist, so absorbing it inside the clock model would hide a
corpus-level property in one feature rather than remove it. The correct place
to handle it is the evaluation protocol, which is gate 5.

One consequence worth recording: now that the control arm emits a genuinely
flat clock (section 3.7), the constant arm's own length artifact is exactly
0.0000 sd at every one of the rates above, measured the same way. Before that
fix it carried an artifact of the opposite sign, so gate 4's two ROC-AUC
numbers could not be compared cleanly. Gate 4 is interpretable now; it was not
before.

### 3.6 Label blindness, which is the point that will get asked about

By default the synthesis **never reads `move_is_substituted`**. Engine-assisted
players do have distinctive timing, and it would be easy to make substituted
plies faster and more uniform. Doing so by default would be planting the
signal the anomaly detector is supposed to discover, and any resulting ROC-AUC
would be measuring a leak this script inserted. The self-test asserts that
passing labels through the default path changes nothing.

`--cheater-timing` enables the labelled variant. It exists so the sensitivity
question can be answered on purpose, as a separately-reported arm, never as the
primary corpus. The choice is recorded per game in a `clock_synthesis` block
alongside the seed, method, base, increment and bucket, so no output file is
ambiguous about which variant produced it.

### 3.7 The control arm

`--method constant` emits a flat clock at the time control's base value for
every ply. It is the clock-ablation control: run the whole anomaly evaluation
twice, once on synthesized clocks and once on constant clocks. If the ROC-AUC
conclusions are the same under both, the synthesis assumptions are not
load-bearing for the anomaly result, and the question "how do you know your
invented clocks did not drive this?" has a measured answer instead of an
argument. This pairing should be treated as required, not optional.

### 3.8 Rating-band (Maia skill) dependence

Added 2026-08-28 (`analysis/anomaly-clock-recalibrate.md` deliverable 2, review
section 1.6). Before this, one phase profile and one `sigma` served all nine
Maia bands, so a 1100-rated and a 1900-rated player had statistically identical
timing on the corpus's primary design axis. The band prefix entered synthesis
only as a `game_streams` seed component, which changes which draws come out,
not what distribution they come from. It is now a real parameter of the
parametric think-time layer, built from Sigman et al.'s own published
band-varying regressions rather than invented values.

Normalized skill: `s = clamp((band - 1100) / 800, 0, 1)`, so `s = 0` at band
1100, `s = 1` at band 1900, `s = 0.5` at the 1500 centre. `s` is a function of
band alone, so it is constant across the five substitution-rate arms of a band
and varies only across the nine band groups (section 3.5).

**Dispersion.** Sigman's Results section: "for low rated players, regression:
SD = 0.1 s + 0.91 <RT>, for high rated players SD = 0.6 s + 1.36 <RT>". The
slope is the dimensionless dominant term and reads as the linear-space
coefficient of variation of RT once the mean RT is more than a second or two.
Interpolating the two slopes linearly in skill and treating the result as the
coefficient of variation of the unit-mean multiplier `X`:

```
CV_X(s)      = 0.91 + (1.36 - 0.91) * s
sigma_log(s) = sqrt(ln(1 + CV_X(s)^2))
```

which runs 0.78 (band 1100), 0.91 (band 1500), 1.02 (band 1900). The
sub-second intercepts (0.1 s, 0.6 s) are dropped: at the mean think times in
this corpus's buckets (about 1 to 24 s) they move `CV_X` by at most about 0.2,
and folding them in needs an assumed reference RT, that is, a new invented
constant. `ClockParams.sigma_log` (default 0.91) is used only when no band is
supplied and equals `sigma_log(0.5)`.

**Phase shape.** Sigman: "High rated players amplify the variations of RTs
during the game: they play faster than lower rated players during the opening
games, and slower during the middle game." The phase profile carries a small,
monotone-in-band see-saw: with `tilt = 2s - 1` (so -1 at band 1100, +1 at band
1900), the opening knots (`k < 7`) are scaled by `1 - 0.11 * tilt` and the
middlegame knots (`7 <= k <= 45`) by `1 + 0.06 * tilt`; later knots are
unchanged, and nothing changes at the centre band. The two gains are calibrated
so the fraction of a 180 s budget consumed after White's 40th move runs from
about 0.747 at band 1100 to about 0.783 at band 1900, matching Sigman's own "at
move 40" figures (low-rated 74.7 +/- 0.4%, high-rated 77.9 +/- 0.3%).

**Measured effect**, 180+0, 6000 games per band (`anomaly-clock-recalibrate.md`
section 2.4): the 40th-move budget fraction rises monotonically 0.749, 0.754,
0.758, 0.763, 0.766, 0.768, 0.774, 0.777, 0.779 across bands 1100 to 1900, and
the realized think-time coefficient of variation rises 0.99, 1.03, 1.09, 1.13,
1.17, 1.22, 1.26, 1.29, 1.33. Before the change every band was identical in
distribution (mean 0.883, CV about 1.0).

**Threading.** `_share_for_move` and `synthesize_game_clocks` take a `band`
argument. `retrofit_corpus` passes the band it already parses from the case
directory name (`{band}_{engine}_r{rate}`); `simulate` passes its per-game
band; the self-tests pass the band directly and need no corpus. The empirical
`--fit-from` layer is unchanged and carries no band coordinate, so `band`
modulates the parametric layer only.

**Scope.** This reproduces Sigman's first-order band-varying pacing (the "at
move 40" budget) and band-varying dispersion (the SD regressions). It does not
add the second-order serial correlation between successive response times that
the same Results section reports (review section 1.6 main point); that gap is
unchanged by this work and is handled by a drafted Limitations item, not code.

---

## 4. Why this is defensible in front of a panel

- Every value is produced by real Lichess clock arithmetic. Base, minus think
  time, floored at zero, plus increment, per player, alternating into one flat
  list. Nothing is a smoothed curve or a hand-drawn decay.
- The distributional shape has a published source for its qualitative form
  (Sigman et al. 2010), the two places where the parametric layer departs from
  that source are stated rather than hidden, and both are exactly what the
  empirical fit removes. As of the 2026-08-28 recalibration the phase amplitude
  and the band-dependent dispersion are also tuned *to* that source: the model
  now reproduces Sigman's published 74.7 to 77.9 percent budget consumed by
  White's 40th move at 180+0 (band 1100 to band 1900), where the pre-change
  model overspent to about 88 percent. See section 3.8 and
  `analysis/anomaly-clock-recalibrate.md`.
- The parameters are not tuned to make the corpus look good. In particular
  they were not tuned to hit the model's own normalization constants; section
  7 reports the mismatch that results, and section 8 explains why chasing it
  would have been the wrong move. They were tuned to match one published human
  number (the 40th-move budget), which moved the normalized-feature mean from
  -0.408 to -0.34 as a side effect, not as a target.
- The synthesis is label-blind, so it cannot manufacture the anomaly signal.
- It is seed-matched across substitution-rate arms, which removes the *time
  control* as a between-arm confound by construction. It does **not** remove
  game length, and length is not random with respect to the label: higher
  substitution rates end games sooner, so their clocks are read off a shorter
  countdown and sit systematically higher. Measured under the shipped default
  mix, this is worth up to +0.0525 sd of the normalized clock feature between
  the 0 percent and 60 percent arms, roughly 90 percent of the feature's entire
  stochastic component, and it points monotonically in substitution rate. See
  section 3.5 and gate 5; do not describe the clock channel as confound-free.
- It is reproducible from a single integer seed, it preserves the original
  engine think-times under `Clocks_engine_seconds` rather than destroying them,
  it stamps provenance into every game, and it ships with a clock-ablation
  control arm that tests whether the whole exercise mattered.
- The honest limitation is stated up front: these are modeled clocks on
  bot-vs-bot games, not observed human clocks, and Chapter 4 must say so. That
  is the same class of caveat the manuscript already carries for the corpus
  itself at `chapters/chapter4.tex:199`.

---

## 5. Retrofit versus regenerate

**Recommendation: retrofit. Do not regenerate.** This is not a close call.

The decisive point is the one the task already identified, and the code
confirms it. Maia is called with `chess.engine.Limit(nodes=1)`
(`generate_anomaly_corpus.py:98`): one policy-network lookup, fixed cost,
identical for a forced recapture and for a critical middlegame decision. A
wall-clock budget imposed on that call cannot produce difficulty-dependent
think time, because the engine does not do more work on harder positions. It
would produce a `sleep()`. So a regenerated corpus would still need the exact
same "how long does a human spend on this move" model developed above, and
would then apply it through a slower and less controllable channel. Regeneration
buys nothing on the axis that actually matters.

Everything else points the same way:

- **Cost.** The pilot measured 1.136 seconds per game averaged over both
  engines (`analysis/anomaly-corpus-pilot.md:99`) and projected 90,000 games at
  about 28.4 hours sequential, 10 to 14 hours at 2 to 3 workers (`:130-133`).
  The retrofit's arithmetic, measured here, is 4,249 games per second single
  core, so 90,000 games of clock synthesis takes about 21 seconds. The rest of
  the retrofit is one pass of reading the corpus.
- **Risk.** The generation run needed a repair pass. `1100_lc0_r00` segfaulted
  from CUDA contention and `1200_lc0_r060` hung for six hours at 855/1000
  games (`analysis/anomaly-corpus-generated.md:58-79`). A second full pass
  re-enters that failure surface for no gain, on a box that will also be
  running the relaunched corpus pipeline.
- **It would destroy work that is already correct.** Regeneration re-rolls the
  RNG, so positions, moves and `move_is_substituted` labels all change. The
  22.96 GiB currently on disk (`analysis/anomaly-corpus-generated.md:16-18`),
  the verified per-case counts, and anything computed on them become stale.
  The retrofit changes exactly one field and adds one missing key.
- **Reversibility.** The retrofit keeps the original engine think-times under
  `Clocks_engine_seconds` and, in the recommended sidecar mode, does not modify
  the corpus at all. A regeneration is not reversible.
- **Storage.** Regeneration wants a second 22.96 GiB while the old copy is
  validated. `/home` had 586 GB free (`analysis/hpc-snapshot.txt:3`) and is
  shared, with the 1.2M pipeline about to consume a large and not-yet-known
  amount of it.
- **Writing it up.** "We applied a documented, seeded, reversible transform to
  a frozen corpus" is a cleaner Chapter 3 sentence than "we regenerated the
  corpus with artificial per-move sleeps".

The one genuine argument for regeneration, stated so it is not lost: only
regeneration can make the move process and the time process *jointly*
dependent, for instance letting a simulated cheater consult the engine only
when they have time to spare. That is a different and more ambitious research
question about cheater behaviour, it is outside the protocol Chapter 3 commits
to, and if it is ever wanted it should be a small purpose-built corpus rather
than a re-run of these 90,000 games.

There is one scenario where a regeneration happens anyway, and it changes
nothing about this recommendation. `analysis/schema-completeness-check.md:277-305`
raises a separate methodological question: the manuscript describes Test 5 as
running on "matched clean-versus-substituted pairs"
(`CCS Thesis - Integrated/chapters/chapter4.tex:107`), and the corpus does not
contain cross-case pairs. If the captain or adviser rules that cross-case pairs
are required, the corpus gets regenerated with a shared per-game seed prefix for
*that* reason, at the 10 to 14 hours already costed. Even then the clock should
**not** be produced at play time. The `nodes=1` argument is unchanged: a
wall-clock budget on a fixed-cost policy lookup is still a sleep, not a think.
The right move in that scenario is to have the regenerated writer call this same
synthesis at write time, which costs 21 seconds of the run and keeps one model
of human time usage rather than two. So the recommendation is stable under both
outcomes of that open question: synthesize the clock, never play it out.

A third option also deserves naming and rejecting: skip clocks entirely, feed a
constant, and treat the anomaly evaluation as clock-free. That is cheaper and
perfectly honest, but it puts every corpus game outside the distribution the
model was trained on, and it throws away a feature the architecture explicitly
concatenates into the BiLSTM input (`chess_rating_net.py:269-270`). The better
use of that idea is as the control arm in section 3.7, which is what the script
implements.

---

## 6. What was implemented

`analysis/scripts/synthesize_anomaly_clocks.py`, about 1,260 lines including the
self-test, standard library only. It does not import torch, numpy or python-chess, so it runs and
self-tests anywhere. It re-implements `seconds_to_clock`, `time_to_seconds` and
`categorize_time_control` as deliberate copies of
`generate_anomaly_corpus.py:63-67` and `format_data.py:33-36,98-109`; any
divergence there is a bug and the self-test pins the behaviour.

Modes:

| Mode | What it writes | Size |
|---|---|---|
| `--mode sidecar` (default) | `clocks.json` per case directory, holding `Clocks`, `Time` and a provenance block per game. Corpus untouched. | roughly 70 MB total |
| `--mode rewrite` | A copy of each pickle with `Clocks` replaced, `Time` added and the original preserved as `Clocks_engine_seconds`. | another 22.96 GiB |
| `--mode inplace` | Same, editing the corpus files. | no extra |

Sidecar is the default on purpose. It costs one read of the corpus and cannot
damage it. A Chapter 4 evaluation script can merge the sidecar at load time
without any change to `prototype/src/`, which also keeps this work off the
critical path of the training pipeline.

The script degrades gracefully when the corpus is absent, which is the state of
this clone: it prints where the corpus actually lives and exits 3 rather than
crashing. `--fit-from` and `--compare-real` do the same. `--self-test` and
`--simulate` need no data at all.

Because the corpus pickles hold torch tensors in `Positions`, the retrofit and
the fitter must run where torch imports (conda env `ratingnet2` on the HPC).
The script detects the missing module and says so instead of failing obscurely.

---

## 7. Sanity checks that were actually run

All numbers below were produced on this machine today by the script itself.
None of them involve real Lichess data, which is the whole content of section 8.

### 7.1 Self-test

`python3 synthesize_anomaly_clocks.py --self-test` passes eleven groups:

1. Every entry of the shipped time-control mix lands in the bucket its comment
   claims, under the pipeline's own 29/179/479/1499 thresholds.
2. Clock-arithmetic invariants over all 12 controls and game lengths 1, 2, 7,
   40 and 100: length matches ply count, every string parses through the
   pipeline's integer-only `time_to_seconds`, no negative value, no zero value
   (no synthesized flag), each player's own series never rises by more than one
   increment, each player's series never rises at all when the increment is
   zero, and the first reported value never exceeds base plus increment.
3. Determinism: the same seed reproduces the same time control and the same
   clock strings.
4. Increment games really do go up somewhere in a 900+10 game, so the
   arithmetic is not accidentally monotone.
5. Label blindness: passing `move_is_substituted` through the default path
   changes nothing, and `--cheater-timing` does change the output.
6. `--method constant` produces a genuinely constant clock.
7. The KS implementation returns D = 0 and p = 1 for a sample against itself
   (including a heavily tied sample), and D > 0.9 with p < 1e-6 for a
   six-sigma shift. Against two independent 2,000-point standard normals it
   returns D = 0.038, p = 0.109; against a 0.2-sigma shift, D = 0.080,
   p = 1e-5.
8. End-to-end retrofit on a constructed fixture corpus of two case directories:
   the sidecar is written, the same game index in the 0 percent and 60 percent
   cases receives the identical time control, the shorter game's clocks are a
   prefix of the longer game's, rewrite mode preserves `Clocks_engine_seconds`,
   adds `Time`, and produces `len(Clocks) == len(Positions) == len(Moves)`,
   which is the guard at `format_data.py:78-80` that would otherwise reject the
   game.
9. The fitter round-trips: fitted on 120 synthesized games it recovers
   plausible per-bin share medians, sets the empirical method, and the fitted
   parameters then drive synthesis without producing a flag.
10. First-order pacing calibration (added 2026-08-28): 2500 games at 180+0 to
    80 plies at the mid band consume a mean of about 0.767 of the 180 s budget
    after White's 40th move, asserted to be inside [0.74, 0.79] (Sigman et
    al.'s published 74.7 to 77.9 percent). Fails loudly if `DEFAULT_PHASE_KNOTS`
    or `sigma_log` is reverted.
11. Rating-band dependence (added 2026-08-28): from 2000-game probes at bands
    1100 / 1500 / 1900, the 40th-move budget fraction is materially higher and
    monotone in band (Sigman's direction), the per-move think-time dispersion
    is materially larger and monotone in band, the mid band still passes
    [0.74, 0.79], and a fixed RNG seed gives different clocks at bands 1100 and
    1900 (the band is a parameter, not just a seed component).

### 7.2 Distribution report, 9,000 simulated games at 100 plies

`--simulate 9000 --seed 42`, after the 2026-08-28 recalibration. Realized
bucket mix: blitz 0.4936, bullet 0.3692, rapid 0.1269, classical 0.0079,
ultrabullet 0.0024, against the targets 0.489, 0.373, 0.128, 0.0073, 0.0024
(the time-control draw is unchanged by the recalibration).

Remaining time, seconds:

| Bucket | mean | median | p05 | p95 | max |
|---|---|---|---|---|---|
| ultrabullet | 10.8 | 10.6 | 2.6 | 19.7 | 19.9 |
| bullet | 54.5 | 48.5 | 9.7 | 120.8 | 127.7 |
| blitz | 133.7 | 126.7 | 30.7 | 290.4 | 321.9 |
| rapid | 388.0 | 381.8 | 91.7 | 790.1 | 967.1 |
| classical | 1072.9 | 1071.2 | 272.2 | 1819.4 | 1897.8 |
| all | 143.8 | 97.2 | 15.0 | 491.6 | 1897.8 |

Implied think times per move, median: 0.21 s ultrabullet, 0.99 s bullet, 2.57 s
blitz, 7.50 s rapid, 21.4 s classical. Those are recognizable numbers for each
control, which is the first thing to check by eye. They are a little lower than
before the recalibration (which reported 0.22 / 1.13 / 2.87 / 8.42 / 23.9)
because the model now paces closer to Sigman's real players.

Averaged over the game, each bucket now retains 54 to 62 percent of its
weighted base time over 100 plies (ultrabullet 54.0, blitz 59.1, classical
59.6, rapid 60.4, bullet 62.0), up from the pre-recalibration 44 to 51 percent
because the model no longer overspends. The fifth percentile still lands in low
double-digit seconds for the fast controls, which is the right end-of-game
picture for bullet and blitz.

### 7.3 The normalization mismatch, reported rather than tuned away

Under the model's constants (mean 273, std 380, `chess_rating_net.py:63-64`),
the synthesized feature has mean -0.34, std 0.44, and maximum 4.28
(`--simulate 9000 --seed 42`, after the 2026-08-28 recalibration; it was -0.41,
0.41, 4.23 before). So the corpus still sits below the normalization centre and
is narrower than unit variance, but by less than the review's -0.408 figure.

The review's section 1.7 was right that the old -0.408 was not purely "a
property of the constants": part of it was the think-time model draining the
clock faster than Sigman's real 3+0 players do. Re-apportioning honestly, of
the roughly 0.41 units of leftward shift the old model showed:

- about 0.07 was that overspend, a property of the *parameters*, and the
  2026-08-28 recalibration removes it (this is the -0.41 to -0.34 move);
- the remaining -0.34 is a property of the 273/380 constants and the corpus,
  for the reasons that still hold:
  - the mix is blitz-and-bullet dominated, so most plies have well under 273
    seconds remaining by construction;
  - the 100-ply cap (`generate_anomaly_corpus.py:60`, `chess_rating_net.py:81`)
    truncates before the long tail of a classical game;
  - `analysis/170k-verification.md:291-294` establishes that 273/380 has no
    published source and no derivation comment, inherited from the released
    implementation, and `:305-309` shows the sibling rating constants do not
    fit this project's corpus either (1665.7/395.8 measured against 1514/366).
    The maximum of 4.28 here is the same phenomenon `:326` describes for a
    classical opening clock.

The parameters were **not** adjusted to hit -0.0 and 1.0. They were tuned to
one published human number (Sigman's 40th-move budget); the -0.41 to -0.34 move
is a side effect of that, not a target. Reverse-engineering the unpublished
273/380 would make the corpus agree with the normalizer while disagreeing with
reality. The right comparison target is the real corpus's own clock
distribution, which the script computes on demand and which nobody has measured
yet. That measurement belongs in section 8, not here.

### 7.4 The validation harness works, on a stand-in

The real-versus-synthetic comparison could not be run against real data, so it
was exercised against a **stand-in**: 1,500 games synthesized with deliberately
different timing parameters (sigma 1.15, n_expected 28, a slower opening
profile) and then treated as if they were real. This tests the harness, not the
plausibility claim. The distinction matters and it is not glossed here.

The tables below were produced before the 2026-08-28 pacing recalibration, so
the "synthetic median" and "KS D" columns for the *default* parameters no
longer match the current defaults exactly (the recalibrated default paces
closer to the stand-in, so the pre-fit D values are somewhat smaller now). The
three conclusions the section draws are structural and unaffected: the
fit-then-resynthesize loop closes, ultrabullet and classical need a stratified
fit or the parametric fallback, and D is the statistic to read, not p. Re-run
this table on the HPC as part of gate 1, against real games rather than a
stand-in.

Default parameters versus the stand-in:

| Bucket | KS D | stand-in median | synthetic median |
|---|---|---|---|
| blitz | 0.276 | 38.0 | 94.7 |
| bullet | 0.281 | 16.0 | 36.9 |
| rapid | 0.248 | 115.0 | 279.0 |
| classical | 0.280 | 394.0 | 805.7 |
| ultrabullet | 0.363 | 4.0 | 8.0 |

After `--fit-from` on the same stand-in and re-comparing:

| Bucket | KS D before | KS D after | stand-in median | fitted median |
|---|---|---|---|---|
| blitz | 0.276 | **0.070** | 38.0 | 38.9 |
| bullet | 0.281 | **0.124** | 16.0 | 11.9 |
| rapid | 0.248 | **0.029** | 115.0 | 126.6 |
| classical | 0.280 | **0.075** | 394.0 | 430.1 |
| ultrabullet | 0.363 | 0.435 | 4.0 | 7.3 |

Three things this establishes, none of which is "the clocks are realistic":

1. The fit-then-resynthesize loop closes. Where the fitter has data it pulls
   the synthetic distribution onto the target, D dropping by a factor of 4 to 8
   for blitz, rapid and classical.
2. Bullet stays at D = 0.124 and ultrabullet gets *worse*. Ultrabullet is 0.24
   percent of a mix-proportional sample, so 1,500 games gave it roughly 4
   games, far under the 30-observation-per-bin floor, and the neighbour-fill
   made it worse than the parametric default. **The empirical fit must be run
   on a bucket-stratified sample, not a mix-proportional one**, or ultrabullet
   and classical must be left on the parametric fallback and that must be said
   in the write-up. This is a concrete finding, and it would have been invisible
   without running the harness.
3. p-values are useless at this scale. D = 0.029 came with p = 1.2e-07 and
   D = 0.070 with p = 1.3e-159, because each bucket carries 10^4 to 10^5 ply
   observations. Report D as the effect size and pre-register a threshold.
   Based on the numbers above, **D below 0.10 per bucket** is a reasonable bar
   and is achievable by the fitted model.

Also worth noting: integer-second rounding creates heavy ties at low remaining
values, which is part of why bullet stays higher. The KS implementation handles
ties correctly (both samples advance past every equal value), and the self-test
pins that.

---

## 8. What is still needed before Chapter 4 can rely on this

The implementation is done. The **full validation is not**. As of 2026-08-28
one piece of it has been done on a laptop: the first-order pacing is now
calibrated against Sigman et al.'s published "at move 40" budget figure (74.7
to 77.9 percent), which the model reproduces across the bands (section 3.8,
`analysis/anomaly-clock-recalibrate.md`). That is a partial "by measurement"
check against the literature, not against real Lichess clock traces. The
distributional plausibility claim against real games still rests on construction
plus a stand-in. Five gates, in order. All five are short.

**Gate 1: fit and compare against real Lichess clocks.** On the HPC, in
`ratingnet2`, against the existing ~170k subset. This is the single step that
converts "plausible by construction" into "plausible by measurement", and it
does not need the 1.2M corpus to finish.

```
python3 analysis/scripts/synthesize_anomaly_clocks.py \
    --fit-from ~/<workdir>/data/processed_games/2024-02 \
    --fit-out analysis/clock_params_170k.json \
    --compare-real ~/<workdir>/data/processed_games/2024-03 \
    --report-json analysis/clock_ks_report.json
```

Fit on one month, compare on a different month so the comparison is not
in-sample. Accept if D is below 0.10 for blitz, bullet and rapid. If bullet
will not come under 0.10 because of second-granularity ties, say so with the
number rather than moving the bar.

**Gate 2: settle the two open assumptions with real data in hand.**
(a) Re-measure the time-control mix from the real
`.sampling_summary.json` files (`CLAUDE.md:26`) and replace the derived
`170k-verification.md:683` figures. (b) Grep one real PGN to confirm whether
`[%clk]` includes the move's own increment, and set `report_after_increment`
accordingly. Both are minutes of work and both are currently assumptions.

**Gate 3: measure the real corpus's own clock mean and standard deviation, and
decide what to do about 273/380.** After the 2026-08-28 recalibration the
synthesized feature sits at -0.34 with std 0.44 (was -0.408 / 0.410; section
7.3 re-apportions the difference). The recalibration removed the part of the
old shift that was a parameter overspend; what remains is a property of the
constants and the mix, and only the real corpus can say whether the real games
sit in the same place. If they do, 273/380 is simply mis-centred for everyone
and the synthetic corpus is consistent with the training data, which is the
outcome that matters. If the real corpus sits near 0 and 1, then either the mix
or the think-time model is still wrong and gate 1 will have shown it. This
connects directly to the open normalization question in
`analysis/170k-verification.md:311-340` and to the captain's 2026-08-17
decision recorded at `CLAUDE.md:34`, and it should be resolved in that thread
rather than separately here.

**Gate 4: run the anomaly evaluation twice.** Once with synthesized clocks,
once with `--method constant`. Report both ROC-AUC numbers in Chapter 4. If
they agree, the synthesis is not load-bearing for the anomaly conclusion and
the panel question is answered with data. If they disagree, that is a finding
in itself and it needs explaining before either number is trusted.

**Gate 5: count plies per arm, and condition on them.** The game-length
confound in section 3.5 is currently *derived from storage*, not counted: the
plies have never been counted directly, and that must happen in one pass over
`~/<workdir>/data/anomaly_corpus/` on the HPC before any of this reaches
the manuscript. Then report, for every ROC-AUC in Chapter 4, the mean and
median ply count of the arm it was computed on, so the reader can see the
gradient. If the counted gradient matches the storage-implied one, also report
ROC-AUC within ply-count strata, so the headline comparison is conditioned on
length rather than confounded by it. A per-arm AUC that survives stratification
is a result; one that does not survive it is a length effect wearing the
detector's clothes. This gate cannot be run from a laptop clone: it needs the
corpus and the trained weights, both of which live on the HPC.

Two smaller items that are not gates but should not be forgotten:

- The parametric fallback over-spends in classical, where a scale-free share
  model has no absolute floor and real players blitz out many long-control
  moves. Classical is 0.79 percent of the mix so the impact is small, and the
  per-bucket empirical fit corrects it automatically. Do not fix it by hand.
- Ultrabullet needs either a stratified fit or an explicit "left on the
  parametric fallback" note, per section 7.4.

Until gate 1 passes, this work should be described as a *designed and
implemented but unvalidated* clock synthesis. It should not be cited in
Chapter 4 as though the distributions had been checked against real games,
because they have not.

---

## 9. Runbook

```bash
# No data needed. Runs anywhere.
python3 analysis/scripts/synthesize_anomaly_clocks.py --self-test
python3 analysis/scripts/synthesize_anomaly_clocks.py --simulate 9000 \
        --report-json /tmp/clock_sim.json

# On the HPC, in conda env ratingnet2 (torch must import to read the pickles).
# Gate 1: fit on real games, then compare out-of-sample.
python3 analysis/scripts/synthesize_anomaly_clocks.py \
        --fit-from ~/<workdir>/data/processed_games/2024-02 \
        --fit-out analysis/clock_params_170k.json
python3 analysis/scripts/synthesize_anomaly_clocks.py \
        --params analysis/clock_params_170k.json \
        --compare-real ~/<workdir>/data/processed_games/2024-03 \
        --report-json analysis/clock_ks_report.json

# Retrofit the corpus. Sidecar first: one read, nothing modified.
python3 analysis/scripts/synthesize_anomaly_clocks.py \
        --corpus ~/<workdir>/data/anomaly_corpus \
        --params analysis/clock_params_170k.json \
        --mode sidecar --out ~/<workdir>/data/anomaly_clocks

# Control arm for gate 4.
python3 analysis/scripts/synthesize_anomaly_clocks.py \
        --corpus ~/<workdir>/data/anomaly_corpus \
        --method constant \
        --mode sidecar --out ~/<workdir>/data/anomaly_clocks_constant

# Only if a Chapter 4 script really needs the field inside the pickles.
python3 analysis/scripts/synthesize_anomaly_clocks.py \
        --corpus ~/<workdir>/data/anomaly_corpus \
        --params analysis/clock_params_170k.json \
        --mode rewrite --out ~/<workdir>/data/anomaly_corpus_clocked
```

Start every real run with `--limit 5` on one case directory to time the read
and confirm the environment before committing to a full pass.

---

## 10. Reference

Sigman, M., Etchemendy, P., Fernandez Slezak, D., and Cecchi, G. A. (2010).
Response Time Distributions in Rapid Chess: A Large-Scale Decision Making
Experiment. *Frontiers in Neuroscience*, 4:60.
https://pmc.ncbi.nlm.nih.gov/articles/PMC2965049/

Used for the qualitative shape of the phase profile only (fast opening, slow
middlegame, time-pressure-driven speed-up). Their FICS database holds more than
2.8 million lightning and blitz games and more than 200 million moves, but that
is the raw download across 1 to 15 minute controls; the analysis is filtered to
3-minute games without increment, for which the paper lists 91,340 games
analyzed. Quote the filtered figure, not the download, when citing the sample.
Their reported power-law tail behaviour on fast moves
is a known departure from the lognormal used in the parametric fallback, and is
stated as such in section 3.2. If this ends up cited in the manuscript, read
the paper directly and confirm the wording rather than relying on this summary.
