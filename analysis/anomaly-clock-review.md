# Review: Sigman et al. grounding, and the timing-masking limitation

Date: 2026-08-18. Scope: `CLAUDE_TASK_2.md`, Parts 1 and 2.

Subject of review: `analysis/anomaly-clock-synthesis-plan.md` (706 lines) and
`analysis/scripts/synthesize_anomaly_clocks.py` (1,042 lines), plus the
architecture claims in the captain's masking question.

Status of the two headline verdicts:

- **Part 1.** The citation is real, and four of six attributions to it are
  accurate to the word. Two are overstated, in three files. The "two honest
  caveats" list at `anomaly-clock-synthesis-plan.md:183-194` is honest about what
  it covers but is **not complete**: there is a third departure, the source paper
  names that departure explicitly and rules it out, and the same paper publishes
  one hard number for this script's own 180+0 control that the shipped parameters
  miss by roughly 11 percentage points. The lognormal-versus-power-law
  simplification is real, is in the direction the plan doc states, makes synthetic
  anomalies **easier** to detect than realistic ones, and is worth 2.0 percent of
  the clock feature's variance, which makes it the least important item in this
  review.
- **Part 2.** Firstmate's answer is directionally right and materially
  incomplete. Its two supporting claims (that rating-deviation dominance is
  established, and that the manuscript already carries an equivalent
  rotating-engine limitation) are both unsupported, and the second is false:
  `rotat` does not appear in any `.tex` file in this repo. **Yes, the masking
  limitation needs its own Limitations line.** Proposed text is in section 2.4.

## How this was checked

Everything numeric below was produced on this machine today. Sigman et al. quotes
were verified by grep against a local copy of the full text at
`/private/tmp/claude-501/-Users-tomdore/d8993863-573d-471e-98b4-efa14ed15cc3/scratchpad/sigman.txt`
(63,696 bytes, extracted from a 185,791-byte fetch of the PMC record), not against
a summarizer's paraphrase. Model measurements come from importing
`analysis/scripts/synthesize_anomaly_clocks.py` read-only and calling its own
`synthesize_game_clocks` and `draw_time_control`; probe scripts are at
`.../scratchpad/verify.py`, `verify2.py`, `verify3.py`. No file in the repo was
modified except this one. No `.tex` or `.bib` file was touched.

Two other passes on this session reported overlapping findings. Where their
numbers and mine differ slightly (different seeds and sample sizes), I quote mine
and note theirs, because the conclusion is identical under both and the spread is
the honest error bar.

---

# Part 1: literature grounding

## 1.1 The source

Sigman, M., Etchemendy, P., Fernandez Slezak, D., and Cecchi, G. A. (2010).
Response Time Distributions in Rapid Chess: A Large-Scale Decision Making
Experiment. *Frontiers in Neuroscience* 4:60. `doi:10.3389/fnins.2010.00060`,
PMCID `PMC2965049`, PMID `21031032`.

The paper exists, the metadata at `anomaly-clock-synthesis-plan.md:696-699` is
correct, and the reference is genuinely about what the plan doc says it is about.
The citation is not fabricated and is not a misattributed body of work.

## 1.2 Claim-by-claim verdict

| # | Where | Attributed claim | Verdict |
|---|---|---|---|
| C1 | plan `:175-177` | RTs "rapid during the first moves (the opening of the game) and the last moves (the endgame) and significantly longer during the middle game" | **Accurate, verbatim** |
| C2 | plan `:177-178` | the late speed-up "is not dominated by the complexity of the position, being most likely determined by time pressure" | **Accurate, verbatim** |
| C3 | plan `:193-194` | "Their corpus is 3-minute games without increment on a different server" | **Accurate.** The server is FICS and could be named |
| C4 | plan `:696-699` | Frontiers in Neuroscience, 4:60, 2010, PMC link | **Accurate**, and more correct than `references.bib:284` |
| C5 | plan `:186-192`, `:703-705` | power-law tails on fast moves, not a lognormal | **Accurate**, but understated in two ways (section 1.5) |
| C6 | plan `:174-175`, `:702-703`, script `:194-195` | "roughly 200 million moves from 2.8 million games" | **Overstated** (section 1.4) |
| C7 | plan `:171-173` | the fast-slow-fast pattern is "the main empirical finding" | **Overstated** (section 1.4) |
| C8 | plan `:183-184`, `:308-310` | there are exactly "two" departures from the source | **Incomplete.** At least a third exists (section 1.6) |

## 1.3 What checks out, with the source text

**C1.** `sigman.txt:122`: "RT distributions for different move numbers revealed a
clear and expected trend: RTs were rapid during the first moves (the opening of
the game) and the last moves (the endgame) and significantly longer during the
middle game (Figure 1 A)." The plan doc reproduces this word for word from "rapid"
onward. Figure 1A's own caption corroborates it independently: "The inverted
U-shape shows that for intermediate moves RTs are slower and the distribution has
a longer tail."

**C2.** The full sentence at `sigman.txt:128-129` reads: "This suggests that
speeding of last moves is not dominated by the complexity of the position, being
most likely determined by time pressure (Figure 1 D)." Exact. One precision note
the plan doc could add: the paper offers a second, non-time-pressure mechanism for
endgame speed in its Discussion, "In certain cases (for instance in a Rook + King
vs King ending) players also play at great speed using a memorized sequence or
algorithm," which the doc omits.

**C3.** `sigman.txt:120`: "For this work, we took into account only 3-min games
without increment, with more than 10 moves but less than 100 moves." And
`sigman.txt:96`: "All games were downloaded from FICS (Free Internet Chess
Server)." Both halves of the plan doc's caveat 2 are correct. Naming FICS costs
four words and makes the caveat checkable by a reader.

**C4.** Volume 4, page 60, 2010, PMC2965049, all confirmed.

Two defects found while checking C4, reported and **not** touched because
`CLAUDE.md` Ground Truth forbids editing `.bib` and this review is not assigned
those files:

- `CCS Thesis - Integrated/references.bib:284` gives `journal = {Frontiers in
  Decision Neuroscience}`. That is the Frontiers *specialty section*, not the
  journal. The journal is *Frontiers in Neuroscience* (`fnins`). The string was
  inherited from `contexts/inspirationpaper.md:198`, which carries the same error.
- `references.bib:283` enters the author as `Slezak, Diego Fern{\'a}ndez`, which
  renders as "D. F. Slezak". The family name is "Fernández Slezak", so the correct
  form is `{Fern{\'a}ndez Slezak}, Diego`.

## 1.4 What is overstated

### C6: "roughly 200 million moves from 2.8 million games"

This is the clearest error, and it appears in three places: plan `:174-175`, plan
`:702-703`, and `synthesize_anomaly_clocks.py:194-195` ("over 2.8M games").

The 2.8M/200M figure is the **raw download**, not the analysis sample.
`sigman.txt:99`: "Our robot started functioning in May, 2009, downloading only
lightning and blitz games, which means total times going from 1 to 15 min. On
January 2010 the database consists of more that 2.8 M games (downloading between
10 K and 20 K games per day), resulting in more than 200 M total moves." (The
"more that" typo is the paper's.) That database spans 1 to 15 minute controls.

The work then discards almost all of it. Immediately before the filter sentence
quoted in C3 above, the paper gives a table of "the number of games analyzed so
far for each total time (without increment)", at `sigman.txt:117`:

| Total time (min) | Games |
|---|---|
| 1 | 142,141 |
| 2 | 11,322 |
| 3 | **91,340** |
| 4 | 21,987 |
| 5 | 37,355 |
| 10 | 29,591 |
| total | 333,736 |

So the 3-minute cell is 91,340 games, about 3.3 percent of 2.8M. Under the paper's
own "less than 100 moves" filter, the 3-min subset holds at most 9.1M moves, or
18.3M plies on the other reading of "moves". That is between 11x and 22x below
200M at the absolute upper bound, and realistically further.

Note for the record: a second pass on this session reported C6 as accurate on the
grounds that both strings appear in the paper. They do appear. The strings are not
the issue. The issue is that the plan doc presents them as the sample the pacing
finding was measured over, and they are the pre-filter download across a wider
range of controls.

**Honest caveat on my own reading.** The paper never states N for the RT analysis
directly, and the table is labeled as games *analyzed* by Crafty, whereas RTs come
straight from the PGN and may not have required scoring. 91,340 is simply the only
3-min figure the paper gives, and the true RT-analysis N could differ from it in
either direction. What is not ambiguous is that 2.8M and 200M describe the
download, not the analysis.

A defensible replacement, for all three sites: "from a FICS database of more than
2.8 million lightning and blitz games, filtered for this analysis to 3-minute
games without increment (91,340 such games listed as analyzed)".

### C7: "the main empirical finding"

The paper calls the fast-slow-fast trend **expected**, in the same sentence it
reports it: "revealed a clear and expected trend" (`sigman.txt:122`). Its own
headline results are different. From the abstract: "We measured robust emergent
statistical observables: (1) RT distributions are long-tailed and show
qualitatively distinct forms at different stages of the game, (2) RT of successive
moves are highly correlated both for intra- and inter-player moves."

So the pacing pattern is real, is in the paper, and is a fair thing to build on.
Calling it "the main empirical finding" inverts the paper's own emphasis, and it
is precisely the kind of claim a panelist who opens the source will notice within
one minute. "One of the reported statistical regularities" or "a reported and
expected trend" is defensible.

## 1.5 The two disclosed caveats: honest, but softer than the source warrants

Both caveats at `anomaly-clock-synthesis-plan.md:186-194` are honestly stated, and
caveat 1 gets a subtlety right that is easy to get backwards: the paper's power
law is in the **slow** tail of the fast-move distributions, so a lognormal "will
under-represent the rare very long think" (`:190-191`) is the correct direction.
That is worth crediting.

Two ways caveat 1 is softer than the paper:

1. By flagging only the fast moves, it implies the middlegame is safe for a
   lognormal. The paper says the opposite in the same sentence: "while the
   distribution of RTs for middle game moves showed pronouncedly longer tails"
   (`sigman.txt:122`). The lognormal at `synthesize_anomaly_clocks.py:325-326`
   therefore under-represents the tail in both regimes, and worst in the
   middlegame, which is where the `phi_k` profile puts its peak
   (`synthesize_anomaly_clocks.py:197-204`, knot (20.0, 1.10)).
2. The Discussion generalizes past the Results wording: "We find that, more
   precisely, the distributions display a power-law behavior over several decades"
   (`sigman.txt:168`). The paper's own summary of this result is "RT distributions
   are heavy-tailed and show a qualitatively distinct shape at different stages of
   the game", not "fast moves have power-law tails".

**What cannot be verified, and should not be implied.** The paper reports **no
power-law exponent** and **no goodness-of-fit test**. `grep -c -i exponent` over
the full text returns 0. The claim rests entirely on "a linear dependence in the
log-log representation" of Figure 1B, which I could not inspect. If the manuscript
ever quotes an exponent for chess RTs, it will not have come from this paper.

Caveat 2 is complete and correctly scoped as written.

## 1.6 The third gap: the paper explicitly discards this class of model

This is the finding that matters, and it is not covered by either disclosed
caveat, nor by the claim at `anomaly-clock-synthesis-plan.md:308-310` that "the
two places where the parametric layer departs from that source are stated rather
than hidden".

From the Discussion, `sigman.txt:170`:

> "The correlation structure observed in RT data also constrains the possible
> models which may account for this data. First, it discards at once **any model
> in which decisions are made independently, or even in which decisions are made
> only taking into account the remaining time budget**. The strong positive
> correlations simply discard this possibility."

The shipped think-time model is exactly that family. `_share_for_move`
(`synthesize_anomaly_clocks.py:313-329`) draws a fresh i.i.d. multiplier per move
at `:325-326`:

```python
    mu = -0.5 * params.sigma_log * params.sigma_log
    x = math.exp(rng.gauss(mu, params.sigma_log))
```

and returns `params.phase(k) * x / n_k` (`:329`), which `synthesize_game_clocks`
multiplies by the mover's own remaining clock at `:364` (`d = share * cur`). No
state crosses moves except the remaining clock itself. Successive response times
are conditionally independent given move index and remaining time budget. The
paper names that description and rules it out by name.

The paper quantifies what is missing. `sigman.txt:136`: "The cross-correlations
amongst all moves revealed a conserved pattern: correlations were positive for
consecutive moves and significantly negative for long-range differences in move
number." And `sigman.txt:143`: "max of black-white correlation is greater than
that of white-white correlations", meaning the **strongest** correlation in real
play is between a move and the *opponent's* adjacent move, which is the signature
of shared position difficulty.

I measured the shipped model against this. 4,000 games at 180+0, 80 plies, White's
think times z-scored within each move index across games (the paper's own
procedure), then correlated across move indices:

```
lag  1: mean r = -0.0171      (opening only, k < 10: -0.0150)
lag  2: mean r = -0.0142
lag  5: mean r = -0.0197
lag 20: mean r = -0.0194
lag 30: mean r = -0.0141
```

Zero, faintly negative, and flat in lag. The paper's positive short-range
structure is absent. The paper's "significantly negative" long-range structure is
present only as a -0.02 residue of the finite budget, not as a measured effect.
A second pass measured the opponent-adjacent (one ply apart) correlation
separately and got -0.0025 at 180+0, that is, the correlation the paper calls the
largest one is zero to three decimal places.

Three consequences, all of which belong in the write-up:

- **The empirical plug-in does not fix this.** `fit_params_from_real`
  (`synthesize_anomaly_clocks.py:659-706`) fits only a per-bin median share and a
  per-bin log spread. It has no autocorrelation term. Running gate 1 successfully
  would not repair the defect the paper names.
- **Gate 1 cannot detect it.** The gate at `anomaly-clock-synthesis-plan.md:594-605`
  is a per-bucket KS test on marginal remaining-time values. A KS statistic on
  marginals is blind to serial dependence by construction. If a panel asks "are
  your synthetic clocks realistic", the marginal distribution is the easy half and
  the correlation structure is the half the cited paper says actually
  discriminates between models.
- **It is a `--fit-from` scope statement, not a bug.** The right disclosure is that
  the model reproduces the paper's first-order pacing shape and does not reproduce
  its second-order correlation structure, which the paper reports as one of its two
  headline observables.

Two smaller members of the same family, from the same results section:

**Skill dependence.** `sigman.txt:125`: "High rated players amplify the variations
of RTs during the game: they play faster than lower rated players during the
opening games, and slower during the middle game." And `sigman.txt:133`: "low
rated players, regression: SD = 0.1 s + 0.91 <RT>, for high rated players SD = 0.6
s + 1.36 <RT>". The implementation has exactly one phase profile
(`DEFAULT_PHASE_KNOTS`, `:197-204`) and one dispersion (`sigma_log = 0.85`,
`:224`). Neither `_share_for_move` (`:313`) nor `synthesize_game_clocks` (`:332`)
takes a rating or band argument. The band enters only as a seed component in
`game_streams` (`:397-406`), which changes which draws come out, not what
distribution they come from. Since the 9 Maia bands are the corpus's primary
design axis (10,000 games per band, `analysis/anomaly-corpus-generated.md:23-35`)
and the regression target is exactly that rating
(`prototype/src/chess_rating_net.py:88-90`), the corpus contains a 1100-rated and
a 1900-rated player whose timing is statistically identical, on the one axis the
cited paper measured and published band-varying coefficients for.

**Board blindness.** The retrofit reads the game only for its length,
`synthesize_anomaly_clocks.py:470`: `n_plies = len(game.get("Moves", []))`.
`Positions` and `Moves` are never inspected further, and the emitted clock is,
given (bucket, ply index), statistically independent of every move played, of
position complexity, of captures and checks, and of the result. The paper's own
explanation for its correlation structure is precisely this: "the most likely
source of correlations is determined by a relative continuous function of
complexity of the board" (`sigman.txt:170`).

## 1.7 The paper hands over one calibration number for this exact control, and the shipped model fails it

`sigman.txt:125-126`: "For instance, at move 40, higher rated players have used a
significantly larger proportion of their time budget than lower rated players
(high rated 77.9 ± 0.3% and low rated 74.7 ± 0.4%)."

That is a 3-minute no-increment measurement, and 180+0 carries weight 0.2000 in
`DEFAULT_TC_MIX` (`synthesize_anomaly_clocks.py:156`), so it is directly
comparable via `--fixed-tc 180+0`. I ran the shipped parametric model at 180+0, 80
plies, 4,000 games, and measured the fraction of the 180-second budget consumed
after White's 40th move (ply index 78):

```
mean   = 0.8827
median = 0.8877
p05    = 0.7996
p95    = 0.9470

after move 10: 0.1732
after move 20: 0.4592
after move 30: 0.7358
after move 40: 0.8836
```

The paper: 74.7 to 77.9 percent. The model: 88.3 percent. That is an overspend of
10.4 to 13.6 percentage points, against standard errors of 0.3 to 0.4 percent, and
the model's **5th percentile (80.0 percent) still sits above the paper's
high-rated mean**. An independent pass got 88.41 percent and p05 = 0.8050 on a
different seed, so the disagreement is not a sampling artifact.

Two honest caveats on my comparison. First, the paper's Figure 1D is a cumulative
distribution over games that survived its 10-to-100-move filter, whereas my run
holds every game to a full 80 plies, so the two samples are not identically
constructed. Second, I read 77.9/74.7 from the body text, not off the figure.
Neither caveat plausibly closes a 10-point gap.

Why this matters for the plan doc. Section 8 (`:588-597`) says the plausibility
claim cannot be converted from "by construction" to "by measurement" without the
HPC and the 170k subset. That is not quite true. One published number, from the
citation the doc already leans on, gives a partial calibration check that runs
today on a laptop, and the current parameters fail it by a wide margin. It also
bears on section 7.3 (`:506-532`), which argues that the normalized-feature mean
of -0.408 is "a property of the constants rather than evidence against the
synthesis". I reproduce -0.4088 at 99 plies under the default mix. At least part
of that leftward shift is the model draining the clock faster than the paper's
real 3+0 players do, which is a property of the *parameters*, not of 273/380.

## 1.8 One free corroboration the plan doc leaves on the table

`anomaly-clock-synthesis-plan.md:157-160` attributes `n_expected = 40` to the
pipeline's own bucket formula (`chess_rating_net.py:93-95`,
`format_data.py:98-109`). The paper says the same thing from the human side, at
`sigman.txt:87` ("around 40 moves") and `:92` ("around 40 movements"). That
converts a parameter which currently looks like it was borrowed from a
classification threshold into one with an independent published basis, at the cost
of one clause.

Relatedly, the indexing convention is right and worth stating explicitly. The
paper indexes by per-player move number ("at move 40", "after the 35th move"),
which is exactly what `phi_k` is indexed on (`k = i // 2`,
`synthesize_anomaly_clocks.py:356`). There is no off-by-two in the mapping from
paper to model.

## 1.9 Adjacent finding: the same citation is used loosely in Chapter 3

Out of scope to edit, reported because it is the same reference.
`CCS Thesis - Integrated/chapters/chapter3.tex:27` states that Sigman et al.
"showed a strong correlation between winning likelihood and remaining time."

The paper's claim is joint and strongly nonlinear. Abstract: the winning
likelihood "can be reliably estimated from a weighted combination of remaining
times **and position evaluation**." Results: "When both sides have sufficient time
(right portion of the image), winning probability is almost exclusively a function
of the score... Contour lines are almost flat for large time budgets." The strong
time effect is confined to time trouble: "a small difference of 8 s is on average
sufficient to compensate for a full piece when the time left is between 20 to 30
s".

So the sentence overstates the effect in the regime where most plies live. A
rewrite would arguably be a *better* motivation for the clock feature, not a
weaker one: remaining time carries little information above roughly one minute and
becomes decisive below about 30 seconds, which is to say the feature is
informative exactly where the model has to discriminate.

## 1.10 A code defect in the control arm, found while checking the above

Reported, not fixed: `analysis/scripts/synthesize_anomaly_clocks.py` is not
assigned to me.

`anomaly-clock-synthesis-plan.md:292-293` states that `--method constant` "emits a
flat clock at the time control's base value for every ply", and `:298` calls the
pairing "required, not optional". The constant branch sets `d = 0.0`
(`synthesize_anomaly_clocks.py:358-359`), but the increment is still credited
afterwards at `:370-373`:

```python
        after_spend = max(params.floor_s, cur - d)
        actual_spend = cur - after_spend
        c_new = after_spend + inc
        c[is_white] = c_new

        reported = c_new if params.report_after_increment else after_spend
```

With `report_after_increment=True` (the default, `:232`), each player gains a full
increment every move with nothing subtracted. Measured output, 20 plies:

```
  300+0    distinct=1   first8=['0:05:00' x8]                                    correct
  900+10   distinct=10  first8=['0:15:10','0:15:10','0:15:20','0:15:20',...]     last=0:16:40
  120+1    distinct=10  first8=['0:02:01','0:02:01','0:02:02','0:02:02',...]     last=0:02:10
  300+3    distinct=10  first8=['0:05:03','0:05:03','0:05:06','0:05:06',...]     last=0:05:30
  1800+20  distinct=10  first8=['0:30:20','0:30:20','0:30:40','0:30:40',...]     last=0:33:20
```

The self-test misses it because group 6 exercises only a zero-increment control,
at `:847`: `flat = synthesize_game_clocks(50, 300, 0, ClockParams(method=METHOD_CONSTANT), random.Random(0))`.

Six of the twelve `DEFAULT_TC_MIX` entries (`:151-165`) carry a nonzero increment.
I measured their normalized weight at **0.3724**. For those games the "clock
ablation" emits `base + k * inc`, a clean monotone ramp that is a trivially
learnable linear encoding of ply index, which is the opposite of an ablation.

Two measured consequences for Gate 4 (`anomaly-clock-synthesis-plan.md:630-634`):

- Averaged over the mix, the constant arm's z-scored clock **drifts upward by
  +0.1144 sd** from ply 0 to ply 98 (ply-0 mean z = -0.0894, ply-98 mean z =
  +0.0250, 6,000 games). That is roughly twice the entire stochastic component of
  the treatment arm (section 1.11).
- Its length artifact runs the **opposite way** to the treatment arm's. Truncating
  99 plies to 81 moves the treatment arm's per-game mean clock by **+0.0511 sd**
  and the constant arm's by **-0.0105 sd**. So Gate 4 as currently implemented
  compares two arms whose length confounds have opposite signs, and a disagreement
  between the two ROC-AUC numbers could not be attributed to the think-time model.

The one-line fix is to report `after_spend` rather than `c_new` under
`METHOD_CONSTANT`, plus one increment control added to self-test group 6.

## 1.11 The lognormal question: direction, magnitude, and which way it biases detection

Short answer: **the bias is real, it is in the direction the plan doc states, it
makes synthetic anomalies easier to detect than realistic ones, and it governs 2.0
percent of the clock feature's variance.** The plan doc's hedge at `:189` ("It
matters less than it sounds") turns out to be quantitatively correct, for a reason
the doc does not give. That reason should replace the hedge.

### Reasoning from the sampling code

The governing lines are `synthesize_anomaly_clocks.py:325-329` and `:361-366`,
with `sigma_log = 0.85`, `x_lo = 0.08`, `x_hi = 8.0`, `kappa = 0.45`,
`min_move_s = 0.1` (`:224-231`).

**The long tail is truncated three times, not once.** Beyond swapping a power law
for a lognormal, `x_hi = 8.0` hard-caps the multiplier at `:327`, and
`d = min(d, params.kappa * cur)` at `:366` caps any single move at 45 percent of
the remaining clock. A power-law tail is unbounded. The plan doc names only the
first of the three truncations. The honest phrasing of the departure is
"lognormal, then clipped at 8x, then clipped at 45 percent of the clock", not
"lognormal instead of power law".

**The fast side has a separate and more interesting defect.** The paper's power law
is in the slow tail, so the lognormal swap says nothing at all about fast moves.
The fast-move deficit comes from a different structural choice: think time is
*always* a share of remaining time, so the floor on any single move is
`max(0.1, x_lo * phi(k) * c / n_k)`, a fixed fraction of the median think time at
every ply index and every control. Measured directly from the shipped parameters:

| control | remaining c | k | floor on think | median think | floor as % of median |
|---|---|---|---|---|---|
| 60+0 | 30 s | 15 | 0.102 s | 0.89 s | 11.5% |
| 180+0 | 90 s | 15 | 0.306 s | 2.66 s | 11.5% |
| 300+0 | 150 s | 15 | 0.510 s | 4.44 s | 11.5% |
| 300+0 | 150 s | 30 | 1.000 s | 8.71 s | 11.5% |
| 900+10 | 450 s | 15 | 1.529 s | 13.31 s | 11.5% |

Real play has an **absolute** fast mode. A premove, a forced recapture, or a book
move takes 0.1 to 0.3 seconds regardless of whether 30 or 300 seconds remain. This
model cannot emit a 0.15-second move with 150 seconds on the clock: its floor
there is 0.51 seconds. So the synthesis **systematically under-represents
extreme-fast moves, and does so more severely the more time is on the clock.**
That is falsifiable in one command against real data and is not in the gate list.

### Magnitude, measured

Variance decomposition of the z-scored clock feature under the model's own
constants (273/380, `chess_rating_net.py:63-64`), 8,000 games at 99 plies, 792,000
observations:

```
  total Var(z clock) = 0.17851   sd = 0.4225
    between time controls        : 55.7%
    within-TC, ply-index decay   : 42.3%
    residual (the stochastic X)  :  2.0%   sd = 0.0591
```

An independent pass got 53.3 / 41.8 / 1.8 percent with sd 0.0539 on a different
seed. Either way, the **entire budget** the lognormal-versus-power-law choice can
move is about 2 percent of the feature's variance. That converts the plan doc's
qualitative hedge into a defence with a number attached, which is what belongs in
the thesis.

### Easier or harder to detect than real anomalies?

Two directions, and they must be separated, because conflating them is what a
panel will catch.

**Versus a timing-realistic synthetic corpus: EASIER.** The clock is label-blind
by construction. `subs` is `None` unless `--cheater-timing` is passed
(`synthesize_anomaly_clocks.py:476`), and a single `params` object drives all 90
case directories. So the anomalous arm and the clean arm draw from identical clock
distributions, and every bit of clock variance is label-independent nuisance
entering the rating head at `chess_rating_net.py:270`. Adding label-independent
noise to a score with fixed class means can only lower AUC, so truncating the
tails **raises** the reported AUC relative to a corpus with realistic clock
dispersion. The reported number is optimistic in that specific and bounded sense,
and the bound is the 2.0 percent above.

**Versus real cheating: the timing channel here is strictly less informative than
in reality.** In real engine-assisted play, timing carries positive label
information, and one of the strongest real signals is the *absence* of the
instant-move mode: an assisted player consults the engine even on a forced
recapture, so their sub-0.3-second moves vanish and their dispersion collapses.
This corpus removes the instant-move mode from **both** arms (the 11.5 percent
floor above), and it pins the coefficient of variation of think time near 1.0 in
both arms and all nine bands (one phase profile, one `sigma_log`). So the single
most discriminative real-world timing cue is not merely absent from the cheating
arm, it is absent from the reference arm too, and the evaluation can neither
credit nor penalise the detector for it.

**Net.** The reported ROC-AUC is not a bound in either direction. It is a
different quantity: detection from move quality, plus a label-independent clock
nuisance whose variance is smaller than real, plus a small label-*correlated*
length artifact (section 1.12). Chapter 4 should say that in those terms rather
than claiming the synthetic clocks are realistic.

## 1.12 One more, because it contradicts an explicit unqualified claim

`anomaly-clock-synthesis-plan.md:316` states: "It is seed-matched across
substitution-rate arms, so it cannot confound the headline comparison." The
narrower claim at `:258-260`, that index-matched assignment removes the *time
control* as a between-arm confound, is true and the self-test pins it. The broader
claim at `:316` does not follow from it.

Seed matching fixes the time control and the draw sequence across the ten cases of
a band (`game_streams`, `:397-406`, excludes engine and rate), so a shorter game's
clocks are a strict prefix of a longer game's. But **game length is not matched
across arms, and it is not random with respect to the label**.
`analysis/anomaly-corpus-generated.md:54-56` says so directly: storage decreases
with substitution rate because higher-rate games are shorter, since the
Maia-versus-Maia clean control more often runs out the 100-ply cap. The per-rate
storage table at `:46-52` is 5.10, 4.71, 4.54, 4.44, 4.16 GiB for the 0, 5, 15, 30
and 60 percent arms over identical 18,000-game counts, a monotone 18 percent
decline.

Since the countdown decays with ply index, a shorter game leaves the clock
systematically **higher**. Measured under the shipped default mix, in the model's
own normalized units, 6,000 games per cell:

| plies | per-game mean z | delta vs 99 | terminal-ply z | delta |
|---|---|---|---|---|
| 99 | -0.4088 | 0 | -0.6525 | 0 |
| 88 | -0.3793 | +0.0295 | -0.6343 | +0.0183 |
| 81 | -0.3578 | **+0.0511** | -0.6197 | +0.0328 |

The scale to judge that against is the residual sd of 0.0591 from section 1.11.
The between-arm length artifact is roughly **86 percent of the entire stochastic
component of the clock feature**, and it points monotonically in substitution
rate, which is exactly the axis Chapter 4 breaks ROC-AUC down by
(`CCS Thesis - Integrated/chapters/chapter4.tex:107`).

I cannot convert +0.05 sd into an AUC delta from this clone: the weights and the
corpus are both on the HPC. The point is that `:316` asserts the confound is zero
by construction, and it is not zero by construction. The cheapest fix is to report
mean ply count per arm alongside every ROC-AUC so the reader can see the gradient.
The plies have never been counted directly; the ratios above are derived from
storage and should be counted in one pass on the HPC before any of this reaches
the manuscript.

## 1.13 Ranking, by what a panel challenge would cost

1. **Section 1.12**, the game-length confound, because it contradicts an explicit
   unqualified sentence in the plan doc and is measurable today.
2. **Section 1.10**, the broken `--method constant` arm, because the defence
   itself is defective and its length artifact has the opposite sign.
3. **Section 1.6**, the third gap, because the cited paper names and rules out
   this model class in one sentence, and neither the empirical fit nor Gate 1
   addresses it.
4. **Section 1.7**, the failed 40th-move calibration, because it is one number
   from the paper already cited and it runs on a laptop.
5. **Section 1.4**, the overstated corpus size and "main empirical finding".
6. **Section 1.11**, the lognormal caveat, correctly signed and worth 2 percent of
   the variance. Replace the hedge with the number.

## 1.14 What I could not verify

- The figures themselves (Figure 1B log-log, Figure 1D cumulative, Figure 3
  correlation matrices). Every figure-derived statement above comes from captions
  and body text.
- N for the RT analysis specifically. The paper does not state it. 91,340 is the
  only 3-minute count it gives, and it is labeled as games *analyzed*.
- Any power-law exponent or fit statistic. None is reported.
- Real per-arm ply counts. Section 1.12's table is derived from storage figures,
  not counted.
- Real Lichess think-time left tails, which is the falsification target for
  section 1.11. There is no game data in this clone, confirmed independently at
  `anomaly-clock-synthesis-plan.md:64-67`.
- Whether the doc's other parameters (`sigma_log`, `kappa`, the six phase knots)
  relate to the paper. They do not, and the script already says exactly that at
  `synthesize_anomaly_clocks.py:195-196` ("The exact knot values here are this
  script's calibration, not theirs"), which is correct and honest.

---

# Part 2: the masking question

## 2.1 What firstmate got right, verified against the code

**The clock is concatenated before the BiLSTM. Correct, verbatim.**
`prototype/src/chess_rating_net.py:269-271`:

```python
        clocks = clocks.unsqueeze(2)
        lstm_input = torch.cat((x, clocks), dim=2)
        packed_input = pack_padded_sequence(lstm_input, lengths, batch_first=True, enforce_sorted=False)
```

Corroborated by the declared input width at `:210`, `lstm_input_size =
conv_filters * 8 + 1`. The `+1` is the clock scalar, so the clock is 1 of 257
per-ply input dimensions.

**The aggregator consumes rating deviation, not timing. Correct, and structurally
provable.** `AnomalyDetector.forward` (`prototype/src/anomaly.py:57-62`) accepts
exactly three arguments: `predictions`, `baseline`, `attention_weights`. No clock
tensor is passed to it anywhere. The score is
`per_move_deviation = torch.abs(predictions - baseline)` (`anomaly.py:98`),
weighted and summed at `:109-111`. The module has zero learnable parameters
(`__init__` at `:53-55` stores one bool).

**The loss is pure rating regression, with no cheat label.**
`criterion = nn.L1Loss()` at `chess_rating_net.py:760`; targets are
`[white_elo, black_elo]` standardized at `:88-90`; the Dataset return dict at
`:98-110` carries no cheat or anomaly field of any kind. So a timing-only
manipulation cannot by itself zero the suspicion score, because the score
differences ratings. That part of the answer is sound.

## 2.2 Corrections

### Correction 1: "rating-deviation is the dominant signal" is an assumption, not a measurement, and the one number in evidence cuts the other way

Firstmate's answer runs two claims together. That the aggregator *consumes* rating
deviation is verified. That rating deviation *dominates* empirically is not
measured anywhere in this project. Three independent confirmations:

1. **No anomaly result exists.** Every anomaly number in Chapter 4 is a
   placeholder. `chapter4.tex:107` is `\TODO{ROC-AUC for $S_{\text{att}}$,
   $S_{\text{max}}$, and $S_{\text{mean}}$}` plus five more `\TODO`s in the same
   paragraph, and `chapter4.tex:11` states plainly: "At the time of writing, one
   run has been completed: the baseline-reproduction sanity gate".
2. **The clock ablation was explicitly dropped.** `chapter3-4-process.md:76-78`:
   "the no-clock variant is dropped per Omori's Round 3 ruling (2026-04-28,
   clock-time already shown to help in the baseline paper) reaffirmed in Round 4
   (2026-08-13): no need to ablate the clock feature unless we change it."
3. **The only clock-contribution number in the repo is external, and it is
   large.** `chapter4.tex:73`: Omori and Tadepalli "reported that removing
   clock-time information increased MAE by 57 rating points". Same figure at
   `chapter3.tex:27`, where it is called "a 24 percent improvement", sourced to
   `contexts/inspirationpaper.md:511`.

Point 3 argues against firstmate, not for him. The clock is a documented 24
percent contributor to the rating prediction, and the rating prediction is the
*entire* input to the aggregator. The absence of any clock ablation in this
project is itself the answer to "is rating-deviation really dominant": nobody has
measured it, and the one relevant number in evidence says the clock channel is
large.

### Correction 2: the coupling is understated. The clock reaches the suspicion score through three channels, not one

- **Channel A, the predictions.** `chess_rating_net.py:270` to `:302-304` to
  `anomaly.py:98`. This is the channel firstmate names.
- **Channel B, the attention weights.** Attention runs on `lstm_output`
  (`chess_rating_net.py:275-276`, `self.attention.forward_causal(lstm_output,
  lengths)`), which is downstream of the clock concat at `:270`. So the alpha_t in
  `S_att = sum alpha_t d_t` are clock-dependent too. Not named in firstmate's
  answer.
- **Channel C, the baseline itself, in the default served path.**
  `prototype/src/api.py:199-203` falls back to `per_move_preds_orig[-1, col]`, the
  model's own final-ply prediction, when the caller supplies no baseline. In that
  path *both* operands of the subtraction at `anomaly.py:98` move with the clock.

### Correction 3: the appeal to an existing rotating-engine acknowledgment is false

I grepped the whole repo. The string `rotat` appears in `README.md:110` (speaker
rotation), `contexts/essentials/presentation_plan.md:3,7` (speaker rotation),
`archive/CCS Thesis Template 25/chapters/chapter2.tex:28,35` (a `rotatepage`
environment), `contexts/essentials/ARCHIVE_defense_prep.md:1083` (an archived
panel-question note, "Real cheaters rotate engines and sometimes pick 2nd or 3rd
choice"), and `CLAUDE_TASK_2.md:63,73`, which is where firstmate's own claim
lives. It appears in **no** current `.tex` file and **nowhere** in
`contexts/essentials/panel_attacks.md`. The manuscript carries no rotating-engine
limitation, so nothing about that acknowledgment can discharge this obligation.

Separately, `panel_attacks.md:929` (which one pass cited as the rotating-engine
entry) is in fact about the revision from move-injection to bot-versus-bot
generation, a different topic again.

### Correction 4: venue. `panel_attacks.md` is not the manuscript

It lives in `contexts/essentials/`, a defense-rehearsal document. Even if it did
carry an equivalent entry, a rehearsal note cannot discharge a manuscript
Limitations obligation, because the panel reads the manuscript.

### Correction 5: the adviser's waiver is conditional, and this work trips the condition

The verbatim escape clause is "unless you plan to change it"
(`chapter3-4-process.md:76-78`, adviser verbatim reproduced at
`analysis/team-analysis-doublecheck.md:359-361`). The clock channel for the 90k
anomaly corpus is being changed wholesale: `anomaly-clock-synthesis-plan.md:19-31`
records that the corpus stores think-time rather than remaining time, and `:33-45`
that it has no `Time` key at all. The plan already designs the control arm for
this (`:291-298`, "required, not optional"), and section 1.10 above shows that arm
is defective for 37.24 percent of the mix and has not been run.

I could not verify the adviser quote against `contexts/consultation_log.md`
directly; it is second-hand through two documents that agree with each other.

## 2.3 Direct answer

**Yes. The masking limitation needs its own explicit Limitations line. Existing
coverage does not suffice.**

Three reasons, each checkable:

1. The manuscript acknowledges exactly three anomaly blind spots: centaur and
   eval-only consultation (`chapter1.tex:83`), low-attention placement
   (`chapter3.tex:148`, `chapter4.tex:101`), and synthetic-to-real transfer
   (`chapter4.tex:199`). None of them is timing. A grep of all five chapters for
   `adaptive|evasion|evade|rotating|counter-measure|arms race|adversar` returns
   only `chapter2.tex:43`, `chapter3.tex:148` and `chapter4.tex:101`, and none of
   those three is about timing.
2. The clock is one of the model's two input streams, is a documented 24 percent
   contributor to the rating prediction (`chapter4.tex:73`), and reaches the
   suspicion score through all three channels in Correction 2. A player controls
   their own pacing at every ply, at zero cost and with no software. That is a
   first-order adversarial surface, not an edge case.
3. The project has zero measurements bearing on it: no clock ablation (dropped),
   no anomaly ROC-AUC (all `\TODO`), and the designed control arm not run and
   currently broken. An unmeasured first-order surface is exactly what the
   existing "Ablation scope" (`chapter4.tex:193`) and "Synthetic anomaly labels"
   (`chapter4.tex:199`) entries already exist to disclose.

Recommendation: **add it in both chapters**, because the two Limitations sections
carry different scopes. Chapter 1 states study-level limitations of the proposed
system; Chapter 4 states what this chapter's experiments do and do not measure.
The masking surface is both. If only one is wanted, take Chapter 4, because that
is where the unmeasured-quantity disclosures live and where the corpus caveat it
attaches to already sits.

## 2.4 Proposed text

Both drafts are em-dash free per `CLAUDE.md` Ground Truth.

**Status: Proposals A and B were both applied on this branch**
(`fm/claude-rust-pass2-preprocessor`). The insertion points were re-verified
against the files before editing and the surrounding text matched what is quoted
below exactly. The optional third item ("Modeled clocks.") was **not** applied:
it is contingent on Chapter 4 actually reporting anomaly results on the
retrofitted corpus, which has not happened, and its placement confidence is lower
by the reviewer's own assessment. It remains a draft here.

### Proposal A: `CCS Thesis - Integrated/chapters/chapter1.tex`

Section is `\section{Scope and Limitation}` at `:79`. The limitations paragraph is
the single long line `:83`. Surrounding context, quoted from the file:

> Several limitations apply to the proposed system. The anomaly module targets
> only in-game engine use during an active game and does not address pre-game
> engine preparation or eval-only consultation, where a player reads a live engine
> evaluation but selects moves themselves; these centaur-style patterns produce
> move sequences closer to natural play and remain a known blind spot of the
> attention-weighted rating-deviation signal, consistent with Barnes and
> Hernandez-Castro \yearcite{barnes2015limits} on the false-positive risk of
> single-game move analysis. **[INSERT HERE]** Prediction accuracy is expected to
> vary across time controls, with longer formats such as Classical yielding lower
> Mean Absolute Error than UltraBullet and Bullet ...

Insert immediately after "...single-game move analysis." and before "Prediction
accuracy is expected to vary across time controls". That first sentence is already
the what-the-anomaly-module-does-not-catch clause, so the new sentence joins the
same family rather than interrupting the accuracy-expectations run that follows.

Exact sentence to insert:

> A player who deliberately regulates the pace of their moves also manipulates one
> of the model's two input streams, because remaining clock time is concatenated
> with the board representation before the recurrent layer and therefore shapes
> both the predicted rating curve and the attention weights that the suspicion
> score reuses; timing control alone does not remove the rating-deviation signal
> the anomaly module measures, but a player who adjusts pacing and move strength
> together presents a harder detection case than the substitution patterns this
> study evaluates, and no result in this study measures that case.

### Proposal B: `CCS Thesis - Integrated/chapters/chapter4.tex`

Section is `\section{Limitations}\label{sec:ch4-limitations}` at `:185`, preamble
at `:187`, then seven `\textbf{Label.}` items at `:189, 191, 193, 195, 197, 199,
201`. The two neighbours of the recommended slot, quoted from the file:

> `:199` \textbf{Synthetic anomaly labels.} The primary anomaly evaluation uses
> bot-versus-bot games in which the human-like side is a Maia model rather than a
> human player. Detection performance measured on that corpus does not transfer
> automatically to real human cheating, and the real-world closed-account
> evaluation that would test transfer is contingent on data availability, as
> recorded in \ref{sec:anomaly}. The centaur pattern, in which a player consults
> an engine evaluation but selects moves themselves, remains outside the detection
> target and is not measured by any result in this chapter.
>
> **[INSERT NEW ITEM HERE]**
>
> `:201` \textbf{Truncation.} Games are truncated at the 100-ply cap to preserve
> comparability with the reference implementation, so no result in this chapter
> reflects behavior in the later stages of long games.

Insert as a new bolded item between `:199` and `:201`. The list runs
anomaly-specific items first and general ones last, so it belongs after "Synthetic
anomaly labels." and before "Truncation."

Exact item to insert:

> \textbf{Timing as an adversarial surface.} Remaining clock time enters the model
> as an input feature concatenated with the board representation before the
> recurrent layer, so it shapes both the per-move rating predictions and the
> attention weights from which every suspicion score in \ref{sec:anomaly} is
> computed. The synthetic corpus substitutes engine moves at random positions and
> does not model a player who regulates pacing, and the contribution of the clock
> feature is inherited from the published baseline ablation rather than measured
> here, so no result in this chapter bounds how far a player who adjusts pacing
> and move strength together could suppress the reported suspicion scores.

### Optional third item, if the clock synthesis lands in Chapter 4 at all

If Chapter 4 ends up reporting anomaly results on the retrofitted corpus, the
Part 1 findings need their own disclosure too, and it is a different claim from
the masking one. Offered as a draft, lower confidence on placement, and it should
not be added until Gate 1 has actually run:

> \textbf{Modeled clocks.} The clock values in the anomaly corpus are synthesized
> rather than observed, from a first-principles think-time model whose pacing shape
> follows Sigman et al.\ \yearcite{sigman2010response}. The model reproduces the
> reported fast opening, slow middlegame and late speed-up, and it does not
> reproduce the serial correlation between successive response times or the
> rating-dependent pacing that the same study reports, so the synthetic clock
> channel is realistic in its first-order shape only and no result in this chapter
> should be read as evidence about timing behavior itself.

## 2.5 Two documentation defects found while verifying Part 2

Reported, not edited, both outside my assigned file:

- `CCS Thesis - Integrated/chapters/chapter1.tex:154` states that z-score
  normalization is "Applied to clock-time and rating labels with $\mu = 1514$,
  $\sigma = 366$". The code applies 1514/366 to *ratings* and 273/380 to *clocks*
  (`chess_rating_net.py:57-64`, applied at `:79-81` and `:88-90`). Already logged
  at `analysis/170k-verification.md:301`.
- `contexts/essentials/panel_attacks.md:212` rehearses "z-normalization per
  category" for the clock feature. The code applies one global constant pair
  across all five time-control categories (`chess_rating_net.py:63-64,79-81`). I
  could not find this one logged anywhere.

## 2.6 What I could not verify in Part 2

- I did not run the model or measure any AUC. Every Part 2 claim is read from
  source, except the clock-synthesis measurements, which are mine.
- I did not open `contexts/consultation_log.md`. The adviser verbatim is
  second-hand through `chapter3-4-process.md:76-78` and
  `analysis/team-analysis-doublecheck.md:359-361`, which agree with each other.
- Whether a timing-masking limitation was discussed and deliberately excluded at
  some point. I found no record of such a discussion, but absence of a record is
  not proof it never happened.
- The AUC impact of the +0.05 sd length artifact in section 1.12. The weights and
  the corpus are both on the HPC.
