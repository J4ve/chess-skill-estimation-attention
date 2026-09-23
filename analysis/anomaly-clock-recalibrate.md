# Anomaly-clock synthesis: pacing recalibration and Maia-band dependence

Date: 2026-08-28. Follow-up to `analysis/anomaly-clock-review.md` sections 1.6
(the skill-dependence sub-point) and 1.7 (the failed 40th-move calibration).
Sits on top of `analysis/anomaly-clock-fixes.md` (commit `5923aac`), which
closed review items 1.4, 1.9, 1.10, 1.12 and Part 2. Those are not revisited
here.

Two deliverables:

1. Re-tune the first-order pacing so the fraction of a 180 s budget consumed
   after White's 40th move at 180+0 matches Sigman et al. (2010)'s published
   74.7 to 77.9 percent, instead of the shipped model's ~88.3 percent.
2. Thread the Maia rating band into the think-time model, using Sigman's own
   published band-varying regressions, so a 1100-rated and a 1900-rated player
   no longer have statistically identical timing.

All numbers below were produced on this machine with pure-Python probes that
import `analysis/scripts/synthesize_anomaly_clocks.py` and call its own
`synthesize_game_clocks`, `draw_time_control` and `sigma_log_for_band`. There
is no numpy, no torch, no corpus and no LaTeX toolchain on this box, which
bounds what could be checked; each limit is stated where it applies. Primary
seed 42, with seeds 7 and 1337 as independent cross-checks. Probe scripts live
in the task scratchpad and are not committed (they follow the same shape as the
fixes-doc probes).

---

## 1. Deliverable 1: first-order pacing recalibration (review 1.7)

### 1.1 The defect

Sigman et al. (2010), the source this script's pacing shape is cited to, report
that at White's 40th move in 3-minute no-increment games real players have
consumed 74.7 to 77.9 percent of their clock budget (low-rated 74.7 +/- 0.4,
high-rated 77.9 +/- 0.3). The shipped parametric model consumed about 88.3
percent at that point, with its 5th percentile (80.0 percent) already above the
paper's high-rated mean. This is a parameter miscalibration, not a missing
feature.

### 1.2 The change

Two knobs, both named in the review as the ones to prefer:

- `DEFAULT_PHASE_KNOTS`: the old shape `(0.15, 0.45, 1.00, 1.10, 0.95, 0.85)`
  scaled down by about 0.68 to `(0.10, 0.30, 0.68, 0.75, 0.65, 0.58)` at the
  same knot positions `(0, 3, 7, 20, 35, 50)`. The fast-open, slow-middlegame,
  late-speed-up shape is unchanged; only the amplitude drops.
- `ClockParams.sigma_log` default `0.85` to `0.91`. This is the mid-band value
  of the Sigman dispersion mapping introduced for deliverable 2 (section 2), so
  the band-agnostic path and the `band=1500` path agree at the corpus centre.

`n_expected` stays at 40 (review 1.8 gives it an independent published basis;
the knots alone reached the target so it did not need to move). `kappa`,
`n_floor`, `x_lo`, `x_hi`, `min_move_s`, `floor_s` unchanged.

### 1.3 Before and after: 40th-move budget fraction

180+0, 80 plies, mid band (1500), 4000 games, fraction of the 180 s budget
consumed after ply index 78:

| | mean | median | p05 | p25 | p75 | p95 | sd |
|---|---|---|---|---|---|---|---|
| before | 0.8829 | 0.8871 | 0.8035 | 0.8550 | 0.9158 | 0.9469 | 0.0442 |
| after (seed 42) | 0.7653 | 0.7668 | 0.6552 | 0.7213 | 0.8116 | 0.8693 | 0.0650 |
| after (seed 1337) | 0.7670 | 0.7686 | 0.6586 | 0.7241 | 0.8116 | 0.8700 | 0.0645 |

Target for this deliverable: mean in [0.755, 0.775] (centre of the paper's
band), bulk roughly inside [0.70, 0.83]. The mean lands at 0.765 to 0.767. The
central half (p25 to p75) is [0.72, 0.81], inside the target; the wider p05 to
p95 span is [0.655, 0.869], a little broader than [0.70, 0.83] because
mid-band `sigma_log` is now 0.91 (from Sigman's RT dispersion, deliverable 2),
where the pre-change model's narrower spread was partly an artifact of the
distribution saturating against the 100 percent ceiling and the `kappa` cap at
88 percent consumed. The variance is now closer to Sigman's reported RT
dispersion, not smaller than it.

Cumulative fraction consumed by move number (White), 180+0, mid band:

| after move | before | after | review-reported (before) |
|---|---|---|---|
| 10 | 0.173 | 0.120 | 0.1732 |
| 20 | 0.460 | 0.340 | 0.4592 |
| 30 | 0.737 | 0.594 | 0.7358 |
| 40 | 0.884 | 0.768 | 0.8836 |

### 1.4 Regression guard (self-test group 10)

New self-test group 10 synthesizes 2500 games at 180+0 to 80 plies at the mid
band, computes the mean fraction consumed after ply index 78, and asserts it is
in [0.74, 0.79]. It is additive: the 9 pre-existing groups are untouched and
still pass, and the harness now prints `SELF-TEST PASSED (11 groups)`.

Reverting `DEFAULT_PHASE_KNOTS` and `sigma_log` to their pre-change values on a
scratch copy fails loudly:

```
SELF-TEST FAILED
  - 40th-move budget fraction at 180+0, mid band = 0.8824, expected in [0.74, 0.79] (Sigman et al. 2010: 0.747 to 0.779)
  - mid band 40th-move fraction 0.8818 left [0.74, 0.79] under the band model (deliverable 1 must still pass at the mix / mid band)
exit=1
```

### 1.5 Drift, length and variance probes (acceptance 3)

The review and `anomaly-clock-fixes.md` use a `(time control, ply index)`-cell
variance decomposition (between time controls / within-TC ply-index decay /
residual stochastic term) plus a ply-0-vs-ply-98 drift probe and a 99-vs-81-ply
per-game length-delta probe. Re-run over the default TC mix, 8000 games at 99
plies, seed 42, in the model's own normalized units (273/380):

| quantity | before | after |
|---|---|---|
| normalized-feature mean (grand mean z, 99 plies) | -0.4054 | -0.3377 |
| total Var(z clock) | 0.16926 | 0.19833 |
| between time controls | 55.6% | 70.5% |
| within-TC ply-index decay | 42.4% | 27.7% |
| residual (stochastic X) | 1.9% | 1.9% |
| residual sd | 0.0574 | 0.0610 |
| drift: ply-0 mean z | -0.0882 | -0.0875 |
| drift: ply-98 mean z | -0.6508 | -0.5836 |
| drift: ply-0 to ply-98 delta | -0.5626 | -0.4962 |
| per-game mean z, 99 plies | -0.4054 | -0.3377 |
| per-game mean z, 81 plies | -0.3541 | -0.2882 |
| length delta (81-ply minus 99-ply per-game mean z) | +0.0513 | +0.0495 |

The "before" column reproduces the review (55.7 / 42.3 / 2.0 percent, residual
sd 0.0591) and the fixes doc (54.1 / 44.0 / 1.9 percent, residual sd 0.0569)
within the seed spread.

What moved and why:

- The numbers moved, as the brief said they would. Nothing regressed
  qualitatively.
- The between-arm game-length artifact is essentially unchanged: +0.0513 to
  +0.0495 sd. This recalibration neither fixes it nor worsens it, exactly as
  expected: it is a corpus property produced by `generate_anomaly_corpus.py`,
  and it stays gate 5's job. Relative to the residual sd it is now about 81
  percent (was about 90 percent), because the residual sd rose slightly.
- The ply-0 to ply-98 drift is shallower (-0.5626 to -0.4962) because the
  countdown now drains less over the game, which is the whole point of the
  recalibration.
- The residual (stochastic) share of variance is unchanged at about 2 percent
  and is still the smallest of the three terms. Its absolute sd rose from 0.057
  to 0.061 because mid-band `sigma_log` is now 0.91 rather than 0.85 and the
  clock is less saturated against the `kappa` cap, so more of the multiplier's
  spread reaches the reported value.
- The between-TC share rose (55.6 to 70.5 percent) and total variance rose
  (0.169 to 0.198) because the within-game ply-index decay shrank in absolute
  terms (shallower countdown) while the between-control mean differences did
  not, so the same between-control spread is now a larger fraction of a smaller
  within-game component.

### 1.6 Plan doc sections 7.3 and 8 (acceptance 4)

`--simulate 9000 --seed 42` now reports a normalized-feature mean of -0.34
(was -0.408), std 0.444 (was 0.410), max 4.276 (was 4.227).

Re-apportioning the old -0.408, honestly: of the roughly 0.41 units of leftward
shift, about 0.07 was the think-time model draining the clock faster than the
paper's real 3+0 players do (an overspend in the parameters), and this change
removes that part. The remaining -0.34 is a property of the 273/380 constants
themselves (which `analysis/170k-verification.md:291-294` shows have no
published source and do not fit this project's corpus), compounded by the
blitz-and-bullet-dominated mix (most plies have well under 273 s remaining) and
the 100-ply cap (truncates before the long tail of a classical game). The
plan doc sections 7.3 and 8 are updated to state the new -0.34 and drop the
implication that the whole -0.408 was "a property of the constants".

---

## 2. Deliverable 2: Maia-band rating dependence (review 1.6)

### 2.1 The defect

The corpus's primary design axis is the 9 Maia bands (1100 to 1900, 10,000
games each). The shipped model used one phase profile and one `sigma_log` for
every band, so a 1100-rated and a 1900-rated player had statistically identical
timing, on the exact axis Chapter 4 breaks ROC-AUC down by and the exact axis
Sigman published band-varying coefficients for. The band entered synthesis only
as a seed component in `game_streams`, which changes which draws come out, not
what distribution they come from.

### 2.2 The mapping

A normalized skill scalar `s = clamp((band - 1100) / 800, 0, 1)`: 0 at band
1100, 1 at band 1900, 0.5 at 1500. `s` is a function of band alone, so it is
identical across the five substitution-rate arms of a band (game i of
`1500_..._r00` and `1500_..._r060` get the same `s`); it varies only across the
nine band groups. Seed matching across rate arms is therefore preserved, and
the confound the review's section 1.12 is about is not reintroduced.

**Dispersion.** Sigman's Results section gives the within-player SD of response
time against its mean: "for low rated players, regression: SD = 0.1 s + 0.91
<RT>, for high rated players SD = 0.6 s + 1.36 <RT>". The slope is the
dimensionless dominant term, and is the linear-space coefficient of variation
of RT once the mean RT exceeds a second or two. The two slopes are interpolated
linearly in skill and read directly as the coefficient of variation of the
unit-mean multiplier X:

```
CV_X(s) = 0.91 + (1.36 - 0.91) * s
sigma_log(s) = sqrt(ln(1 + CV_X(s)^2))
```

which runs 0.78 (band 1100), 0.91 (band 1500), 1.02 (band 1900). The
sub-second intercepts (0.1 s, 0.6 s) are dropped: at the mean think times in
this corpus's buckets (about 1 to 24 s) they move CV_X by at most about 0.2,
and folding them in requires assuming a single reference RT, i.e. introducing a
new invented constant, which the review specifically warns against.
`ClockParams.sigma_log` (now 0.91) is used only when no band is supplied and
equals `sigma_log(0.5)`.

**Phase shape.** Sigman: "High rated players amplify the variations of RTs
during the game: they play faster than lower rated players during the opening
games, and slower during the middle game." The phase profile gets a small,
monotone-in-band see-saw:

```
tilt = 2*s - 1                      # -1 at band 1100, +1 at band 1900
k < 7      : phi_k *= (1 - 0.11 * tilt)     # opening: faster as band rises
7 <= k <= 45 : phi_k *= (1 + 0.06 * tilt)   # middlegame: slower as band rises
k > 45     : phi_k unchanged
```

The two gains (0.11, 0.06) are calibrated so the 40th-move budget fraction runs
from about 0.747 at band 1100 to about 0.783 at band 1900, matching Sigman's
own "at move 40" figures (74.7 percent low-rated, 77.9 percent high-rated). The
adjustment is zero at the centre band, so deliverable 1's mid-band calibration
is unaffected by it.

### 2.3 Threading

`_share_for_move` and `synthesize_game_clocks` take a `band` argument.
`retrofit_corpus` passes the band it already parses from the case directory
name (`{band}_{engine}_r{rate}`) straight through; the self-tests construct the
band directly and need no corpus. `simulate` passes the per-game band it
already assigns. The empirical `--fit-from` layer is unchanged and carries no
band coordinate; `band` modulates the parametric layer only.

### 2.4 Before and after: the band now changes the distribution

180+0, 80 plies, 6000 games per band, seed 42 (seed 7 in parentheses where it
differs). "RT CV" is the realized coefficient of variation of the per-move
think times.

| band | sigma_log | move-40 budget frac | sd of that | RT CV |
|---|---|---|---|---|
| 1100 | 0.777 | 0.749 (0.750) | 0.056 | 0.99 |
| 1200 | 0.812 | 0.754 | 0.058 | 1.03 |
| 1300 | 0.846 | 0.758 (0.759) | 0.061 | 1.09 |
| 1400 | 0.879 | 0.763 (0.762) | 0.063 | 1.13 |
| 1500 | 0.910 | 0.766 (0.767) | 0.066 | 1.17 |
| 1600 | 0.940 | 0.768 (0.770) | 0.068 | 1.22 |
| 1700 | 0.969 | 0.774 (0.775) | 0.069 | 1.26 |
| 1800 | 0.997 | 0.777 (0.776) | 0.071 | 1.29 |
| 1900 | 1.023 | 0.779 (0.779) | 0.074 | 1.33 |

Before this change every row was identical in distribution (mean 0.883, RT CV
about 1.0, sd 0.044), differing only in the random draws.

Against Sigman: the paper's move-40 figures are 74.7 percent (low-rated) and
77.9 percent (high-rated); the model gives 74.9 and 77.9. The paper's SD slopes
are 0.91 (low) and 1.36 (high); the model's realized RT CV runs 0.99 to 1.33.
Both the direction and the magnitude match. The realized RT CV sits slightly
above the raw slope at the low end because the phase profile contributes a
little dispersion of its own on top of the multiplier.

### 2.5 Regression guard (self-test group 11)

New self-test group 11 asserts, from 2000-game probes at bands 1100 / 1500 /
1900:

- the 40th-move budget fraction is materially higher at 1900 than 1100
  (Sigman's direction), and monotone across the three;
- the mid band still lands in [0.74, 0.79] (deliverable 1 still holds at the
  mix / mid band);
- the per-move think-time dispersion is materially larger at 1900 than 1100,
  and monotone;
- at a fixed RNG seed, bands 1100 and 1900 produce different clock strings
  (the band reaches the model through a parameter, not just the seed).

Neutralising the band model on a scratch copy (`PHASE_OPEN_GAIN` and
`PHASE_MID_GAIN` to 0, `SIGMAN_CV_HI` equal to `SIGMAN_CV_LO`) fails loudly:

```
SELF-TEST FAILED
  - 40th-move budget fraction not materially higher for band 1900 than band 1100 (1100=0.7695, 1900=0.7702); Sigman reports high-rated higher (74.7% vs 77.9%)
  - 40th-move budget fraction not monotone in band (1100=0.7695, 1500=0.7686, 1900=0.7702)
  - per-move think-time dispersion not materially larger for band 1900 (CV: 1100=1.005, 1900=0.993); Sigman SD slopes are 0.91 (low) vs 1.36 (high)
  - think-time dispersion not monotone in band (CV: 1100=1.005, 1500=1.007, 1900=0.993)
  - band did not change the clocks at a fixed RNG seed (still seed-only)
exit=1
```

---

## 3. Scope note for the plan doc, not written here

The rating-dependence work touches the same results section of Sigman et al.
that carries the serial-correlation observable (review 1.6 main point), which is
explicitly out of scope for this task and is being handled by a drafted
Limitations item, not code. The plan doc's new band-dependence subsection
(section 3.8) carries one sentence recording that the band model reproduces
Sigman's first-order band-varying pacing and dispersion but not the
second-order serial correlation between successive response times, so a reader
does not infer that threading the band closed that gap. No manuscript `.tex`
was touched.

---

## 4. What could not be checked here

- No torch and no corpus on this box, so the retrofit entrypoint
  (`retrofit_corpus`) was exercised only through self-test group 8's constructed
  fixture (plain-list Positions), not against the real 90k pickles. The band it
  passes through is the one it already parsed; the change is a single extra
  keyword argument on a call it already made.
- No real Lichess clock traces, so the Sigman comparison is against the paper's
  published body-text figures (74.7 / 77.9 percent at move 40; SD slopes 0.91 /
  1.36), not a re-derived dataset. `--fit-from` still has no real input.
- No LaTeX toolchain, so no manuscript numbers were recomputed and nothing was
  rebuilt.
- Converting the recalibration or the band spread into a ROC-AUC delta needs
  the trained attention checkpoint and the corpus, both on the HPC (gates 1 and
  5). That is a separate downstream step and is unchanged by this task.
