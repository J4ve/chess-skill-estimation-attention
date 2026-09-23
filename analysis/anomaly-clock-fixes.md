# Anomaly-clock synthesis: fixes for the 2026-08-18 review

Follow-up to `analysis/anomaly-clock-review.md`. Five items, each either fixed
with a re-measurement or explicitly ruled not-a-fix with reasoning.

All numbers below were produced on this machine with pure-Python probes that
import `analysis/scripts/synthesize_anomaly_clocks.py` and call its own
`synthesize_game_clocks` / `draw_time_control`. There is no numpy, no torch, no
corpus and no LaTeX toolchain on this machine, which bounds what could be
verified; each limit is stated where it applies. Where the review's seed and
mine differ the spread is small and both are quoted.

**Note on the starting point.** The task brief states that
`fm/claude-rust-pass2-preprocessor` was "already merged to `main`". It was not:
`git merge-base --is-ancestor` reports it is not an ancestor of `main`, and all
three subject files (the review, the plan doc, the script) existed only on that
branch. This branch therefore merges it into `main` first, then applies the
fixes on top, so the fix commits read as a clean diff.

---

## 1. The broken `--method constant` control arm: FIXED

**Defect.** With `--method constant` the think time `d` is set to `0.0`, but the
increment was still credited afterwards, so each player gained a full increment
per move with nothing subtracted. Six of the twelve `DEFAULT_TC_MIX` entries
carry a nonzero increment; measured normalized weight **0.3724**, matching the
review exactly.

Reporting `after_spend` instead of `c_new`, which is the one-line fix the review
proposes, is **not sufficient on its own**: `after_spend` is derived from `cur`,
and `cur` is itself the previous ply's `c_new`, so the ramp survives with a
one-increment offset. The control arm must not accrue the increment at all. The
fix zeroes the increment under `METHOD_CONSTANT`, which makes both the reported
and the carried clock sit flat at the base value under either setting of
`report_after_increment`.

**Before**, 20 plies (matches the review's table exactly):

```
  300+0    distinct=1   0:05:00 ... 0:05:00        correct
  900+10   distinct=10  0:15:10 -> 0:16:40         ramp
  120+1    distinct=10  0:02:01 -> 0:02:10         ramp
  300+3    distinct=10  0:05:03 -> 0:05:30         ramp
  1800+20  distinct=10  0:30:20 -> 0:33:20         ramp
```

**After**: all five report `distinct=1`, each pinned at its base value
(`0:05:00`, `0:15:00`, `0:02:00`, `0:05:00`, `0:30:00`).

**Drift across a game**, averaged over the mix, 6,000 games at 99 plies, in the
model's own normalized clock units (273/380):

| | ply-0 mean z | ply-98 mean z | drift |
|---|---|---|---|
| before | -0.0999 | +0.0090 | **+0.1089 sd** |
| after | -0.1021 | -0.1021 | **+0.0000 sd** |

The review measured +0.1144 sd on its seed; mine is +0.1089 on a different seed.

**Length artifact of the control arm** (per-game mean z, 81 plies vs 99):
before **-0.0099 sd** (review: -0.0105), after **exactly 0.0000** at every arm
length. This matters for gate 4 beyond the ramp itself: the control arm's
artifact previously ran *opposite* in sign to the treatment arm's, so the two
ROC-AUC numbers gate 4 compares were not cleanly comparable. They are now.

**The fix is scoped to the control arm.** The treatment arm is numerically
identical before and after: per-game mean z at 99 plies -0.4061, the 81-ply
delta +0.0525, and the variance decomposition 54.1 / 44.0 / 1.9 percent with
residual sd 0.0569, all unchanged to four decimal places.

**Regression guard.** Self-test group 6 previously exercised only a
zero-increment control (`300+0`), which is exactly the case that cannot see this
bug. It now also checks `900+10`, `120+1`, `300+3` and `1800+20` for a single
distinct value, for sitting at the base value, and for zero reported think time.
Reverting the fix on a scratch copy now fails loudly:

```
SELF-TEST FAILED
  - constant method is not constant at 900+10: 25 distinct values, 0:15:10 -> 0:19:10
  - constant method at 900+10 does not sit at the base value
```

Full self-test passes, 9 groups.

## 2. The length confound: plan doc CORRECTED, plus a new gate. Deliberately not a code fix.

**The claim was false and is now corrected.** `anomaly-clock-synthesis-plan.md`
asserted the synthesis "is seed-matched across substitution-rate arms, so it
cannot confound the headline comparison". Index matching fixes the *time
control*. It does not fix *game length*, and length is not random with respect
to the label.

Converting the review's illustrative 99-to-81 truncation into the five-point
gradient the corpus's own per-rate storage table implies (treating storage as
proportional to ply count, since each game stores per-ply position tensors,
and anchoring the clean arm at the 99-ply cap), 6,000 games per cell:

| rate | GiB | rel. length | implied plies | per-game mean z | delta vs 0% |
|---|---|---|---|---|---|
| 0% | 5.10 | 1.0000 | 99 | -0.4061 | 0 |
| 5% | 4.71 | 0.9235 | 91 | -0.3820 | +0.0240 |
| 15% | 4.54 | 0.8902 | 88 | -0.3742 | +0.0319 |
| 30% | 4.44 | 0.8706 | 86 | -0.3723 | +0.0338 |
| 60% | 4.16 | 0.8157 | 81 | -0.3536 | **+0.0525** |

Residual stochastic sd of the same feature is 0.0581 under this seed, so the
0-to-60 percent artifact is about **90 percent of the clock feature's entire
stochastic component**, and it is monotone in substitution rate, which is the
axis Chapter 4 breaks ROC-AUC down by. The review measured +0.0511 on its seed;
mine is +0.0525. Unlike the label-independent clock nuisance, this artifact is
label-*correlated*, which is the kind that can move an AUC.

**So the confound is real, and a bare wording softening would not have been
enough.** But length matching inside `synthesize_anomaly_clocks.py` is the wrong
remedy, and was rejected for four reasons:

1. The length difference is produced by `generate_anomaly_corpus.py`, in how
   games end. The synthesis only retrofits clocks onto plies that already exist
   and has no freedom to change how long a game ran.
2. Matching by truncation would discard roughly 18 percent of the plies in the
   clean arm, and those are endgame plies, where both the clock signal and the
   move-quality evidence the detector actually uses are richest. That trades a
   0.05 sd clock artifact for damage to the primary evidence.
3. Matching by padding would fabricate plies that were never played.
4. The confound is not clock-specific. Mean game length differs between arms
   whether or not clocks exist, so absorbing it inside the clock model would
   hide a corpus-level property inside one feature rather than remove it.

The remedy belongs in the evaluation protocol, so the correction ships with
**gate 5**: count plies per arm directly on the HPC (they have still never been
counted; the gradient above is storage-derived), report mean and median ply
count alongside every ROC-AUC, and if the counted gradient confirms the
storage-implied one, report ROC-AUC within ply-count strata so the comparison is
conditioned on length rather than confounded by it.

**Not verifiable here.** Converting +0.05 sd into an AUC delta needs the corpus
and the trained weights, both of which are on the HPC. That is gate 5's job.

## 3. Citation overstatements: FIXED at all three sites

- **"roughly 200 million moves from 2.8 million games"** is the raw FICS
  download across 1 to 15 minute controls, not the analysis sample. The paper's
  3-minute cell is **91,340 games, about 3.3 percent of 2.8M**. Corrected in
  `anomaly-clock-synthesis-plan.md` section 3.2 and section 10, and in the
  phase-knot comment in `synthesize_anomaly_clocks.py`.
- **"the main empirical finding"** inverts the paper's own emphasis: it calls
  the fast-slow-fast trend "expected" in the same sentence that reports it, and
  its headline results are the long tails and the serial correlation. Now
  described as "one of the reported statistical regularities", with the paper's
  actual headline results named.

**Repo sweep for other repeats: none found.** The task asked specifically about
`contexts/essentials/` and chapter text; neither overstatement appears there.
Two superficially similar hits are unrelated and were correctly left alone:
`contexts/inspirationpaper.md:385` ("GPT-2 was fine-tuned on 2.8 million chess
games", a different paper and a coincidental number) and
`contexts/essentials/ARCHIVE_defense_prep.md:512` (200 million Chess.com
members).

## 4. Two manuscript/doc defects: FIXED, plus one mirror the brief did not name

**4a. `CCS Thesis - Integrated/chapters/chapter1.tex:154`.** Stated z-score
normalization is "Applied to clock-time and rating labels with mu = 1514, sigma
= 366". Verified against the code: `chess_rating_net.py:61-64` defines two
separate pairs, ratings 1514/366 and clocks 273/380, applied at `:79-81`
(clocks) and `:85-90` (rating labels and targets). The sentence now names both
pairs separately. No em dashes introduced.

**Mirror check.** `revised/chapter1_v3.md:123` repeated the identical error and
has been fixed to match, since Ground Truth requires the markdown mirrors stay
in sync. `revised/chapter3_v3.md` does **not** repeat it: it writes the clock
constants symbolically as `mu_c` / `sigma_c` and attaches 1514/366 only to the
target ratings, which is correct, so it was left unchanged. Two legacy copies
(`revised/LEGACY_PRE_RSC/chapter1_v2.md:248` and
`contexts/essentials/ARCHIVE_defense_prep.md:729`) carry older phrasings of the
same conflation and were deliberately left alone as superseded records.

**4b. `contexts/essentials/panel_attacks.md:212`.** Rehearsed "z-normalization
per category". Verified against the code: normalization uses one global constant
pair for every game (`chess_rating_net.py:63-64`, applied at `:79-81`), and the
time-control category is computed only after that, purely to carry a label for
per-time-control error reporting (`:95,103,449-504`). Never used to normalize.
The rehearsed answer now says so, and the delivery note was updated to match so
the answer is not delivered the old way from muscle memory.

## 5. Two `.bib` defects: VERIFIED INDEPENDENTLY, then FIXED

Checked against the publisher record rather than the review's claim, as the
brief required. Both defects are real.

- **Journal.** `Frontiers in Decision Neuroscience` is the Frontiers *specialty
  section*, not a journal. The PMC record for PMC2965049 gives the journal as
  **Frontiers in Neuroscience**, DOAJ lists the article under Frontiers in
  Neuroscience, and the DOI itself (`10.3389/fnins.2010.00060`) carries the
  `fnins` journal code. Corrected.
- **Author.** `Slezak, Diego Fern{\'a}ndez` parses under the BibTeX name grammar
  as family "Slezak", given "Diego Fernández", rendering **"D. F. Slezak"**. The
  published form is **Diego Fernández Slezak**, a compound family name.
  Corrected to `{Fern{\'a}ndez Slezak}, Diego`, where the braces hold the family
  name together as one token.

**Scope note.** `CLAUDE.md` Ground Truth forbids editing `.tex`/`.bib` "as part
of documentation cleanup". These were treated as in scope, per the brief, as
content corrections traceable to a specific review finding rather than routine
cleanup. Only the two defective fields were touched; no fields were added.

**Verification, and its limit.** There is no `pdflatex` or `biber` on this
machine, so the PDF was **not** rebuilt and the rendered bibliography line was
not visually confirmed. Instead the entry was parsed with a probe implementing
the relevant BibTeX rules (split on " and " and "," at brace depth zero, braced
group as one indivisible token):

```
before:  family='Slezak'                 given="Diego Fern{\'a}ndez"  -> D. F. Slezak
after:   family="Fern{\'a}ndez Slezak"   given='Diego'                -> D. Fernández Slezak
```

Brace balance checked on both. A PDF rebuild on a machine with the toolchain is
still the last step before this reaches the manuscript.

**Not edited, deliberately.** `contexts/inspirationpaper.md:198` and `:613` carry
the same "Frontiers in Decision Neuroscience" string and are the origin the
review traced the error to, but that file is a verbatim reproduction of the
Omori and Tadepalli paper. The error is that paper's, and correcting it in place
would misrepresent the source document.
