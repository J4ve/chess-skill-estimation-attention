# Audit: the stage-2 hyperparameter sweep's methodology

Date: 2026-08-24. Scope: `CLAUDE_TASK.md`.

Subject of review: `analysis/stage2-tuning-preregistration.md` (66 lines),
`analysis/stage2-sweep-results.md` (62 lines) and
`analysis/lr3e4-5seed-confirmation-results.md` (47 lines), audited as primary
sources. `chapter3-4-process.md` §4.0 was read for orientation and is audited
here only where it restates a number.

## Headline verdict

**The adoption decision is correct and survives every correction below.**
`lr=3e-4` really is the one cell that clears the pre-registered bar, it clears it
by 12–21× the relevant noise, the 5-seed confirmation really does separate from
the stage-1 attention baseline with no overlap, and the sweep's guardrails were
real guardrails, not retrofitted ones. Nothing here calls for re-running anything.

**But five stated numbers are wrong, one statistical term is misused, and one
framing is more favourable than the evidence supports.** In order of what a panel
challenge would cost:

1. The 5-seed **test MAE mean is arithmetically wrong**: the doc says 215.38, the
   five listed per-seed values average **215.54**. The standard deviation (2.71)
   is right, which means the mean was mistyped or miscomputed, not the whole
   summary. §4.1
2. The 5-seed doc's **test-set claim has no comparator anywhere in the document** —
   "well below what would be expected under the baseline" is asserted against a
   number that is never stated. The claim turns out to be *true and significant*
   once the comparator is supplied (paired Δ = **−6.50 rating points**, t = −4.28,
   p ≈ 0.013), but as written it is an unsupported assertion. §4.2
3. **The headline "~10 Elo" improvement is the validation number.** The test-set
   improvement is **6.50 points, 64% of it.** Validation was the selection
   criterion, so the test number is the honest headline. §4.3
4. What the doc calls **"confidence intervals" are mean ± 1 sample SD**, not
   confidence intervals. Real 95% t-intervals still do not overlap, so the verdict
   stands, but the term is wrong in a document whose whole purpose is statistical
   defensibility. §4.4
5. **`--patience` is not early stopping.** The pre-registration calls factor #1
   "early-stopping patience"; `--patience` drives `ReduceLROnPlateau`. This makes
   factors #1 and #3 *both learning-rate controls*, so the one-factor-at-a-time
   design has two non-independent factors in it. §5.1
6. **The winning learning rate sits on the edge of its grid**, with a monotone
   trend running into that edge, and no pre-registered rule for extending a
   boundary hit. The sweep cannot exclude that the optimum is past 3e-4. §5.2
7. `chapter3-4-process.md` §4.0's **"13.4 points inside the bar"** is a units
   error: 13.4 is the margin in **σ_seed**; in points the margin inside the bar is
   **9.92**. §4.5
8. Two trivial rounding slips: the bar is **221.64**, not "precisely 221.63"; the
   5-seed best-val is **213.23 ± 0.78**, not 213.24 ± 0.79. §4.6

## How this was checked

Every number below was recomputed on this machine from the per-cell and per-seed
values as they appear in the source documents — nothing was taken from a summary
line. Statistics were computed with Python's `statistics` module (sample std,
n−1) and hand-checked t-intervals; the probe scripts are in this session's
scratchpad. Loss semantics and the `--patience` question were resolved by reading
`prototype/src/chess_rating_net.py`, not by reading the prose about it. No `.tex`
file was touched; proposed manuscript text is quoted for a human to place.

The one thing I could not do is run anything: no HPC access, so every claim about
what a re-run would show is reasoning, and is labelled as such.

---

# Part 1: What the loss number actually is

Worth settling first, because three of the findings depend on it.

`chess_rating_net.py:760` sets `criterion = nn.L1Loss()`, and the criterion is
applied to **de-normalised** outputs — `criterion(outputs * ratings_std +
ratings_mean, targets * ratings_std + ratings_mean)` at `:417` and `:469`.

So "best-val loss = 223.38" is **mean absolute error in rating points**, on the
same scale as test MAE. The two are directly comparable, and describing a
val-loss delta in "Elo points" is dimensionally correct.

(Terminology nit, not a finding: the corpus is Lichess, whose ratings are
Glicko-2, not Elo. The docs use "Elo points" and "rating points"
interchangeably. Harmless, but a panel that cares will notice.)

# Part 2: Were the guardrails actually followed?

## 2.1 Pre-registration before results — holds

`analysis/stage2-tuning-preregistration.md` records its own state honestly: the
stage-1 val losses were known (224.24 ± 1.64 vs 223.38 ± 0.87), the headline test
MAE was not yet computed, and no tuning run had started. The trigger condition it
states is genuinely decidable from what was known at that moment — the arms tie on
val loss, so stage 2 fires — and it does not depend on the unpeeked test number.

The document also does the thing that most distinguishes real pre-registration
from decoration: it pre-commits to the *negative* result. "If no configuration
clears 2 × σ_seed, we say so plainly," with the exact sentence it would have
published. That sentence is not the one that got published, which is what you
want to see.

## 2.2 The 2σ bar — arithmetic correct, one digit misreported

Bar = 223.38 − 2(0.87) = **221.64**. `stage2-sweep-results.md` says "≈ 221.6
(precisely 221.63)"; `chapter3-4-process.md` §4.0 repeats "≈ 221.63". Off by
0.01. Immaterial to every decision it feeds, but "precisely" is doing work in
that sentence, and it is precisely wrong.

Exactly one of the 17 cells clears 221.64 — cell 9, `tune_lr3e-4`, at 211.7163.
Verified against all 17 values; the next-best cell (14, at 222.3076) misses the
bar. That part of the table is correct as published.

## 2.3 Checkpoint selection — followed

The pre-registered rule is that best-validation checkpoint is primary with the
terminal epoch-60 number reported alongside, "since it is the one that checkpoint
selection cannot inflate." The results table carries both columns for all 17
cells. Cell 9 leads on both (211.7163 best-val, 218.4956 terminal — the best
terminal number in the table by 10.4 points). The winner is therefore not an
artifact of checkpoint selection, which is the specific failure the rule exists
to catch.

## 2.4 One-factor-at-a-time — followed as stated, but see §5.1

17 = 3+3+3+3+2+3 across the six declared factors, matching the pre-registered
grid exactly, in the pre-registered order. The control values are held as
declared. No extra cells appear, and none of the 17 is missing.

The design was executed as written. Whether "one factor at a time" is a true
description of what varied is a separate question, and the answer is no — §5.1.

---

# Part 3: The 17 cells, read as evidence rather than as a table

## 3.1 The control cells corroborate the stage-1 baseline

Cells 1, 4, 8, 10, 14 and 16 are configurationally identical to each other and to
the stage-1 attention seed-0 config. Their spread:

| cell | best-val loss |
|---|---|
| 1 `tune_patience5` | 223.03 |
| 4 `tune_wd1e-5` | 223.52 |
| 8 `tune_lr1e-4` | 223.3288 |
| 10 `tune_bs32` | 223.2054 |
| 14 `tune_drop05` | 222.3076 |
| 16 `tune_ad64` | 223.7538 |

**223.19 ± 0.50** (n=6, sample std). The stage-1 attention 5-seed mean is
223.38 ± 0.87. The two agree to 0.19 points. The claim that these cluster tightly
around ~223 and are consistent with stage-1 is **correct**, and it is a genuinely
useful internal control — it says the sweep harness reproduces the stage-1 result
it is being measured against.

**A second thing falls out of it that the docs do not draw.** These six runs share
a seed (`--seed 0`) *and* a configuration. In a fully deterministic pipeline they
would be identical. They differ by ±0.50 — which is **57% of the 0.87 seed-to-seed
standard deviation** the entire adoption bar is built on. So most of what is being
called "seed-to-seed variance" at this scale is really run-to-run
nondeterminism (cuDNN kernel selection, atomics, dataloader worker interleaving)
rather than seed effects. This does not move the bar — 0.87 is the right empirical
number to build it from either way — but it means **a single-seed cell value
carries roughly ±0.5 of noise at minimum**, which matters for §3.2.

## 3.2 Could a different seed have produced a different winner? No.

| rank | cell | best-val loss |
|---|---|---|
| 1 | 9 `tune_lr3e-4` | **211.7163** |
| 2 | 14 `tune_drop05` | 222.3076 |
| 3 | 5 `tune_wd1e-3` | 223.0199 |
| 4 | 1 `tune_patience5` | 223.03 |
| 5 | 10 `tune_bs32` | 223.2054 |

Winner-to-runner-up gap = **10.59 points**. Against every noise estimate
available:

- 21.2 × the within-config sd (0.50, §3.1)
- 12.2 × the stage-1 seed-to-seed sd (0.87)
- 13.6 × the lr=3e-4 seed-to-seed sd (0.78, §4.6)
- 4.5 × even the *worst-case* single-run discrepancy observed anywhere in this
  work (2.34 points, §3.3)

There is no plausible seed draw under which cell 14 beats cell 9. The
"close second" risk the brief asks about **does not exist here**, and that is
worth stating positively: the runner-up is not close, it is 10.59 points away in
a field whose entire non-winning range spans 225.09 − 222.31 = 2.78 points. The
winner is not a member of the same distribution as the other 16 cells.

Note also that the runner-up, cell 14 (`tune_drop05`), *is a control cell* — it is
the control configuration, not a distinct treatment. So the honest reading is
sharper still: **no non-lr treatment cell beat the control.**

## 3.3 The winning cell's number is not reproducible to better than ~2.3 points

This one is not flagged anywhere and I think it should be.

- Sweep cell 9: `--seed 0`, `--num_workers 4` → best-val **211.7163** at epoch 25.
- 5-seed confirmation, seed 0: `--seed 0`, `--num_workers 8` → best-val
  **214.0545** at epoch 30.

Nominally the same configuration and the same seed, run twice, 2.34 points apart —
**4.7× the within-config sd** of §3.1, and larger than the entire 5-seed spread
(212.00–214.05) of the confirmation itself. The only declared difference between
them is `--num_workers`, which changes dataloader worker interleaving and hence
batch composition order.

Nothing about the verdict changes: 214.05 still clears the 221.64 bar by 7.6
points. But it does mean **211.72 is the least reproducible number in the
chain**, and it is the number `chapter3-4-process.md` §4.0 headlines. The
defensible figure to quote for the adopted configuration is the 5-seed mean,
**213.23 ± 0.78**, not the single-seed sweep cell. Recommended fix in §7.

---

# Part 4: Independent recomputation of the 5-seed confirmation

Recomputed from the five per-seed rows, not from the summary lines.

## 4.1 Test MAE mean is wrong

Per-seed test MAE: 219.8945, 212.7914, 214.1221, 214.8340, 216.0566.

- Mean = 1077.6986 / 5 = **215.5397**
- Sample sd (n−1) = **2.7061**

The document states **215.38 ± 2.71**. The sd matches to two decimals; the mean is
low by **0.16**. Since the sd is right, this is a transcription/arithmetic slip in
the mean alone, not a different dataset.

Direction matters slightly: the error makes the result look **better** than it is.
Small, but it is the one place where a wrong number flatters the finding.

## 4.2 The test-set claim is asserted without a comparator

The document says test MAE "is well below what would be expected under the
baseline, validating that the improvement is genuine and not an artifact of
overfitting to the validation set." **No baseline test MAE appears anywhere in
that document.** As written, the sentence compares a number to nothing.

The comparator exists — `analysis/seed-rerun-results.md:37` gives stage-1
attention test MAE = 222.05 ± 1.04 across the same five seeds, on the same frozen
17,014-game test split. Because the seeds match, this supports a *paired*
comparison, which is the right test:

| seed | lr=3e-4 test MAE | stage-1 attention test MAE | Δ |
|---|---|---|---|
| 0 | 219.8945 | 221.20 | −1.31 |
| 1 | 212.7914 | 223.57 | −10.78 |
| 2 | 214.1221 | 221.53 | −7.41 |
| 3 | 214.8340 | 221.24 | −6.41 |
| 4 | 216.0566 | 222.68 | −6.62 |

Paired Δ = **−6.50 ± 3.40** (sd), se = 1.52, **t = −4.28, df = 4, p ≈ 0.013**,
95% CI **[−10.72, −2.29]**, excluding zero. Attention-with-lr=3e-4 beats stage-1
attention on the held-out test set in **5 of 5 seeds**.

So the claim is **true and defensible** — it just is not made. Supplying this
table is a strict improvement to the document.

One caveat worth carrying: **seed 0 is an outlier** at Δ = −1.31 against
−6.41…−10.78 for seeds 1–4. Seed 0 is also the seed with the sweep/confirmation
discrepancy of §3.3 and the worst test MAE of the five. Not a problem for the
verdict (the paired test already accounts for the spread), but if anyone asks
"which seed is doing the work," the answer is that seed 0 is doing the *least*,
and the result does not depend on it.

## 4.3 The headline improvement is the validation number, and the test number is smaller

- Best-val loss: 223.38 → 213.23, **Δ = −10.15**  ← the quoted "~10 Elo points"
- Test MAE: 222.05 → 215.54, **Δ = −6.50** (paired)

The test gain is **64% of the validation gain**; 3.64 points do not transfer.

This is exactly what you expect when validation is the selection criterion — the
val number is optimistically biased by the checkpoint selection and the tuning
decision itself, the test number is not. Nothing is wrong here. But the thesis
should headline **6.5 points on held-out test**, not 10, because 10 is the number
the selection procedure had its thumb on.

## 4.4 "Confidence intervals" are not confidence intervals

The document writes: "with **no interval overlap** (confidence intervals:
[212.45–214.03] vs [222.51–224.25])."

Those are **mean ± 1 sample standard deviation**:
213.24 ± 0.79 = [212.45, 214.03]; 223.38 ± 0.87 = [222.51, 224.25]. A ±1 SD
interval is not a confidence interval; for n=5 it is roughly a 65% interval for a
single draw, not an interval on the mean.

The actual 95% t-confidence intervals on the means (t₄ = 2.776, se = s/√5):

- lr=3e-4: 213.23 ± 0.97 → **[212.26, 214.20]**
- stage-1 attention: 223.38 ± 1.08 → **[222.30, 224.46]**

**Still no overlap, by 8.1 points.** The conclusion is untouched — and in fact
holds under the stricter reading, which is the useful thing to be able to say.
Only the label is wrong. Fix in §7.

## 4.5 "13.4 points inside the bar" is a units error

`chapter3-4-process.md` §4.0 item 4: "best-val loss **211.72** at epoch 25 …
13.4 points inside the bar, not a marginal pass."

- Points inside the bar: 221.64 − 211.7163 = **9.92**
- Margin below the stage-1 mean: 223.38 − 211.7163 = **11.66** points
- That margin expressed in σ_seed: 11.66 / 0.87 = **13.41 σ**

13.4 is the **sigma** figure, labelled as points. The sentence is not wrong in
spirit — it is a large margin either way — but it mixes units in the one place the
chapter is arguing about statistical margins.

## 4.6 Best-val summary, minor rounding

Per-seed best-val: 214.0545, 213.6439, 213.4261, 212.0000, 213.0395 →
mean **213.2328**, sd **0.7808**. Published as 213.24 ± 0.79; correct to
**213.23 ± 0.78**. Trivial in magnitude, but 213.24 is the number quoted
downstream, so it is worth fixing at the source rather than in five places later.

---

# Part 5: The design's real limitations

## 5.1 `--patience` is a learning-rate control, so two of the six factors are not independent

The pre-registration lists factor #1 as **"early-stopping patience"**, grid
{5, 10, none}. In the code:

- `chess_rating_net.py:619` — `--patience`, help text `"ReduceLROnPlateau patience"`
- `:759` — `scheduler = ReduceLROnPlateau(optimizer, "min", patience=params["patience"], factor=params["lr_factor"])`
- There is **no early-stopping break** in the training loop.

`stage2-sweep-results.md` half-catches this in its own notes — "`patience = none` is
realized as `--patience 60` … so the `ReduceLROnPlateau` LR scheduler never steps"
— and the results table confirms it: every one of the 17 cells reports a terminal
epoch-60 loss, including the `patience 5` cells. **Nothing ever stopped early.**
`--patience` controls only how many stagnant epochs pass before the LR is halved.

Two consequences:

1. **The pre-registration's factor name is factually wrong**, in the document whose
   authority comes from having been written first. That is the kind of thing a
   panel enjoys finding.
2. **Factors #1 and #3 are both learning-rate controls.** "One factor at a time"
   assumes the factors are separable things. Patience-vs-lr is not a speculative
   interaction — it is the same knob at two time scales. The patience sweep asked
   "how fast should the LR decay from 1e-4," and the answer it got cannot be
   carried to a run that starts at 3e-4, because the decay schedule interacts with
   the initial value by construction.

**And the adopted configuration takes patience=5 from that sweep and pairs it with
lr=3e-4 — a combination no cell tested.** Cell 9 ran patience=5 at lr=3e-4 (it
inherits the control), so the *specific pair* was run; what was never tested is
whether patience=5 is still the right patience at the higher lr. Given that a
higher initial LR generally wants a different decay cadence, this is the single
most likely place where an untested interaction is leaving performance on the
table.

## 5.2 The winning learning rate is on the boundary of its grid

| lr | best-val loss |
|---|---|
| 5e-5 | 231.7523 |
| 1e-4 | 223.3288 |
| **3e-4** | **211.7163** |

Monotone decreasing, and the winner is **the largest value tested**. The
pre-registration declares its budget "not extended" and contains no rule for what
to do when the optimum lands on a grid edge — which is the standard
pre-registration escape hatch for exactly this case.

So the honest statement is not "lr=3e-4 is optimal" but **"lr=3e-4 is the best of
the three learning rates tested, and the trend was still improving at the edge of
the grid."** The sweep cannot exclude that 1e-3 is better. Given the monotone run
and that 3e-4 → 1e-3 is the same ~3× step as the two the grid already took, this
is a live possibility, not a pedantic one.

I would not re-run the sweep for it. But the manuscript should not claim
optimality it did not test, and "does lr keep improving past 3e-4" is a cheap
addition to the scale-transfer check already registered as
`hyperparam-scale-transfer-2m`.

## 5.3 The five non-lr null results were all measured at a learning rate now known to be wrong

Every non-lr cell held lr at the control value **1e-4**. Cell 8 (`tune_lr1e-4`,
the control) scored 223.3288; cell 9 scored 211.7163. So the entire rest of the
sweep was conducted **11.6 points away** from the regime the model actually ships in.

The sweep's conclusion — "every other swept factor stayed within noise of the
control" — is therefore correctly stated *only* with the qualifier **at lr=1e-4**.
Whether batch size, weight decay, dropout or attention_dim matter at lr=3e-4 is
untested. The claim in `chapter3-4-process.md` §4.0 that "learning rate, not the
other five factors, is the one lever documented as mattering" is defensible about
what was *documented*, and would be indefensible if read as "the other five do not
matter." The current wording is close to that line; proposed tightening in §7.

This is the generic weakness of one-factor-at-a-time designs, but it is not
generic here: the design tuned the factor with the largest main effect **last
among the first three**, and having found it, never revisited the four factors
swept before or after it under the new value.

---

# Part 6: What I could not verify

Stated plainly rather than glossed:

- **No HPC access.** Every per-cell and per-seed number here is taken as published
  in the analysis docs. I recomputed the summary statistics *from* those numbers;
  I could not verify the numbers themselves against training logs, TensorBoard
  event files or checkpoints.
- **Whether test MAE was truly unpeeked** requires the commit history — audited
  separately in §8.
- **§3.3's attribution to `--num_workers`** is the only declared difference between
  the two seed-0 runs; I cannot rule out an undeclared difference (different node,
  different driver, concurrent load). The 2.34-point discrepancy is a fact; the
  cause is an inference.
- **§5.2's "1e-3 might be better"** is reasoning from a monotone trend of three
  points, not evidence. It could equally diverge.
