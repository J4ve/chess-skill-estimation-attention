> **Status update (2026-08-17, `fm/verify-followup`).** This audit was written when all 6
> headline findings below were open. Status now:
>
> 1. **Best-val-selection effect, no seed control** — **resolved.** Seed-controlled rerun
>    (5 seeds/arm, `--split_seed 42`) landed: best-val loss 224.24±1.64 (baseline) vs
>    223.38±0.87 (attention), a statistical tie; test MAE 223.64±1.74 vs 222.05±1.04.
>    `analysis/seed-rerun-results.md`, `chapter3-4-process.md` "Current status (2026-08-16)".
> 2. **Arms not controlled (no seed anywhere)** — **resolved**, same rerun as (1); seeds are
>    now fixed and swept explicitly.
> 3. **Split fragility / paired bootstrap uncomputable (no per-game errors saved)** —
>    **resolved.** Per-game error dumps now exist (`models/<exp>_per_game_errors.csv`, HPC)
>    and the paired bootstrap has been computed: attention beats baseline in 4/5 seeds with
>    the per-seed CI excluding 0 in those 4 (seed 3 is the exception, CI straddles 0). See
>    "Paired bootstrap (per-game test errors)" in `analysis/seed-rerun-results.md` and
>    `analysis/scripts/paired_bootstrap.py`. Note the seed-averaged ("seed-blind") version of
>    this same bootstrap reproduces almost exactly the false-confidence failure mode this
>    audit predicted at §G (a tight CI entirely below 0) — the per-seed table, not that
>    number, is the one to cite.
> 4. **Player leakage un-measurable / about to become permanently unfixable** — **resolved.**
>    White/Black usernames are now recorded at preprocessing (landed separately, prior to
>    this task; verify current `prototype/src/format_data.py` if re-auditing).
> 5. **Live demo does not reproduce (random-init attention head)** — **resolved.** The demo
>    now serves the baseline checkpoint instead of a random-init attention head (landed
>    separately, prior to this task).
> 6. **Task-brief premise wrong: baseline is already bidirectional, not a BiLSTM "swap-in"**
>    — **acknowledged in the manuscript**, not a code fix. `chapter3.tex` §3 (bidirectionality
>    subsection, around line 333) documents the inherited lookahead property and the
>    final-ply/retrospective-curve distinction directly. `chapter1.tex:19` was additionally
>    softened (this task) so its move-by-move framing does not imply causal/streaming
>    inference beyond what §L1 below establishes. Prefix-recompute (the fix that would make
>    per-move inference genuinely causal) remains **out of scope / not implemented** — see
>    §L1 below for the analysis and recommendation.
>
> Two items from this audit are explicitly **out of scope** for the above: the "deeper CNN"
> claim (F2) - resolved separately, 2026-08-17: Omori's FINAL ruling keeps the deeper CNN as a
> planned, sequenced stage-2 step rather than dropping it (see chapter3-4-process.md §4.3
> "Current status (2026-08-17)"), not left open - and prefix-recompute itself (§L1's
> recommendation (a), not implemented - wording was adjusted instead of the behavior, which
> remains open by design).

# 170k Verification — methodology audit, evaluation design, and adviser questions

Senior-review pass over the 170k baseline-vs-attention ablation and the manuscript,
prepared as input to the full ~1.2M paper-faithful run. Read-only: no training was
run, no code or `.tex` was modified.

All line references are to this repository at commit `9e62325`. Claims are marked
**[verified]** where I read the cited line myself, **[derived]** where the number is
arithmetic over quoted artifacts, and **[open]** where the repository cannot settle it.

---

## Headline conclusions

1. **The "~2.5 Elo attention win" is a best-val-selection effect, not a converged-model
   effect.** At epoch 60 the two arms differ by **0.49 Elo** (228.952 vs 228.462), not 2.45.
   The headline lives entirely in the gap between where each run's best-val pick landed
   (e33 vs e18). Each pick is individually sound — attention's ≈223.2 level recurs at four
   separate epochs — but with no seed control the 2.45 cannot be attributed to attention
   rather than to the initialisation draw. §B, §G.
2. **The two ablation arms were not controlled.** No seed is set anywhere in
   `prototype/src/` — the arms differ in weight init, dropout masks and batch order as
   well as in the attention flag. The claim that they "differ **only** in `--use_attention`"
   (`prototype/experiments/results/170k-ablation/README.md:37`) is false. §B.
3. **The split fragility is real but did not bite this run.** Loss-level arithmetic
   excludes a wholesale re-shuffle across the reboot. Identity of the two arms' test sets
   is nonetheless *unproven*, and no per-game errors were saved, so the paired bootstrap
   **cannot be computed from the committed artifacts at all**. §G.
4. **Player leakage is real, is currently un-measurable, and is about to become
   permanently un-fixable** — but the full run has to re-preprocess anyway, so the fix is
   free if taken now and forfeited if not. §A.
5. **The live demo does not reproduce and never did.** It serves baseline predictions
   corrupted by randomly-initialised attention, and because nothing seeds that init, the
   same PGN gives different answers after every API restart. §M1.
6. **One task-brief premise is wrong:** the thesis did not "swap in" a BiLSTM. The
   baseline paper is itself bidirectional. The lookahead problem is inherited, which
   converts the biggest liability into a contribution. §L1.

---

## A. Player-leakage audit (panel-critical)

### A.1 What the code does

The split is over *files*, not players, and the file list is unsorted:

- `prototype/src/chess_rating_net.py:540` — `all_files = [... for f in os.listdir(data_dir) ...]` **[verified]**
- `prototype/src/chess_rating_net.py:544-545` — nested `train_test_split(..., random_state=42)`, test 10% then val 20% of the rest **[verified]**

Realised sizes for N = 170,138: train 122,499 / val 30,625 / test 17,014 — exactly
72.00 / 18.00 / 10.00 % **[derived]**.

### A.2 The same player's games *can* land in both train and test — and almost certainly do

Player identity is destroyed at preprocessing. `parse_game` returns
`{WhiteElo, BlackElo, Result, Clocks, Positions, Moves, Time}` and never reads the
`White` / `Black` PGN headers (`prototype/src/format_data.py:72-83`) **[verified]**;
the on-disk schema confirms it (`hpc-data-setup.md:66-68`) **[verified]**. So the
leakage cannot even be *measured* on the existing 170k, let alone controlled.

The severity is much higher than a naive "30k out of ~90M games/month" estimate
suggests, because of *how* each month was sampled. The preprocessor streams the monthly
archive and kills `curl` once `--max-games` is reached (`hpc-data-setup.md:81-83`,
`chapter3-4-process.md:26`) **[verified]**, and the logs show a 98.8% keep rate —
`DONE scanned=30351 kept=30000` (`hpc-data-setup.md:54`) **[verified]**. That is a
sequential read from the head of the archive: the 30,000 games are the *first* 30,000
of the month, not a sample spread across it.

Lichess monthly dumps are ordered chronologically. If that holds, 30,000 games out of
roughly 90–100M in a month is ≈0.03% of the month — a contiguous window on the order of
**10–15 minutes of site-wide play** **[derived, one assumption]**. Within a 15-minute
window an active bullet player appears repeatedly, so repeat players are not a tail risk,
they are the common case. A game-level random split then scatters one player's games
across train, val and test.

> **Verification, one command, no retraining:** re-stream the first 30k games of any one
> month and print the `UTCDate`/`UTCTime` of the first and last kept game, plus
> `len(set(usernames))` vs `2 × n_games`. This settles both the window width and the
> repeat rate. Do this before the full run.

### A.3 Direction and rough size of the inflation

Leakage of this kind inflates apparent accuracy: the model can learn player-specific
style→rating shortcuts that do not generalise. I will not put a number on it —
the honest position is that it is **unmeasured**, and §A.2 gives the measurement.

Two things bound the concern usefully:

- The label is a *per-player* quantity (Glicko-2 rating at game start), and it is nearly
  constant across a player's games within a 15-minute window. So a memorised
  username→rating mapping is worth almost the entire label. This is the worst-case
  structure for game-level splitting.
- Both the baseline paper and the reference code split by game with no player control
  (`contexts/inspirationpaper.md:407-408`, plain random 80/20) **[verified via audit lane]**.
  So the published 182 carries the same optimism.

### A.4 Prescription

**Do not** simply argue it away. Do this instead:

1. **Record usernames at preprocessing time for the 1.2M run.** Add
   `"White": game.headers.get("White"), "Black": game.headers.get("Black")` to the dict at
   `prototype/src/format_data.py:75-83`. This is a two-line change and it costs nothing
   *now* because the full run re-preprocesses from scratch regardless. If it is skipped,
   the player-disjoint split becomes impossible for the rest of the thesis without a
   second 40-month re-stream.
2. **Report both splits.** Game-level random (comparability with 182) as the primary
   table, player-disjoint as the honesty check, with the delta stated. Framing:
   *"the published benchmark's split convention inflates both its number and ours by the
   same mechanism; here is the size of that inflation, which the baseline never measured."*
   That is a contribution, not a concession.
3. **Keep the disclosure.** `chapter3.tex:177` and `chapter3.tex:211` already state the
   limitation plainly **[verified]** — an earlier over-claim ("no single Lichess username
   appears in more than one partition") was already softened, per
   `contexts/model-improve-findings.md:56-63`. Once (1) is done, upgrade that paragraph
   from an apology to a measurement.
4. **Fix the rehearsed panel answer, which currently contradicts the manuscript.**
   `contexts/essentials/panel_attacks.md:472` (entry M7) still answers the leakage question
   by asserting the player-level guarantee that `chapter3.tex:211` explicitly disclaims
   **[verified via audit lane]**. Delivering that answer at the defense would be a false
   statement to the panel about the thesis's own content, and it cites a section under a
   name that no longer exists (renamed "Data Partitioning",
   `contexts/essentials/revisions_official_td_rsc.md:102`). Replace it with the `ch3:211`
   concession plus the schema explanation. This is the highest-priority correction in the
   panel-prep set.

---

## B. Multi-seed protocol

### B.1 The ablation was not a controlled comparison

There is **no seeding anywhere in the training path**. Repo-wide, `torch.manual_seed`
appears only in throwaway scripts under `prototype/experiments/`; `prototype/src/` has
none, and there is no `cudnn.deterministic`, no DataLoader `generator=`, and no
`worker_init_fn` **[verified]**. `random_state=42` at
`chess_rating_net.py:544-545` seeds **only the data split** **[verified]**.

So the baseline and attention arms differed in:

- initial weights (including `v = torch.randn(...)`, `attention.py:64`),
- dropout masks (`dropout_rate=0.5`, `chess_rating_net.py:187`),
- minibatch order (`shuffle=True`, `chess_rating_net.py:554`).

`prototype/experiments/results/170k-ablation/README.md:37` — "They differ only in the
attention flags" — is therefore **incorrect** and must be corrected before the manuscript
inherits it.

The manuscript already promises what the code cannot deliver: "Training runs will use
fixed random seeds documented in the code repository. Each ablation will be repeated with
at least three random seeds" (`chapter3.tex:153`), repeated at `chapter4.tex:25` and in
the ablation table caption `chapter4.tex:69` **[verified]**. And `chapter3.tex:275`
conflates the two seeds outright: "Random seed: fixed at 42 (`random_state=42` in the
reference split code)" — that is the *split* seed and it does not touch initialisation.

### B.2 Why this is decisive rather than pedantic

The measured effect is smaller than the machinery used to detect it:

| Comparison | Baseline | Attention | Δ |
|---|---|---|---|
| best-val **val** loss | 225.9011 (e33) | 223.1487 (e18) | 2.752 |
| best-val **test** MAE | 224.185 | 221.734 | **2.451** |
| terminal e60 **test** MAE | 228.952 | 228.462 | **0.490** |

**[verified from `logs/abl_baseline_170k.log`, `logs/abl_attention_170k.log`,
`logs/abl_attention_170k_resume.log`, and the three eval logs]**

The test delta (2.451) essentially inherits the val-selection delta (2.752). At
convergence the arms are 0.49 apart — 0.2%.

An adversarial re-check of this section is worth recording, because it cuts against the
obvious reading. Attention's 223.1487 at e18 is **not** a lucky one-epoch dip: e19, e23 and
e30 give 223.3838 / 223.2956 / 223.2834, so the ≈223.2 level **recurs four times**
**[derived]**. Within this run the attention arm genuinely reached a lower validation
minimum than the baseline arm (whose e33 best of 225.9011 is itself only 1.56 Elo better
than e25's 227.4614). The best-val *selection* is therefore robust. The problem lies
elsewhere, and is worse:

1. **Attribution.** With no seed control (§B.1) the two arms differ in initialisation,
   dropout masks and batch order as well as in the attention flag. A 2.75 Elo difference
   in validation minima between two *single* runs cannot be attributed to attention rather
   than to the draw. One seed cannot separate them, and no amount of care in reading the
   curves will fix that.
2. **Convergence.** Whatever the best-val picks show, the two architectures end up **0.49
   Elo apart** at epoch 60. An effect that shrinks by 80% with more training is not a
   capacity effect; it is consistent with attention acting as a mild regulariser that
   changes *when* the model peaks more than *how well* — which is exactly what the training
   curves show (attention peaks at e18 with a 14.0 train-val gap, baseline at e33 with
   29.6) **[derived]**.

One provenance caveat on the whole table: the mapping from each reported test MAE to a
training epoch rests on operator annotation, not on the artifacts. All four eval
invocations print the identical checkpoint path, and the two baseline eval header lines are
byte-identical while yielding different Test Loss values **[verified]** — direct evidence
that an unlogged manual `cp` swapped the file between invocations (§F.3.3).

### B.3 Prescription

**Separate the two seeds.** Add `--split_seed` (default 42, *never* varied — it is what
preserves comparability with 182) and `--seed` (varied), and at the top of `main()`:

```python
torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
g = torch.Generator(); g.manual_seed(args.seed)   # pass to the train DataLoader
```

**N = 5 seeds, not 3.** Three seeds give a standard deviation on 2 degrees of freedom —
too wide to rule anything in or out at the 0.5–2.5 Elo scale in play. Five is the
smallest N that yields a usable sd. Run the seed replicates **at the 170k scale**, not at
1.2M: variance is a property of the training procedure and transfers; at 4.2 h/run on one
A5000, 5 seeds × 2 arms = 10 runs ≈ 10.5 h wall-clock on 4 GPUs **[derived]**. At 1.2M
the same sweep would be ~7× that and is not affordable.

**Report `mean ± sd` over seeds for both the best-val and terminal checkpoints.** The
terminal number is the one that is not contaminated by selection.

**Folding seed variance into significance.** There are two independent noise sources and
the game-level bootstrap only sees one:

- *test-sampling* noise — what the paired bootstrap over games measures;
- *training* noise — seed to seed, which the bootstrap is blind to.

With n = 17,014 and a per-game |error| sd of ~170, the paired bootstrap's standard error
on the difference is ≈0.6 Elo (assuming the two models' per-game errors correlate at
ρ≈0.9) **[derived, one assumption]**. A 2.45 Elo difference is then ~4 SE and the
bootstrap will call it "highly significant" — while the converged-model difference is
0.49 and the seed sd is unknown and plausibly larger than both. **A game-level bootstrap
alone would produce a confidently wrong answer here.**

Use a hierarchical (two-level) resample instead: on each of the 10,000 iterations,
resample *seed pairs* with replacement **and** resample *test games* with replacement,
then take the mean paired difference. The resulting interval carries both sources. Report
it alongside the plain game-level interval and say explicitly which one the claim rests on.

---

## C. Normalization decision

### C.1 What is hardcoded, and where

- `chess_rating_net.py:52-56` — `ratings_mean=1514, ratings_std=366, clocks_mean=273, clocks_std=380` as `ChessGamesDataset` defaults **[verified]**
- `chess_rating_net.py:547-549` — the datasets are constructed as `ChessGamesDataset(train_files)` with **no** normalization arguments, so the defaults always win; there are no CLI flags for them in `build_parser` (`:462-490`) **[verified]**
- `prototype/src/api.py:43-46` — the same four constants re-declared as separate literals, a second source of truth that can silently drift **[verified]**

### C.2 Provenance is asymmetric — and the manuscript contradicts itself

The **rating** constants are published: the paper gives "The mean rating is 1514, with a
standard deviation of 366" (`contexts/inspirationpaper.md:439`) **[verified via audit lane]**.

The **clock** constants are not. The paper says the clock feature is z-scored but never
publishes μ or σ (`contexts/inspirationpaper.md:437`) **[verified via audit lane]**. So
`273 / 380` has no published source and no derivation comment in the code. It is inherited
from the released implementation.

Two manuscript defects follow:

- `chapter3.tex:33` states μ_c and σ_c are "the mean and standard deviation across the
  training set" **[verified]** — they are not; they are fixed constants, as
  `chapter3.tex:180` itself then says. Chapter 3 contradicts itself two hundred lines apart.
- `chapter1.tex:154` applies the *rating* constants to the *clock* feature
  ("Applied to clock-time and rating labels with μ = 1514, σ = 366") **[verified via audit
  lane]** — wrong on its face and contradicted by `CLAUDE.md`'s own key-constants note.

### C.3 The constants do not fit this corpus

The project's own 6,000-game profile measures **Elo mean 1665.7, sd 395.8**
(`contexts/model-improve-findings.md:95-97`) **[verified]** against the hardcoded
1514 / 366. The corpus mean is 152 Elo — **0.41 sd** — above the normalisation centre.

### C.4 Recommendation: reuse the paper's constants for the control run; re-fit only in the tuning stage

**Reuse, for three reasons.**

1. For the **rating** target it is nearly a no-op anyway. The loss de-standardises before
   the L1 (`chess_rating_net.py:350-353`), so `|ŷσ+μ − (yσ+μ)| = σ|ŷ−y|`; a mis-centred μ
   only shifts what the output head must learn, which is a learnable bias. The measured
   0.41 sd offset costs a little early convergence, not final accuracy.
2. Comparability with 182 is the entire justification for the control run
   (`chapter4.tex:19`), and changing target scaling is a gratuitous divergence.
3. Re-fitting on the *whole corpus* would be a mild leak. If re-fitted at all it must be
   on the **training split only**.

**But fix the plumbing regardless**, because the *clock* case is different — clocks are an
**input**, so scale errors degrade conditioning rather than being absorbed by a bias:
with μ=273, σ=380 a classical game's opening clock normalises to (3600−273)/380 = **8.75**,
a large outlier into the LSTM **[derived]**. The 40-month corpus has a different
time-control mix from the 6-month tail, so this gets worse, not better, at 1.2M.

Concretely:
- add `--ratings_mean/--ratings_std/--clocks_mean/--clocks_std`, default to the current
  values, and thread them into `ChessGamesDataset` at `:547-549`;
- import them into `api.py` from a single module instead of re-declaring at `:43-46`;
- compute train-split statistics **and print them** on every run as a diagnostic, without
  using them in the control arm;
- fix `chapter3.tex:33` and `chapter1.tex:154`; cite the clock constants to the released
  implementation, not to the paper.

Then, in the **tuning stage only** (§I), test train-split-refitted clock constants and a
`log1p` clock transform as candidate improvements, reported separately.

---

## D. Data sampling scheme for the full run

### D.1 Uniform across the window — the manuscript and the paper already require it

The paper: 1.2M games, April 2021 – July 2024, 30,000 per month
(`contexts/inspirationpaper.md:404-405`) **[verified via audit lane]**. The manuscript
commits to the same: `chapter3.tex:157`, and `chapter3.tex:169-171` ("April 2021 – July
2024 (40 months)", "≈1,200,000", "30,000 games / month") **[verified]**. 40 × 30,000 =
1,200,000 exactly.

The current 170k is the **6-month tail** 2024-02..2024-07 (`hpc-data-setup.md:27-34`)
**[verified]** — not paper-faithful. **Recommendation: uniform 30k/month across all 40
months.** This is not a judgement call; it is already the written protocol.

### D.2 The within-month sampling must also change

Per §A.2, each month currently contributes the *first* ~30k games in the archive — a
~15-minute chronological slice. Beyond player leakage this is a representativeness
defect: one narrow window is one time-of-day, i.e. one geographic peak, with its own
time-control and rating mix. Forty such windows do not average out; they systematically
sample the same hour-of-day if the dumps start at a consistent point.

**Fix:** reservoir-sample 30,000 from the full month, or take every k-th qualifying game
with k = ⌊month_total / 30,000⌋. Reservoir sampling costs one full decompression pass per
month (no extra storage). Given that streams already had to be retried up to 3× per month
(`hpc-data-setup.md:84-87`), budget for this.

### D.3 Data size vs test MAE — the extrapolation supports 1.2M and predicts ~184–190

Two measured points from this project:

| N games | best-val test MAE | source |
|---|---|---|
| 50,138 | 249.1 | `chapter3-4-process.md:96`, `chapter4.tex:51` |
| 170,138 | 224.2 | `logs/abl_baseline_eval.log` |

**[verified]**

Fitting the two forms a panel would ask about **[derived]**:

- **log-linear** (MAE ∝ −ln N): slope = −24.9 / (ln 170138 − ln 50138) = −20.4 Elo per
  e-fold → at 1.2M: 224.2 − 20.4 × 1.953 = **184.4**
- **power law** (MAE ∝ N^−b): b = 0.0862 → at 1.2M: **189.5**

So the project's own scaling curve projects **≈184–190 at the full corpus, against the
published 182**. That is a strong, defensible answer to "why do you need 1.2M":

- at 170k you are ~42 Elo short and the gap is dominated by corpus size, not by a pipeline
  defect — consistent with train MAE already reaching 176–184 while test sits at 224;
- the projection only closes near 10⁶, so a subset is *not* defensible for the headline
  comparison — and `chapter4.tex:19` has already pre-committed to exactly that rule;
- but note the projection lands **2–8 Elo above 182**, so the manuscript must not promise
  to hit 182. `chapter4.tex:17` already handles this correctly ("a target, not a
  guarantee"); `chapter3.tex:236,238` does not (see §K).

Two measured points cannot validate a functional form. **Insert one intermediate run at
~400k** before committing the full preprocessing budget: if it lands near the predicted
~200 (log-linear) the curve is trustworthy and the full run is justified; if it plateaus
early, that is a finding in itself and saves ~150 GB and a week.

### D.4 Storage

35 GB / 170,138 games → **≈247 GB at 1.2M** **[derived]**, against ~587 GB free on a
`/home` pool that is already ~92% full (`hpc-data-setup.md:79-80`). See §F.2 — the
one-pickle-per-game layout is the thing to change, and changing it solves three other
problems at once.

---

## E. Clock / time fidelity

**Confirmed faithful.** The per-move clock is concatenated into the BiLSTM input exactly
as the manuscript's equation `z_t = [CNN(X_t) ‖ ĉ_t]` (`chapter3.tex:46`) describes:

- `chess_rating_net.py:190` — `lstm_input_size = conv_filters * 8 + 1` **[verified]**
- `chess_rating_net.py:249-250` — `clocks.unsqueeze(2)`, `torch.cat((x, clocks), dim=2)` **[verified]**

`Time` is used for the per-time-control breakdown, and matches the manuscript:

- `chess_rating_net.py:85-87` — `initial + 40*increment` → `categorize_time_control` **[verified]**
- `format_data.py:88-96` — thresholds 29/179/479/1499 **[verified]**, identical to
  `chapter3.tex:172` and `chapter3.tex:209` **[verified]**, and to the paper
  (`contexts/inspirationpaper.md:415-416`) **[verified via audit lane]**

**Non-issue, stated so it is not re-raised:** `collate_fn` pads clocks with 0.0 in
*normalised* space (= 273 s raw), which looks alarming, but `pack_padded_sequence`
(`:251`) means padded steps never enter the LSTM. Harmless.

### Deviations to flag

1. **Alignment guard missing** — see §M6. This is the one real defect in the clock path,
   and it affects training data, not just inference.
2. **The paper is internally inconsistent about what the feature is.** Five sites say
   *remaining* clock time, four say time *spent* (`contexts/inspirationpaper.md:432, 437`
   vs `:451-452, 454-455`) **[verified via audit lane]**. The implementation is
   unambiguously *remaining* (`format_data.py:61-64` harvests `[%clk]`, the post-move
   reading), and `chapter3.tex:27` follows the same reading — **so the thesis is correct**.
   Footnote the divergence, because a panelist reading the baseline's §5.2 will see
   "time spent" and ask.
3. **Time-spent is not fed in.** It is derivable from consecutive remaining values plus the
   increment, and is arguably the more behaviourally informative signal (hesitation).
   Worth one tuning-stage ablation (§I), not a control-run change.
4. **Clock normalisation constants** — §C.
5. **Low severity:** `time_to_seconds` (`format_data.py:31-34`) does `int(parts[2])` and
   will raise on a fractional-second clock (`0:00:59.9`). It has not fired, but ultrabullet
   is exactly where tenths appear. One `float()` fixes it.

---

## F. Full-scale (1.2M) run design

### F.1 Hyperparameters

**Control arm: change nothing.** batch 32, lr 1e-4, weight decay 1e-5, dropout 0.5, Adam,
lr_factor 0.5, patience 5, epochs 60, `val_batch_size 512`. Two fixes that are corrections,
not changes:

- **Set `--epochs` default to 60** (`chess_rating_net.py:467` currently 100,
  `prototype/example_config.yaml:6` also 100) **[verified]**. Every run to date passed
  `--epochs 60` explicitly, so the default silently contradicts the documented control
  protocol — a reproducibility trap and the substance of §M5.
- **Set `val_batch_size` default to 512** (`chess_rating_net.py:475` is 8192,
  `example_config.yaml:9` is 8192) **[verified]**. The shipped default OOMs a 24 GB A5000
  (`CLAUDE.md`), so the committed config cannot be run as-is.

**Add real early stopping.** `patience` is currently *only* `ReduceLROnPlateau`'s patience
(`chess_rating_net.py:594`) **[verified]**; the epoch loop at `:631` runs unconditionally
to the end. The manuscript's "patience 5" reads as early stopping and is not. The measured
cost: the baseline's last improvement was epoch 33, so **27 of 60 epochs (45%) were wasted**
and produced a *worse* model **[derived]**. At 1.2M those are ~13 h of A5000 time per run.
Either implement early stopping with a documented patience, or state in Ch3 that training
is fixed-length and `patience` refers to the LR scheduler only.

**Weight decay is effectively inert — and this is a real finding.** The loss is computed in
Elo points (`:350-353`), so gradients are ~366× larger than in standardised space. Adam is
invariant to a global gradient rescaling, *except* that `torch.optim.Adam` applies
`weight_decay` by adding `wd·θ` to the gradient **before** the adaptive normalisation.
The regularisation is therefore ~366× weaker in relative terms — effectively
`1e-5 / 366 ≈ 2.7e-8` **[derived]**. Given the measured overfitting (train 176.4 vs val
229.8 at e60), this is worth one tuning-stage sweep (§I). Do **not** change it in the
control arm — the reference implementation has the same property, so it is part of what
"faithful" means.

**Attention config:** keep `attention_dim=64`, `attention_type=bahdanau`. The module adds
16,448 parameters (`128→64` key + `128→64` query + a 64-vector, `attention.py:62-64`)
against a ≈0.76 M-parameter baseline — **+2.2%** **[derived]**.

### F.2 Memory and throughput — the two things that actually bite

**The attention op, not the model, sets the batch size.** `forward_causal` materialises
`combined` of shape `(batch, seq_q, seq_k, attention_dim)` (`attention.py:129`)
**[verified]**. At batch 32, T=100, d=64 that is 32·100·100·64·4 B = **82 MB**, and
autograd retains it for the backward pass. Batch 256 → **655 MB** per copy. The `tanh`
makes it non-decomposable, so the options are: chunk over query blocks, wrap the attention
block in `torch.utils.checkpoint`, or keep the batch small. **Recommendation:** gradient
checkpointing on the attention block, then batch 64–128. This also explains the measured
~8% per-epoch slowdown of the attention arm (4.55 vs 4.22 min/epoch) **[derived]**.

**One pickle per game does not survive 1.2M.** 1.2M small files means inode pressure, an
`os.listdir` over 1.2M entries on every process start, and a per-item `open()`+`unpickle`
in the DataLoader. **Repack into ~240 shards of 5,000 games** (a binary blob plus an index,
or WebDataset tar shards / LMDB). One change fixes four problems simultaneously:

- **throughput** — sequential shard reads instead of 1.2M random opens;
- **split determinism** — the split comes from a committed manifest, not `os.listdir` (§G.3);
- **reboot survival** — re-staging 240 files instead of re-symlinking 1.2M;
- **storage** — no symlink farm at all.

**Storage plan.** Canonical sharded corpus on `/home` (~247 GB, survives reboot); stage to
`/tmp` NVMe for the active run. Checkpoints are cheap — ~9 MB each, ~550 MB per 60-epoch
run (`results/170k-ablation/README.md:118-119`) **[verified]** — so they are not the
constraint. Do **not** hold two full 247 GB copies plus scratch on a shared pool at 92%
full; stage per-shard-group if `/tmp` is tight.

**DDP: no.** Use **four independent single-GPU runs, one per configuration**, pinned with
`CUDA_VISIBLE_DEVICES`. Reasons: (a) the ablation needs ≥4 configurations anyway, so
one-run-per-GPU is embarrassingly parallel and gives the same 4× throughput with **zero**
methodological risk; (b) DDP changes the effective batch size, and holding the global batch
at 32 to preserve comparability would leave 8 per GPU, which is inefficient and still
perturbs BatchNorm statistics. DDP buys nothing here and costs comparability.

**Mixed precision: measure before adopting.** `CLAUDE.md` records that training is
**I/O-bound** (~85 s/epoch on NVMe vs 10+ min on HDD), so AMP may not help at all. Profile
one epoch first. If adopted, use bf16 and restrict it to the tuning stage — it perturbs
numerics against the reference.

**`num_workers`:** 32 cores ÷ 4 concurrent runs = 8 each; the runs used 8
(`results/170k-ablation/README.md:46`). Keep 6–8.

### F.3 Resilience runbook

The 170k attention run was interrupted by a node reboot at epoch 58 and the recovery left
three scars (§G.1, §L11 below). Fix them before the full run:

1. **Persist scheduler and `best_epoch`.** `save_checkpoint` (`:427-445`) stores epoch,
   model, optimizer, params, `best_val_loss` — but **not** `scheduler.state_dict()` and
   **not** `best_epoch` **[verified]**. The scheduler is constructed fresh at `:594`
   *before* the resume block at `:602-606` and is never restored, so `num_bad_epochs`/`best`
   reset on resume. And `best_epoch` (initialised 0 at `:599`) is never restored — which is
   why the resume log's trailer reads `best val epoch: 0`, a value that is simply false.
2. **Fix the `best_val_loss` off-by-one.** `:648-649` writes the checkpoint with the
   *pre-update* `best_val_loss`; the update happens at `:652-653` **[verified]**. Resuming
   from `latest.pth` after a best-setting epoch restores a stale, higher best.
3. **Never overwrite `model_55.pth`.** The eval path hardcodes that filename (`:612`) with
   no `--checkpoint` flag, so the operator workflow was `cp best_model.pth model_55.pth`
   (`results/170k-ablation/README.md:62-68`) **[verified]** — but `model_55.pth` is *also*
   the genuine epoch-55 checkpoint written by training. A skipped or failed `cp` would
   silently evaluate epoch 55 and produce a byte-identical log. **Add `--checkpoint`.**
4. **Log the environment every run:** file count, split sizes, manifest hash, `data_dir`,
   resolved hyperparameters. Currently the trainer prints none of these — which is exactly
   why §G.1 cannot be settled from the artifacts.
5. tmux + `nohup`, pinned `CUDA_VISIBLE_DEVICES`, `--resume models/<exp>/latest.pth`,
   sharded data staged from `/home` (F.2) so re-staging after a wipe is a copy, not a rebuild.

### F.4 Is the full 1.2M required?

**For the headline comparison against 182: yes**, and the manuscript has already
pre-committed to that rule at `chapter4.tex:19` ("the comparison is only numerically
meaningful on the full corpus"). §D.3 shows why: at 170k you are ~42 Elo short and the
scaling curve only closes near 10⁶.

**For everything else: no.** Run the seed replicates (§B.3), the 2×2 architecture ablation
(§L7) and the tuning sweep (§I) at 170k, where a run is 4.2 h rather than ~30 h, and
transfer the *variance* estimate to the 1.2M comparison. That is the compute-optimal split
of the budget and it is defensible: variance is a property of the procedure, the headline
is a property of the corpus.

---

## G. Evaluation fixes (the three known gaps)

### G.1 Split integrity — fragility confirmed, contamination excluded, identity unproven

**The mechanism is real.** `os.listdir` (`:540`) returns entries in filesystem order;
`train_test_split` (`:544-545`) applies a fixed-seed permutation to *that order*. A
different input order gives a different partition of the same files. The inference path
falls through the *same* `:540-545` construction (`:608-625`) with no persisted split, so
every one of the five process invocations (2 trainings, 3 evals) rebuilt the split
independently **[verified]**.

**The reboot made this live.** `/tmp` was wiped at epoch 58 and the flat directory was
re-symlinked before resuming (`chapter3-4-process.md:121-122`) **[verified]**. The resume
then ran at **35.16 min/epoch** against the baseline's 4.22 (70.32 min / 2 epochs, resume
log:10) **[derived]**.

That 8× gap is **not** evidence that the file set or its ordering changed — an earlier draft
of this audit overstated it, and the correction matters because it is the kind of overreach
a panelist would catch. Two caveats: the ratio divides an *attention* epoch by a *baseline*
epoch, and attention's own pre-reboot per-epoch time was never logged (the process was
killed before the timer printed), so the comparison cannot be isolated. And the magnitude is
fully explained by storage alone — `CLAUDE.md` puts the HDD path at ~10+ min/epoch at 50k
scale; scaling by the 3.39× data ratio gives ≈34 min/epoch against the observed 35.16
**[derived]**. The sound inference is that the *storage backing* changed (the re-created
symlinks resolved to the slow `/home` HDD rather than NVMe), which says nothing either way
about file ordering.

**But a wholesale re-shuffle is excluded, by arithmetic.** If the resumed run's val split
had been redrawn, ~72% of the new val files would be files the model had trained on for 58
epochs, so val loss would collapse toward the train level: expected
≈ 0.72 × 183.5 + 0.28 × 228.7 = **196.2**. Observed at epoch 59: **228.4919** — inside the
e54–e58 band (227.75–228.87) **[derived]**. The same argument applies to the three eval
processes: a redrawn test split would have pulled test MAE to ≈191; observed 228.95 and
228.46. Residual implied contamination is ≈1.7% (baseline) and ≈0.6% (attention) — the same
size as genuine val-vs-test variation, so that is a **ceiling, not a measurement**.

**Verdict:** gross leakage ruled out; **identity of the two arms' test partitions is
unproven and must be re-established before any paired test is run.** The logs cannot settle
it — the trainer prints no file count, no split sizes, no data_dir echo, and the only
per-epoch timing goes to TensorBoard event files that were deliberately not committed
(`results/170k-ablation/README.md:109-114`) **[verified]**.

**Fix (deterministic, reboot-safe):**

```python
all_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".pkl"))   # :540
```

plus: write `split_manifest.json` (the three file lists and a SHA-256 over the sorted
basenames) on first run; on every subsequent run **load and verify** it rather than
recomputing; print the counts and the hash. `sorted()` alone fixes ordering; the manifest
is what makes the split auditable and lets an eval process prove it scored the right games.

### G.2 Paired bootstrap

**First, a blocker:** the committed artifacts contain **no per-game errors**. `test()`
computes `mae_per_item` (`:415`) and immediately folds it into five running sums
(`:416-418`), then discards it; the eval prints two lines (`:623-624`) **[verified]**.
Per-time-control *counts* are computed but never printed. **No paired test can be computed
from what exists — the eval must be re-run with per-game dumps.**

Add to `test()`: write `per_game_errors.csv` with `game_id, white_err, black_err,
time_control, white_elo, black_elo`. That one file unlocks the bootstrap, the rating-bracket
eval, the per-bin counts, the calibration plot (§L6) and the move-index curve (§L13).

**Method.**
- **Unit = game**, not move. Moves within a game are not independent (§L2), and the label
  is per-game constant, so a move-level bootstrap would inflate effective N by ~65×
  (mean 64.9 plies/game, `contexts/model-improve-findings.md:95-96`). Once usernames exist
  (§A.4), upgrade to a **player-level cluster bootstrap** — cluster by player, not game.
- **Paired**: both models scored on the *identical* test partition (which §G.1 must first
  guarantee). Per game *i*, `d_i = e_i^attn − e_i^base`.
- **Resample** games (later: player clusters) with replacement, B = 10,000, recompute
  `mean(d)`, take the 2.5th/97.5th percentiles.
- **Report the CI on the difference**, not against 182. `chapter4.tex:19` already specifies
  this correctly; `chapter3.tex:238` ("95% CI does not cross the 182 baseline") is
  **mis-specified** and must be fixed (§K).
- **Hierarchical version** per §B.3 — resample seed pairs as well as games.

**What "significant" should mean for the panel.** State all three:
1. the 95% CI on mean(d) excludes zero (statistical significance);
2. the effect exceeds seed-to-seed sd (reproducibility);
3. the effect is meaningful against the label's own noise floor — Glicko-2 uncertainty for
   provisional players is tens of Elo, so **2.45 Elo on a 224 baseline (1.1%) is
   practically negligible** even if it clears (1) and (2).

The honest headline is available already, and `chapter4.tex:163` has pre-written the
sentence for it: *"the extended architecture performs comparably to the baseline on rating
estimation while adding the anomaly-localization capability that the baseline does not
provide."* Lead with that. It is a stronger position than defending 2.45 Elo.

### G.3 Rating-bracket evaluation

Not implemented: `test()` (`:389-424`) breaks out time control only **[verified]**, while
`chapter3.tex:149,173,239` and the scaffolded table at `chapter4.tex:135-157` promise five
brackets **[verified]**.

**One design subtlety the implementation must get right:** the time-control bucket is a
property of the *game*, but a rating bracket is a property of a *player*. One game yields
two observations that can fall in different brackets. So the bracket table must aggregate
over **player-observations (2N)**, not games (N) — unlike the existing per-time-control code
which averages the two sides first (`mae_per_item`, `:386`). Report `n` per bracket.

```python
BRACKETS = [(0,1200),(1201,1600),(1601,2000),(2001,2400),(2401,10000)]
# per side s in {white, black}: bracket by the TRUE rating, accumulate |err|, count
```

**Report counts, because the existing per-time-control table is already unreliable at the
edges.** Using the 6,000-game mix (`contexts/model-improve-findings.md:95-97`), the
17,014-game test set contains roughly **8,320 blitz / 6,343 bullet / 2,183 rapid / 125
classical / 40 ultrabullet** **[derived]**. With a per-game |error| sd of ~170 that gives a
standard error of **±27 for ultrabullet** and **±15 for classical**. The reported
attention-vs-baseline differences in those two buckets — ultrabullet 223.1→211.3 (−11.8) and
classical 167.3→176.3 (+9.0) — are **0.4 σ and 0.6 σ. They are noise** **[derived]**, and
the manuscript is currently on track to report them as findings. Add per-bin `n` and CIs to
both subgroup tables, and expect the same problem in the 2401+ bracket.

---

## H. Anomaly head — architecture and ROC-AUC protocol

### H.1 Verified defects

**(a) The head is non-parametric and unreachable from training.** `AnomalyDetector`
(`anomaly.py:25-122`) has zero `nn.Parameter`s — it is pure arithmetic **[verified]**. It is
only invoked on the `return_attention=True` branch (`chess_rating_net.py:292-294`), and
`train_one_epoch` calls `model(positions, clocks, lengths)` with the default
`return_attention=False` (`:333`) **[verified]**. Therefore `--use_anomaly` changes **nothing**
about training.

> **Consequence for the planned ablation battery:** `chapter3-4-process.md:63-64` schedules
> four runs — baseline, attention-only, anomaly-head-only, full. Runs 3 and 4 would produce
> results **identical** to runs 1 and 2 at equal seed. Two of the four planned GPU-weeks buy
> nothing. Either give the head learnable parameters and a loss, or drop those two arms and
> say why.

**(b) The attention weights fed to the head are dominated by a positional artefact.**
`chess_rating_net.py:274-277` collapses the causal `(batch, seq_q, seq_k)` weight tensor to
per-key importance by **averaging over the query dimension** **[verified]**. Under the
causal mask, key *s* is only visible to queries *t ≥ s*, so column *s* has (L−s) non-zero
entries out of L. Even with uniform within-row weights, the column mean goes as
`(1/L)·Σ_{t≥s} 1/(t+1) ≈ ln(L/s)/L` — **monotonically decreasing in s** **[derived]**. Early
plies structurally receive the most "importance" regardless of content, and late plies are
suppressed. Since engine assistance concentrates in the middlegame and endgame, the
weighting actively works against the detector's purpose. The code comment at `:271-273` even
weighs taking the diagonal and picks the mean — the wrong choice.

**(c) The baseline is self-referential.** `api.py:196-201` defaults `R_baseline` to the
model's own final-ply prediction **[verified]**, so `d_t = |R̂_t − R̂_T|` measures the curve's
deviation from its own endpoint. A consistently mispredicted player scores *low*. This
**contradicts the manuscript**, which defines R_baseline properly at `chapter3.tex:213`
("the player's Lichess Glicko-2 rating for the specific time control of the game at game
start") **[verified]**. Note this corrects the scout claim in §M4: the manuscript is not
silent; the code disagrees with it.

**(d) Two of the three specified aggregations do not exist.** `chapter3.tex:126-132` defines
`S_max` and `S_mean` alongside `S_att`, with an explicit evasion rationale at `:124`, and
`chapter4.tex:93` says all three are reported jointly **[verified]**. `anomaly.py:110-120`
computes only the attention-weighted sum; `api.py:236-238` exposes only that
**[verified]**. `per_move_deviation` is already returned, so both are two lines away.

**(e) The whole per-move curve is retrospective** — see §L1. The anomaly module consumes
`per_move_preds` (`api.py:190`), every element of which has seen the rest of the game. This
is the deepest issue in the section: the current "move-level" suspicion signal is
retrospective annotation, not live detection.

### H.2 How attention should feed the detector

1. **Stop averaging over queries.** Use `weights[:, T-1, :]` — the final query's
   distribution over keys — which is a genuine "looking back from the end of the game,
   which plies mattered" profile and carries no positional artefact. Alternatively train a
   dedicated single-query attention head for the anomaly branch.
2. **Give the head parameters and a loss**, or stop calling it a head. If it stays
   non-parametric, describe it in Ch3 as a *post-hoc scoring rule* over the rating curve —
   which is what it is — and delete the anomaly ablation arms.
3. **Require `R_baseline`.** Remove the self-prediction fallback; return HTTP 400 when it is
   absent. The `_ELO_SCALE_FLOOR` assertion (`anomaly.py:88`) does not catch this, because a
   self-predicted Elo passes the scale check.
4. **Implement `S_max` and `S_mean`** and report all three, as Ch3 already promises.
5. **Fix causality before claiming move-level detection** (§L1).

### H.3 ROC-AUC evaluation protocol

*(Corpus design is out of scope per the brief; this is the evaluation protocol only.)*

- **Unit** = one game-side (one player in one game). Label = substituted / clean.
- **Score** = each of `S_att`, `S_max`, `S_mean`, reported as three separate ROC curves.
- **Pairing.** The corpus is matched clean/substituted pairs, so use a paired analysis:
  report AUC with a bootstrap CI over **pairs** (10,000 resamples), not an unpaired DeLong
  test, and keep both members of a pair in or out of a resample together.
- **Stratification.** Report AUC by substitution rate (5/15/30/60%), by engine (SF16 / Lc0),
  and by Maia band. Detection at 5% substitution is the number that matters; a pooled AUC
  dominated by the 60% case is not informative. `chapter4.tex:99` already scaffolds exactly
  this breakdown.
- **Thresholds calibrated on validation only**, never on test — `chapter4.tex:99` states
  this; enforce it in code.
- **Operating point.** Report precision/recall/FPR at a justified threshold, not just AUC
  (§L16). Report AUC descriptively; no numeric threshold is set.
- **Prevalence caveat.** Cheating prevalence on Lichess is far below the ~50% of a matched
  corpus. Report precision at realistic prevalence (e.g. 0.1–1%) as well, or the precision
  numbers will not survive contact with a panelist.
- **A control set the current plan lacks:** strong-titled-player brilliancies (§L14). Without
  it, "surprisingly strong play" and "engine use" are not separable.

---

## I. Pre-registered hyperparameter tuning plan

Pre-registered = written down, committed to git, and dated **before** the runs.

**Stage 0 — fix defects, then re-baseline.** §M1, §M6, §G.1, §G.2, §B.3, plus the
`--epochs`/`--val_batch_size` defaults. Nothing below is meaningful until the split is
deterministic and per-game errors are dumped.

**Stage 1 — establish the noise floor before tuning anything.** 5 seeds × {baseline,
attention} at 170k. **Deliverable: the seed-to-seed sd, σ_seed.** This number is the
denominator for every later claim. Budget: 10 runs ≈ 10.5 h wall on 4 GPUs.

**Stage 2 — tune in a fixed order, one factor at a time, budget declared in advance.**
Order chosen by expected effect size given the measured overfitting (train 176 vs val 230):

| # | Hyperparameter | Grid | Runs | Rationale |
|---|---|---|---|---|
| 1 | early-stopping patience | {5, 10, none} | 3 | 45% of epochs currently wasted (§F.1) |
| 2 | weight decay | {1e-5, 1e-3, 1e-2} | 3 | effectively inert at the Elo loss scale (§F.1) |
| 3 | learning rate | {5e-5, 1e-4, 3e-4} | 3 | interacts with (2); tune after |
| 4 | batch size | {32, 64, 128} | 3 | bounded by attention memory (§F.2) |
| 5 | dropout | {0.3, 0.5} | 2 | second-order once (2) is real |
| 6 | attention_dim | {32, 64, 128} | 3 | cheapest capacity knob |

**Total: 17 runs at 170k ≈ 18 h wall on 4 GPUs.** Declare this budget in advance and do not
extend it. One seed per cell during search; **re-run the winner at 5 seeds** before
believing it.

**Success criterion (pre-registered):** a configuration is adopted only if it beats the
stage-1 attention mean by **more than 2 × σ_seed**. Anything smaller is not distinguishable
from restarting training.

**Checkpoint-selection rule (pre-registered, closes §L5):** best-validation checkpoint is
primary, terminal epoch-60 is the secondary check — already stated at `chapter3.tex:270` and
`chapter4.tex:25` **[verified]**. Given §B.2, **also** report the terminal number in the
headline table, since it is the one selection cannot inflate.

**Honest interpretation, agreed in advance:**
- Report **all 17 runs**, not the winner. A table of every cell is the evidence that the
  winner was not cherry-picked.
- Any tuned result is reported as a **separate row** from the control run, exactly as
  `chapter3.tex:278` already requires.
- If no configuration clears 2 × σ_seed, **say so**. "Attention did not measurably improve
  MAE at this scale; its contribution is the interpretable per-move weighting that the
  anomaly module requires" is a defensible, publishable finding.

---

## J. Adviser consultation questions

Five Round 4 questions are still pending (`chapter3-4-process.md:78-80`): Q1 scope, Q3
attention epoch budget, Q4 eval bar, Q5 dataset scope, Q6 pre-oral timing — Q3 and Q5 gate
the full run **[verified]**. Three procedural notes before the list:

- **Number these `R5-Q1 … R5-Q18`, not `Q1 … Q18`.** Four incompatible Q1–Q6/Q1–Q10
  schemes are already live in the repo (the consultation log, the archived adviser list,
  the deck plan's *panel* questions at `pre-oral-deck-plan.md:182-218`, and Part M) —
  Part K's Q5 is *resolved* while Round 4's Q5 is *pending*, and the same string means
  both **[verified via audit lane]**. A fresh prefix keeps every downstream document
  citeable.
- **The Round 4 question text was never transcribed** — only the six topic labels survive,
  so the five pending items cannot be quoted back verbatim **[verified via audit lane]**.
  The list below therefore *re-asks those topics deliberately and sharpens them*, rather
  than assuming the adviser remembers the original wording.
- **Do not re-ask what is already ruled.** Dataset scale (2026-04-25): the paper's 1.2M is
  correct and the released `max_game_per_month = 100` was a debug setting. Epoch budget
  (2026-04-25): the released code's 60/patience-5 is authoritative over the paper's
  appendix 50/10, **and more epochs for the larger model were already permitted in
  principle** **[verified via audit lane]**.

**Binding constraint on everything below** (2026, Stage-1 ruling): the control run must use
unmodified RatingNet hyperparameters for 182-comparability; retuning is permitted only in
Stage 2, and both must be reported jointly **[verified via audit lane]**. §I is written to
respect this.

**Data scale and sampling**
1. Round 4's Q5 (dataset scope) is still open and now blocks preprocessing. Confirm: full 40-month 1.2M uniform
   (30k/month), or an approved subset? Our own scaling curve projects **184–190** at 1.2M
   (§D.3) — i.e. we likely land 2–8 Elo *above* the published 182 even at full scale. Is
   that acceptable given Ch4 already frames 182 as a target rather than a threshold?
2. We discovered each month's 30k games are the **first** 30k in the archive — roughly a
   15-minute window of site activity, not a spread sample (§A.2). We propose reservoir
   sampling across each full month. This diverges from whatever the baseline did (the paper
   does not say). Approve?
3. May we insert a **400k intermediate run** to validate the scaling extrapolation before
   committing the full preprocessing budget?

**Player leakage**
4. The per-game pickles store no usernames, so a player-disjoint split is currently
   impossible (§A). Adding them costs two lines and is **free only if done now**, since the
   full run re-preprocesses anyway. Approve recording usernames and reporting both splits
   (game-level for comparability, player-disjoint for honesty)?

**Epochs for the larger attention model (Round 4 Q3)**
5. You already granted extra epochs in principle at Round 1c ("you might need more epochs
   which is fine"), so this asks for a **number or a stopping criterion**, not permission.
   Our data says the binding issue is the opposite of more epochs: the baseline's last
   improvement was epoch 33 of 60 and attention's was epoch 18, so 27 and 42 epochs
   respectively bought nothing and ended in a worse model (§B.2, §F.1). Should we
   (a) keep 60 fixed for comparability, (b) add early stopping with a documented patience,
   or (c) run 60 and report both the best-val and terminal numbers?

**Evaluation bar (Round 4 Q4) — this one is urgent**
6. There is **no ruling on the evaluation bar**; the bar currently in force is the one we
   wrote into Chapter 3 ourselves, and it is hostile to our own results: "≤ 182 MAE"
   (`chapter3.tex:236`), a bootstrap CI that must not cross 182 (`:238`), and "each novel
   extension contributes a non-zero MAE improvement" (`:237`). Measured: 224.2 baseline /
   221.7 attention at 170k, and §D.3 projects 184–190 even at 1.2M. **As written, Chapter 4
   would report failure against thresholds we set for ourselves and that nobody required of
   us.** Please ratify a revised bar — we propose: primary claim is the delta against our
   own reproduced baseline on the identical partition, with 182 cited as a reference point
   rather than a threshold (which is already Chapter 4's framing at `ch4:17-19`).

**Normalization constants (never asked)**
7. This question was drafted but never sent, and there is no ruling on it; ready-made
   wording exists at `contexts/essentials/ARCHIVE_defense_prep.md:101` — reuse it so the
   history stays traceable. The rating constants (1514/366) are published; the **clock
   constants (273/380) are not** — they appear nowhere in the paper (§C.2). Our corpus
   measures Elo mean 1665.7 / sd 395.8. Reuse the paper's constants for the control run and
   report re-fitted values as a diagnostic — agreed?

**Attention contribution framing**
8. At convergence the attention and baseline arms differ by **0.49 Elo**; the 2.45 Elo figure
   is a best-val selection artefact, and the two arms were not seed-controlled (§B). Is the
   panel-safe framing: *"attention does not measurably improve rating MAE at this scale; its
   contribution is the interpretable per-move weighting the anomaly module requires"*?
9. The "deeper CNN" novelty claim has no implementation (§M2) — the trunk is the baseline's
   4-block CNN. Do we **build and ablate** it, or **scope the claim down** to attention plus
   the anomaly module?

**Statistical significance — and a direct conflict with the panel's own revision**
10. **Panel revision 10 asked us to remove the statistical tools and the statistical test from
   Materials/Evaluation Methods. Gaps (1) and L2 require a paired bootstrap.** We removed the
   section heading and the standalone "Statistical Test" subsection, but retained the paired
   bootstrap and KS test inside the evaluation table (`chapter3.tex:238,241`) and SciPy at
   `:194` (§K). Please confirm this reading — that the panel objected to a *social-science
   statistical-tools section*, not to reporting confidence intervals on a regression result.
   This is the single most consequential open interpretation in the manuscript.
11. `chapter3.tex:238` currently specifies the CI against the **182 published figure**, while
    `chapter4.tex:19` correctly specifies it against our own **reproduced baseline** on the
    identical partition. Confirm we fix Ch3 to match Ch4.

**Rating brackets**
12. With the measured time-control mix, the ultrabullet and classical cells of the subgroup
    tables hold ~40 and ~125 test games — differences there are noise (§G.3). Report them
    with `n` and CIs, merge the thin cells, or drop them?

**Anomaly validation and the engine-substitution schedule**
13. The Ch3 protocol specifies ~450,000 generated games (9 Maia bands × 5 rates × 2 engines ×
    5,000) at Stockfish/Lc0 depth 5–20 (`chapter3.tex:254`). Has this been costed in GPU/CPU
    hours? It plausibly exceeds the entire model-training budget. What is the minimum
    defensible corpus?
14. The anomaly head has **zero learnable parameters** and is unreachable from the training
    loop, so the planned "anomaly-head-only" and "full model" ablation arms are numerically
    identical to the baseline and attention arms (§H.1a). Drop those two arms, or give the
    head parameters?
15. Ground truth from engine-correlation bans, scored with engine-correlation-like features,
    is circular (§L14). Do we need a **strong-titled-player brilliancy control set** to
    separate "surprisingly strong" from "engine-assisted"?

**Causality — the highest-priority item**
16. The BiLSTM is inherited from the baseline, and the baseline paper *itself* claims
    prefix-causal per-move semantics while using a bidirectional architecture (§L1). Our
    per-move outputs therefore see the future. Preferred resolution: (a) prefix-recompute
    inference mode for the demo, (b) a causal forward-only head alongside the retrospective
    BiLSTM, or (c) document it as inherited and reframe "real-time" as latency-only (which is
    what `chapter1.tex:140` already defines)? **We recommend (a)+(c) and treating the
    discovery as a contribution.**
17. **Q10 (live inference architecture) is still recorded as open and as blocking the
    Development objective sign-off, with the decision due "before prototype implementation
    begins"** **[verified via audit lane]**. The prototype has already shipped a full
    bidirectional whole-game pass, so that decision has been made implicitly by the code
    without your input. Please ratify or overturn it explicitly — it is the same
    architectural choice as question 16 and should be answered once, for both.

**Reproducibility of our own numbers**

18. Q-E (exact split reproduction) is still open and needs its scope upgraded. It was filed
    as "seed 42 alone may not reproduce *Omori's* partition without the input-file list" —
    but per §G.1 the unsorted `os.listdir` at `chess_rating_net.py:540` means seed 42 does
    not reliably reproduce **our own** partition between our own runs either. We propose
    `sorted()` plus a committed split manifest before any further training. Confirm this is
    acceptable under the Stage-1 comparability constraint — our reading is that it *improves*
    comparability rather than breaking it, since it changes only the ordering fed to
    `train_test_split`, not the split proportions or the seed.

---

## K. Manuscript audit

### K.1 Panel revisions — 8 applied, 3 partial, 3 still missing

Only the incomplete ones are listed, as instructed. **Strong corroborating signal:** the
items found unaddressed are exactly the items that have **no entry** in the official tracker
`contexts/essentials/revisions_official_td_rsc.md` (whose summary table at `:159-187` reads
"24 DONE, 1 PARTIAL, 0 PENDING"). The team worked the tracker faithfully; the gaps are asks
that never reached it.

**Caveat on the tracker itself:** its `.tex` line pointers are systematically stale, so it
cannot be used to verify revision status **[verified via audit lane]**. Every verdict below
was checked against the current `.tex` directly, and against the pre-RSC revisions in git
history where "was it changed?" was the question.

**Still missing**

- **Rev 1 — playable platform.** Every description of the Develop objective is a
  *reviewer-facing analysis viewer*: it ingests finished PGNs or observes third-party Lichess
  games (`chapter1.tex:66`, `chapter1.tex:81`, `chapter3.tex:198`). Nothing in Ch1/Ch3/Ch4
  says a user *plays* on the platform. This is a scope change, not a wording fix — raise it
  with the adviser before building anything.
- **Rev 6 — "evaluation of the algorithm" section.** A repo-wide grep finds the phrase only
  in the task brief. The merged subsection at `chapter2.tex:11-25` is a chronological
  narrative (Elo → Glicko → TrueSkill → WHR → Glicko-2 → IPR → Maia → RatingNet) with no
  comparative assessment. `chapter2.tex:91` (Synthesis) predates the defense unchanged.
- **Rev 9 — Figures 1 & 2 captions.** Both captions (`chapter3.tex:98`, `chapter3.tex:139`)
  are byte-identical to the pre-defense text apart from a float specifier change
  (`[H]` → `[!ht]`), and both remain 4–5 sentence discussions rather than labels.

**Partial**

- **Rev 2 — objectives.** Sub-items removed and verbs converted to nouns, but the
  "Specifically, this study aims:" enumerate of three specific objectives remains
  (`chapter1.tex:61-67`). The tracker recorded the ask as "remove sub-objectives **or**
  change verb to noun" and both halves of that reading were executed. Under the stricter
  panel wording the list itself should go. **Confirm which reading is authoritative.**
- **Rev 10 — statistical tools/tests.** Title and subsection done (the section heading no
  longer reads "Materials and Statistical Tools / Evaluation Methods"; the standalone
  "Statistical Test" subsection is deleted). The tests themselves remain at
  `chapter3.tex:149, 238, 241` and SciPy at `:194` — **deliberately**, and the tracker records
  the retention as intentional. See adviser question 9: this is a genuine conflict between the
  panel's ask and the methodological requirement for a significance test.
- **Rev 11 — Instrument.** Components removed (`chapter3.tex:149` is now single prose). The
  sample-dataset **figure** was not added — what was added is a *table*
  (`chapter3.tex:159-186`). `grep includegraphics chapters/chapter3.tex` returns only four
  hits, none a dataset sample.

### K.2 Methodology chapter vs. code — inconsistencies

**Match (no action):** 12-plane encoding (`chapter3.tex:19` ↔ `format_data.py:15-28`);
clock concatenation (`:46` ↔ `chess_rating_net.py:249-250`); 100-ply cap (`:215` ↔ `:73,79`);
72/18/10 with `random_state=42` (`:211` ↔ `:544-545`); `pack_padded_sequence` (`:215` ↔ `:251`);
time-control thresholds (`:172,209` ↔ `format_data.py:88-96`); best-val-primary reporting
(`:270` ↔ `chapter4.tex:25`).

**Promises the code does not keep:**

| # | Manuscript | Code | Ref |
|---|---|---|---|
| 1 | "at least three random seeds" (`ch3:153`, `ch4:25,69`); "Random seed: fixed at 42" (`ch3:275`) | no seeding at all in `prototype/src/`; 42 seeds only the split | §B.1 |
| 2 | rating-bracket subgroup eval (`ch3:149,173,239`; `ch4:135-157`) | `test()` breaks out time control only (`:389-424`) | §G.3 |
| 3 | paired bootstrap, 10,000 resamples (`ch3:149,238`; `ch4:19,89`) | not implemented; per-game errors not even saved (`:415-418`) | §G.2 |
| 4 | `S_max`, `S_mean` reported jointly (`ch3:126-132`; `ch4:93`) | only `S_att` computed (`anomaly.py:110-120`) | §H.1d |
| 5 | RMSE (`ch3:236`; `ch4:74`) | not computed anywhere | — |
| 6 | offline-vs-live agreement "within floating-point tolerance" (`ch3:243`) | impossible: API attention is randomly re-initialised each start | §M1 |
| 7 | "same train and test split definition across all configurations" (`ch3:153`) | split rebuilt from unsorted `os.listdir` in each of 5 processes | §G.1 |
| 8 | latency benchmark (`ch3:242`; `ch4:175`) | not measured | §L15 |
| 9 | R_baseline = pre-game Glicko-2 (`ch3:213`) | falls back to the model's own last-ply prediction (`api.py:196-201`) | §H.1c |

**Numbers and hyperparameters that disagree:**

- **"deeper CNN"** — claimed at `ch3:37, 98, 149, 237, 248, 278` and `ch4:23, 65, 77, 79, 163`
  (11 sites). The trunk is the baseline's 4-block CNN (`chess_rating_net.py:177-186`, comment
  "identical to baseline"), and `build_parser` has no depth flag. §M2.
- **Attention equations describe a different mechanism than the code implements.** §M3.
- **Epochs: three different values.** Paper appendix 50, reference code 60, this fork's
  default 100 (`chess_rating_net.py:467`, `example_config.yaml:6`). `ch3:270` asserts 60 and
  cites `chess_rating_net.py:255`, which in this repo is a comment. §M5.
- **Loss space.** `ch3:104` says L1 on *standardized* ratings; the code de-standardizes to Elo
  first (`:350-353`). §M7.
- **Clock normalisation.** `ch3:33` says training-set-derived; `ch3:180` says fixed constants.
  `ch1:154` applies the rating constants to the clock feature. §C.2.
- **182 framing is internally inconsistent.** `ch3:236` ("≤ 182 MAE") and `ch3:238` ("CI does
  not cross the 182 baseline") read as pass/fail; `ch4:17` explicitly disclaims this ("a
  target, not a guarantee") and `ch4:19` correctly bootstraps against the *reproduced*
  baseline. **Ch4 is right; fix Ch3.** §M8.
- **Split provenance.** `ch3:211` says 72/18/10 matches "the nested split in the reference
  RatingNet implementation" — true of the *code*, but the *paper* states a plain random 80/20
  with no validation set (`contexts/inspirationpaper.md:407-408`). Record this in the same
  paper-vs-code divergence paragraph as the epochs discrepancy.
- **Bidirectionality vs. real-time.** `ch3:65` states each step incorporates "context from both
  past and **future** moves", while `ch1:19` claims "as each move is submitted, the system
  produces an updated rating prediction". These cannot both be true of the current
  architecture. §L1.

**Chapter 4 is stale.** `ch4:11` says "one run has been completed"; three are complete (50k
sanity gate, 170k baseline, 170k attention). The 170k ablation appears **nowhere** in the
manuscript — only in `chapter3-4-process.md:104-138` and the results README. Chapter 4's
discipline is otherwise exemplary: every unmeasured quantity is a `\TODO` and `ch4:5-8`
forbids filling one with an estimate. Keep that rule when the 170k numbers go in, and label
them a pipeline check, not a benchmark comparison (per `ch4:19`).

### K.3 Defense-material corollaries

Not the manuscript, but they will be read out loud at the pre-oral, so they belong in the
same audit **[all verified via audit lane]**:

- **`panel_attacks.md:472` (M7) contradicts the manuscript on player leakage.** §A.4 item 4.
  Highest priority in the prep set.
- **The causality attack is absent from all 146 catalogued entries.** §L1, final paragraph.
- **Eleven rehearsed answers (M-CF1…M-CF11, `panel_attacks.md:73-127`) defend the Conceptual
  Framework** that the panel ordered removed and that has been deleted. They are orphaned
  rather than wrong; rehearsing them wastes prep time and, worse, answering "walk me through
  the conceptual framework" re-raises a section the panel already closed. Retarget the
  salvageable ones (M-CF3, M-CF4, M-CF7) onto the retained Theoretical Framework figure.
- **Two rehearsed answers promise ablations that cannot run.** `panel_attacks.md:413` (A3)
  contradicts the adviser's Round 3 ruling *and* its own document's M-O5 at `:39`; `:418`
  (A4) promises a deeper-CNN-vs-baseline comparison the code cannot produce (§M2). Both also
  cite "Objective 8", which no longer exists after the RSC objective restructure.
- **`panel_attacks.md:864` (HH2) still says the synthetic protocol is "pending the adviser
  meeting"** — locked at Round 2b on 2026-04-26. Narrow it to the one thing genuinely still
  open (the engine depth schedule) rather than deleting it.
- **The parameter-count TODO (`panel_attacks.md:150`, M-SK5) can now be closed** for the
  architecture that actually exists: ≈0.76 M parameters in the baseline trunk, **+16,448**
  for Bahdanau attention (+2.2%) **[derived, §F.1]**. It cannot be closed while the answer's
  "deeper CNN adds Y million" clause presupposes a trunk that does not exist — drop that
  clause. The dropout TODO (`:191`, M-SK15) is 0.5 (`chapter3-4-process.md:44`).

---

## L. Second-opinion scrutiny items

### L1 — BiLSTM causality **[CONFIRMED, but the premise needs correcting — highest priority]**

**The premise in the brief is wrong.** The thesis did **not** swap in a BiLSTM; it inherited
one. `contexts/essentials/thesis_explained.md:607` already says so ("BiLSTM is retained from
the baseline, not added"). Delete the "swaps in a BiLSTM" framing from all defense material —
a panelist who checks will find it false, on the item that matters most.

**State the inheritance carefully, because the paper's own prose is self-inconsistent.** The
paper calls the recurrence "Bidirectional" in the abstract, method and architecture
description (`contexts/inspirationpaper.md:240, 270, 444`), yet ten lines after the last of
those it describes the input as "the hidden state representation at the previous move"
(`:454`) — a strictly forward recurrence — and "CNN-LSTM" appears unqualified in the title,
the running headers, the Figure 1 caption (`:473-474`) and the conclusion (`:542`)
**[verified via audit lane]**. Only the released code settles it: it is bidirectional
(`contexts/essentials/ratingnet_code_audit.md:41`), and this fork mirrors it
(`chess_rating_net.py:196`) **[verified]**.

So the defensible formulation is: *"the baseline's text is self-inconsistent about
directionality; its released code is bidirectional; we therefore treat the baseline as
non-causal and cite the code."* Do not assert flatly either that the paper is bidirectional
or that it is forward-only — both are refutable from the paper itself.

**The underlying defect is real, and it is inherited.**

- `chess_rating_net.py:191-197` — `nn.LSTM(..., bidirectional=True)` **[verified]**
- `chess_rating_net.py:251-253` — `pack_padded_sequence` over the **whole game in one pass**,
  then `pad_packed_sequence` **[verified]**
- `chess_rating_net.py:282-284` — the rating head is applied to every position of that output

In PyTorch, `output[:, t, :hidden]` is the forward state (has seen plies 1..t) and
`output[:, t, hidden:]` is the backward state (has seen plies t..T). So **the prediction at
ply t sees every ply after t.** It is whole-game, not per-growing-prefix.

**A precise and reassuring corollary the audit should state plainly:** the *training and
evaluation* objective uses only the **last** ply (`:287` `last_time_step_output`, `:350-353`,
`:410`) **[verified]**. At t = T the backward stream has consumed exactly one ply — the
current one. **The reported 224.2 / 221.7 MAE figures are therefore effectively causal and are
not contaminated by lookahead.** What *is* contaminated is the per-move curve, monotonically
more so toward the start of the game (at t = 1 the backward stream has seen everything) — and
that curve is precisely what the demo plots and what the entire anomaly module consumes
(`api.py:190`, `anomaly.py:98`).

**The causal attention does not fix it.** `attention.py:100-139` masks correctly
(`:132-133`), but it operates on already-bidirectional features. Causal masking over
non-causal inputs is not causal. The module docstring's claim of "introducing no lookahead"
(`attention.py:5-8`) is false **at the system level**, and should be corrected in place.

**The baseline paper has the same contradiction:** it describes its per-move output as
conditioned on "that move and all the previous moves" (`contexts/inspirationpaper.md:496-497`)
**[verified via audit lane]** — prefix-causal semantics that its own bidirectional
architecture does not deliver.

**Methods paragraph to adopt (drop-in for Ch3, after `:65`):**

> The BiLSTM is retained unmodified from the reference architecture of Omori and Tadepalli,
> in which the backward pass at ply *t* consumes plies *t* through *T*. Consequently the
> per-ply rating estimate ⟨R̂_t⟩ for *t* < *T* is conditioned on the complete game and is
> retrospective rather than prefix-causal. This property is inherited from the baseline,
> whose published description of its per-move outputs as conditioned on "that move and all the
> previous moves" is not enforced by its architecture. Three consequences are reported here.
> First, the headline test MAE is unaffected: it is evaluated at the final ply, where the
> backward pass has consumed only the current ply. Second, the per-ply curve and the
> attention-weighted anomaly score derived from it are retrospective annotations, correctly
> interpreted as post-hoc review aids rather than live detection. Third, for genuinely
> prefix-causal operation the prototype additionally exposes a recompute mode that evaluates
> the network on the growing prefix *z₁..z_t* at each ply, at *O(T)* forward passes per game;
> the per-move latency of this mode is reported in §4.6, and the divergence between the two
> modes is reported as a measure of how much retrospective information the BiLSTM is using.

**Recommendation.** Adopt (a) prefix-recompute for the demo — 100 forward passes/game is
trivially affordable at inference and makes the "as each move is submitted" claim in
`chapter1.tex:19` *true* — plus (c) keep `chapter1.tex:140`'s latency-based definition of
"real-time". Do **not** retrain unidirectionally for the control arm; that would break
comparability with 182. **Measure and report the divergence between whole-game and
prefix-recompute per-move curves** — that measurement is a genuine, publishable contribution
that the baseline never made.

**This is currently the single largest uncovered attack surface in the defense material.**
`contexts/essentials/panel_attacks.md` catalogues 146 question-level entries, and the
causality/look-ahead attack is **not among them** **[verified via audit lane]**. The three
nearest entries — `panel_attacks.md:187` (M-SK14), `:413` (A3) and `pre-oral-deck-plan.md:194`
(deck Q4) — all treat bidirectionality as a *latency* question and argue that it is
beneficial; none treats it as an *information-leakage* question. So the honest answer does
not exist anywhere in the prep set. Write it (the methods paragraph above is a starting
point) and rehearse it before the pre-oral.

### L2 — Per-move sample non-independence **[CONFIRMED]**

Games average 64.9 plies (`contexts/model-improve-findings.md:95-96`) **[verified]**, so a
move-level i.i.d. test would inflate effective N by ~65×. The label is constant within a
game, so the correlation is near-total. **Prescription:** game-level (and, once usernames
exist, player-level) cluster bootstrap — §G.2. Note the current code never produces move-level
errors anyway (`:410` evaluates the last ply only), so this is a risk to avoid rather than a
mistake already made.

### L3 — Metric aggregation **[CONFIRMED — and worse than stated]**

**The paper's aggregation convention is undocumented.** Its §4.2 says only that MAE is used
and reported on de-standardized ratings; Table 1's row label is the bare "Average Test Loss"
(`contexts/inspirationpaper.md:422-423`) **[verified via audit lane]**. The reference
implementation is not in this repo, so it cannot be recovered from source here.

**Worse, the paper is internally inconsistent.** It says "The average MAE across the time
controls is 182" (`:484`), but an unweighted macro-average of its own per-time-control table
gives **(186+182+183+182+151)/5 = 176.8** **[derived via audit lane]**. So the 182 must be a
*pooled, mix-weighted* mean in which bullet and blitz dominate. Any reproduction whose
time-control mix differs is not comparing like with like — and per L10 the paper's mix is
unpublished. (The paper also prints 182 in the abstract/conclusion and 183 in the
introduction.)

**Our own number is a third thing again.** The headline is last-ply-only, averaged over both
players, and **a mean of per-batch means** rather than a per-game mean: `test()` accumulates
`loss.item()` per batch and divides by `len(test_loader)` — the batch count (`:424`)
**[verified]**. With n = 17,014 and `val_batch_size 512` that is 34 batches with a final batch
of **118** games, which receives weight 1/34 = 2.94% instead of its true 118/17014 = 0.69% —
an **over-weight of 4.24×** **[derived]**. The per-time-control figures, by contrast, are
correct per-game means (`:416-422`). So the headline and the subgroup rows use different
estimators. This also injects roughly ±0.4–0.7 Elo of avoidable noise into a 2.45 Elo
comparison **[derived]**.

**Fix:** divide by game count, not batch count; state the convention explicitly in Ch4; report
the headline **both** pooled and macro-averaged over time controls, with the mix, so the
comparison to 182 is auditable in either convention. And state in Ch4 that the published
182's aggregation is undocumented, making the comparison indicative rather than exact — or
email Omori and settle it.

### L4 — Glicko-2 label noise **[CONFIRMED; the thesis's stated rationale is an inference]**

The paper documents **no** player-inclusion filter — no provisional filter, no minimum-games
threshold, no rating-deviation filter. "Provisional" appears once, as a justification for not
feeding the opponent's rating in, not as a data filter (`contexts/inspirationpaper.md:462`)
**[verified via audit lane]**. So `chapter3.tex:175,213` ("Provisional ratings ... retained
(not filtered)", "matching the approach taken by Omori and Tadepalli") is **an inference, not
a documented match** — reword to "the baseline documents no inclusion filter; we likewise
apply none." The paper's label-noise discussion is qualitative and never quantifies an
irreducible floor.

**Prescription:** do not add a filter (it would break comparability). Instead **quantify the
floor**: Lichess publishes rating deviation; report the test-set distribution of RD and the
MAE restricted to low-RD (established) players as a supplementary row. If the low-RD MAE is
materially better, that bounds the label-noise contribution and reframes the whole 182-vs-224
discussion — a cheap, strong result.

### L5 — Checkpoint selection **[ALREADY ADDRESSED — keep and strengthen]**

`chapter3.tex:270` and `chapter4.tex:25` already pre-register best-val as primary with
terminal epoch-60 as secondary **[verified]**. Given §B.2 (the entire headline effect lives in
the selection), promote the terminal number into the headline table rather than a footnote.
Formalise in the §I pre-registration.

### L6 — Shrinkage to the mean **[CONFIRMED as a real risk; not yet measurable]**

Aggregate MAE cannot detect compression toward the population mean, and compression is exactly
what would suppress the anomalies the detector must catch. Supporting evidence that the risk is
live: the naive predict-mean floor is 346 (`chapter3.tex:236`) and the model sits at 224 — it
is well above the floor, but that does not bound per-bracket bias. **Prescription:** report
**signed bias per rating bracket** (mean of `ŷ − y`, not `|ŷ − y|`) and a predicted-vs-true
calibration scatter with the y = x line. Both fall out of the per-game error dump in §G.2 at
no extra compute. Expect positive bias at the bottom bracket and negative at the top; report
the regression slope of ŷ on y as a single compression statistic.

### L7 — Confounded ablation **[REFUTED as stated — the confound does not exist, because the deeper CNN does not]**

The brief says the comparison "bundles a deeper CNN with attention". It does not: both 170k
arms use the identical 4-block trunk (`chess_rating_net.py:177-186`), and the runs differ only
in `--use_attention` (`results/170k-ablation/README.md:44-53`) **[verified]**. The 170k
ablation is a **clean attention-only ablation** — modulo the seed problem in §B.1, which is
the real confound.

The 2×2 design is still the right thing to build, because the manuscript claims *two*
novelties and has only ever tested one:

| | no attention | attention |
|---|---|---|
| **4-block CNN (baseline)** | ✅ done (224.2) | ✅ done (221.7) |
| **deeper CNN** | ❌ not implemented | ❌ not implemented |

Implementing the deeper trunk requires a `--cnn_blocks` flag and a config; `e6_capacity.py`
sketched options but nothing was adopted **[verified via audit lane]**. See adviser question 8:
build it or scope the claim down.

### L8 — Reproduction gap **[ALREADY ADDRESSED in Ch4; strengthen with the scaling curve]**

`chapter4.tex:19` already forbids comparing subset runs to 182, and `chapter4.tex:59-61`
already explains the 50k gap as an overfitting penalty rather than a pipeline defect
**[verified]**. Strengthen it with §D.3: two measured points give a scaling law projecting
**184–190** at 1.2M, which quantifies the explanation instead of asserting it. Add the 170k
point (224.2) to that table — it is currently absent from the manuscript entirely.

### L9 — Board/move encoding fidelity **[CONFIRMED faithful — no divergence from the baseline]**

`board_to_array` (`format_data.py:15-28`) produces 12 planes and nothing else: no castling
rights, no en passant, no repetition or 50-move counters, no history stacking, no side-to-move
**[verified]**. The paper's encoding is identically bare — greps for castling, en passant,
repetition, history, 50-move over the paper return zero hits
(`contexts/inspirationpaper.md:429-430`) **[verified via audit lane]**. **So there is no
divergence to fix; this is a shared limitation.**

One wording fix: the paper's "similar input representation as AlphaZero" is loose, and
`chapter3.tex:19` repeats it. AlphaZero's chess input is 119 planes with 8 steps of history,
repetition counts, castling rights, side-to-move and a no-progress counter. Say **"the
piece-plane component of the AlphaZero representation"**. Then state the consequence as a
limitation: the model cannot see castling legality or repetition, and cannot identify the side
to move from the tensor alone (only indirectly via ply parity through the LSTM) — which also
bears on L13. Adding a side-to-move plane is a cheap, well-motivated extension for the tuning
stage.

### L10 — Time-control mix **[UNVERIFIABLE — the paper publishes no mix]**

Tables 1 and 2 break results out by time control but publish no sample counts
(`contexts/inspirationpaper.md:415-416`) **[verified via audit lane]**, so there is nothing to
match. Combined with L3-2 (182 is mix-weighted), the comparison to 182 carries an
**unquantified mix confound**. Mitigate rather than solve: report our own realised mix (the
`\TODO` at `chapter4.tex:29` already asks for it), and report the headline **both** pooled and
macro-averaged over the five controls, which makes the comparison robust to mix in one
direction.

### L11 — Compute-budget parity **[CONFIRMED, small]**

Attention adds 16,448 parameters to a ≈0.76 M baseline (**+2.2%**) and cost ~8% more per epoch
(4.55 vs 4.22 min) at equal epochs **[derived]**. So the comparison is close to iso-compute
already; the honest statement is *iso-epoch, +8% wall-clock, +2.2% parameters*. Report those
three numbers and skip a separate iso-compute ablation — it would not change the picture given
the converged difference is 0.49 Elo. Do note the *memory* asymmetry (§F.2): the
`(B,T,T,d)` intermediate is the real cost and it constrains batch size.

### L12 — Chronological split **[MISSING — recommended, cheap]**

No temporal holdout exists anywhere in code or manuscript. This is genuinely valuable for a
system claiming deployment realism: hold out the **final month (2024-07)** entirely as a
second, test-only partition and report MAE on it beside the random-split test MAE. The gap
between them is the honest estimate of deployment degradation, and it costs one extra eval
pass — no retraining. It also partially mitigates player leakage (§A), since players active in
the training window are less likely to dominate a disjoint later month.

### L13 — Move-index curve and side-to-move **[MISSING — falls out of the same dump]**

Neither is reported. Both come free from the per-game error dump of §G.2 once it is extended to
per-ply predictions on a sample of test games:
- **MAE and signed bias vs ply index** — expect a monotone improvement with more plies, and
  this curve is *also* the direct diagnostic for L1: under whole-game bidirectional inference
  the early-ply error will be anomalously *low*, and the gap against prefix-recompute
  inference measures the lookahead advantage quantitatively.
- **Split by side to move** — white/black asymmetry. Note the model has no side-to-move plane
  (§L9), so any asymmetry found is learned from ply parity alone, which is worth saying.

### L14 — Anomaly credibility traps **[CONFIRMED — all four are live]**

1. **Circular ground truth.** Closed-account labels come from Lichess's Kaladin/Irwin, which
   are themselves engine-correlation detectors — and `chapter3.tex:258` already documents this
   pipeline **[verified]**. Scoring engine-correlation-derived labels with an
   engine-correlation-like feature measures agreement with Lichess's detector, not truth. The
   manuscript acknowledges the pipeline; it must also state this as a **limitation on what the
   KS test can conclude**.
2. **Survivorship bias.** Banned accounts are the cheaters who were *caught*, i.e. the
   least-careful ones. Report it as a stated bound on generalisation.
3. **Conflated anomaly types.** Sandbagging (deliberate underperformance), boosting,
   brilliancies, and full-engine play produce very different signatures, and a single
   `d_t = |R̂_t − R_baseline|` (`chapter3.tex:115`) treats them identically — it is *unsigned*,
   so sandbagging and engine use are indistinguishable by construction. **Report signed
   deviation** as well, which separates the two at zero cost.
4. **"Surprising = suspicious" is unfalsifiable** without a control for legitimate surprise.
   **Prescription: a strong-titled-player brilliancy control set** — games by titled players
   containing engine-agreeing brilliancies, which must score *low*. Without it the detector
   cannot distinguish skill from cheating, and a panelist will ask exactly this.

### L15 — Real-time latency **[PROMISED, NOT MEASURED]**

`chapter3.tex:242` and `chapter4.tex:175` scaffold it; no measurement exists **[verified]**.
Measure and report: mean and **p95/p99** ms/move on an A5000, batch 1, plus the CPU-only
number (a reviewer's laptop is the realistic deployment). Report it for **both** inference
modes (whole-game and the prefix-recompute mode of §L1) — the latter is ~T× more forward
passes and is the number that substantiates the title.

### L16 — Legibility wins **[recommended, in priority order]**

1. **Effect-size framing.** State 2.45 Elo as **1.1% of 224**, next to the converged 0.49 Elo
   and next to Glicko-2's own uncertainty. Own the smallness; it reads as rigour.
2. **One concrete anomaly operating point.** Precision/recall/FPR at a justified threshold,
   plus precision at realistic prevalence (§H.3). One honest operating point beats an AUC.
3. **Attention-heatmap case studies.** 2–3 games with the per-ply weights overlaid. Caveat
   them with §H.1b (the current query-averaged weights carry a positional artefact) — fix that
   first, or the heatmaps will show a decaying ramp and a panelist will notice.

---

## M. Scout-found concrete defects

### M1 — Demo serves baseline predictions corrupted by random attention **[CONFIRMED — worse than reported]**

- `api.py:99-110` — constructs `ChessEloPredictor(..., use_attention=True, use_anomaly=True)` **[verified]**
- `api.py:112` — `model.load_base_state_dict(saved["model_state_dict"], strict=False)` **[verified]**
- `chess_rating_net.py:303-305` — `load_base_state_dict` calls `load_state_dict(..., strict=False)` and **discards the return value**, so `missing_keys` is never inspected and nothing warns **[verified]**
- `attention.py:62-64` — the unloaded parameters, including `v = torch.randn(...)` **[verified]**
- `chess_rating_net.py:269` — `rating_input = lstm_output + attn_context` **[verified]**

**The magnitude is not a small perturbation.** `attn_context` is a softmax-weighted average of
`lstm_output` rows, so it has the *same scale* as `lstm_output`. The head therefore receives
roughly **twice** the activation magnitude it was trained on, plus an arbitrary temporal
smear. Every per-move rating the demo displays is out of distribution.

**Three consequences the brief did not name:**

1. **The demo is not reproducible across restarts.** Nothing seeds the random init (§B.1), so
   the same PGN yields different ratings after every API restart. `RESULT_CACHE`
   (`api.py:54`) hides this within a process, and the "stable cache key across Python process
   restarts" comment (`api.py:165`) is stable over an *unstable* function.
2. **The audit trail logs unreproducible numbers.** `api.py:9-10, 68-72` writes
   `predictions.jsonl` "for reproducibility and fairness-review audit trails". It is currently
   an audit trail of noise.
3. **It makes an Ch3 evaluation type impossible.** `chapter3.tex:243` promises offline-vs-live
   agreement "within floating-point tolerance" on 100 matched games. That test cannot pass.

**Fix:** set `use_attention=params.get("use_attention", False)` and
`use_anomaly=params.get("use_anomaly", False)` at `api.py:106,109` so the served architecture
matches the checkpoint; capture and assert on `missing_keys` in `load_base_state_dict`
(`chess_rating_net.py:303-305`) and refuse to start if attention weights are requested but
absent; seed the process. Until a trained attention checkpoint exists, **serve the baseline**.
This is the defense centerpiece — fix it first.

### M2 — "Deeper CNN" claimed but not implemented **[CONFIRMED]**

`chess_rating_net.py:177-186` is a 4-block trunk with the comment "CNN trunk (identical to
baseline)"; `build_parser` (`:462-490`) exposes `--conv_filters` but **no depth option**
**[verified]**. The claim appears at 11 sites (§K.2). `e6_capacity.py` costed options; none was
adopted **[verified via audit lane]**. **Fix:** implement `--cnn_blocks` and run the 2×2 of
§L7, or scope the novelty claim down to attention + anomaly module. Adviser question 8.

### M3 — Attention equations describe a different mechanism **[CONFIRMED]**

- Manuscript (`chapter3.tex:71-81`): `e_t = vᵀ tanh(W_a h_t + b_a)` — **content-only**, no query
  term; `α_t = softmax` over `t = 1..T` — **global**, whole sequence; `s = Σ α_t h_t` — a
  **single** context vector; then `ŷ = FC₂(Dropout(f(FC₁(s))))` at `:90` — **one prediction per
  game** **[verified]**
- Code (`attention.py:127-133`): `e(t,s) = vᵀ tanh(W_q h_t + W_k h_s)` — full **query-key**
  additive, with a **causal** mask, producing a **per-ply** context; then
  `rating_input = lstm_output + attn_context` (`chess_rating_net.py:269`) — a **residual add**,
  per-move output **[verified]**

Three distinct divergences: content-only vs query-key; global vs causal; one-per-game vs
per-move. The manuscript's equations also carry a bias `b_a` that the code's projections do not
have (`bias=False`, `attention.py:62-63`). **And the manuscript's own math contradicts the
real-time claim**: a single `s` over `t = 1..T` is by construction one retrospective prediction
per game.

**Correction to the brief:** the claim that `e7` "tested and rejected" this variant is not
supported. `e7_attn_wiring.py` did enumerate and reject a global-pooling wiring, but **[verified
via audit lane]**: (a) e7 has **no captured output anywhere in the repo**; (b) it cannot be
re-run as committed — it and every other `e*` script hardcode
`sys.path.insert(0, "/tmp/opencode/lab/src")`, a directory that no longer exists; (c) it tested
a **superseded** attention module (it dereferences `s.att.query`, which the current
`BahdanauAttention` does not define, so it would now raise `AttributeError`); and (d) the
wiring actually shipped (residual add) is **not** the wiring e7 tested (concatenation with a
doubled `fc1`).

Relatedly, **there is no gradient audit proving the attention parameters train.**
`e1_grad_audit.py` is a pre-fix artifact with no captured output. The strongest available
evidence is static: `:269` puts `attn_context` into `rating_input` and `:282` feeds it to
`fc1`, so autograd must reach the attention parameters **[verified]**. **Fix:** rewrite the
Ch3 equations to match the code, and add a two-line unit test asserting
`model.attention.v.grad is not None` after a backward pass. Do not tell a panel "we audited
the gradient" until that test exists and its output is committed.

### M4 — Anomaly baseline self-referential **[CONFIRMED in code; the manuscript claim is REFUTED]**

`api.py:196-201` **[verified]** — confirmed. But the brief's claim that "`chapter3.tex:115`
never defines where the baseline comes from" is **wrong**: `chapter3.tex:213` defines it
precisely ("the player's Lichess Glicko-2 rating for the specific time control of the game at
game start") **[verified]**. The defect is therefore sharper than reported — **the code
contradicts a specification the manuscript already got right.** Fix the code, not the
manuscript; add a forward reference from `:115` to `:213` so the definition is not 98 lines
away from the equation. §H.1c.

### M5 — Stale file:line refs and the epochs contradiction **[CONFIRMED]**

`chapter3.tex:270-274` cites `chess_rating_net.py:255` (epochs), `:257` (patience), `:293`
(Adam) **[verified]**. In this repo line 255 is a comment inside the attention block, 257 is a
comment, and 293 is `if self.anomaly_detector is not None...`. The real locations are `:467`,
`:478` and `:589-593` **[verified]**. Epochs are specified **three** ways: paper appendix 50,
reference code 60, this fork's default **100** (`:467`, `example_config.yaml:6`) **[verified]**;
the paper's appendix also gives patience 10 against the code's 5.

**Fix:** change the fork's `--epochs` default to 60 so the executable default agrees with the
documented control protocol; replace the `.tex` file:line citations with a citation to the
committed run command (`results/170k-ablation/README.md:44-53`), which is stable, or drop line
numbers entirely. Record the 50/60/100 divergence in the paper-vs-code paragraph.

### M6 — PGN parser clock/position misalignment **[CONFIRMED — but located differently than reported]**

`format_data.py:54-67` appends a position on **every** ply but a clock **only** when a
`[%clk ...]` comment matches; the sole guard is `if not clocks: return None` (`:69-70`)
**[verified]** — it rejects games with *zero* clocks, not games with *partial* clocks. The
upstream length-equality guard is gone.

**Correction to the brief:** it is not "the inference PGN parser" specifically, and it does not
always fail silently.

- **At inference** it fails **loudly**: `api.py:153-160` builds `positions` of length L and
  `clocks` of length C; if C ≠ L, `torch.cat` at `chess_rating_net.py:250` raises and the
  endpoint returns HTTP 400 (`api.py:265-267`) **[verified]**.
- **In training it can fail silently**, and this is the real exposure — the HPC preprocessing
  uses **the same `parse_game`** (`hpc-data-setup.md:38-42`) **[verified]**, so the defect is in
  the training corpus, not just the demo. `collate_fn` (`:104-105`) pads positions and clocks
  *independently* to their own batch maxima. If some other game in the batch has C = L = max,
  the shapes match, `cat` succeeds, and the short game's clocks are silently wrong. If the
  missing annotation is mid-game, every subsequent clock is **shifted by one ply**.

In practice Lichess annotates every ply, so this is latent rather than active — but it is
data-dependent and unlogged, so nobody would know if it fired.

**Fix:** in `parse_game`, `if len(clocks) != len(positions): return None`, and log the reject
count per month so the rate is known rather than assumed.

### M7 — Loss described as standardized, computed in Elo **[CONFIRMED]**

`chapter3.tex:104` says L1 "between predicted and actual **standardized** ratings"
**[verified]**; `chess_rating_net.py:350-353` de-standardizes both sides first **[verified]**.
For L1 this is a constant factor σ = 366 on the loss, so the optimum is unchanged — but it is
**not** cosmetic, for two reasons: (a) it makes `weight_decay` effectively ~366× weaker under
Adam (§F.1), which matters given the measured overfitting; (b) `chapter3.tex:264` claims the
implementation "matches the hyperparameters of the released RatingNet implementation exactly",
and that claim cannot be evaluated without knowing which space upstream computed its loss in —
which this repo cannot establish. **Fix:** correct `:104` to say the L1 is computed on
de-standardized (Elo-point) values, note the reported "loss" is therefore in rating points
(which is what makes the log values directly comparable to MAE), and flag the weight-decay
consequence.

### M8 — 182 as a pass/fail threshold **[PARTIAL — refuted for Ch4, confirmed for Ch3]**

Chapter 4 explicitly **disclaims** the pass/fail framing: "This is a target, not a guarantee to
beat the baseline... that contribution does not depend on producing a lower MAE"
(`chapter4.tex:17`), and `chapter4.tex:19` restricts the comparison to the full corpus and
bootstraps against the *reproduced* baseline rather than the published figure **[verified]**.
Chapter 3 has not caught up: `chapter3.tex:236` lists "≤ 182 MAE" as a **Threshold**, and
`chapter3.tex:238` defines significance as "95% CI does not cross the 182 baseline"
**[verified]** — which is also statistically wrong, since 182 is a point estimate from a
different corpus with an undocumented aggregation (§L3).

**Fix:** propagate Ch4's framing into Ch3 — change the `tab:eval-metrics` threshold column to
"reference point (not a pass/fail threshold)" and restate the significance row as a CI on the
paired difference against the reproduced baseline. This is a one-table edit and it removes the
strongest available "you failed your own criterion" line of attack.

---

## Priority ordering

**Before the full run — blocking**

1. §M1 demo (defense centerpiece, currently non-reproducible)
2. §G.1 `sorted()` + split manifest
3. §G.2 per-game error dump (nothing statistical is possible without it)
4. §B.3 seed plumbing (`--seed` separate from `--split_seed`)
5. §A.4 record usernames — **free now, impossible later**
6. §M6 clock/position length guard
7. §F.1 `--epochs` and `--val_batch_size` defaults; §F.3 resume fixes

**Before Chapter 4 is written**

8. §B.3 5-seed replication at 170k → σ_seed
9. §G.3 rating-bracket eval with per-bin counts
10. §G.2 paired bootstrap, hierarchical
11. §L1 causality paragraph + prefix-recompute mode + divergence measurement
12. §K.2 the nine manuscript-vs-code corrections; §M8 Ch3/Ch4 alignment

**Manuscript, independent of compute**

13. §K.1 panel revisions 1, 6, 9 (and confirm the rev-2 and rev-10 readings with the adviser)
14. §J adviser questions — send R5-Q6 (evaluation bar), R5-Q10 (statistical-test conflict)
    and R5-Q16/17 (causality and live inference) first; the evaluation bar is the one that
    determines whether Chapter 4 reads as a pass or a failure

---

*Prepared from a read-only clone. Source of every claim is cited inline; where the repository
could not settle a question it is marked **[open]** or **UNVERIFIABLE** rather than resolved by
assumption.*
