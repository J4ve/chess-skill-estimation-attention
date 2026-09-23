# Fix review — the eight fixes, methodology/objectives strictness, deviations, cross-check, and the anomaly-corpus design

Review of commits `31c347b` (training/eval/api core) and `08ae9f0` (`format_data.py`),
evaluated against the prescriptions in `analysis/170k-verification.md` (the reference
audit), the manuscript (`CCS Thesis - Integrated/chapters/*.tex`), and the live HPC
snapshot (`analysis/hpc-snapshot.txt`). Repository state reviewed: HEAD `4b415c4`.
Read-only review: no code was modified. All file:line references are to HEAD.

**Headline.** Six of the eight fixes are present and correct. Two are subtly
incomplete: the per-game error CSV stores only *absolute* errors, so two of the four
analyses the commit message claims it enables are not actually possible from it
(§1.6); and the "observable reject count" is a counter that nothing reads —
observability was the entire point of that prescription (§1.8). The fixes also
surface one consequential new hazard nobody has written down: under the new sorted
split, **re-evaluating the existing 170k checkpoints is invalid** — roughly 72% of the
new test partition was training data for those runs (§5.3-N1). The methodology
machinery the manuscript promises (rating brackets, paired bootstrap, RMSE, latency,
S_max/S_mean) remains unimplemented; the fixes built the *inputs* to those analyses,
not the analyses (§2). Section 6 provides the requested full design of the bot-vs-bot
anomaly-validation corpus.

---

## 1. The eight fixes — verdicts against the prescriptions

| # | Fix (task brief) | Prescription | Verdict |
|---|---|---|---|
| 1 | Demo serves checkpoint's architecture | §M1 | **Correct** |
| 2 | Missing-keys warning | §M1 | **Correct** (warns; audit prescribed refuse-to-start — see 1.2) |
| 3 | `--seed` / `--split_seed` + full seeding | §B.3 | **Correct** |
| 4 | `--epochs` 60, `--val_batch_size` 512 defaults | §F.1/§M5 | **Correct** |
| 5 | `sorted()` split + verified `split_manifest.json` | §G.1 | **Correct**, two durability qualifications (1.5) |
| 6 | `test()` writes `per_game_errors.csv` | §G.2 | **Partial** — absolute errors only (1.6) |
| 7 | Resume: scheduler + `best_epoch` + post-update best + `--checkpoint` + env logging | §F.3 | **Correct** (edge notes in 1.7) |
| 8 | Usernames + partial-clock guard + observable reject count | §A.4/§M6 | **Partial** — reject count is not observable (1.8) |

### 1.1 Demo serves the checkpoint's architecture — correct

`api.py:109,112` now builds the served model with
`use_attention=params.get("use_attention", False)` and
`use_anomaly=params.get("use_anomaly", False)`, so the baseline `model_55.pth`
(whose params lack those keys) is served as a pure baseline and the loaded weights
match the architecture exactly. The anomaly-scoring path is fully guarded: every use
of the `anomaly` result is inside `if MODEL.anomaly_detector is not None:`
(`api.py:209-228`), with neutral zeros otherwise, and `seq_len` is bound before the
branch (`api.py:208`). The demo's per-move outputs are now deterministic across
restarts for a correctly-loaded checkpoint (no randomly initialized module
participates; `model.eval()` at `api.py:116` disables dropout), which restores
feasibility of the `chapter3.tex:243` offline-vs-live agreement test for the baseline.

Residuals, carried forward (none blocks the fix as specified by the task brief):

- The audit's §M1 prescribed *refusing to start* when attention weights are requested
  but absent; the implementation only warns (see 1.2). A misconfigured deployment
  still serves corrupted predictions, now with a log line.
- The API process is still unseeded; moot while served modules always have loaded
  weights, live again the day a partially-loaded config slips through the warning.
- The §H.1c self-referential `R_baseline` fallback is untouched (`api.py:199-204`):
  absent baselines still default to the model's own final-ply prediction,
  contradicting `chapter3.tex:213`. Not in the eight; still open.
- The four normalization constants remain re-declared in `api.py:43-46` (§C.4
  plumbing; not in the eight; still open).

### 1.2 Missing-keys warning — correct

`chess_rating_net.py:323-344` captures the `load_state_dict` result and warns on
missing keys filtered to `attention.` / `anomaly_detector.` prefixes, gated on
`self.use_attention` / `self.use_anomaly`. The prefixes match the real module
attribute names (`self.attention` at `:222`, `self.anomaly_detector` at `:236`); the
anomaly arm of the filter is vacuous since `AnomalyDetector` has no parameters, which
is harmless. The warning is actually emitted in both contexts: the API configures
logging at startup (`api.py:60-65`, before `_load_model`), and a bare CLI run emits
WARNING-level records to stderr through Python's last-resort handler.

One asymmetry worth noting: **unexpected** keys are not warned. Evaluating an
attention checkpoint through the CLI eval path *without* `--use_attention` silently
drops the attention weights and scores an attention-trained trunk as if it were a
baseline (`chess_rating_net.py:726-737` builds the eval model from CLI flags, not
from the checkpoint's stored params; `:777` loads with `strict=False`). The API side
fixed exactly this class of mismatch by reading params from the checkpoint; the CLI
eval path did not adopt the same pattern. Filed as new-issue N3 (§5.3).

### 1.3 Seed separation — correct

`--seed` (default 0, varied) and `--split_seed` (default 42, never varied) exist with
accurate help text (`chess_rating_net.py:610-611`). `main()` seeds torch, numpy and
`random`, and builds a `torch.Generator` passed to the train DataLoader only
(`:644-648`, `:706`). The split is correctly isolated from `--seed`:
`load_or_create_split` uses `train_test_split(..., random_state=split_seed)`
(`:582-583`), which ignores global RNG state. Both seeds are recorded in `params`
(`:672-673`) and echoed by the environment log (`:692`). This delivers exactly the
§B.3 prescription and un-conflates `chapter3.tex:275`'s "seed 42" on the code side.

Two scope notes, not defects: no `worker_init_fn` is set, which is fine because
`ChessGamesDataset.__getitem__` contains no randomness — only shuffle order needs the
generator; and no `cudnn.deterministic` is set, so runs are seed-controlled but not
bitwise reproducible on GPU. That is sufficient for the 5-seed protocol (§B.3), which
needs statistical, not bitwise, control; say "seed-controlled" rather than
"deterministic" in Chapter 3.

### 1.4 Defaults — correct

`--epochs` 60 (`:599`), `--val_batch_size` 512 (`:608`), both mirrored in
`prototype/example_config.yaml:6,9`. Repo-wide grep finds no surviving
`epochs: 100` / `8192` configuration. The committed defaults now run the documented
control protocol as-is and no longer OOM a 24 GB A5000.

### 1.5 Deterministic auditable split — correct, with two durability qualifications

`main()` lists sorted basenames (`:683`) and routes every run — training *and* the
inference/eval path — through `load_or_create_split` (`:687`), which freezes the
first split into `split_manifest.json` (three file lists + SHA-256 over the sorted
basenames, `:555-589`) and on later runs verifies both the hash (`:562-567`) and
exact file-list consistency (`:570-573`) before use, failing loudly with an
actionable message on any mismatch. Counts and hash are printed every run
(`:692-693`). The nested proportions are unchanged (10% test, then 20% val →
72/18/10, `:582-583`), so this delivers §G.1 as specified — determinism from
`sorted()` + seed, auditability from the manifest, and the eval process can now
prove it scored the frozen partition.

Qualifications:

1. **The manifest lives in `data_dir`, and on the HPC that is `/tmp`, which is wiped
   on every reboot** (per `analysis/hpc-snapshot.txt`, 8 reboots since Jul 31). After
   a wipe-and-restage the manifest is silently *recomputed*, not verified. The split
   itself survives (same sorted names + same seed give the same partition, and any
   difference in the restaged file set changes the hash of a *surviving* manifest —
   but a wiped manifest can't catch a short restage). The audit-trail property §G.1
   wanted — "lets an eval process prove it scored the right games" — currently lasts
   exactly one reboot. Fix is one line of operational policy: write (or copy) the
   manifest into `model_dir` on `/home`, or commit it; verify against the durable
   copy.
2. At 1.2M files the manifest is a ~50 MB JSON of 1.2M basenames — workable but
   unwieldy to diff or commit. Storing per-list SHA-256 digests plus the file lists
   in a compressed sidecar, or moving to the §F.2 shard manifest, avoids that.

### 1.6 Per-game error dump — partial: absolute errors only

`test()` accepts `per_game_csv` and writes one row per game — `game_id, white_err,
black_err, time_control, white_elo, black_elo` — with errors in Elo points and a
`finally`-guarded close (`chess_rating_net.py:449-492`); the eval path wires it to
`<model_dir>/per_game_errors.csv` (`:778-779`). Elo values are properly numeric
(`:88`), game_id is the pickle basename stem (`:107`). That much is correct.

But the errors are stored as **absolute values** (`torch.abs(...)`, `:476-478`),
and the docstring plus commit message claim the file enables "paired-bootstrap /
bracket / calibration / move-index analyses." Two of those four are actually enabled;
two are not:

| Analysis | Needs | From \|err\| alone? |
|---|---|---|
| Paired bootstrap on MAE deltas (§G.2) | per-game \|err\| for both arms | **Yes** |
| Rating brackets + per-bin n (§G.3) | \|err\| + true Elo per side | **Yes** |
| Signed-bias / calibration / compression slope (§L6) | signed `ŷ−y` or predictions | **No** |
| Move-index curve (§L13) | per-ply predictions | **No** |

§L6 is not decorative — it is the shrinkage-to-the-mean check the anomaly module
depends on. The fix costs two columns: store `white_pred, black_pred` (or signed
errors) instead of, or beside, the absolute values; `abs()` is recoverable
downstream, sign is not. The move-index curve legitimately needs a separate per-ply
dump and can stay out of this CSV, but then the docstring should stop claiming it.
Also note `test()` is invoked with the CSV only on the eval path; a training run
still ends without a test pass (`main()` ends at `:827-831`), so the dump requires
the separate eval invocation — consistent with the historical workflow, worth one
sentence in the runbook.

### 1.7 Resume fixes — correct

- `save_checkpoint` persists `scheduler.state_dict()` and `best_epoch`
  (`:501-522`); `load_checkpoint` restores both when present (`:525-538`); the
  resume block passes the scheduler in (`:752-757`). `num_bad_epochs`/`best` no
  longer reset on resume, and the trailer's `best val epoch` (`:830`) is no longer
  the constant 0 after a resume.
- The epoch/`latest.pth` checkpoints are now written **after** the best-val update
  (`:804-825` — best first, then epoch/latest), so a resume from `latest.pth` sees
  the post-update `best_val_loss`; the §F.3.2 off-by-one is gone.
- `--checkpoint` exists on the eval path (`:604`, `:763-766`), removing the
  `cp best_model.pth model_55.pth` operator ritual that §F.3.3 flagged as the source
  of the byte-identical-eval-headers incident.
- Environment logging prints data_dir, file count, split sizes, manifest hash and
  the fully resolved hyperparameters on every run (`:692-694`), which is what §G.1
  needed to make future split questions settleable from logs.

Edge notes (acceptable, record them): resuming from a *pre-fix* checkpoint silently
restores neither scheduler nor best_epoch (keys absent → fresh scheduler, best_epoch
0) — that is the legacy behavior, but it is now silent where a one-line "legacy
checkpoint: scheduler state unavailable" warning would prevent misreading a resumed
run's trailer; and `best_epoch` is stored 0-based while the per-epoch prints are
1-based ("Epoch 34" is `best_epoch=33`), a pre-existing cosmetic off-by-one in the
trailer worth normalizing when it next matters.

### 1.8 format_data — usernames and guard correct; reject count is not observable

- Usernames: `White`/`Black` headers recorded into the pickle dict
  (`format_data.py:88-89`), exactly the §A.4 two-line change, unlocking the
  player-disjoint split for the re-preprocessed corpus. Downstream consumers are
  key-tolerant (`ChessGamesDataset` reads specific keys; extra keys are ignored).
- Guard: `len(clocks) != len(positions)` rejects the game (`:78-80`). Positions are
  appended on every ply and clocks only on a `[%clk]` match (`:63, :66-69`), so a
  missing annotation *anywhere* — including mid-game, the silent-shift case §M6
  identified — produces a length mismatch and a rejection. The inference path uses
  the same `parse_game` (`api.py:152`), so the API is covered too.
- **Reject count: dead code as shipped.** `rejected_games` (`format_data.py:14`) is
  incremented on both reject paths (`:75, :79`) and **read by nothing in the
  repository** — no print, no log, no import. The §M6 prescription was "log the
  reject count per month **so the rate is known**"; a write-only module global does
  not deliver that. Two further gaps compound it: the corpus-critical driver is
  `preprocess_lichess.py` **on the HPC, outside this repo** (`hpc-data-setup.md:38-42`
  — it prints only aggregate `scanned=`/`kept=`, which conflates clock rejects with
  Elo/time-control filter rejects), and if that driver parallelizes with
  multiprocessing a module global undercounts. Concrete completion: have `parse_game`
  return a reject *reason* (or expose `get_reject_counts()`), have the driver print
  `rejected_no_clocks=` and `rejected_clock_misalign=` per month, and sync the fixed
  `format_data.py` to the HPC copy **before** the 1.2M preprocessing — the fix
  protects the corpus only if the corpus is built through it.

---

## 2. Methodology strictness

### 2.1 Rating evaluation — what the manuscript promises vs. what now exists

Status of the audit's nine "promises the code does not keep" (§K.2) at HEAD:

| # | Manuscript promise | Status at HEAD |
|---|---|---|
| 1 | Fixed seeds, ≥3 seeds per ablation (`ch3:153`, `ch4:25,69`) | **Machinery kept** (1.3); the seed *runs* remain to be executed; `ch3:275`'s split/init conflation still needs the doc fix |
| 2 | Rating-bracket subgroup eval (`ch3:149,173,239`; `ch4:135-157`) | **Still unimplemented** — `test()` breaks out time control only (`chess_rating_net.py:446-447,469-473`); no bracket code anywhere in `prototype/src` |
| 3 | Paired bootstrap, 10,000 resamples (`ch3:149,238`; `ch4:19,89`) | **Input now exists** (per-game CSV), **analysis code does not**; the hierarchical seed-aware variant (§B.3) also absent |
| 4 | `S_max`, `S_mean` reported jointly (`ch3:126-134`; `ch4:93`) | **Still unimplemented** — `anomaly.py` computes only the attention-weighted sum; note the manuscript now *specifies* all three with the evasion rationale, so the code lags the spec |
| 5 | RMSE (`ch3:236`; `ch4:74`) | **Still unimplemented** (recoverable from the CSV only if predictions/signed errors are added — see 1.6) |
| 6 | Offline-vs-live agreement (`ch3:243`) | **Now feasible** for the baseline checkpoint (1.1); still unmeasured |
| 7 | Same split across all configurations (`ch3:153`) | **Kept** — frozen verified manifest (1.5) |
| 8 | Latency benchmark (`ch3:242`; `ch4:175`) | **Still unmeasured**; no timing code in the prototype |
| 9 | `R_baseline` = pre-game Glicko-2 (`ch3:213`) | **Still broken** — self-prediction fallback intact (`api.py:199-204`) |

So the fixes closed rows 6 and 7 and built the *input* for rows 3 and 5; rows 2, 3,
4, 5, 8, 9 remain open implementation debt, and row 1 remains open run debt. Two
design subtleties from the audit are still nowhere in code or manuscript and should
ride along with row 2/3 implementation: the bracket table must aggregate over
**player-observations (2N)**, not games, with per-bin `n` and CIs (§G.3 — the thin
ultrabullet/classical cells are noise at current sizes), and the headline MAE is
still a mean of per-batch means (`test():498` divides by `len(test_loader)`), the
§L3 estimator defect — now cheaply fixable downstream from the CSV, but the in-code
headline remains the wrong estimator and the Ch4 convention statement is still owed.

### 2.2 Anomaly evaluation — designed or deferred?

**The bot-vs-bot protocol is genuinely designed, not deferred.** `chapter3.tex:254`
commits to: the 9-model Maia ladder; Stockfish 16 + Lc0 as substitution engines;
difficulty by depth 5/10/20; substitution rates 0/5/15/30/60% at uniformly random
positions introduced at generation time; ~5,000 games per case per band per engine
(≈450,000 total); randomized 6-ply openings; 100-ply cap; ROC-AUC plus precision at
1% and 5% FPR; thresholds calibrated on validation only (`ch3:260`); Kaggle
cross-evaluation and a prioritized real-world closed-account path with a KS test and
an explicit drop-contingency (`ch3:256`). `ch4:95-103` scaffolds the reporting,
including per-rate/per-engine/per-band breakdowns.

**But the design does not yet survive the four §L14 credibility traps**, and its
compute is uncosted. Trap-by-trap:

1. **Circular ground truth.** The *synthetic* path is independent of Lichess's
   detectors by construction — good. The *real-world* path remains circular
   (closed-account labels come from Kaladin/Irwin, which are engine-correlation
   detectors); `ch3:258` documents the pipeline but never states the limitation on
   what the KS test can conclude. Defense exists for one path, is unstated for the
   other. **Partially designed.**
2. **Survivorship bias.** Nothing anywhere addresses it. **Deferred.**
3. **Anomaly-type conflation.** `d_t` is unsigned (`ch3:115`), so sandbagging and
   engine-boosting are indistinguishable by construction; S_max/S_mean address
   *evasion*, not *type*. No signed deviation, no per-type reporting anywhere.
   **Deferred.**
4. **"Surprising = suspicious."** No brilliancy/legitimate-surprise control set
   anywhere in ch3, ch4, or the planning docs. **Deferred.**

Compounding blockers, all confirmed still present at HEAD: the anomaly head has zero
learnable parameters and is unreachable from training (`anomaly.py` — no
`nn.Parameter`; `train_one_epoch` calls `model(...)` without `return_attention`,
`chess_rating_net.py:372`), so the planned 4-arm battery still schedules two no-op
arms (`chapter3-4-process.md:63-64`); the attention weights feeding `S_att` still
carry the §H.1b query-averaging positional artefact (`chess_rating_net.py:294-297` —
the mean-over-queries choice the audit showed is monotonically biased toward early
plies); and the 450k-game corpus has no compute budget (audit question J.13). One
protocol under-specification: "depth 5/10/20" is well-defined for Stockfish but not
for Lc0, whose strength knob is visits/nodes. Section 6 addresses all of this.

---

## 3. Objectives coverage

Objectives as stated at `chapter1.tex:63-67`: (1) *Design* of the attention-augmented
CNN-BiLSTM for real-time skill estimation and move-level anomaly detection;
(2) *Evaluation* of prediction accuracy and anomaly-detection performance on the
Lichess Open Database; (3) *Development* of a real-time web prototype surfacing
per-move ratings, suspicion scores, and critical moves to a fair-play reviewer.
Operationally these resolve to three required results: a defensible **skill-estimation
MAE**, a defensible **anomaly ROC-AUC**, and a working **real-time platform**.

### 3.1 Skill-estimation MAE — plan produces it; five named gaps

Required: full-corpus (≈1.2M) MAE for reproduced-baseline and extended arms, with
seeds, per-time-control and per-bracket subgroup tables, and a paired bootstrap
against the reproduced baseline, with 182 as reference (`ch4:17-19`).

The plan now credibly produces this: the pipeline is validated end-to-end (50k
sanity gate: terminal train MAE 181.3 ≈ the paper's 182; 170k ablation complete),
seeds/split/resume/CSV machinery landed, and the audit's scaling fit projects
184–190 at 1.2M. Missing, in dependency order: (a) the 1.2M corpus itself —
preprocessing not started; storage confirmed feasible (§5.4) but only with the §F.2
sharding; (b) the 5-seed replicates at 170k (σ_seed is the denominator for every
later claim); (c) bracket-eval and bootstrap *code* (§2.1 rows 2-3); (d) the ~400k
intermediate run validating the scaling extrapolation before the full preprocessing
spend; (e) ratification of the evaluation bar — `ch3:236,238` still defines success
as ≤182 with a CI that must not cross 182, which the projection says will read as
self-inflicted failure (adviser question R5-Q6, the urgent one).

### 3.2 Anomaly ROC-AUC — plan does not yet produce it

Required: AUC with precision at 1%/5% FPR on a labeled corpus (`ch3:240`),
KS on the real-world sample when available (`ch3:241`).

Currently not producible: **no labeled corpus exists and none is costed**; the
scoring head is untrainable/unreachable (so "anomaly-detection performance" of the
*trained* system is not yet a defined quantity — it is a post-hoc scoring rule);
`S_max`/`S_mean` are unimplemented; the `S_att` signal is contaminated by the
query-averaging artefact; and the four credibility traps are undefended (§2.2). The
anomaly-corpus track is the critical path for objective 2 and has the longest lead
time after the 1.2M corpus itself. Section 6 is the concrete plan; its prerequisites
(§6.7) name the code decisions that gate it.

### 3.3 Real-time platform — demo now honest, three gaps to "real-time"

Required: live per-move ratings, suspicion scores and critical moves with measured
latency (`ch1:66`, `ch1:140` defines real-time as soft-real-time).

Fix 1 made the served predictions *correct and reproducible* — the precondition for
everything else. Remaining gaps: (a) **causality** — the per-move curve is
retrospective (bidirectional LSTM; §L1), the prefix-recompute inference mode the
audit recommends does not exist (no such code in `prototype/src`), and adviser
questions R5-Q16/17 that ratify the resolution are still open; (b) **latency** —
promised (`ch3:242`), unmeasured, and the prefix-recompute mode is ~T× the forward
passes, so the latency number that substantiates the title depends on the causality
decision; (c) **live Lichess streaming** — `ch1:81`/`ch3:198` promise live game-ID
streaming; the prototype exposes PGN text/upload endpoints only (`api.py:266-315`);
suspicion scores are neutral zeros until an anomaly-capable checkpoint exists
(1.1); and the panel's Rev-1 "playable platform" scope question (audit §K.1) is
still an unresolved scope risk for what "platform" must mean.

---

## 4. Deviation classification

Classification rule used: **minor** = the approved method's intent is preserved and
only the written description must change; **major** = changes the corpus, what is
measured, the comparison target, or the claim — requires the adviser's sign-off.

| # | Deviation from approved Ch3 | Class | Basis |
|---|---|---|---|
| D1 | `sorted()` listing + frozen verified split manifest | **minor** | `ch3:211` specifies proportions/seed, not listing order; this *delivers* `ch3:153`'s same-split promise. Document in Data Partitioning + Reproducibility (`ch3:288`). Confirm under the Stage-1 comparability constraint — drafted as R5-Q18 |
| D2 | `--seed`/`--split_seed` separation; 5 seeds not 3 | **minor** | `ch3:153` already promises ≥3 seeds; `ch3:275` must be rewritten (it documents the split seed as "the" seed). 5 vs 3 is a strengthening within the stated protocol |
| D3 | Hierarchical (seed × game) bootstrap replacing the plain game-level CI, and CI computed against the reproduced baseline, not 182 | **major** | Changes the significance test itself and its target; `ch3:238` as written specifies CI-vs-182. `ch4:19` already uses the correct target — Ch3 must be aligned. R5-Q10/Q11 |
| D4 | Evaluation bar: 182 as reference point, not pass/fail threshold | **major** | `ch3:236-238` vs `ch4:17`. The single most consequential ratification; projection says 184–190 at full scale. R5-Q6 |
| D5 | Within-month sampling: reservoir/uniform instead of first-30k head-of-archive | **major** | Changes the corpus relative to the (undocumented) baseline practice; `ch3:157/169-171`'s 40-month uniform coverage is compliance, but the within-month mechanism is a corpus-definition choice. R5-Q2 (plus R5-Q3's 400k checkpoint) |
| D6 | Usernames recorded in per-game data | **minor** | Schema addition, no method change — but `ch3:177` ("no player identifiers stored") and `ch4:187` become **false** for the rebuilt corpus and must be rewritten |
| D7 | Reporting a player-disjoint split beside the game-level split | **major** | Adds a reported evaluation and reframes the leakage disclosure (`ch3:211`) from apology to measurement. R5-Q4 |
| D8 | Partial-clock game rejection (inclusion criterion tightened) | **minor** | Corpus-inclusion change that is a correctness guard; expected near-zero rate — which is exactly why the per-month reject count (1.8) must be reported, so "near-zero" is measured, not assumed |
| D9 | Bracket subgroup aggregated over 2N player-observations with per-bin n and CIs; thin-cell handling | **minor** | `ch3:149/173` names brackets but not the unit; must be documented when implemented. Thin-cell merge/drop decision: R5-Q12 |
| D10 | Headline MAE estimator: per-game pooled mean (and macro-average reported beside it) instead of mean-of-batch-means | **minor** | Estimator correction toward what `ch3:219-222` already defines; convention statement owed in Ch4 (§L3) |
| D11 | Prefix-recompute inference mode + retrospective framing of the per-move curve | **major** | Resolves the `ch3:65` (future context) vs `ch1:19` (as-each-move-is-submitted) contradiction; changes what "real-time" claims. R5-Q16/Q17 |
| D12 | Anomaly head: give it parameters and a loss, or reframe as a post-hoc scoring rule and drop the two no-op ablation arms | **major** | Changes either the architecture or the claim, and halves the planned battery (`chapter3-4-process.md:63-64`). R5-Q14 |
| D13 | Deeper CNN: implement `--cnn_blocks` + 2×2 ablation, or scope the claim down | **major** | The claim exists at 11 manuscript sites incl. `ch3:37`; the code has no depth flag. R5-Q9 |
| D14 | Documentation-only corrections: loss space (`ch3:104` says standardized; code computes in Elo), clock-constant provenance (`ch3:33` vs `ch3:180`; `ch1:154` applies rating constants to clocks), stale code-line citations (`ch3:270-274` — now *more* stale after the fix shifted line numbers) | **minor** | §M7, §C.2, §M5. No method change |

**Adviser questions implied.** Every major deviation above maps onto the already
drafted R5 set (audit §J): R5-Q2/Q3 (sampling + 400k), R5-Q4 (dual split), R5-Q6
(evaluation bar — urgent), R5-Q9 (deeper CNN), R5-Q10/Q11 (bootstrap and its
target), R5-Q12 (thin cells), R5-Q14 (anomaly arms), R5-Q16/Q17 (causality/live
inference), R5-Q18 (split reproduction). **The fixes introduce no deviation that
needs a question not already drafted** — the correct move is to send the drafted
R5 list, leading with Q6, Q16/17, and Q14, since those gate Chapter 4's framing, the
platform architecture, and half the ablation budget respectively.

---

## 5. Cross-check: the reference audit vs. the fixed code

### 5.1 The audit's blocking list (§ priority items 1–7)

| Priority item | Status |
|---|---|
| 1. §M1 demo | **Resolved** (1.1/1.2), with warn-vs-refuse and H.1c residuals |
| 2. §G.1 sorted + manifest | **Resolved** (1.5), manifest-durability qualification |
| 3. §G.2 per-game dump | **Partially resolved** (1.6 — absolute-only) |
| 4. §B.3 seed plumbing | **Resolved** (1.3) |
| 5. §A.4 usernames | **Resolved in-repo**; must be synced to the HPC driver before preprocessing (1.8) |
| 6. §M6 clock guard | **Resolved** (1.8), reject count not yet observable |
| 7. §F.1 defaults + §F.3 resume | **Resolved** (1.4/1.7) |

Items 8–14 (5-seed replication, bracket eval, hierarchical bootstrap, causality
paragraph + prefix mode, the nine manuscript corrections, panel-revision gaps,
adviser questions) are all **still open** — unchanged by the fixes, as expected.

### 5.2 §M and §H finding status

M1 resolved (residuals 1.1) · M2 open (no depth flag; claim at 11 sites) · M3 open
(`ch3:71-81` equations still content-only/global/single-vector vs the code's
query-key/causal/per-move residual) · M4 open (`api.py:199-204`) · M5 partially
resolved (defaults fixed; `ch3:270-274` citations still stale, now doubly so) ·
M6 resolved (observability gap) · M7 open (doc) · M8 open (doc; D4). H.1a open
(zero-parameter head unreachable from training; two no-op arms still scheduled) ·
H.1b open (`chess_rating_net.py:294-297`) · H.1c open · H.1d open (code side;
manuscript now specifies all three aggregations) · H.1e/L1 open (no prefix mode).
Also still uncorrected: the false "differ only in the attention flags" controlled-
comparison claim at `prototype/experiments/results/170k-ablation/README.md:37` and
its echo at `chapter3-4-process.md:114` (§B.1) — both must be softened to "differ in
the attention flag *and* in uncontrolled init/dropout/batch-order randomness."

### 5.3 New issues introduced by the fixes

- **N1 (major). Historical checkpoints cannot be legitimately evaluated under the
  new split.** Sorting changed the input order to `train_test_split`, so the frozen
  manifest partition differs from every pre-fix run's (irrecoverable, unsorted
  `os.listdir`-order) partition. Expected overlap: ~72% of the new test set was
  *training* data for the existing 170k checkpoints. Consequence: the audit §G.2
  instruction "the eval must be re-run with per-game dumps" is now **unsound for the
  existing checkpoints** — re-running eval on them under the new manifest produces
  silently contaminated (optimistic) numbers with no warning. The paired bootstrap
  for the 170k comparison therefore requires **retraining both arms under the
  frozen manifest**, which the Stage-1 5-seed replication does anyway — fold them
  into one job. Cheap guard worth adding: the eval path warns (or refuses) when the
  loaded checkpoint's params carry no `split_seed`/manifest hash matching the
  active manifest.
- **N2 (minor).** Manifest durability on wiped `/tmp` (1.5).
- **N3 (minor).** CLI eval path silently drops unexpected (attention) keys —
  arch-mismatch eval without warning (1.2).
- **N4 (minor).** CSV absolute-only; docstring/commit overstate what it enables (1.6).
- **N5 (minor).** `rejected_games` is write-only; the observability half of fix 8
  is not delivered (1.8).

### 5.4 HPC sanity check — the audit's storage claims vs. live numbers

| Claim | Live numbers | Verdict |
|---|---|---|
| ~205 KB/game → ≈247 GB at 1.2M (§D.4) | 35 GB processed / 170,138 games = 205.7 KB/game → 247 GB at 1.2M | **Holds** |
| Fits `/home` free space; no second full copy (§F.2) | 586 GB free at 92% on the shared 7.3 TB pool → one canonical copy (247 GB) + working headroom fits; two copies (494 GB) would consume ~84% of the remaining shared space | **Holds, sharpened: the no-second-copy rule is binding, not advisory** |
| Stage to `/tmp` NVMe for the active run (§F.2) | root NVMe shows 318 GB available → a full 247 GB stage is 78% of free space | **Tight — stage per shard-group, not the full corpus** |
| Reboot resilience matters (§F.3) | 8 reboots since Jul 31, 5 in the last 11 days; `/tmp` wiped each time; a 1.2M run is ~30 h/arm → expect ≥1 interruption per multi-day campaign | **Holds, strengthened: sharding (240 shards vs 1.2M files) and the now-landed resume fixes are load-bearing, not nice-to-have** |
| Checkpoints not a constraint (~550 MB/run) | models dirs: 543–555 MB per completed run | **Holds** |
| N = 170,138 provenance | per-month counts 5×30,000 + 20,138 (2024-06 short stream) = 170,138 | **Consistent** |
| 4× A5000 24 GB / 32 cores / 125 GB RAM assumptions (§F.2 worker math, §B.3 wall-clock) | matches snapshot exactly | **Holds** |

One addition the audit did not make: at the observed reboot cadence, re-staging 1.2M
individual files to `/tmp` after each reboot (the current flat-file layout) costs
hours per incident and is the dominant operational risk of the full run — the §F.2
repack into ~240 shards converts each re-stage into a ~247 GB sequential copy and
should be treated as a **precondition** of the 1.2M campaign, not an optimization.

---

## 6. Bot-vs-bot anomaly-validation corpus protocol — full design

This section is the requested design document. It upgrades the committed protocol at
`chapter3.tex:254-260` rather than replacing it; divergences from that text are
marked **[Δ]** and each is a candidate manuscript edit. No code here — this is the
pre-registrable protocol.

### 6.1 Engine-substitution schedule

**Unit.** One *game-side* (one player in one game). Substitution is applied to
exactly one side per substituted game; the opponent plays clean Maia. Labels are
per-side.

**When substitution happens.** At generation time (as `ch3:254` specifies) — the
substituted side's move is chosen by the strong engine *at the board actually
reached*, avoiding the continuity violations of post-hoc move splicing.

**Rates.** 0% (clean) / 5% (light) / 15% (moderate) / 30% (heavy) / 60% (full
cheat), per `ch3:254`. The 5% stratum is the operationally realistic case and the
number that matters (`ch4:167` already says so).

**Placement patterns [Δ — ch3 has only uniform-random].** Three patterns per
nonzero rate, reported separately:

- **P1 uniform-random** (ch3's pattern): substitution plies drawn uniformly from
  plies 7–100 of the substituted side. Hardest-to-detect baseline.
- **P2 critical-moment**: substitute only when the position is decision-dense —
  operationalized as |engine eval swing available| ≥ 150 cp between the best and the
  Maia-preferred move. This is the realistic centaur pattern `chapter1.tex:83`
  names as the known blind spot; measuring it directly converts the blind spot from
  an assertion into a number.
- **P3 block-onset**: contiguous substitution from onset ply *k* to the end of the
  cap, *k* ~ Uniform{10..60}. Models "panic cheating" from a worsening position;
  the onset ply is recorded so detection latency (plies from onset to first flag)
  can be reported.

Never substitute plies 1–6 (the randomized opening book, `ch3:254`).

**Clock realism [Δ — unstated in ch3, and it is the biggest validity hole].** The
model consumes clocks as an input feature, so a synthetic corpus with degenerate
clock tracks lets the detector win by reading timestamps, not chess. Each generated
game must carry a synthetic clock track: fit a per-band, per-time-control move-time
distribution (lognormal over fraction-of-remaining-time, plus increment mechanics)
from real Lichess games of that band, and sample every move's time from it — for
substituted moves too (a careful cheater masks timing). Additionally generate a
**naive-timing stratum** (substituted moves get near-constant engine-latency times)
as an explicitly easier detection condition. Report the two strata separately; the
gap between them measures how much of the detector's performance is timing tells
versus move quality.

### 6.2 Bot-strength ladder

- **Human-like side:** the full 9-model Maia ladder, 1100–1900 in 100-Elo steps
  (McIlroy-Young et al.), run as Lc0 with Maia weights at the canonical
  single-node/policy-head setting (Maia is defined by its policy, not by search).
- **Opponent:** same-band clean Maia (matched-strength pairings), so game
  trajectories are band-realistic and clean/substituted games are drawn from the
  same generator except for the substitution operator.
- **Substitution engines:** Stockfish 16 at UCI depth 5 / 10 / 20 (weak / medium /
  strong), per `ch3:254`; Lc0 with a strong network at **100 / 1,000 / 10,000
  visits** **[Δ: `ch3:254`'s "depth" is not Lc0's strength knob — specify visits]**.
- **Grid discipline [Δ]:** Stockfish covers the full grid; Lc0 runs a *probe*
  subgrid (bands {1100, 1500, 1900} × rates {5, 30} × patterns {P1, P2}) as the
  engine-generalization check rather than a full duplicate grid — §6.5 shows the
  full Lc0 grid is what breaks the compute budget, and per-engine ablation needs a
  probe, not a mirror.

### 6.3 Ground-truth labeling scheme

- **Label = construction.** A game-side is labeled *cheating* iff the generator
  substituted ≥1 of its moves (rate > 0 arm); *clean* iff rate = 0. No Lichess
  moderation signal touches any synthetic label (trap 1, §6.6).
- **Per-side labels**, with full generation metadata persisted per game: band,
  engine, depth/visits, rate, pattern, onset ply (P3), timing stratum, and the
  **per-ply substitution mask**. The mask upgrades the corpus from game-level
  classification to *move-level localization* ground truth — the thing the "critical
  moves" objective (`ch1:66`) actually needs and which no game-level label provides.
- **Clean pool reuse.** Clean games depend only on band (not engine/rate/pattern),
  so one clean pool per band serves every matched comparison — 45,000 clean sides
  (9 × 5,000) rather than a clean arm per cell.

### 6.4 Evaluation metrics

- **Game-level:** ROC-AUC per score (`S_att`, `S_max`, `S_mean`, reported jointly
  per `ch3:134`); precision at 1% and 5% FPR with thresholds calibrated on the
  corpus's validation split only (`ch3:260`); paired bootstrap CIs over matched
  clean/substituted pairs (10,000 resamples, pairs kept together — audit §H.3).
- **Stratified:** every metric broken down by rate × engine × band × pattern ×
  timing stratum, with per-cell n (pooled AUC dominated by the 60% arm is
  uninformative — audit §H.3).
- **Per-anomaly-type precision/recall** at the calibrated operating points, over
  the type set {engine-boost P1/P2/P3, sandbagging (§6.6-T3), clean, brilliancy
  control (§6.6-T4)} — this is the trap-3 deliverable.
- **Move-level:** localization AUC of per-ply flags against the substitution mask,
  plus detection latency (plies from P3 onset to first flag). Report under
  retrospective scoring as "post-hoc review aid" unless/until the prefix-recompute
  mode exists (§6.7).
- **Real-world (if a sample materializes):** KS two-sample test as `ch3:256`,
  reported as *agreement with the platform's moderation pipeline*, never as truth
  (§6.6-T1).
- **Prevalence honesty:** report precision at realistic prevalence (0.1–1%) beside
  the matched-corpus numbers (audit §H.3) — the matched corpus is ~50% positive and
  precision numbers quoted from it will not survive a panelist otherwise.

### 6.5 Game count and compute budget

**Count [Δ — replaces ch3's flat 450k with a powered, staged build].** With AUC
around 0.80, ~1,000 substituted sides per cell bound the per-cell AUC
standard error near ±0.015 — sufficient for per-cell reporting; pooled and
per-factor marginals get much tighter for free.

| Component | Cells | Games |
|---|---|---|
| Stockfish full grid | 9 bands × 4 rates × 3 patterns × 2 timing × 1,000 | 216,000 |
| Lc0 probe | 3 bands × 2 rates × 2 patterns × 2 timing × 1,000 | 24,000 |
| Clean pool | 9 bands × 5,000 | 45,000 |
| Brilliancy controls (§6.6-T4) | real titled-player sides + synthetic upsets | ~4,000 |
| **Total** | | **≈289,000** |

Build in two stages: **Stage A pilot** = 10% of every cell (~29k games) to shake out
the generator, the clock model, and first AUCs; **Stage B** = the remainder, only
after Stage A numbers and the §6.7 prerequisites are green. Pre-register the grid
and the stopping rule (no post-hoc cell additions without a logged amendment).

**Compute (4× A5000, 32 cores, no SLURM — `analysis/hpc-snapshot.txt`).** Per game
≤100 plies: ~100 Maia policy evaluations (milliseconds each, batchable) + ≤60
substituted-move searches. Stockfish NNUE at depth ≤20 averages ~0.1–0.3 s/move/core:
216k games × ~30 engine moves × 0.2 s ≈ **360 CPU-core-hours** — about half a day
across 32 cores. Lc0 at 10k visits ~1–2 s/move on one A5000: 24k × 30 × 1.5 s ≈
**300 GPU-hours** — 3–4 days on one GPU; this is why Lc0 is a probe, not a full
grid (a full Lc0 mirror would be ~2,700 GPU-hours and would eclipse the entire
training budget, confirming audit question J.13's concern). Eval-swing computation
for P2 adds one shallow search per ply on the substituted side (depth 10, ~50 ms):
≈120 CPU-core-hours. **Total: under ~500 CPU-core-hours + ~300 GPU-hours, roughly
one week wall-clock alongside training, on hardware already in hand.**

**Storage:** PGN + metadata + masks ≈ 2 KB/game → < 1 GB. Score the corpus by
streaming PGN through `parse_game` at evaluation time rather than materializing
pickles — materialized, it would cost ≈60 GB against a `/home` pool where §5.4 shows
every gigabyte is contended.

### 6.6 The four credibility traps — concrete design choices

- **T1 — Independent label source.** All ROC/precision/PR claims rest exclusively
  on construction labels (§6.3). Lichess-ban-derived data appears only in the KS
  agreement analysis, and `ch3:258` gains one sentence stating the circularity
  limitation explicitly: *closed-account labels descend from engine-correlation
  detectors, so the KS result measures agreement with that pipeline, not
  ground-truth cheating.* **[Δ: manuscript edit]**
- **T2 — Survivorship-bias stratification.** The synthetic grid *constructs* the
  players survivorship hides: the careful-cheater strata (5% rate, P2 placement,
  masked timing) are exactly the population that bans under-sample. Report detection
  per care-level stratum — the naive-vs-careful gap is the measured bound on how
  much a ban-derived evaluation would overstate detection. If a real closed-account
  sample materializes, additionally stratify it by games-played-before-closure as a
  care proxy, and state that it represents *caught* cheaters only.
- **T3 — Per-type reporting.** Report **signed** deviation alongside |d_t|
  (`ch3:115` is unsigned; sign separates engine-boost, predicted ≫ baseline, from
  sandbagging, predicted ≪ baseline, at zero cost — audit §L14.3). Add a
  **sandbagging arm**: clean Maia-band-b games evaluated against a baseline of
  b+400 (the deflated-account signature); the detector must flag them as
  anomalous-low, and they enter the per-type P/R table as their own class.
  **[Δ: adds one cheap arm and one manuscript equation note]**
- **T4 — Brilliancy control set for the false-positive rate.** ~2,000 real
  game-sides by titled players (public Lichess API, 2401+ bracket; evaluation-only,
  never training), plus a synthetic surprise arm (clean Maia-1900 beating
  Maia-1100 — legitimate "surprisingly strong vs. expectation" games). Requirement,
  pre-registered: at the calibrated 1%/5% FPR thresholds, FPR on the brilliancy set
  must be reported, and if it exceeds ~2× the calibrated rate the honest conclusion
  is that the score measures *surprise*, not *assistance* — reported as such, not
  tuned away. **[Δ: new control set; answers audit §H.3's missing control]**

### 6.7 Prerequisites this design inherits (gate Stage B on all four)

1. **Head decision (R5-Q14):** parameterize the anomaly head with a training loss,
   or reframe it in Ch3 as a post-hoc scoring rule and drop the two no-op ablation
   arms. The corpus is valid either way; what the AUC *means* differs.
2. **Fix the attention feed (§H.1b):** use the final-query attention row (or a
   dedicated head), not the query-mean — otherwise `S_att` inherits a positional
   ramp and the per-band results will partly measure the artefact.
3. **Require `R_baseline` (§H.1c):** the self-prediction fallback must go before
   any suspicion score is evaluated; every synthetic game has a defined baseline
   (the Maia band's nominal rating), so the corpus never needs the fallback.
4. **Causality labeling (L1 / R5-Q16):** game-level AUC is safe under retrospective
   scoring; any *move-level* localization claim must either use prefix-recompute
   inference or be labeled a post-hoc review aid, matching whatever Ch3 framing the
   adviser ratifies.

---

*Prepared read-only against HEAD `4b415c4`. Every verdict cites the current line it
rests on; where a fix is called incomplete, the completing change is named in the
same paragraph.*
