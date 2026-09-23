# Comparison methodology for the 1.2M evaluation — settled

> **Status note (2026-08-19).** Written 2026-08-16 against the then-current 1.2M /
> April 2021-July 2024 scope. The three-way comparison logic below is unchanged and
> still settled; only the corpus figures are superseded by the 2026-08-17 adviser
> ruling (~2M games, April 2021 through present). Read "1.2M" throughout as "the full
> corpus", now ~2M.

**Assessment date:** 2026-08-16. Assessment only; no manuscript or code edits.
Sources: this repo (cited as `file:line`) plus, where noted, the author's public
release repo `AstroBoy1/RatingNet` (verified 2026-08-16).

**Bottom line first.** Both prior informal answers were right, and they are compatible.
`model_55.pth` **is** the author's released checkpoint: the paper links his code repo
(`contexts/inspirationpaper.md:552`), and that repo's README publicly distributes the
weights — "Download model_55.pth and put it in models/cnn_bilstm_clocks_all. We also
provide a direct download from google drive at this link:
https://drive.google.com/drive/folders/164qXisHsNAKSM6R7ZMeTeJjPpnZ7s5Rt" — the
identical folder ID recorded in this repo (`README.md:159`, `prototype/README.md:21`).
So yes, we can score his model on our test set, under the conditions in §2. But that
score is a *comparison against his model*, never a *reproduction of his 182*: his
dataset, split, and aggregation are not recoverable (§4). The defensible structure
after 1.2M is: **primary = our full model vs our own retrained baseline on the
identical frozen split; supporting = his checkpoint scored on that same split; 182
quoted only as an indicative reference.** That matches what the repo has already ruled
twice (`chapter3-4-process.md:75-76`, `analysis/170k-verification.md:849-851`). One
piece of hygiene remains before (a) is panel-proof: hash-verify our local copy against
the author's Drive release (§1).

---

## 1. Provenance of the checkpoint

**Verdict: author-released, publicly distributed; our local copy's integrity is the
only unverified link.**

What is **proven**:

- The paper's artifact section links the author's code repo: "The code:
  https://github.com/AstroBoy1/RatingNet" (`contexts/inspirationpaper.md:552`). The
  paper text itself never mentions weights — but the linked repo's README does,
  explicitly: "Download model_55.pth and put it in models/cnn_bilstm_clocks_all. We
  also provide a direct download from google drive at this link:
  https://drive.google.com/drive/folders/164qXisHsNAKSM6R7ZMeTeJjPpnZ7s5Rt?usp=sharing"
  (AstroBoy1/RatingNet README, "Installation and Setup"; verified 2026-08-16). The
  checkpoint is a released artifact of the paper.
- The Drive folder this repo records — "Mirror / download: [Google Drive folder]"
  (`prototype/README.md:21`) and "download from the [Google Drive folder]"
  (`README.md:159`) — carries the **same folder ID** (`164qXisHsNAKSM6R7ZMeTeJjPpnZ7s5Rt`)
  as the author's README link. It is the author's own distribution channel, not a
  team upload.
- The checkpoint's internal format matches the author's save code exactly. His
  original `src/chess_rating_net.py` hardcodes
  `best_path = "models/cnn_bilstm_clocks_all/model_55.pth"` (line 246, matching our
  audit's note at `contexts/essentials/ratingnet_code_audit.md:50`) and saves
  `torch.save({'model_state_dict': model.state_dict(), 'params': params}, best_path)`
  (lines 316-318; verified 2026-08-16). Our `api.py` reading `saved["params"]` and
  `saved["model_state_dict"]` successfully (`prototype/src/api.py:93-94`, `:115`;
  live load at `pre-oral-deck-plan.md:15`) is therefore evidence the file is in the
  author's own format.
- The file size is authenticity evidence, not a concern: 3,058,200 bytes
  (`prototype/README.md:19`) fits an fp32 state dict of the ~760k-parameter baseline
  plus a small params dict — the author's optimizer-free format — whereas every
  checkpoint our own trainer writes always embeds `optimizer_state_dict`
  (`prototype/src/chess_rating_net.py:507-528`, `prototype/README.md:108-109`) and
  would be roughly 3× larger. The frozen file therefore cannot be a mislabeled
  team-trained checkpoint.
- The author confirmed the epoch story (hedged): "I think it was training up to 60 and
  epoch 55 was the best validation with patience of 5 epoch and maybe forgot to update
  the appendix" (`contexts/consultation_log.md:122`). So "best-val epoch of his
  60-epoch run" is author-confirmed, and the paper's stated "epochs: 50"
  (`contexts/inspirationpaper.md:550`) is an appendix error by his own account.

What is **assumed** (the one open link): that our local file — staged 2026-08-12 from
`/tmp/opencode/model_55.pth` (`prototype/README.md:19`; commit `f807e93`) — is
byte-identical to the author's Drive release. No hash is recorded anywhere in the
repo, and a `strict=False` load succeeds silently even for a wrong file
(`prototype/src/chess_rating_net.py:323-330`), so a successful load does not prove
integrity.

**To close it** (one sitting): download `model_55.pth` fresh from the author's Drive
link, SHA-256 both files, record the match (and the hash) in
`prototype/README.md` and `contexts/consultation_log.md`. Only if the hashes differ,
ask Omori which is authoritative. Two hygiene notes while there: the phrase "Option C
\"both\" weight strategy, captain-approved" (`prototype/README.md:15-16`) references a
decision documented nowhere in the repo or git history — either document it or delete
it; and downstream docs should cite the author's README as the provenance source
rather than each other (`contexts/essentials/thesis_explained.md:231`,
`pre-oral-deck-plan.md:104`). Beware also that team-trained files named
`model_55.pth` exist in the ablation dirs — the eval workflow copies checkpoints over
that name (`prototype/experiments/results/170k-ablation/README.md:62-63`) — so never
identify this artifact by filename alone; identify it by hash.

## 2. Can we score the checkpoint on our test set?

**Yes — architecturally loadable and scoreable.** `api.py` loads it with
`torch.load(checkpoint_path, map_location=device, weights_only=False)` then
`params = saved["params"]` (`prototype/src/api.py:93-94`), builds `ChessEloPredictor`
with `use_attention`/`use_anomaly` defaulting to False
(`prototype/src/api.py:102-113`), and loads weights via
`load_base_state_dict(..., strict=False)` with no key remapping
(`prototype/src/api.py:115`, `prototype/src/chess_rating_net.py:323-330`). The trunk
is the baseline: "The base architecture matches the released RatingNet checkpoint.
When ``use_attention`` is False the forward pass is identical to the baseline"
(`prototype/src/chess_rating_net.py:159-162`), with CNN/BiLSTM/head blocks marked
"identical to baseline" (`prototype/src/chess_rating_net.py:197-233`), matching the
audited upstream dims (4 conv layers 32→64→128→256; BiLSTM hidden 64 ×3
bidirectional; LSTM input 257) (`contexts/essentials/ratingnet_code_audit.md:40-41`).
A live load and 16–31 ms inference is on record (`pre-oral-deck-plan.md:15`).

**The test-MAE path exists.** Running `chess_rating_net.py` without `--train` loads
the checkpoint and calls `test()` (`prototype/src/chess_rating_net.py:782-789`), which
computes `nn.L1Loss` on **de-standardized** ratings —
`criterion(outputs * ratings_std + ratings_mean, targets * ratings_std + ratings_mean)`
(`prototype/src/chess_rating_net.py:469`) — i.e., MAE in rating points, the paper's
metric ("The MAE is given for the pre-standardized rating",
`contexts/inspirationpaper.md:422-424`), plus per-time-control MAE and a per-game CSV.
The split mechanism is the nested 72/18/10
`train_test_split(..., random_state=split_seed)`
(`prototype/src/chess_rating_net.py:583-584`) with `--split_seed` defaulting to 42 and
marked "NEVER vary" (`prototype/src/chess_rating_net.py:617`), frozen by a committed
manifest. Note the scale: the existing manifest (train 122,499 / val 30,625 /
test 17,014, sha256 `4be0f9e8…`) freezes the **170,138-game** corpus
(`hpc_seed_rerun_RUNBOOK.md:1-11`, `hpc-operations-guide.md:134-141`); the 1.2M corpus
does not exist yet (§3), so a new 1.2M manifest must be generated and committed
**before** anything is scored on it.

**Normalization constants.** `api.py` hardcodes `RATINGS_MEAN=1514.0`,
`RATINGS_STD=366.0`, `CLOCKS_MEAN=273.0`, `CLOCKS_STD=380.0`, `MAX_PLIES=100`
(`prototype/src/api.py:42-47`); the same values are constructor defaults in the
dataset (`prototype/src/chess_rating_net.py:57-64`) and are used with defaults in the
train/eval path (`prototype/src/chess_rating_net.py:703-705`). Against the paper:

- Rating mean/std **match the paper**: "The mean rating is 1514, with a standard
  deviation of 366" (`contexts/inspirationpaper.md:438-439`).
- Clock mean/std and the ply cap are **not stated in the paper** — it says only that
  clock time "is standardized by subtracting the mean and dividing by the standard
  deviation" (`contexts/inspirationpaper.md:437`). The 273/380 values
  (`contexts/essentials/ratingnet_code_audit.md:56`, which flags "No derivation
  comment") and the 100-ply cap (`contexts/essentials/ratingnet_code_audit.md:44`)
  come from the author's code per our audit. Since the checkpoint is his code's
  output (§1), his code's constants are by construction the right inference-time
  constants — and they are exactly what is hardcoded.

**Exact conditions for a meaningful score of his checkpoint:** (1) hash-verified copy
(§1); (2) key-list the loaded state dict once and record that it contains exactly the
baseline tensor names — `strict=False` would otherwise mask a wrong file; (3) score
through the same `test()` path, on the committed 1.2M manifest split, with the default
constants; (4) report it with the §3(a) caveats attached.

**Two operational gaps.** The checkpoint is not in the repo (gitignored; absent from
`/tmp/opencode` on this machine) and must be fetched from the Drive folder
(`prototype/README.md:16-21`); and the eval needs a local `.pkl` corpus
(`prototype/src/chess_rating_net.py:689-691`).

## 3. The three-way comparison after 1.2M

All three must be scored on the same committed 1.2M-manifest test split, MAE in rating
points. The 1.2M corpus deliberately mirrors the paper's window — "**April 2021 – July
2024** (the paper's window), ~30,000 games/month" (`chapter3-4-process.md:22`) — and
its remaining 34 months are built by `corpus_stream.sh` + reservoir sampling
(`corpus_stream.sh:2-4`, `prototype/src/preprocess_lichess.py:10-16`), gated on the
seed rerun finishing (`corpus_stream.sh:6-11`).

**(a) His released checkpoint on OUR test set** — *supporting result.*
- **Proves:** how the author's actual trained model performs on our data distribution
  — the only *direct* model-vs-model evidence against the published baseline, on
  identical test data with (b) and (c). Uniquely, it needs no retraining compute.
- **Does NOT prove:** anything about the number 182 (his test set, split, and
  aggregation differ — §4), and it cannot serve as the Ch4 baseline: "the Ch4 numbers
  must come from the retrained models, not the frozen checkpoint"
  (`chapter3-4-process.md:75-76`).
- **Caveats:** (i) hash verification first (§1); (ii) **train-on-test contamination**
  — his training window (Apr 2021–Jul 2024, `contexts/inspirationpaper.md:404-405`)
  contains *every* month of our mirrored corpus, so some of our test games may have
  been in his training sample. With both sides sampling ~30k/month from the tens of
  millions of monthly Lichess games, the expected overlap is of order tens of games
  per month — roughly ≲0.1% of the test split — small, but exactly quantifiable only
  with his game list, which was never released (§4); say the caveat rather than
  claiming zero. (iii) Time-control mix: the paper publishes per-control MAEs but no
  sample counts, so any aggregate carries "an unquantified mix confound"
  (`analysis/170k-verification.md:1270-1273`).

**(b) Our reproduced baseline (same architecture, retrained by us) on our test set.**
- **Proves:** a like-for-like control at our data scale, our split, our seeds — the
  correct denominator for every claim about our contributions. The seed-controlled
  170k rerun (baseline ×5 / attention ×5, `--split_seed 42` fixed,
  `hpc_seed_rerun_RUNBOOK.md:1-11`) is the existing instance of this discipline; the
  1.2M run needs its own explicit (and costed) seed plan, which no repo document yet
  commits to.
- **Does NOT prove:** equivalence to the published model. At sub-1.2M scale it is
  strictly worse (best-val test MAE 249.1 at 50k, `chapter3-4-process.md:95-97`;
  224.2 at 170k, `chapter3-4-process.md:124-127`, verified from logs at
  `analysis/170k-verification.md:160-167`); the project's own scaling projection is
  ≈184–190 at 1.2M against the published 182
  (`analysis/170k-verification.md:347-362`).
- **Caveats:** our corpus is a different sample of Lichess than his (reservoir-sampled
  30k/month for new months, `prototype/src/preprocess_lichess.py:10-16`, vs the
  head-of-month kill-at-cap sampling of the original six months,
  `analysis/170k-verification.md:65-67`; his per-month sampling method is unstated);
  and both his 182 and our numbers use game-level splits with no player control, so
  both carry the same player-leakage optimism
  (`analysis/170k-verification.md:92-95`).

**(c) Our full model (baseline + attention + anomaly) on our test set.**
- **Proves:** the thesis contribution — the (c) − (b) delta on the identical
  partition, seed-controlled, with the bootstrap Ch4 already specifies. The 170k
  lesson makes seed control non-negotiable: the apparent 2.45-point attention win
  shrank to 0.49 at epoch 60, and "with no seed control the 2.45 cannot be attributed
  to attention rather than to the initialisation draw"
  (`analysis/170k-verification.md:15-19`).
- **Does NOT prove:** superiority over the *published* model — that inference runs
  through (a) and (b), with their caveats.

## 4. What is impossible

**Reproducing his exact 182 is impossible.** Confirmed, on four independent grounds:

- **No dataset.** The paper calls its dataset open source but gives no URL
  (`contexts/inspirationpaper.md:377-378`), and no dataset release accompanies the
  code repo.
- **No split.** "The data is split randomly with 80% used for the training set and
  20% used for the test set" (`contexts/inspirationpaper.md:407-408`) — no seed, no
  validation set. Our nested 72/18/10 matches his *code*, not the paper's text
  (`analysis/170k-verification.md:1015-1018`); bit-for-bit reproduction would need
  his input-file list and sklearn version, which we do not have
  (`contexts/essentials/thesis2_plan.md:48-52`).
- **No selection protocol.** The paper's appendix says "epochs: 50"
  (`contexts/inspirationpaper.md:550`) while the author says the run went to 60 with
  epoch 55 best-val (`contexts/consultation_log.md:122`) — the published protocol and
  the actual run demonstrably differ.
- **The 182 itself is fuzzy.** The intro says 183 (`contexts/inspirationpaper.md:275-276`),
  the results table's "Average Test Loss" says 182
  (`contexts/inspirationpaper.md:464-471`), and a macro-average of the paper's own
  per-control MAEs gives 176.8 (`analysis/170k-verification.md:1163-1168`) — the
  aggregation convention is undocumented, making any comparison "indicative rather
  than exact" (`analysis/170k-verification.md:1183-1185`).

Therefore the planning-doc language "Reproduce Omori 2024's 182 MAE on the same
dataset split" (`contexts/essentials/thesis2_plan.md:25`, repeated at `:74`) and the
deck claim that our split keeps "the published 182 MAE benchmark directly comparable"
(`pre-oral-deck-plan.md:83`) are **not defensible** and should not be said to the
panel. The manuscript's Ch3 pass/fail framing against 182 has the same defect; the
audit's ruling stands: "Ch4 is right; fix Ch3"
(`analysis/170k-verification.md:1011-1014`).

**Comparing against his model is possible — and easy.** The distinction in the task
brief is confirmed: scoring his released checkpoint on our test set yields a number
*comparable in kind* to 182 (same metric: MAE in raw rating points over both players'
ratings, `contexts/inspirationpaper.md:422-424`, `:409-410`) but never *identical* to
it (different test games, different time-control mix, undocumented aggregation).
"Compare against his model" (via the checkpoint) and "reproduce his 182" (impossible)
are different claims; keep them separated in every panel-facing sentence.

## 5. The panel-safe headline

**Primary result: (c) vs (b).** Say:

> "Our primary result is the improvement of the extended model over our own reproduced
> RatingNet baseline — identical architecture, identical 1.2M-game corpus, identical
> committed test split, seed-controlled runs, bootstrap confidence intervals. We cite
> the published 182 MAE as an indicative external reference, not a pass/fail
> threshold, because the original paper does not document its dataset, split, or
> aggregation precisely enough for an exact comparison."

(Before saying "identical committed test split," generate and commit the 1.2M split
manifest — the existing manifest covers the 170k corpus only,
`hpc-operations-guide.md:134-141`; and decide the 1.2M seed count explicitly rather
than implying the 170k rerun's five-per-arm transfers.)

This is the framing the repo has already converged on: the drafted adviser question
proposes "primary claim is the delta against our own reproduced baseline on the
identical partition, with 182 cited as a reference point rather than a threshold"
(`analysis/170k-verification.md:849-851`), and the manuscript pre-commits that the 182
comparison "is only numerically meaningful on the full corpus"
(`analysis/170k-verification.md:520-522`).

**Supporting result: (a), after the hash check.** Add:

> "As a secondary check, we score the author's publicly released checkpoint on our own
> test set under identical evaluation — placing his model and ours on the same footing
> even though the published 182 was measured on his own, unreleased test split."

Describe the artifact as "the author's released checkpoint" (that is now the
documented fact — his README distributes it) with the local copy hash-verified per §1.
The adviser *is* the author: he will recognize his own Drive link, so the team should
know this provenance story cold.

**(b) at full scale is the bridge, not the headline:** it shows our reproduction lands
where the scaling curve predicts (≈184–190 projected at 1.2M,
`analysis/170k-verification.md:347-362`), which is what makes the (c) − (b) delta
credible as the contribution.
