# Pre-registration: White versus Black separation retrain

Date: 2026-09-28. Written and committed **before** either run was launched, and
before the smoke tests. Decision cutoff: **2026-10-02, evening (Philippine
time)**. The defense is 2026-10-05.

Nothing in the manuscript and nothing in the served model changes as part of
this experiment. Both new options are off by default, so every existing number
stays the product of an unchanged code path.

## Why

`analysis/side-mirroring-results.md` reports that the model's White and Black
estimates are nearly the same number inside every game. Re-verified from the
committed per-game dumps before writing this note, with the same script:

| Quantity | `attn_tuned` (served) |
| --- | --- |
| median per-game \|predicted White minus Black\| | 0.7126 |
| maximum, over all 255,000 test games | 4.4336 |
| standard deviation of the signed predicted gap | 0.95 |
| standard deviation of the signed real gap | 126.90 |
| Pearson r, signed predicted gap against signed real gap | **0.00092** |
| overall MAE | 171.9168 |
| overall MAE, both sides replaced by their per-game average | 171.9157 |
| MAE on the 10,544 games whose players differ by 300 or more | 276.7769 |

The correlation of 0.00092 is the sharpest statement of the problem: the tiny
difference the model does express carries no information about the real one.

Two causes, both verified. The network is shared up to a final
`Linear(fc1_h, 2)` whose two rows have cosine similarity 0.999867, and Lichess
pairs close opponents, so the median real gap is 34 points and one shared
estimate costs a mean-absolute objective almost nothing. The adviser asked for
the sides to be separate and named the closely matched training data as the
challenge.

## The two arms

Both are trained from scratch on the full 2,550,000-game corpus in the adopted
configuration: attention on (Bahdanau, `attention_dim` 64), learning rate 3e-4,
batch size 32, weight decay 1e-5, dropout 0.5, patience 5, 60 epochs, split seed
42, training seed 0, and the committed split manifest. Each arm changes exactly
one thing. Neither scheme is tuned: the values below are fixed here and are not
revisited after seeing any result.

### Arm A, mismatch weighting (`mirror_arm_a_gapweight`)

Per-game loss weight, on the raw rating scale:

```
g_i = |white_elo_i - black_elo_i|
w_i = min(1 + g_i / 100, 10)
L   = sum_i(w_i * l_i) / sum_i(w_i),  where l_i = mean over the two sides of |pred - target| in rating points
```

Justification. The weight is linear and monotone in the gap, which is the
simplest scheme that does what is asked. The scale of 100 is set so that the
300-point threshold already used as the wide-gap cut in the mirroring analysis
receives weight 4, that is, four times an evenly matched game; the median game
(gap 34) keeps weight 1.34, so games at or beyond the threshold count about
three times as much as a typical one. The cap of 10 bounds the rare extremes:
the widest real gap in the test partition is 1,837 points, which would otherwise
carry over 19 times the weight of an even game and let a few hundred games
dominate the gradient. Dividing by the summed weight rather than by the count
makes this a weighted **mean**, so the loss stays in rating points and on the
same scale as the unweighted arm and the learning rate carries over unchanged.

### Arm B, difference-aware head (`mirror_arm_b_diffhead`)

Architecture. White and Black each get their own rating head: their own
`Linear(lstm_output_dim, fc1_h)` followed by their own `Linear(fc1_h, 1)`,
concatenated back to the baseline's `(batch, seq, 2)` output.

The branch starts at `fc1`, not only at the last layer, and that is deliberate.
Splitting `Linear(fc1_h, 2)` into two `Linear(fc1_h, 1)` layers would be an
exact reparameterization: the two rows of `fc2` are already independent
parameters, so that change alone cannot make the sides differ. The mirroring is
a property of the objective and of the shared features, not of head capacity.
Giving each side its own hidden projection is the smallest change that actually
adds side-specific capacity.

Objective, on the raw rating scale:

```
L = L_side + lambda * mean_i |(pred_white_i - pred_black_i) - (true_white_i - true_black_i)|
```

with `L_side` the usual mean absolute per-side error and **lambda = 0.5**, fixed.

Justification for lambda. `L_side` averages over both outputs of every game, so
each output receives gradient of magnitude 1/(2B); the difference term averages
over games, so each output receives `lambda/B`. At lambda = 0.5 those are
exactly equal, which is the one value at which neither objective dominates the
other, and it needs no tuning to arrive at. This is checked by autograd in
`tests/test_side_separation.py::test_lambda_one_half_gives_the_two_objectives_equal_per_output_gradient`
in the prototype repository. The risk that equal weight costs overall accuracy
is real, and criterion (a) below is what tests it.

### What stays untouched in both arms

Validation and the best-checkpoint choice keep the plain unweighted per-side
mean absolute error, the same quantity that selected the checkpoint behind
171.9168, and the learning-rate scheduler still steps on it. Only the training
loss changes. Evaluation is unchanged in every respect.

## Pass mark

Scored on the frozen 255,000-game held-out **test** partition, from each arm's
**best-validation** checkpoint. An arm passes only if **all three** hold.

**(a) Overall accuracy is not sacrificed.** Paired bootstrap of the per-game
mean absolute error against the served arm's `attn_tuned__best.csv`
(MAE 171.9168), 10,000 resamples, seed 12345, via
`analysis/scripts/paired_bootstrap_heldout.py` with `A` the new arm and `B`
`attn_tuned`. Let the 95 percent CI of the mean delta be `[lo, hi]`, negative
meaning the new arm is better.

> **PASS(a) iff `lo <= 0`**, that is, the CI either contains zero or lies
> entirely below it. Failing means the CI lies entirely above zero, so the arm
> is worse by more than test-set resampling noise.

**(b) The wide-gap games actually improve.** MAE restricted to the 10,544 games
whose players differ by 300 or more rating points, field `mae_gap_ge_300` from
`analysis/scripts/side_mirroring_check.py` (not the side-averaged variant).

> **PASS(b) iff `mae_gap_ge_300 <= 266.78`**, a drop of at least **10 rating
> points** from the served arm's 276.7769.

**(c) The two estimates actually separate.** Both of the following, from the
same script:

> **PASS(c) iff `pred_gap_median >= 10.0` rating points AND
> `signed_gap_pearson_r >= 0.30`.**

Both are required because either alone is gameable. A head that emits
meaningless per-side jitter would raise the median without carrying information,
and a correlation can be positive while the magnitude stays negligible. The
median threshold of 10 is more than an order of magnitude above the current
0.7126 and above the current **maximum** of 4.4336, so the present behavior
cannot reach it by chance; it is also about 30 percent of the real median gap of
34, so the model must express a substantial share of the typical difference. The
correlation threshold of 0.30 stands against a current 0.00092.

**If both arms pass:** adopt the one with the lower overall test MAE. **If both
fail:** the mirroring stands as a reported limitation, unchanged, and the
manuscript's existing Chapter 1, 4 and 5 text on it stays as it is.

**Cutoff.** Any result produced after the evening of **2026-10-02** (Philippine
time) is not used in the thesis, whatever it says.

**Scope of the bootstrap.** It covers test-set resampling noise only, not
seed-to-seed training variance, which ran about 1 rating point in the 170k
seed-controlled rerun. A near miss on (a) is within training variance, and the
results note should say so rather than treat the boundary as exact.

**If a run is unfinished at evaluation time** (no `FINISHED` marker and no
`Training duration (min):` line in its log): do not score a partial run against
this pass mark. Record the epoch reached, and report that arm as INCOMPLETE. An
arm that reached at least 40 of its 60 epochs may additionally be scored from
its best-validation checkpoint and reported as an **indicative, not
pre-registered** number, clearly labeled as such.

## Provenance

- Prototype code: this repository at commit
  `2ae26a5577c24535b3a37c624224fd311bfa8862`. Both options are off by default.
  The study repository carries this same note and consumes this repository as
  its `prototype` submodule.
- Deployed trainer on the HPC, `~/thesis2/src/chess_rating_net.py`,
  sha256 `cda3b08835f811b3e81940d1a2be264d98e49fd3e1637312302ab6d32b0c96f1`.
  The pre-change file is kept beside it as `chess_rating_net.py.bak_20260928`.
- Corpus store `/tmp/ratingnet_store/corpus.sqlite`, 2,550,000 games, restored
  from `data/corpus_store_backup.sqlite`; split manifest placed from the
  committed `data/split_manifest_2p55M_seed42.json`, sha256
  `dee7d11779df12bfc1242445079f9903b1c48b0787103db63f7b410bd9ed2b3b`,
  1,836,000 / 459,000 / 255,000.
- `analysis/scripts/score_test_split.py` now passes `separate_heads` through
  from `ckpt["params"]`, defaulting to False, so it can rebuild Arm B's head
  under its `strict=True` load. Checkpoints written before this experiment have
  no such key and score exactly as they did.
- `analysis/scripts/side_mirroring_check.py` gained `pred_gap_sd`, `real_gap_sd`
  and `signed_gap_pearson_r`. Every previously reported field is unchanged.

## Where everything lives

All paths are relative to `~/thesis2` on
`<user>@<hpc-host>` (key `~/.ssh/<key>`).

| | Arm A | Arm B |
| --- | --- | --- |
| experiment name | `mirror_arm_a_gapweight` | `mirror_arm_b_diffhead` |
| GPU | 2 | 3 |
| tmux session | `mirror_a` | `mirror_b` |
| log | `logs/mirror_arm_a_gapweight.log` | `logs/mirror_arm_b_diffhead.log` |
| checkpoints | `models/mirror_arm_a_gapweight/` | `models/mirror_arm_b_diffhead/` |
| per-epoch checkpoint | `models/<exp>/model_<N>.pth`, plus `latest.pth` | same |
| checkpoint to score | `models/<exp>/best_model.pth` | same |
| completion marker | `models/mirror_arm_a_gapweight/FINISHED` | `models/mirror_arm_b_diffhead/FINISHED` |
| completion log line | `Training duration (min):` | same |
| eval label | `mirror_arm_a` | `mirror_arm_b` |
| eval outputs | `analysis/heldout_test_eval/mirror_arm_a__best.{csv,json}` | `..._b__best.{csv,json}` |

GPUs 0 and 1 carry other people's long-running services and are not touched.

## Unattended operation

The runs have to survive days with nobody watching, including a reboot.

- **Every epoch checkpoints.** The trainer writes `model_<N>.pth`, `latest.pth`
  and, on an improvement, `best_model.pth` after every epoch. This is existing
  behavior, not new.
- **Auto-resume.** The existing watchdog `hpc_training_heal.sh` runs from cron
  every 10 minutes (`*/10 * * * * /bin/bash
  ~/thesis2/hpc_training_heal.sh`, verified installed).
  Both arms were added to its `RUNS` list with their exact flags and GPU, in the
  HPC copy, this repository's copy and the prototype repository's sanitized
  mirror, all three kept in sync. It holds a single-instance lock, rebuilds
  `/tmp/ratingnet_store` from `data/corpus_store_backup.sqlite` when a reboot
  has wiped it, relaunches any listed experiment that is not running and lacks
  the completion line, resuming from `models/<exp>/latest.pth`, and logs to
  `logs/training-heal.log`. The mechanism is unchanged; only the `RUNS` entries
  were added.
- **A reboot cannot change the split.** `ensure_store` restores the corpus but
  not the manifest, so a post-reboot run regenerates it. That regeneration was
  checked against the committed manifest before launch and reproduces it exactly
  (same sha256, same three file lists), so the arms stay on the same partition
  as the 171.9168 reference.
- **A dropped flag cannot pass silently.** A resume rebuilds the model and the
  objective from the command line, not from the checkpoint, so a launcher
  missing `--gap_weighting` or `--diff_loss_weight 0.5` would have kept training
  a different experiment under the same name. This project has already lost one
  run to that class of bug. The trainer now refuses to resume when
  `gap_weighting`, `separate_heads`, `diff_loss_weight`, `use_attention`,
  `deeper_cnn` or `dense_supervision` disagree with the checkpoint.
- **Completion is unambiguous.** On finishing, the trainer writes
  `models/<exp>/FINISHED` (a temp file plus rename, so it is never seen
  half-written) carrying the finish time, epochs completed, best epoch, best
  validation loss and the resolved settings, and still prints
  `Training duration (min):` as its final log lines.

## The evaluation, as one sequence

A fresh worker runs exactly this, from these notes alone. The `analysis/scripts/`
paths are the study repository's copies, which are the ones deployed on the
cluster; this repository holds the trainer and the launch and healing scripts.

```bash
ssh -i ~/.ssh/<key> <user>@<hpc-host>
cd ~/thesis2

# 1. Both arms must be finished. Do not score a partial run (see the rule above).
cat models/mirror_arm_a_gapweight/FINISHED models/mirror_arm_b_diffhead/FINISHED
grep -H "^Training duration (min):" logs/mirror_arm_*.log

# 2. Pick a GPU that is genuinely free. This machine is shared; never kill anything.
nvidia-smi

# 3. The corpus store lives on /tmp and a reboot wipes it. Restore if needed.
[ -f /tmp/ratingnet_store/corpus.sqlite ] || { mkdir -p /tmp/ratingnet_store; \
  cp data/corpus_store_backup.sqlite /tmp/ratingnet_store/corpus.sqlite.tmp && \
  mv /tmp/ratingnet_store/corpus.sqlite.tmp /tmp/ratingnet_store/corpus.sqlite; }
cp -n data/split_manifest_2p55M_seed42.json /tmp/ratingnet_store/split_manifest.json

# 4. Score both arms on the held-out test partition. Same script, same manifest,
#    same settings as analysis/heldout_test_eval. Resumable: it skips any
#    (arm, checkpoint) whose .json already exists, so the six old arms are not
#    rescored. Takes a few minutes per checkpoint. Use tmux.
CUDA_VISIBLE_DEVICES=<free gpu> bash analysis/scripts/run_heldout_test_eval.sh

# 5. Paired bootstrap against the served arm, for criterion (a).
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ratingnet2
python analysis/scripts/paired_bootstrap_heldout.py \
  --pair mirror_arm_a_vs_attn_tuned \
     A=analysis/heldout_test_eval/mirror_arm_a__best.csv \
     B=analysis/heldout_test_eval/attn_tuned__best.csv \
  --pair mirror_arm_b_vs_attn_tuned \
     A=analysis/heldout_test_eval/mirror_arm_b__best.csv \
     B=analysis/heldout_test_eval/attn_tuned__best.csv \
  --out_json analysis/heldout_test_eval/mirror_bootstrap.json

# 6. Separation and wide-gap MAE, for criteria (b) and (c). attn_tuned is
#    included so the comparison numbers are recomputed, not copied.
python3 analysis/scripts/side_mirroring_check.py analysis/heldout_test_eval \
  mirror_arm_a mirror_arm_b attn_tuned \
  | tee analysis/heldout_test_eval/mirror_side_check.txt
```

Then, locally: pull `mirror_bootstrap.json`, `mirror_side_check.txt` and the two
`__best.json` files, and write `analysis/mirror-retrain-results.md` with, per
arm, the three criteria and a PASS or FAIL against the marks fixed above. The
note goes in both this repository and the study repository. The `__best.csv` dumps are about 20 MB each and are gitignored, like the
existing ones.

Report the numbers plainly, including a failure. A failed arm is a real result:
it says the mirroring is not cheap to remove, which is worth more to the
manuscript than silence.
