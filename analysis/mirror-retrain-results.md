# White versus Black separation retrain: scored against the pre-registered pass mark

Scored 2026-10-11 on the HPC, against the pass mark fixed in
`analysis/mirror-retrain-preregistration.md` before either arm was launched.
Both arms had finished training on 2026-10-01, well inside the pre-registered
cutoff; see "On the scoring date" below for why scoring itself happened on
2026-10-11, after that cutoff.

## What was run

On `~/thesis2` on `<user>@<hpc-host>`, in one tmux session (06:10 to 06:33
Philippine time, 2026-10-11):

1. The corpus store at `/tmp/ratingnet_store` was missing (a prior reboot had
   wiped it). Rebuilt it from `data/corpus_store_backup.sqlite` and placed
   `data/split_manifest_2p55M_seed42.json` as `split_manifest.json`, exactly
   as the pre-registration describes. This took about 70 seconds this time
   (06:10:33 to 06:11:42); an earlier restore of the same 11 GB file, logged
   by the watchdog on 2026-09-30, took about 7 minutes, so the time this step
   needs varies with what else the HPC's disk cache is holding.
2. Scored each arm's best-validation checkpoint (and, for the same
   epoch-60 record kept alongside every other arm, its epoch-60 checkpoint)
   on the frozen 255,000-game test partition with
   `analysis/scripts/score_test_split.py` on the HPC. The script enforces
   the expected manifest hash itself and raised no error, confirming the
   rebuilt store reproduces the frozen split
   (`manifest_sha256=dee7d11779df12bfc1242445079f9903b1c48b0787103db63f7b410bd9ed2b3b`,
   255,000 test games, matching the pre-registration exactly).
3. Ran `analysis/scripts/paired_bootstrap_heldout.py` (10,000 resamples, seed
   12345) pairing each arm's best-checkpoint CSV against the served arm's
   `attn_tuned__best.csv`, for criterion (a).
4. Ran `analysis/scripts/side_mirroring_check.py` over `mirror_arm_a`,
   `mirror_arm_b` and `attn_tuned`, for criteria (b) and (c).

Nothing above changed any script's logic, any threshold, or any seed. The
`analysis/scripts/` paths above are the study (cs_thesis) repository's
copies, which are the ones deployed on the HPC; this repository (the
prototype) holds the trainer, and its own flat `analysis/score_test_split.py`
predates this experiment's `separate_heads` passthrough, so it cannot load
Arm B's checkpoint as it stands. The full logs, JSON outputs and the
per-game CSVs (gitignored, about 20 MB each) stay on the HPC at
`analysis/heldout_test_eval/`. The small JSON/text outputs are committed
alongside this note, at the same relative path, in this repository only
(not in the study repository, where only this note is committed).

## Criterion (a): overall accuracy not sacrificed

Paired bootstrap of per-game MAE against `attn_tuned__best.csv` (MAE
171.9168), 10,000 resamples, seed 12345. PASS iff the 95% CI's lower bound is
at or below zero (negative means the new arm is better).

| Arm | MAE (arm) | MAE (`attn_tuned`) | mean delta | 95% CI | PASS(a)? |
| --- | --- | --- | --- | --- | --- |
| A, `mirror_arm_a_gapweight` | 171.5080 | 171.9168 | -0.4088 | [-0.7254, -0.0885] | **PASS** (CI entirely below zero) |
| B, `mirror_arm_b_diffhead` | 172.2724 | 171.9168 | +0.3556 | [+0.0433, +0.6694] | **FAIL** (CI entirely above zero) |

**Scope caveat on Arm B's fail.** The pre-registration flags that the
bootstrap covers test-set resampling noise only, not seed-to-seed training
variance, which ran about 1 rating point in an earlier seed-controlled rerun.
Arm B's observed delta, +0.3556, and the distance of its CI's lower bound
from zero, +0.0433, are both smaller than that 1-point scale. This is a near
miss in the sense the pre-registration anticipated: a different training
seed could plausibly have put this arm on the other side of zero. The
pass/fail call stands as written (FAIL, the CI does not cross zero on this
run), but it should not be read as a wide or decisive margin. Arm A's pass
is comfortably assigned by the letter of the rule (the whole CI sits below
zero), but the same scope caveat applies to it symmetrically: an effect
size of 0.4088 MAE points is on the order of that same seed-to-seed noise,
so the result is distinguishable from test-set resampling noise but not
shown to exceed seed-to-seed training variance.

## Criterion (b): wide-gap games improve

MAE restricted to the 10,544 test games whose players differ by 300 or more
rating points (`mae_gap_ge_300`, not the side-averaged variant). PASS iff
this is at or below 266.78 (at least a 10-point drop from the served arm's
276.7769).

| Arm | `mae_gap_ge_300` | drop from 276.7769 | PASS(b)? |
| --- | --- | --- | --- |
| A | 246.9841 | 29.79 | **PASS** |
| B | 247.1801 | 29.60 | **PASS** |

Both arms clear this by a wide margin, about three times the required
10-point drop.

## Criterion (c): the two estimates actually separate

PASS iff `pred_gap_median >= 10.0` rating points AND
`signed_gap_pearson_r >= 0.30`. Both are required.

| Arm | `pred_gap_median` | `signed_gap_pearson_r` | PASS(c)? |
| --- | --- | --- | --- |
| A | 15.6697 | 0.3889 | **PASS** (both conditions clear) |
| B | 8.0144 | 0.4242 | **FAIL** (median below 10.0, despite the correlation clearing 0.30) |

Arm B's correlation is, if anything, slightly higher than Arm A's, but its
median gap falls short of the 10-point bar. This matches the early warning
on the record from the pre-registration's own smoke test: on a 3,000-game
subset, Arm B's difference term rose across training while its side term
fell, "a genuine early hint that Arm B may fail criterion (c)." That hint
held at full scale, specifically on the median-magnitude half of the
criterion, not the correlation half.

## Overall verdict per arm

| Arm | (a) | (b) | (c) | Overall |
| --- | --- | --- | --- | --- |
| A, `mirror_arm_a_gapweight` | PASS | PASS | PASS | **PASS** |
| B, `mirror_arm_b_diffhead` | FAIL | PASS | FAIL | **FAIL** |

Arm A clears all three pre-registered criteria. Arm B clears one of three.

## The adoption rule, and where it does not quite reach

The pre-registration names two cases: if both arms pass, adopt whichever has
the lower overall test MAE; if both fail, the mirroring stands as a reported
limitation, unchanged. This run lands in neither named case: exactly one arm
passes. There is no choice to make between two passing arms, since there is
only one. Taken at face value, the pre-registered criteria were written to
be the full test of whether an arm is good enough to adopt, and Arm A is the
only arm that clears them.

This note does not take any adoption action itself. Nothing in the
manuscript, the served model, or the deployed prototype changes as part of
writing this note. Whether and how the thesis uses this result is the
captain's decision, not this note's.

## Overall test MAE, for context

| Arm | best-val checkpoint | epoch-60 checkpoint |
| --- | --- | --- |
| `attn_tuned` (served) | 171.9168 | 172.05 |
| A, `mirror_arm_a_gapweight` | 171.5080 | 172.4368 |
| B, `mirror_arm_b_diffhead` | 172.2724 | 173.4584 |

As with every other arm on record, the epoch-60 checkpoint scores worse than
the best-validation checkpoint for both new arms. The best-validation
checkpoint is the one the pass mark is scored on and the only one that
matters for the verdict above; epoch-60 is included only for the same
secondary-record consistency kept for every other arm.

## Provenance

- Ratingnet (prototype) repository commit at the time this note was written:
  `6688607c9de05cbd742865cd5cf31ece2b160bd6`. Checked with
  `git merge-base --is-ancestor`: the pre-registration's own pinned commit,
  `2ae26a5577c24535b3a37c624224fd311bfa8862`, is an ancestor of this commit.
  `git diff --stat` between the two touches only `src/static/app.js`,
  `src/static/index.html` and `src/static/styles.css` (the web UI); nothing
  under `src/chess_rating_net.py` or any other training or scoring code
  changed.
- Deployed trainer on the HPC, `~/thesis2/src/chess_rating_net.py`, sha256
  `cda3b08835f811b3e81940d1a2be264d98e49fd3e1637312302ab6d32b0c96f1`,
  matching both the pre-registration's recorded hash and this repository's
  own `src/chess_rating_net.py` at the commit above, byte for byte:
  unmodified since launch.
- Scoring scripts, sha256 as run on the HPC, matching byte for byte the
  copies at cs_thesis (study repository) commit `dfd0a4415d300e6ee0aff23198875cdaa89c99b2`
  (unchanged there since `9c4a3b0`, "Pre-register the White versus Black
  separation retrain"):
  - `analysis/scripts/score_test_split.py`:
    `8d850085ec3b777750e7c2f2e7717ed4a8eb7596bc84dc0067ea2886d212ec35`
  - `analysis/scripts/side_mirroring_check.py`:
    `d58c6617ea37a5d1264f226f1c8cdb1527792045ecc097b906373be6505cc94a`
  - `analysis/scripts/paired_bootstrap_heldout.py`:
    `081bc406b3fc8897e78010aa3e0d1715cda4886ee51fe1c201d0208df7207802`
  - `analysis/scripts/run_heldout_test_eval.sh`:
    `5ca989d50102776c2a444b379ea6ac3ed929c3b395b1ef5e822f16fe2aff7c89`
- Checkpoints scored (best-validation, as the pass mark requires):
  - `models/mirror_arm_a_gapweight/best_model.pth`, best epoch 58, best
    validation loss 172.1311005557261, sha256
    `97ffa6e77d53a47ee75eaa199670d9d795b66b45dfd687e5c441ebcfd77c0481`
  - `models/mirror_arm_b_diffhead/best_model.pth`, best epoch 55, best
    validation loss 172.7244058224137, sha256
    `263c7f6fb3d65f7ea61170aca27f85fc8ec055efccf58988f15e1fb849774dd4`
  - Both best-epoch and best-validation-loss figures match each arm's
    `FINISHED` marker exactly.
- Split manifest sha256: two different hashes are in play, and both match
  what they should.
  - `dee7d11779df12bfc1242445079f9903b1c48b0787103db63f7b410bd9ed2b3b` is
    the `manifest_sha256` the scorer itself computes and checks: a SHA-256
    over the newline-joined, sorted list of all 2,550,000 game filenames in
    the corpus store, not a hash of the manifest file's bytes. This is also
    the value stored in the manifest JSON's own `"sha256"` field. It matches
    the pre-registration's recorded value exactly, and `score_test_split.py`
    raises `SystemExit` on any mismatch, so this was verified by the script
    itself, not merely asserted here.
  - `d026ac0e18749c62753bc87cb3d49501cd36efe706bb32f1475a274652f9c60e` is the
    plain `sha256sum` of the manifest file `data/split_manifest_2p55M_seed42.json`
    on disk (its raw bytes, 81,600,168 of them). This is a different
    quantity from the one above and is included only so a reader who runs
    `sha256sum` on that file is not thrown by seeing a different hash.
  - The split itself: train 1,836,000 / val 459,000 / test 255,000, all
    2,550,000 games, confirming the rebuilt store reproduces the frozen
    split used for the 171.9168 reference and for every other arm in
    `analysis/heldout-test-eval-results.md`.
- Raw outputs, committed alongside this note in this repository:
  `analysis/heldout_test_eval/mirror_arm_a__best.json`,
  `mirror_arm_b__best.json`, `mirror_bootstrap.json`, `mirror_side_check.txt`.
  Two fields in the two `__best.json` files, `src_dir_used` and
  `per_game_csv`, carried the full HPC account path in the raw output;
  those two fields only were edited to the same `~/thesis2`-relative form
  used elsewhere in this note before committing, since this repository is
  public. Every number in both files is untouched. The matching per-game
  CSVs, and the epoch-60 checkpoint's JSON for each arm (a secondary record
  kept for consistency with every other arm, not part of the pass mark),
  stay on the HPC and are not committed here, as with every other arm's
  equivalent files.

## On the scoring date

Both arms finished training on 2026-10-01 (Arm A at 12:47, Arm B at 14:00,
Philippine time), well inside the pre-registered cutoff of the evening of
2026-10-02. Plainly: neither arm was scored at that time, or at any point
before the cutoff. No session ran the evaluation sequence in the gap
between the arms finishing and this one. The scoring that produced every
number in this note ran on 2026-10-11, between 06:10 and 06:33 Philippine
time, nine days after the cutoff.

The pre-registration states its own rule on this exactly: "Any result
produced after the evening of 2026-10-02 (Philippine time) is not used in
the thesis, whatever it says." This note's numbers were produced after that
date. Whether that rule is applied as written, or whether the captain
decides otherwise given that both runs themselves finished on time, is the
captain's decision, not this note's. This note only reports what the
pre-registered pass mark says about the numbers, once they exist.
