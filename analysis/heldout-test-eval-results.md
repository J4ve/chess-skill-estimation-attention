# Held-out test eval results (Chapter 4 headline numbers)

Run 2026-09-04 on the HPC, frozen split (`split_seed 42`, manifest sha256
`dee7d117...`), 255,000-game held-out TEST partition, scored once per arm on
its best-validation-epoch checkpoint and its epoch-60 checkpoint.
Raw per-arm output: `analysis/heldout_test_eval/*.json` (+ per-game CSVs on
HPC, not committed, too large). Aggregate: `analysis/heldout-test-eval-results.json`.

Architecture freeze (firstmate, 2026-09-04): the reported architecture is the
**tuned attention arm** (4-layer CNN + Bahdanau attention, attention_dim 64,
lr 3e-4, dropout 0.5). The deeper-CNN arm (arm5) finished at val 172.37,
tying the tuned arm (172.38), so it is not promoted and stays an internal
diagnostic.

## Test MAE by arm (rating points)

| arm | val MAE | test MAE, best-val ckpt | test MAE, epoch-60 ckpt |
|---|---|---|---|
| reproduced baseline (arm1) | 175.56 | **175.00** | 176.59 |
| + attention, untuned (arm2, lr 1e-4) | 173.59 | 173.12 | 173.89 |
| **+ attention, tuned (lr 3e-4) [FROZEN ARCH]** | 172.38 | **171.92** | 172.05 |
| deeper CNN (arm5, not promoted) | 172.37 | 172.05 | 172.92 |
| low-dropout diagnostic (internal) | 174.48 | 173.95 | 175.67 |
| baseline + lr 3e-4 (arm6, confound control) | n/a | **173.55** | n/a |
| Omori's released checkpoint (`model_55.pth`, zero-shot on our data) | n/a | **193.47** | n/a |

The best-val-checkpoint column is the number directly comparable to the 182
test MAE reported by Omori and Tadepalli (2024): our reproduced baseline
scores 175.00, consistent with and below 182.

## The attention effect on TEST, confirmed

175.00 (baseline) to 171.92 (tuned attention) is a **3.08 MAE improvement on
the held-out test set**, larger than the ~2.0 MAE gap seen on validation.
Paired-bootstrap (10,000 resamples, `analysis/scripts/paired_bootstrap_heldout.py`)
confirms this is significant: 95% CI [-3.41, -2.76], does not cross zero.
Full results, all pairs: `analysis/heldout_test_eval/bootstrap.json`.

## Omori's actual released weights, scored on our data (closes a gap left open since 2026-08-25)

Ran `models/model_55.pth` (Omori's own released checkpoint, unmodified,
zero-shot - no retraining) through `score_test_split.py` on the identical
frozen 255,000-game test split used for every arm above (manifest hash
`dee7d117...`, verified matching). This is the check planned on 2026-08-30
("run the held-out test once... on the two defended arms plus Omori's
released `model_55.pth` as a zero-shot reference") but never executed during
the 2026-09-04 run - closed 2026-09-12.

**Result: 193.47 MAE.** Worse than our reproduced baseline (175.00), worse
than our tuned-attention arm (171.92), and worse than Omori's own published
182. Paired bootstrap confirms both gaps are real, not noise:
- vs our baseline: delta +18.47, 95% CI [18.02, 18.92], does not cross zero.
- vs our tuned-attention arm: delta +21.55, 95% CI [21.09, 22.01], does not cross zero.

**Reading this honestly.** This is not evidence his architecture is worse -
it is the same architecture (verified via state_dict comparison, 2026-08-30:
identical conv1-conv4 trunk, no conv5, no attention on either side). Both his
checkpoint and our corpus de-standardize on the same constants (mean 1514,
std 366, kept for compatibility per `AGENTS.md`), so normalization is not the
mechanism - both sides use identical constants. The explanation is simpler:
his checkpoint was fit to his own original training sample, roughly 1.2M
games from a narrower 2021-2024 window, and never saw the ~2.55M
independently-sampled games in our corpus (different specific games, broader
2019-2026 date range). A model scores worse zero-shot on data it never
trained on than a model retrained end-to-end on that exact data - which is
exactly what our reproduced baseline and attention arms are. This is a
distribution-shift result, not an architecture comparison, and should be
reported as such.

A smaller-scale version of this same check (170k games, 2026-08-25) found
his checkpoint scoring 198.97 - reasonably consistent with today's full-scale
193.47, both well above our from-scratch reproductions at their respective
scales.

## Epoch-60 vs best-val checkpoint

Every arm's epoch-60 checkpoint scores worse than its best-val checkpoint on
test (mild post-best-epoch overfitting, consistent across arms). Report the
best-val checkpoint as the headline number; epoch-60 is a secondary
robustness figure.

## Reproduction fidelity

Baseline test MAE (175.00) sits close to its validation MAE (175.56): no
train/val/test leakage surprise, the reproduction is faithful.

## Confound-control arm (Benitez Q2) - answered

`fullcorpus_baseline_lr3e4_arm6` isolates the learning-rate change from the
attention effect (lr 3e-4 was previously only ever run with attention).
Finished training, scored on TEST: **173.55 MAE**. Bootstrap:
- `baseline_lr3e4_vs_baseline`: delta -1.45, 95% CI [-1.78, -1.13], does not
  cross zero. The lr bump alone, without attention, is a real but smaller
  improvement over the plain baseline.
- `tuned_attention_vs_baseline_lr3e4`: delta -1.63, 95% CI [-1.94, -1.32],
  does not cross zero. **Attention still helps even after isolating the
  learning-rate change** - this directly answers Benitez's Q2: the effect
  reported above is not just the lr change wearing an attention costume.

## Status

- [x] All 6 arms scored on TEST (best-val checkpoints; arm6 and Omori's
      checkpoint scored best-val only, not epoch-60)
- [x] Paired bootstrap CI - all 7 pairs done, see
      `analysis/heldout_test_eval/bootstrap.json`
- [x] arm6 (confound control) - finished, scored, bootstrapped (Benitez Q2 answered)
- [x] Omori's released checkpoint zero-shot on our test set - closed 2026-09-12,
      see section above

## Reproduce

```
ssh -i ~/.ssh/<key> <user>@<hpc-host>
cd ~/<workdir> && conda activate ratingnet2
bash analysis/scripts/run_heldout_test_eval.sh   # re-scores all arms present under models/
python analysis/scripts/paired_bootstrap_heldout.py --pair NAME A=csv B=csv ... --out_json analysis/heldout_test_eval/bootstrap.json
python analysis/scripts/aggregate_heldout_results.py --eval_dir analysis/heldout_test_eval --bootstrap analysis/heldout_test_eval/bootstrap.json --out_json analysis/heldout-test-eval-results.json
```
