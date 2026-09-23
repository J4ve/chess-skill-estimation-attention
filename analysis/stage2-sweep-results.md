# Stage-2 Hyperparameter Tuning Sweep — Results

**Pre-registration:** `analysis/stage2-tuning-preregistration.md` (committed before any tuning run).
**Executed:** 2026-08-16 on `<user>@<hpc-host>` (4x RTX A5000), conda env `ratingnet2`.

**Status: COMPLETE** — all 17 cells executed and results collected 2026-08-17.

## Fixed across all 17 runs (non-negotiable)

- Data: 170,138 games, `/tmp/ratingnet_data_flat`, split manifest sha256
  `4be0f9e8de8372dcc302f164d271b871841e82b8843c1cefbd4d25b119faacc6`
  (train=122,499 / val=30,625 / test=17,014, `--split_seed 42` **FIXED**).
- Model: ATTENTION (`--use_attention --attention_type bahdanau`), 60 epochs,
  `--val_batch_size 512`, `--num_workers 4`, `--seed 0` (single-seed sweep).
- Control values (held unless swept): batch 32, lr 1e-4, weight decay 1e-5,
  dropout 0.5, patience 5, attention_dim 64.
- Each run varies **only** the one swept flag; last-wins over the control default.

## Success criterion (pre-registered)

A cell is adopted only if its **best-val loss** beats the stage-1 attention mean by
more than **2 × σ_seed**. Stage-1 attention best-val loss = **223.38 ± 0.87**
(sample std, n−1) → bar = 223.38 − 2×0.87 ≈ **221.6** (precisely 221.63).

`best-val epoch` is 0-indexed as logged (value N = the (N+1)-th epoch).

## Results (17 cells, pre-registered order)

| # | experiment | factor | swept value | control? | best-val loss | best-val epoch | terminal epoch-60 loss | clears 2σ bar? |
|---|---|---|---|---|---|---|---|---|
| 1 | `tune_patience5` | patience | 5 | yes | 223.03 | 24 | 230.80 | no |
| 2 | `tune_patience10` | patience | 10 | no | 225.09 | 29 | 230.64 | no |
| 3 | `tune_patience_none` | patience | none (60) | no | 223.34 | 24 | 238.36 | no |
| 4 | `tune_wd1e-5` | weight decay | 1e-5 | yes | 223.52 | 30 | 231.88 | no |
| 5 | `tune_wd1e-3` | weight decay | 1e-3 | no | 223.0199 | 24 | 231.5790 | no |
| 6 | `tune_wd1e-2` | weight decay | 1e-2 | no | 223.7580 | 24 | 228.9050 | no |
| 7 | `tune_lr5e-5` | learning rate | 5e-5 | no | 231.7523 | 32 | 234.6979 | no |
| 8 | `tune_lr1e-4` | learning rate | 1e-4 | yes | 223.3288 | 33 | 229.6658 | no |
| 9 | `tune_lr3e-4` | learning rate | 3e-4 | no | 211.7163 | 25 | 218.4956 | **yes** |
| 10 | `tune_bs32` | batch size | 32 | yes | 223.2054 | 29 | 230.5909 | no |
| 11 | `tune_bs64` | batch size | 64 | no | 226.1886 | 30 | 231.8265 | no |
| 12 | `tune_bs128` | batch size | 128 | no | 228.4962 | 36 | 233.4524 | no |
| 13 | `tune_drop03` | dropout | 0.3 | no | 226.0110 | 24 | 233.2799 | no |
| 14 | `tune_drop05` | dropout | 0.5 | yes | 222.3076 | 27 | 231.6655 | no |
| 15 | `tune_ad32` | attention_dim | 32 | no | 223.6893 | 25 | 231.9632 | no |
| 16 | `tune_ad64` | attention_dim | 64 | yes | 223.7538 | 23 | 232.2389 | no |
| 17 | `tune_ad128` | attention_dim | 128 | no | 224.7478 | 23 | 237.1280 | no |

## Interpretation (filled in at completion)

- The six control-value cells (1, 4, 8, 10, 14, 16) are configurationally identical
  (the stage-1 attention seed-0 config); their spread is a within-batch reproducibility
  check.
- Winner (if any) is re-run at 5 seeds as a **separate** follow-up step (not part of
  this sweep).

## Notes

- `patience = none` is realized as `--patience 60` (≥ max epochs, so the
  `ReduceLROnPlateau` LR scheduler never steps within the 60-epoch budget).
- Per-epoch wall time this run ≈ 5.6 min (vs ~3.8 min in the seed rerun): `--num_workers 4`
  (per the pre-registered constraint) plus concurrent corpus-stream preprocessing on CPU.
