# 170k attention-ablation results — raw evidence

Faithful copies of the small, human-readable artifacts from the 170k
baseline-vs-attention ablation, pulled off the HPC on 2026-08-14. These are the
raw evidence behind the headline numbers already committed to
`chapter3-4-process.md` §4.3 ("Attention ablation (170k) — 2026-08-14").

**HPC source root:** `~/Bacsain/thesis2/` on `<hpc-host>` (user
`<user>`, conda env `ratingnet2`). All files below were copied verbatim
(`scp`) and are read-only on the HPC.

## What is here

| File | Run | Contents | HPC source |
|------|-----|----------|------------|
| `logs/abl_baseline_170k.log` | baseline | Per-epoch train/val loss for all 60 epochs, best-val summary | `logs/abl_baseline_170k.log` |
| `logs/abl_attention_170k.log` | attention | Per-epoch train/val loss, epochs 1–58 (run interrupted by node reboot at epoch 58) | `logs/abl_attention_170k.log` |
| `logs/abl_attention_170k_resume.log` | attention | Resume tail, epochs 59–60 | `logs/abl_attention_170k_resume.log` |
| `logs/abl_baseline_eval.log` | baseline | Best-val + terminal test MAE and per-time-control breakdown | `logs/abl_baseline_eval.log` |
| `logs/abl_attention_best_eval.log` | attention | Best-val test MAE and per-time-control breakdown | `logs/abl_attention_best_eval.log` |
| `logs/abl_attention_terminal_eval.log` | attention | Terminal (e60) test MAE and per-time-control breakdown | `logs/abl_attention_terminal_eval.log` |
| `run_baseline_repro.sh` | (reference) | Reference config script (the 50k sanity-gate baseline). Its flags mirror the 170k runs; the exact 170k commands are below | `run_baseline_repro.sh` |

"Loss" throughout is the model's MAE objective (L1 loss in Elo rating points);
there is no separate loss/MAE distinction in these runs.

Note: these `.log` files are committed intentionally as raw evidence. The repo's
`.gitignore` has a global `*.log` rule (aimed at transient build/training logs),
so they were force-added (`git add -f`). They are tracked files, not transient
output.

## Training config (no yaml on the HPC — CLI flags only)

Both runs used identical control settings (per §4.3): epochs 60, patience 5,
Adam lr 1e-4, batch 32, dropout 0.5, BiLSTM 64×3 bidir, conv 32, fc1 32,
weight_decay 1e-5, lr_factor 0.5, `val_batch_size 512`, split 72/18/10
`random_state=42`. They differ only in the attention flags.

Exact commands recovered from HPC `~/.bash_history` (attention train + best eval)
and reconstructed for the baseline/resume/terminal runs:

```bash
# attention (GPU2)
python -u src/chess_rating_net.py --train --data_dir /tmp/ratingnet_data_flat \
  --experiment abl_attention_170k --use_attention --attention_type bahdanau \
  --epochs 60 --lr 1e-4 --batch_size 32 --val_batch_size 512 --num_workers 8 \
  --model_dir models > logs/abl_attention_170k.log 2>&1

# baseline (GPU1) — identical minus the attention flags
python -u src/chess_rating_net.py --train --data_dir /tmp/ratingnet_data_flat \
  --experiment abl_baseline_170k \
  --epochs 60 --lr 1e-4 --batch_size 32 --val_batch_size 512 --num_workers 8 \
  --model_dir models > logs/abl_baseline_170k.log 2>&1

# attention resume (post-reboot, epochs 59–60)
python -u src/chess_rating_net.py --train --data_dir /tmp/ratingnet_data_flat \
  --experiment abl_attention_170k --use_attention --attention_type bahdanau \
  --resume models/abl_attention_170k/latest.pth --epochs 60 --lr 1e-4 \
  --batch_size 32 --val_batch_size 512 --num_workers 8 --model_dir models \
  > logs/abl_attention_170k_resume.log 2>&1

# eval (best-val): cp the checkpoint over model_55.pth, then run inference
cp models/<exp>/best_model.pth models/<exp>/model_55.pth
python -u src/chess_rating_net.py --data_dir /tmp/ratingnet_data_flat \
  --experiment <exp> [--use_attention --attention_type bahdanau] \
  --model_dir models --val_batch_size 512 --num_workers 8

# eval (terminal): same, but cp model_60.pth over model_55.pth first
```

## Mapping back to §4.3

All headline numbers match `chapter3-4-process.md` §4.3 exactly (no
discrepancy):

| §4.3 headline | Value in artifacts | Source |
|---------------|--------------------|--------|
| baseline best-val epoch 33 (val loss 225.9) | `best val loss: 225.90109939575194`, `best val epoch: 32` (0-indexed → epoch 33) | `logs/abl_baseline_170k.log` |
| baseline best-val test MAE **224.2** | `Test Loss: 224.18523451861213` | `logs/abl_baseline_eval.log` |
| baseline terminal (e60) test MAE 229.0 | `Test Loss: 228.9522202435662` | `logs/abl_baseline_eval.log` |
| attention best-val epoch 18 (val loss 223.1) | `Epoch 18, Validation Loss: 223.1487` | `logs/abl_attention_170k.log` |
| attention best-val test MAE **221.7** | `Test Loss: 221.73424530029297` | `logs/abl_attention_best_eval.log` |
| attention terminal (e60) test MAE 228.5 | `Test Loss: 228.462059469784` | `logs/abl_attention_terminal_eval.log` |

Per-time-control test MAE (best-val) — §4.3 vs artifacts (match):

| Time control | baseline §4.3 | baseline log | attention §4.3 | attention log |
|--------------|---------------|--------------|----------------|---------------|
| ultrabullet | 223.1 | 223.1096 | 211.3 | 211.2654 |
| bullet | 246.5 | 246.4892 | 239.5 | 239.5197 |
| blitz | 210.1 | 210.1367 | 210.3 | 210.2572 |
| rapid | 212.3 | 212.3383 | 214.2 | 214.1715 |
| classical | 167.3 | 167.2707 | 176.3 | 176.3159 |

## Quirks to be aware of when reading the raw files

- **`model_55.pth` in the eval logs:** the eval command first `cp`s the desired
  checkpoint (`best_model.pth` or `model_60.pth`) over `model_55.pth`, then runs
  the inference-only path, which hardcodes `model_55.pth` (the frozen-demo
  convention in `src/chess_rating_net.py`). So the "Loading frozen checkpoint
  from … model_55.pth" line is expected; the actual evaluated weights are the
  best-val or terminal checkpoint, as indicated by each log's header.
- **0- vs 1-indexed epochs:** `best val epoch: 32` in the baseline log is
  0-indexed (loop starts at 0) → printed "Epoch 33" (`model_33.pth`), matching
  §4.3's "33".
- **Resume log's `best val epoch: 0`:** the resume restored `best_val_loss`
  (223.1487) but not the epoch counter, so the resume tail prints `0`. The real
  best-val epoch is 18 (from the pre-reboot `abl_attention_170k.log`).
- **No plots were generated** on the HPC (no PNG/PDF/SVG anywhere under
  `~/Bacsain/thesis2/`), so none are included. Per-epoch loss curves are fully
  reconstructible from the text logs. TensorBoard event files
  (`~/Bacsain/thesis2/runs/abl_{baseline,attention}_170k/events.out.tfevents.*`,
  ~9–10 KB each) also exist on the HPC but are binary and redundant with the
  text logs, so they are not committed here.

## Deliberately excluded

- Model checkpoints (`.pth`): ~550 MB per run under
  `~/Bacsain/thesis2/models/abl_{baseline,attention}_170k/`.
- Raw Lichess data (`/tmp/ratingnet_data_flat`, ~35 GB when fully flattened).
- TensorBoard event binaries (see above).
- No secrets or credentials.
