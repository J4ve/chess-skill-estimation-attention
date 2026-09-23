# Anomaly detection extra arms: HPC runbook

How to find, check, resume and aggregate the runs pre-registered in
`analysis/anomaly-extra-arms-plan.md`. The runs are unattended: every arm writes its own results
and markers, and the aggregator builds the results table from whatever has finished.

## Access

```
ssh -i ~/.ssh/<key> <user>@<hpc-host>
cd ~/<workdir>
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ratingnet2
```

Scripts on the HPC live next to the existing detector scripts in `~/<workdir>/analysis/scripts/`
(copied from this branch's `analysis/scripts/`; `train_anomaly_detector.py` there is byte-identical
to this branch's copy, sha256 `ab5358df...`).

## Sessions, GPUs, outputs

Launched 2026-09-15 13:40 +08:00 (HPC clock) by `analysis/scripts/extra_arms_launch.sh`.

| tmux session | Steps, in order | Device | Output dirs (under `analysis/extra_arms/`) | Log (under `logs/`) | ETA |
|---|---|---|---|---|---|
| `xa_a0` | A0 (writes the gate), then A0g | CPU (4 threads) | `A0/`, `A0g/`, `cache/pooled_features.npz` | `extra_arms_A0.log`, `extra_arms_A0g.log` | A0 about 25 min (gate by about 14:05), A0g about 15 min after |
| `xa_a1` | A1 (waits for the gate) | CPU | `A1/` | `extra_arms_A1.log` | about 20 to 30 min after the gate |
| `xa_a2` | A2, then A2g (wait for the gate) | CPU (4 threads) | `A2/`, `A2g/` | `extra_arms_A2.log`, `extra_arms_A2g.log` | about 20 min each after the gate |
| `xa_a3` | A3 seed 0 (4-config val grid), A3 seed 1, then A3g seed 0 | GPU 2 | `A3_seed0/`, `A3_seed1/`, `A3g_seed0/` | `extra_arms_A3_seed0.log` etc. | about 3 to 4 h after the gate (grid epochs about 10 to 20 s on the full data) |
| `xa_a4` | A4 board-embedding cache, then A4 training (waits for gate and cache) | GPU 3 (+6 CPU readers) | `A4_cache/` (about 10 GB), `A4/` | `extra_arms_A4_cache.log`, `extra_arms_A4.log` | cache about 2 h (smoke read rate 37 games/s), training about 1 h after; done by about 17:30 if nothing stalls |

Each arm directory holds `STARTED`, then either `DONE` or `FAILED` (traceback inside), plus
`results.json`, `scores.npz`, and for A3/A4 `tuning_log.json` / `train_log.json` and model weights.
The A4 cache writes `A4_cache/CACHE_DONE` when complete.

**Gate:** `A0/A0_GATE_PASS` or `A0/A0_GATE_FAIL`. Every downstream arm polls for it once a minute;
on `A0_GATE_FAIL` (or A0 `FAILED`) they exit and write `FAILED` without training. A failed gate
means the anchor did not reproduce 0.807 / 0.685 within 0.005 and needs a decision before
anything else is run.

## Checking progress (read-only)

```
cd ~/<workdir>
tmux ls | grep '^xa_'
ls analysis/extra_arms/*/{DONE,FAILED} analysis/extra_arms/A0/A0_GATE_* analysis/extra_arms/A4_cache/CACHE_DONE 2>/dev/null
tail -5 logs/extra_arms_*.log
nvidia-smi
```

## Aggregating results

Runnable at any time, by anyone; arms that have not finished are listed as such:

```
cd ~/<workdir>
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ratingnet2
python analysis/scripts/aggregate_extra_arms.py --root analysis/extra_arms \
    --out_json analysis/extra_arms/anomaly-extra-arms-results.json \
    --out_md analysis/extra_arms/anomaly-extra-arms-results.md
```

Then copy both files into the repo as `analysis/anomaly-extra-arms-results.json` and
`analysis/anomaly-extra-arms-results.md`:

```
scp -i ~/.ssh/<key> \
  '<user>@<hpc-host>:<workdir>/analysis/extra_arms/anomaly-extra-arms-results.*' analysis/
```

The aggregator's takeaway bullets are mechanical; rewrite that section in plain language after
reading the tables, and add any deviations from the plan (section 9 of the plan) with reasons.
Report every arm, including failed or unfinished ones.

## Resuming a failed or killed arm

A fix that changes no pre-registered setting (a crash, a path, out of memory) may re-run an arm;
log it in the plan's section 9 and in the results file. Never re-run an arm with new settings after
its withheld-condition results exist.

1. Read `analysis/extra_arms/<arm>/FAILED` and the arm's log.
2. Re-run just that step in a fresh tmux session (the run wrapper clears the old markers):

```
cd ~/<workdir>
S=analysis/scripts; R="bash $S/extra_arms_run.sh"
tmux new -d -s xa_redo_A0   "$R A0 $S/extra_arm_a0.py --split v2"
tmux new -d -s xa_redo_A0g  "$R A0g $S/extra_arm_a0.py --split grouped"
tmux new -d -s xa_redo_A1   "$R A1 $S/extra_arm_a1.py"
tmux new -d -s xa_redo_A2   "$R A2 $S/extra_arm_a2.py --split v2"
tmux new -d -s xa_redo_A2g  "$R A2g $S/extra_arm_a2.py --split grouped"
tmux new -d -s xa_redo_A3   "export CUDA_VISIBLE_DEVICES=<idle gpu>; $R A3_seed0 $S/extra_arm_a3.py --seed 0"
tmux new -d -s xa_redo_A3s1 "export CUDA_VISIBLE_DEVICES=<idle gpu>; $R A3_seed1 $S/extra_arm_a3.py --seed 1"   # needs A3_seed0 DONE
tmux new -d -s xa_redo_A3g  "export CUDA_VISIBLE_DEVICES=<idle gpu>; $R A3g_seed0 $S/extra_arm_a3.py --seed 0 --split grouped"
tmux new -d -s xa_redo_A4c  "export CUDA_VISIBLE_DEVICES=<idle gpu>; $R A4_cache $S/extra_arm_a4.py --stage cache --workers 6"
tmux new -d -s xa_redo_A4   "export CUDA_VISIBLE_DEVICES=<idle gpu>; $R A4 $S/extra_arm_a4.py --stage train"
```

Notes:

- Pick GPUs with `nvidia-smi` first; use only idle ones and never touch other users' processes.
- The A4 cache resumes at cell granularity: finished `A4_cache/<cell>.npz` shards are skipped.
- A0's pooled-feature cache (`analysis/extra_arms/cache/pooled_features.npz`) is reused by A0g and
  A2; delete it only if `rating_cheap.npz` changes.
- A3 seed 0 retrains its whole validation grid on a re-run (same seeds, same selection rule).
- HPC `/tmp` is wiped on reboot; all outputs here are on `/home`, so a reboot only kills running
  steps. Re-run those steps as above.

## Smoke tests

Every step was smoke-tested on about 3 percent of games per cell (`--smoke`, outputs under
`analysis/extra_arms_smoke/`, logs `logs/extra_arms_*_smoke.log`) before launch, all exit 0,
including the aggregator. Smoke metrics are not reported or used for any decision (plan section 7,
rule 4; see the plan's section 9 note on the structure check).
