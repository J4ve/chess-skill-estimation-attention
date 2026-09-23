#!/bin/bash
# Launch every extra anomaly arm in its own HPC tmux session. Downstream arms
# wait for analysis/extra_arms/A0/A0_GATE_PASS and exit FAILED on A0_GATE_FAIL.
#   GPU_A3=2 GPU_A4=3 bash analysis/scripts/extra_arms_launch.sh
# Check nvidia-smi first and pick idle GPUs; never touch other users' processes.
set -e
cd ~/thesis2
S=analysis/scripts
R="bash $S/extra_arms_run.sh"
: "${GPU_A3:?set GPU_A3}"; : "${GPU_A4:?set GPU_A4}"
tmux new -d -s xa_a0   "$R A0 $S/extra_arm_a0.py --split v2; $R A0g $S/extra_arm_a0.py --split grouped"
tmux new -d -s xa_a1   "$R A1 $S/extra_arm_a1.py"
tmux new -d -s xa_a2   "$R A2 $S/extra_arm_a2.py --split v2; $R A2g $S/extra_arm_a2.py --split grouped"
tmux new -d -s xa_a3   "export CUDA_VISIBLE_DEVICES=$GPU_A3; $R A3_seed0 $S/extra_arm_a3.py --seed 0 && $R A3_seed1 $S/extra_arm_a3.py --seed 1; $R A3g_seed0 $S/extra_arm_a3.py --seed 0 --split grouped"
tmux new -d -s xa_a4   "export CUDA_VISIBLE_DEVICES=$GPU_A4; $R A4_cache $S/extra_arm_a4.py --stage cache --workers 6 && $R A4 $S/extra_arm_a4.py --stage train"
tmux ls | grep '^xa_'
