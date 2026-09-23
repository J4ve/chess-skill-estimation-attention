#!/usr/bin/env bash
# Launch one stage-2 tuning sweep run in a named tmux session.
# Usage: hpc_sweep_launch_run.sh <session> <gpu> <experiment> [swept-flags...]
#
# Stage-2 control values (held fixed for every cell, per pre-registration):
#   batch 32, lr 1e-4, weight decay 1e-5, dropout 0.5, patience 5, attention_dim 64
#   (attention bahdanau, seed 0, split_seed 42, 60 epochs, val_batch_size 512).
# The single swept flag is passed as extra args (last-wins over the control default).
set -euo pipefail
sess="$1"; gpu="$2"; exp="$3"; shift 3
cd ~/thesis2
mkdir -p logs models
if tmux has-session -t "$sess" 2>/dev/null; then
  echo "session $sess already exists, skipping"
  exit 0
fi
resume=()
if [ -f "models/$exp/latest.pth" ]; then resume=(--resume "models/$exp/latest.pth"); fi
cmd=(python -u src/chess_rating_net.py --train
     --data_dir /tmp/ratingnet_data_flat --experiment "$exp"
     --epochs 60 --lr 1e-4 --batch_size 32 --val_batch_size 512
     --num_workers 4 --seed 0 --split_seed 42 --model_dir models
     --weight_decay 1e-5 --patience 5 --dropout_rate 0.5
     --use_attention --attention_type bahdanau --attention_dim 64
     "$@" "${resume[@]}")
printf -v cmdline '%q ' "${cmd[@]}"
tmux new -d -s "$sess" "cd ~/thesis2 && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ratingnet2 && CUDA_VISIBLE_DEVICES=$gpu $cmdline 2>&1 | tee -a logs/$exp.log; exec bash"
echo "launched $sess (gpu=$gpu exp=$exp extra='$cmdline')"
