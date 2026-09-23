#!/usr/bin/env bash
# Launch one seed-controlled rerun in a named tmux session.
# Usage: launch_run.sh <session> <gpu> <experiment> <seed>
set -euo pipefail
sess="$1"; gpu="$2"; exp="$3"; seed="$4"
cd ~/thesis2
mkdir -p logs models
if tmux has-session -t "$sess" 2>/dev/null; then
  echo "session $sess already exists, skipping"
  exit 0
fi
# Attention arm must match the verified 2026-08-14 abl_attention_170k config
# (checkpoint params: use_attention=True, attention_type=bahdanau, attention_dim=64;
# all other params are trainer defaults already passed below). Without these flags
# the run silently trains the baseline architecture under the attention name.
ATTN=''
LR='1e-4'
case "$exp" in
  abl_attention_*) ATTN='--use_attention --attention_type bahdanau --attention_dim 64' ;;
  conf_lr3e4_*) ATTN='--use_attention --attention_type bahdanau --attention_dim 64'; LR='3e-4' ;;
esac
tmux new -d -s "$sess" "cd ~/thesis2 && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ratingnet2 && R=''; [ -f models/$exp/latest.pth ] && R='--resume models/$exp/latest.pth'; CUDA_VISIBLE_DEVICES=$gpu python -u src/chess_rating_net.py --train --data_dir /tmp/ratingnet_data_flat --experiment $exp --epochs 60 --lr $LR --batch_size 32 --val_batch_size 512 --num_workers 8 --weight_decay 1e-5 --patience 5 --dropout_rate 0.5 --seed $seed --split_seed 42 --model_dir models $ATTN \$R 2>&1 | tee logs/$exp.log; exec bash"
echo "launched $sess (gpu=$gpu exp=$exp seed=$seed)"
