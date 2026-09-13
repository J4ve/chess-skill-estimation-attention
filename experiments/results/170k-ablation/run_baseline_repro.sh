#!/usr/bin/env bash
set -e
cd ~/Bacsain/thesis2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ratingnet2
export CUDA_VISIBLE_DEVICES=1
echo "=== baseline repro start $(date -u) ==="
python -u src/chess_rating_net.py --train \
  --data_dir /tmp/ratingnet_data_flat \
  --experiment cnn_bilstm_clocks_all \
  --epochs 60 --lr 1e-4 --batch_size 32 --val_batch_size 512 \
  --num_workers 8 \
  --model_dir models 2>&1
echo "=== baseline repro end $(date -u) ==="
