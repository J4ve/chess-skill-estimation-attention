#!/bin/bash
# Stage 1 (GPU) of the anomaly-corpus-v2 detector feature extraction.
# Frozen tuned-attention rating-model forward pass over all 134 v2 cells.
# Run from ~/thesis2 on the HPC, inside a tmux window.
#   tmux new -s v2_rating_cheap
#   bash analysis/scripts/run_v2_rating_cheap.sh
# Poll analysis/v2_features/.rating_cheap.DONE for completion.
set -e
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ratingnet2
cd ~/thesis2

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}   # GPU 0 or 2 only, never 1
OUT=analysis/v2_features
mkdir -p "$OUT" logs
rm -f "$OUT/.rating_cheap.DONE" "$OUT/.rating_cheap.FAILED"
echo "START $(date -Is)  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" > logs/v2_rating_cheap.log

if python analysis/scripts/extract_detector_features_v2.py \
      --stage rating_cheap \
      --corpus_dir data/anomaly_corpus_v2 \
      --checkpoint models/preflight_check_2m/best_model.pth \
      --out_dir "$OUT" \
      --resume \
      >> logs/v2_rating_cheap.log 2>&1 ; then
  echo "DONE $(date -Is)" >> logs/v2_rating_cheap.log
  touch "$OUT/.rating_cheap.DONE"
else
  echo "FAILED $(date -Is)" >> logs/v2_rating_cheap.log
  touch "$OUT/.rating_cheap.FAILED"
  exit 1
fi
