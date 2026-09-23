#!/bin/bash
# Stage 2 (CPU, multi-day) of the anomaly-corpus-v2 detector feature extraction.
# Stockfish 16 at fixed depth over substituted + matched-clean plies of all 134
# v2 cells.  Run from ~/thesis2 on the HPC, inside a tmux window.
#   tmux new -s v2_engine
#   SF_WORKERS=10 bash analysis/scripts/run_v2_engine.sh
# Resumable: re-running skips cells whose shard already exists.
# Poll analysis/v2_features/.engine.DONE for completion; engine.npz is
# re-merged from whatever shards exist at the end of every run.
set -e
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ratingnet2
cd ~/thesis2

OUT=analysis/v2_features
SF_WORKERS=${SF_WORKERS:-10}          # leave cores for the training arm + OS (32-core box)
SF_DEPTH=${SF_DEPTH:-20}              # design range 18-20
SF_TIME_CAP=${SF_TIME_CAP:-5.0}      # wall-clock safety cap per position
CLEAN_PLIES=${CLEAN_PLIES:-10}
mkdir -p "$OUT" logs
rm -f "$OUT/.engine.DONE" "$OUT/.engine.FAILED"
echo "START $(date -Is)  workers=$SF_WORKERS depth=$SF_DEPTH cap=${SF_TIME_CAP}s" >> logs/v2_engine.log

if python analysis/scripts/extract_detector_features_v2.py \
      --stage engine \
      --corpus_dir data/anomaly_corpus_v2 \
      --out_dir "$OUT" \
      --sf_bin ~/thesis2/engines/stockfish/stockfish-ubuntu-x86-64-avx2 \
      --sf_depth "$SF_DEPTH" \
      --sf_time_cap_s "$SF_TIME_CAP" \
      --sf_workers "$SF_WORKERS" \
      --clean_plies_per_game "$CLEAN_PLIES" \
      --resume \
      >> logs/v2_engine.log 2>&1 ; then
  echo "DONE $(date -Is)" >> logs/v2_engine.log
  touch "$OUT/.engine.DONE"
else
  echo "FAILED $(date -Is)" >> logs/v2_engine.log
  touch "$OUT/.engine.FAILED"
  exit 1
fi
