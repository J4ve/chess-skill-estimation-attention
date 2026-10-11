#!/bin/bash
# Full-scale run: live rating curves (GPU 2) and Stockfish eval (CPU) over the
# n=3000 sampled TEST games, launched concurrently (they don't contend).
# Run from ~/thesis2/scratch/move_explain inside tmux session
# "fm_holder" (no --experiment in this script's own argv, per AGENTS.md's
# watchdog note -- this is a one-off analysis run, not a training run, but
# the same substring-match hazard applies to any heal script watching this
# session).
set -e
cd ~/thesis2/scratch/move_explain
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ratingnet2

mkdir -p logs
rm -f out/.curves.DONE out/.curves.FAILED out/.engine.DONE out/.engine.FAILED

(
  if python3 scripts/move_explanation_curves.py \
      --src-dir src_snapshot/src \
      --store-dir /tmp/ratingnet_store \
      --games-file out/sample_games_3000.json \
      --served-checkpoint ~/thesis2/models/preflight_check_2m/best_model.pth \
      --mirror-a-checkpoint ~/thesis2/models/mirror_arm_a_gapweight/best_model.pth \
      --device cuda --gpu-index 2 \
      --max-plies 100 \
      --out-jsonl out/curves_3000.jsonl \
      --equivalence-check-n 3 \
      >> logs/curves_3000.log 2>&1 ; then
    touch out/.curves.DONE
  else
    touch out/.curves.FAILED
  fi
) &

(
  if python3 scripts/move_explanation_engine.py \
      --store-dir /tmp/ratingnet_store \
      --games-file out/sample_games_3000.json \
      --sf-bin ~/thesis2/engines/stockfish/stockfish-ubuntu-x86-64-avx2 \
      --sf-depth 17 --sf-time-cap-s 1.0 \
      --workers 10 \
      --max-plies 100 \
      --out-jsonl out/engine_3000.jsonl \
      >> logs/engine_3000.log 2>&1 ; then
    touch out/.engine.DONE
  else
    touch out/.engine.FAILED
  fi
) &

wait
echo "BOTH_FINISHED $(date -Is)" >> logs/run_move_explanation.log
