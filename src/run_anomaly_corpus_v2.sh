#!/usr/bin/env bash
# v2 synthetic anomaly-validation corpus sweep.
# Design: data/anomaly-corpus-v2/design.md.
#
# Main factorial : 9 Maia bands x 2 engines x 7 substitution rates
#                  (0/2/5/10/20/40/60 %), GAMES_PER_CELL games each.
#                  Rate 0 is the clean control and is generated with
#                  --engine none, once per (band, engine arm).
# Hard negatives : bands 1100..1800, strength-mismatch clean cells
#                  ({band}_hardneg), GAMES_PER_CELL games each.
#
# Each cell is resumable via a per-directory .done marker. The SUMMARY line the
# generator prints is captured to <cell>/.cell_summary.json.
#
# GPU: set CUDA_VISIBLE_DEVICES before launch to keep lc0 off the GPU that the
# concurrent deepcnn_arm5 training run is using (GPU 1). The default below
# exposes physical GPUs 0,2,3 only.
set -euo pipefail
cd "$(dirname "$0")"

ENGINES_DIR=${ENGINES_DIR:-$HOME/Bacsain/thesis2/engines}
OPENINGS_DIR=${OPENINGS_DIR:-$HOME/Bacsain/thesis2/data/opening_tables}
OUT_DIR=${OUT_DIR:-$HOME/Bacsain/thesis2/data/anomaly_corpus_v2}
LOG_DIR=${LOG_DIR:-$HOME/Bacsain/thesis2/logs/anomaly_corpus_v2}
GAMES_PER_CELL=${GAMES_PER_CELL:-2000}
WORKERS=${WORKERS:-3}
GLOBAL_SEED=${GLOBAL_SEED:-42}
CLOCK_MODULE_DIR=${CLOCK_MODULE_DIR:-$HOME/Bacsain/thesis2/analysis/scripts}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,2,3}

BANDS=${BANDS:-"1100 1200 1300 1400 1500 1600 1700 1800 1900"}
ENGINES=${ENGINES:-"stockfish16 lc0"}
RATES=${RATES:-"0.0 0.02 0.05 0.10 0.20 0.40 0.60"}
HARD_NEG_BANDS=${HARD_NEG_BANDS:-"1100 1200 1300 1400 1500 1600 1700 1800"}

mkdir -p "$OUT_DIR" "$LOG_DIR"

LC0=$ENGINES_DIR/lc0/build/release/lc0
SF=$ENGINES_DIR/stockfish/stockfish-ubuntu-x86-64-avx2
LC0NET=$ENGINES_DIR/lc0_networks/small.pb.gz
MAIA=$ENGINES_DIR/maia_weights

rate_tag() { python3 -c "import sys;print('r%02d'%round(float(sys.argv[1])*100))" "$1"; }

JOBLIST=$(mktemp)
for BAND in $BANDS; do
  for ENGINE in $ENGINES; do
    for RATE in $RATES; do
      TAG=$(rate_tag "$RATE")
      if [ "$TAG" = "r00" ]; then
        echo "$BAND|none|0.0|${ENGINE}|${BAND}_${ENGINE}_r00|0" >> "$JOBLIST"
      else
        echo "$BAND|$ENGINE|$RATE|${ENGINE}|${BAND}_${ENGINE}_${TAG}|0" >> "$JOBLIST"
      fi
    done
  done
done
for BAND in $HARD_NEG_BANDS; do
  echo "$BAND|none|0.0|hardneg|${BAND}_hardneg|1" >> "$JOBLIST"
done

# de-dup the identical per-engine r00 controls that differ only by dir label?
# no: the design's factorial counts one control per engine arm; keep both.
TOTAL=$(wc -l < "$JOBLIST")
echo "cells=$TOTAL games_per_cell=$GAMES_PER_CELL workers=$WORKERS seed=$GLOBAL_SEED"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  out=$OUT_DIR"

run_cell() {
  local BAND=$1 ENGINE=$2 RATE=$3 LABEL=$4 CELLNAME=$5 HARDNEG=$6
  local CELL_DIR="$OUT_DIR/$CELLNAME"
  local LOG_FILE="$LOG_DIR/$CELLNAME.log"
  if [ -f "$CELL_DIR/.done" ]; then
    echo "SKIP $CELLNAME (done)"; return 0
  fi

  local -a EXTRA=()
  [ "$HARDNEG" = "1" ] && EXTRA+=(--hard-negative)
  [ "$ENGINE" = "lc0" ] && EXTRA+=(--lc0-net-path "$LC0NET")

  if python generate_anomaly_corpus.py \
      --maia-band "$BAND" --engine "$ENGINE" --substitution-rate "$RATE" \
      --num-games "$GAMES_PER_CELL" --output-dir "$CELL_DIR" \
      --opening-table "$OPENINGS_DIR/openings_band${BAND}.json" \
      --maia-weights-dir "$MAIA" --lc0-path "$LC0" --stockfish-path "$SF" \
      --clock-module-dir "$CLOCK_MODULE_DIR" --global-seed "$GLOBAL_SEED" \
      --log-every 200 "${EXTRA[@]}" > "$LOG_FILE" 2>&1; then
    grep '^SUMMARY ' "$LOG_FILE" | tail -1 | sed 's/^SUMMARY //' > "$CELL_DIR/.cell_summary.json" || true
    touch "$CELL_DIR/.done"
    echo "CELL_DONE $CELLNAME $(tail -1 "$LOG_FILE")"
  else
    echo "CELL_FAILED $CELLNAME (see $LOG_FILE)"
  fi
}
export -f run_cell
export OUT_DIR LOG_DIR GAMES_PER_CELL OPENINGS_DIR CLOCK_MODULE_DIR GLOBAL_SEED
export LC0 SF LC0NET MAIA

tr '|' ' ' < "$JOBLIST" | xargs -P "$WORKERS" -L1 bash -c 'run_cell "$@"' _
rm -f "$JOBLIST"
echo "SWEEP_COMPLETE"
