#!/usr/bin/env bash
# Full anomaly-validation synthetic corpus sweep: 9 Maia bands x 2 engines x
# 5 substitution rates, GAMES_PER_CASE games each (default 1000 -> 90,000
# games total; the approved scaled-down target -- see
# analysis/anomaly-corpus-pilot.md for why 5,000/case was not used).
#
# Runs cases through a job queue with modest concurrency (default 3 workers)
# so this stays a light neighbor to whatever else is running on the shared
# HPC box (see CLAUDE.md's num_workers ~= cores / concurrent_runs practice).
# Each individual game-generation process is single-threaded (Threads=1).
set -euo pipefail

cd "$(dirname "$0")"

ENGINES_DIR=${ENGINES_DIR:-~/Bacsain/thesis2/engines}
OUT_DIR=${OUT_DIR:-~/Bacsain/thesis2/data/anomaly_corpus}
GAMES_PER_CASE=${GAMES_PER_CASE:-1000}
WORKERS=${WORKERS:-3}
LOG_DIR=${LOG_DIR:-~/Bacsain/thesis2/logs/anomaly_corpus}

mkdir -p "$OUT_DIR" "$LOG_DIR"

JOBLIST=$(mktemp)
for BAND in 1100 1200 1300 1400 1500 1600 1700 1800 1900; do
  for ENGINE in stockfish16 lc0; do
    for RATE in 0.0 0.05 0.15 0.30 0.60; do
      CASE_ENGINE=$ENGINE
      [ "$RATE" = "0.0" ] && CASE_ENGINE=none
      echo "$BAND|$CASE_ENGINE|$RATE|${ENGINE}" >> "$JOBLIST"
    done
  done
done

echo "Total cases: $(wc -l < "$JOBLIST"), $GAMES_PER_CASE games/case, $WORKERS workers"

run_case() {
  local BAND=$1 CASE_ENGINE=$2 RATE=$3 ENGINE_LABEL=$4
  local RATE_TAG
  RATE_TAG=$(echo "$RATE" | tr -d '.')
  local CASE_DIR="$OUT_DIR/${BAND}_${ENGINE_LABEL}_r${RATE_TAG}"
  local LOG_FILE="$LOG_DIR/${BAND}_${ENGINE_LABEL}_r${RATE_TAG}.log"

  if [ -f "$CASE_DIR/.done" ]; then
    echo "SKIP (already done) $CASE_DIR"
    return 0
  fi

  python generate_anomaly_corpus.py \
    --maia-band "$BAND" \
    --engine "$CASE_ENGINE" \
    --substitution-rate "$RATE" \
    --num-games "$GAMES_PER_CASE" \
    --output-dir "$CASE_DIR" \
    --maia-weights-dir "$ENGINES_DIR/maia_weights" \
    --lc0-path "$ENGINES_DIR/lc0/build/release/lc0" \
    --stockfish-path "$ENGINES_DIR/stockfish/stockfish-ubuntu-x86-64-avx2" \
    --lc0-net-path "$ENGINES_DIR/lc0_networks/small.pb.gz" \
    --log-every 200 > "$LOG_FILE" 2>&1 \
  && touch "$CASE_DIR/.done" \
  && echo "CASE_DONE $CASE_DIR $(tail -1 "$LOG_FILE")" \
  || echo "CASE_FAILED $CASE_DIR (see $LOG_FILE)"
}
export -f run_case
export OUT_DIR LOG_DIR GAMES_PER_CASE ENGINES_DIR

cat "$JOBLIST" | tr '|' ' ' | xargs -P "$WORKERS" -I{} bash -c 'run_case {}'
rm -f "$JOBLIST"
echo "SWEEP_COMPLETE"
