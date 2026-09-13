#!/usr/bin/env bash
# Pilot batch for the anomaly-validation synthetic corpus: one Maia band x
# one engine x all 5 substitution rates (~500 games/case ~= 2500 games).
# Measures real per-game wall-clock cost to extrapolate the full ~450k-game
# target before committing HPC compute/storage to it.
set -euo pipefail

cd "$(dirname "$0")"

ENGINES_DIR=${ENGINES_DIR:-~/Bacsain/thesis2/engines}
OUT_DIR=${OUT_DIR:-/tmp/anomaly_pilot}
BAND=${BAND:-1500}
ENGINE=${ENGINE:-stockfish16}
GAMES_PER_CASE=${GAMES_PER_CASE:-500}

mkdir -p "$OUT_DIR"

for RATE in 0.0 0.05 0.15 0.30 0.60; do
  CASE_ENGINE=$ENGINE
  if [ "$RATE" = "0.0" ]; then
    CASE_ENGINE=none
  fi
  CASE_DIR="$OUT_DIR/${BAND}_${ENGINE}_r$(echo $RATE | tr -d '.')"
  echo "=== band=$BAND engine=$CASE_ENGINE rate=$RATE -> $CASE_DIR ==="
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
    --log-every 50
done
