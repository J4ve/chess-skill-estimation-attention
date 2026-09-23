#!/usr/bin/env bash
# corpus_stream.sh — build the 1.2M-corpus months 2021-04..2024-01 one month at a
# time on the CSPC HPC: download the month's .pgn.zst (~30 GB), preprocess it to
# 30k per-game .pkl files, delete the .zst, then move to the next month.
#
#   Phase A (immediate, network-only — safe while the seed rerun trains):
#           download 2021-04.
#   Phase B: wait until the seed-controlled rerun matrix is fully COMPLETE
#            (10x models/<exp>/COMPLETE + no training processes), so the
#            CPU-heavy preprocess does not contend with training.
#   Phase C: serial loop — download(m), preprocess(m), rm raw(m).
#
# Fully resumable: completed downloads are size-verified (curl -C - resumes
# partial ones), completed months are marked by <outdir>/.corpus_done.
# Peak raw staging ~30 GB (one archive); output ~6 GB/month accumulates under
# data/processed_games/. Never touches data/processed_games/{2024-02..2024-07},
# data/processed_games_flat, or /tmp/ratingnet_data_flat.
#
# Launch: tmux new -d -s corpus_stream \
#   'cd ~/thesis2 && bash corpus_stream.sh; exec bash'
set -u
cd ~/thesis2 || exit 1

RAW=data/raw_zst
OUT=data/processed_games
LOG=logs/corpus_stream.log
PROG=logs/corpus_stream.progress
BASE=https://database.lichess.org/standard
MAXGAMES=30000
mkdir -p "$RAW" logs

log()  { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
prog() { echo "$*" > "$PROG"; }

MONTHS="2021-04 2021-05 2021-06 2021-07 2021-08 2021-09 2021-10 2021-11 2021-12 \
2022-01 2022-02 2022-03 2022-04 2022-05 2022-06 2022-07 2022-08 2022-09 2022-10 \
2022-11 2022-12 2023-01 2023-02 2023-03 2023-04 2023-05 2023-06 2023-07 2023-08 \
2023-09 2023-10 2023-11 2023-12 2024-01"

# Remote sizes verified via Content-Range on 2026-08-15.
declare -A EXPECTED=(
  [2021-04]=32128332169 [2021-05]=32692123854 [2021-06]=29730565470 [2021-07]=29809494467
  [2021-08]=30392855649 [2021-09]=28460843524 [2021-10]=28478996133 [2021-11]=28207869558
  [2021-12]=31107096955 [2022-01]=33232239428 [2022-02]=27971810913 [2022-03]=29397132411
  [2022-04]=28144915874 [2022-05]=28784698676 [2022-06]=28331039141 [2022-07]=29807104208
  [2022-08]=29984806278 [2022-09]=28887786014 [2022-10]=29973814533 [2022-11]=28851245687
  [2022-12]=30161508176 [2023-01]=33465852948 [2023-02]=31838284336 [2023-03]=34864029436
  [2023-04]=32870230722 [2023-05]=33724525615 [2023-06]=31246059602 [2023-07]=30849136644
  [2023-08]=31248072774 [2023-09]=30254169325 [2023-10]=30813798999 [2023-11]=30047560534
  [2023-12]=31654288404 [2024-01]=32379325271
)

for m in $MONTHS; do
  echo "$m" | grep -Eq '^(2021-(0[4-9]|1[0-2])|2022-(0[1-9]|1[0-2])|2023-(0[1-9]|1[0-2])|2024-01)$' \
    || { log "FATAL: month $m outside the approved 2021-04..2024-01 window"; exit 1; }
  [ -n "${EXPECTED[$m]:-}" ] || { log "FATAL: no expected size for $m"; exit 1; }
done

file_size() { stat -c%s "$1" 2>/dev/null || echo 0; }

download_month() {
  local m=$1 f="$RAW/lichess_db_standard_rated_${1}.pgn.zst"
  local url="$BASE/lichess_db_standard_rated_${1}.pgn.zst"
  local exp=${EXPECTED[$1]} sz rc attempt
  for attempt in 1 2 3 4 5 6; do
    sz=$(file_size "$f")
    [ "$sz" -eq "$exp" ] && { log "download $m already complete ($sz B)"; return 0; }
    [ "$sz" -gt "$exp" ] && { log "download $m oversized ($sz > $exp); restarting file"; rm -f "$f"; }
    prog "phase=download month=$m attempt=$attempt size=$(file_size "$f")/$exp"
    curl -fsSL -C - --retry 3 --retry-delay 30 -o "$f" "$url"; rc=$?
    if [ "$rc" -eq 0 ] || [ "$rc" -eq 33 ]; then  # 33 = resume requested past EOF (file complete)
      sz=$(file_size "$f")
      [ "$sz" -eq "$exp" ] && { log "download $m complete ($sz B)"; return 0; }
    fi
    log "download $m attempt $attempt rc=$rc size=$(file_size "$f")/$exp; retrying in 60s"
    sleep 60
  done
  log "FATAL: download $m failed after 6 attempts (size=$(file_size "$f"), expected $exp)"
  prog "phase=failed month=$m reason=download"
  exit 1
}

preprocess_month() {
  local m=$1 zst="$RAW/lichess_db_standard_rated_${1}.pgn.zst" outdir="$OUT/$1" kept attempt
  if [ -f "$outdir/.corpus_done" ]; then
    log "preprocess $m already done; skipping"
    return 0
  fi
  for attempt in 1 2 3; do
    rm -rf "$outdir"; mkdir -p "$outdir"
    prog "phase=preprocess month=$m attempt=$attempt"
    python -u src/preprocess_lichess.py --input "$zst" --output-dir "$outdir" \
      --max-games "$MAXGAMES" --log-every 5000 2>&1 | tee "logs/preprocess_${m}.log"
    kept=$(find "$outdir" -maxdepth 1 -name 'game_*.pkl' | wc -l)
    if [ "$kept" -ge "$MAXGAMES" ]; then
      touch "$outdir/.corpus_done"
      log "preprocess $m OK kept=$kept"
      return 0
    fi
    log "preprocess $m attempt $attempt kept=$kept < $MAXGAMES; retrying"
  done
  log "FATAL: preprocess $m failed after 3 attempts"
  prog "phase=failed month=$m reason=preprocess"
  exit 1
}

EXPS="abl_baseline_s0 abl_baseline_s1 abl_baseline_s2 abl_baseline_s3 abl_baseline_s4 \
abl_attention_s0 abl_attention_s1 abl_attention_s2 abl_attention_s3 abl_attention_s4"

wait_for_training() {
  local done_count procs
  log "waiting for seed rerun to finish before preprocessing"
  while true; do
    done_count=0
    for e in $EXPS; do [ -f "models/$e/COMPLETE" ] && done_count=$((done_count + 1)); done
    procs=$(pgrep -f "chess_rating_net\.py --train" | wc -l)
    if [ "$done_count" -ge 10 ] && [ "$procs" -eq 0 ]; then
      log "training matrix complete (10/10 COMPLETE, 0 training procs)"
      return 0
    fi
    if [ "$procs" -eq 0 ] && ! tmux has-session -t seed_queue 2>/dev/null; then
      log "WARNING: no training procs and seed_queue watcher gone (COMPLETE=$done_count/10); proceeding"
      return 0
    fi
    prog "phase=wait-training complete=$done_count/10 procs=$procs"
    sleep 300
  done
}

source ~/miniconda3/etc/profile.d/conda.sh && conda activate ratingnet2

log "corpus_stream start (months: 2021-04..2024-01, max_games=$MAXGAMES)"

# Phase A: network-only first download, safe while training runs.
download_month 2021-04

# Phase B: no CPU-heavy work until the seed rerun is done.
wait_for_training

# Phase C: serial stream — download, preprocess, delete raw, next month.
for m in $MONTHS; do
  download_month "$m"
  preprocess_month "$m"
  rm -f "$RAW/lichess_db_standard_rated_${m}.pgn.zst"
  log "month $m fully done; raw .zst deleted"
done

total_pkls=$(find "$OUT" -name 'game_*.pkl' | wc -l)
log "ALL 34 MONTHS DONE; pkls under $OUT (incl. 2024-02..07): $total_pkls; du=$(du -sh "$OUT" | cut -f1)"
prog "phase=done months=34 total_pkls=$total_pkls"
