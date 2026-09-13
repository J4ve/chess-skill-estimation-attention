#!/usr/bin/env bash
# fm-corpus-zst-cleanup.sh - deterministic auto-delete for completed corpus months.
#
# Root cause this fixes: a preprocessing worker's declared status routinely
# lags behind its real progress (observed all night on 2026-08-19/20) - the
# .zst for a month sits undeleted even after preprocessing genuinely finished,
# because the worker never noticed its own completion to trigger the delete
# step in its own instructions. This decouples cleanup from worker
# self-reporting entirely: it checks real on-disk state (pkl count + a clean
# DONE line in the preprocess log), not any worker's status file.
#
# Usage: fm-corpus-zst-cleanup.sh [--once] [--interval SECONDS]
#   --once             run a single pass and exit (default: loop forever)
#   --interval SECONDS seconds between passes when looping (default: 300)
set -u

RAW_DIR="/mnt/d/firstmate/projects/cs_thesis_2/local-run/data/raw_zst"
PROCESSED_DIR="/mnt/d/firstmate/projects/cs_thesis_2/local-run/data/processed_games"
LOG_DIR="/mnt/d/firstmate/projects/cs_thesis_2/local-run/logs"
CLEANUP_LOG="/mnt/d/firstmate/data/corpus-zst-cleanup.log"
INTERVAL=300
ONCE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --once) ONCE=1; shift ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

log() {
  echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] $*" >> "$CLEANUP_LOG"
}

pass() {
  local zst month pkl_count log_file done_line freed=0
  for zst in "$RAW_DIR"/lichess_db_standard_rated_*.pgn.zst; do
    [ -e "$zst" ] || continue
    month=$(basename "$zst" | sed -E 's/^lichess_db_standard_rated_([0-9]{4}-[0-9]{2})\.pgn\.zst$/\1/')
    [ -n "$month" ] || continue

    pkl_count=$(find "$PROCESSED_DIR/$month" -maxdepth 1 -name "*.pkl" 2>/dev/null | wc -l)
    if [ "$pkl_count" -ne 30000 ]; then
      continue
    fi

    # find any log whose name contains the month and has a clean DONE line
    done_line=""
    for log_file in "$LOG_DIR"/*"$month"*.log; do
      [ -e "$log_file" ] || continue
      if grep -q "^DONE .*kept=30000" "$log_file" 2>/dev/null; then
        done_line="$log_file"
        break
      fi
    done

    if [ -z "$done_line" ]; then
      continue
    fi

    local size_bytes
    size_bytes=$(stat -c '%s' "$zst" 2>/dev/null || echo "?")
    if rm -f "$zst"; then
      log "deleted $month.zst ($size_bytes bytes) - verified 30000 pkls + clean DONE in $(basename "$done_line")"
      freed=$((freed + 1))
    else
      log "FAILED to delete $zst"
    fi
  done
  if [ "$freed" -gt 0 ]; then
    log "pass complete: freed $freed archive(s)"
  fi
}

if [ "$ONCE" -eq 1 ]; then
  pass
  exit 0
fi

log "cleanup loop started, interval=${INTERVAL}s"
while true; do
  pass
  sleep "$INTERVAL"
done
