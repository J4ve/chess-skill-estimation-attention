#!/usr/bin/env bash
# fm-corpus-autopreprocess.sh - deterministic auto-launch for finished downloads.
#
# Root cause this fixes: a worker's download finishes, but the agent
# responsible for noticing "download done, start preprocessing" either goes
# stale or its whole process dies before it can trigger the next step -
# observed repeatedly all night (loop-c, loop-d, loop-e, loop, loop-b,
# loop-f, loop-g all hit this at least once). This script closes that gap
# without needing any agent to notice anything: it checks real on-disk state
# (does the .zst's size match the remote content-length exactly) and real
# process state (is clkscan already running against this exact file), and
# launches preprocessing itself the moment a download is genuinely complete
# and nothing is already processing it.
#
# Pairs with fm-corpus-zst-cleanup.sh, which handles the other end (deleting
# a .zst once its preprocessing is verified done). Together these two scripts
# mean neither end of "download -> preprocess -> delete" needs a worker
# agent to notice anything - only to kick off the initial download.
#
# Usage: fm-corpus-autopreprocess.sh [--once] [--interval SECONDS]
set -u

RAW_DIR="/mnt/d/firstmate/projects/cs_thesis_2/local-run/data/raw_zst"
PROCESSED_DIR="/mnt/d/firstmate/projects/cs_thesis_2/local-run/data/processed_games"
LOG_DIR="/mnt/d/firstmate/projects/cs_thesis_2/local-run/logs"
CLKSCAN="/mnt/d/firstmate/projects/cs_thesis_2/local-run/bin/clkscan"
PYTHON="/mnt/d/firstmate/projects/cs_thesis_2/prototype/.venv/bin/python"
PREPROCESS_SCRIPT="/mnt/d/firstmate/projects/cs_thesis_2/analysis/scripts/preprocess_onepass.py"
AUTOLOG="/mnt/d/firstmate/data/corpus-autopreprocess.log"
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
  echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] $*" >> "$AUTOLOG"
}

pass() {
  local zst month pkl_count remote_len local_len
  for zst in "$RAW_DIR"/lichess_db_standard_rated_*.pgn.zst; do
    [ -e "$zst" ] || continue
    month=$(basename "$zst" | sed -E 's/^lichess_db_standard_rated_([0-9]{4}-[0-9]{2})\.pgn\.zst$/\1/')
    [ -n "$month" ] || continue

    # already fully preprocessed? nothing to do (cleanup script will delete the zst)
    pkl_count=$(find "$PROCESSED_DIR/$month" -maxdepth 1 -name "*.pkl" 2>/dev/null | wc -l)
    if [ "$pkl_count" -eq 30000 ]; then
      continue
    fi

    # already being preprocessed right now? don't launch a duplicate
    if pgrep -f "preprocess_onepass.py.*$month" >/dev/null 2>&1; then
      continue
    fi

    # is the download actually complete? compare against remote content-length
    local_len=$(stat -c '%s' "$zst" 2>/dev/null || echo "0")
    remote_len=$(curl -fsSL -I "https://database.lichess.org/standard/lichess_db_standard_rated_${month}.pgn.zst" 2>/dev/null | grep -i '^content-length:' | tr -d '\r' | awk '{print $2}')
    if [ -z "$remote_len" ] || [ "$local_len" != "$remote_len" ]; then
      continue
    fi

    # genuinely complete and nothing already processing it - launch it ourselves
    setsid nohup "$PYTHON" "$PREPROCESS_SCRIPT" \
      --input "$zst" \
      --output-dir "$PROCESSED_DIR/$month" \
      --clkscan "$CLKSCAN" \
      --max-games 30000 \
      > "$LOG_DIR/pp_${month}_auto.log" 2>&1 < /dev/null &
    disown
    log "launched preprocessing for $month (verified $local_len bytes matches remote content-length) - log: pp_${month}_auto.log"
  done
}

if [ "$ONCE" -eq 1 ]; then
  pass
  exit 0
fi

log "auto-preprocess loop started, interval=${INTERVAL}s"
while true; do
  pass
  sleep "$INTERVAL"
done
