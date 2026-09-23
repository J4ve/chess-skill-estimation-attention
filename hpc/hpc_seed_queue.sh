#!/usr/bin/env bash
# seed_queue.sh — keep exactly one seed-rerun training job per GPU.
# Per-GPU ordered pipelines below: first item is the current occupant, the rest
# are the queue. When a slot frees (occupant exits), the first non-COMPLETE
# item is launched via launch_run.sh, which auto-adds
# --resume models/<exp>/latest.pth when the checkpoint exists.
set -u
cd ~/thesis2
LOG=logs/seed_queue.log
INTERVAL=60
MAX_EPOCHS=60

declare -A ORDER=(
  [0]="conf_lr3e4_s0:0 conf_lr3e4_s4:4"
  [1]="conf_lr3e4_s1:1"
  [2]="conf_lr3e4_s2:2"
  [3]="conf_lr3e4_s3:3"
)

is_running() { pgrep -f "chess_rating_net\.py --train.*--experiment $1" >/dev/null; }

epoch_of() {
  python - "$1" <<'PY' 2>/dev/null || echo 0
import sys, torch
try:
    print(torch.load(f"models/{sys.argv[1]}/latest.pth",
                     map_location="cpu", weights_only=False).get("epoch", 0))
except Exception:
    print(0)
PY
}

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

declare -A RESTARTS
log "seed_queue watcher started (interval=${INTERVAL}s)"
while true; do
  all_done=1
  for gpu in 0 1 2 3; do
    read -ra items <<< "${ORDER[$gpu]}"
    occ_idx=-1
    for i in "${!items[@]}"; do
      exp="${items[$i]%%:*}"
      if is_running "$exp"; then occ_idx=$i; break; fi
    done
    if [ "$occ_idx" -ge 0 ]; then
      # Any running occupant means work remains — even the last pipeline item
      # must finish and be marked COMPLETE before the watcher may exit
      # (corpus_stream's gate waits on 10/10 COMPLETE markers).
      all_done=0
      continue
    fi
    for i in "${!items[@]}"; do
      exp="${items[$i]%%:*}"; seed="${items[$i]##*:}"
      [ -f "models/$exp/COMPLETE" ] && continue
      if [ "$(epoch_of "$exp")" -ge "$MAX_EPOCHS" ]; then
        touch "models/$exp/COMPLETE"
        log "GPU$gpu: $exp finished $MAX_EPOCHS epochs; marked COMPLETE"
        continue
      fi
      key="${gpu}:${i}"
      RESTARTS[$key]=$(( ${RESTARTS[$key]:-0} + 1 ))
      if [ "${RESTARTS[$key]}" -gt 3 ]; then
        log "GPU$gpu: SKIP $exp after 3 failed restarts (manual check needed)"
        touch "models/$exp/COMPLETE"
        continue
      fi
      ./launch_run.sh "$exp" "$gpu" "$exp" "$seed" >>"$LOG" 2>&1
      log "GPU$gpu slot free -> launched $exp (seed=$seed, restart #${RESTARTS[$key]})"
      all_done=0
      break
    done
  done
  if [ "$all_done" -eq 1 ]; then
    log "all pipelines drained/complete; watcher exiting"
    break
  fi
  sleep "$INTERVAL"
done
