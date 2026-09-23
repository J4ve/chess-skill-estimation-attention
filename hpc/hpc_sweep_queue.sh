#!/usr/bin/env bash
# sweep_queue.sh — stage-2 hyperparameter tuning sweep: keep one run per GPU.
# 17 pre-registered cells (see analysis/stage2-tuning-preregistration.md), run in
# pre-registered order round-robin across GPUs 0-3. Every cell is the ATTENTION
# model at 170k with --split_seed 42 fixed; only the swept flag varies.
set -u
cd ~/thesis2
LOG=logs/sweep_queue.log
RESFILE=logs/sweep_results.tsv
INTERVAL=60
MAX_EPOCHS=60

# Per-GPU ordered pipelines (round-robin, pre-registered order).
declare -A ORDER=(
  [0]="tune_patience5 tune_wd1e-3 tune_lr3e-4 tune_drop03 tune_ad128"
  [1]="tune_patience10 tune_wd1e-2 tune_bs32 tune_drop05"
  [2]="tune_patience_none tune_lr5e-5 tune_bs64 tune_ad32"
  [3]="tune_wd1e-5 tune_lr1e-4 tune_bs128 tune_ad64"
)

# Swept flag per experiment (control defaults live in hpc_sweep_launch_run.sh).
declare -A SWEPT=(
  [tune_patience5]="--patience 5"
  [tune_patience10]="--patience 10"
  [tune_patience_none]="--patience 60"
  [tune_wd1e-5]="--weight_decay 1e-5"
  [tune_wd1e-3]="--weight_decay 1e-3"
  [tune_wd1e-2]="--weight_decay 1e-2"
  [tune_lr5e-5]="--lr 5e-5"
  [tune_lr1e-4]="--lr 1e-4"
  [tune_lr3e-4]="--lr 3e-4"
  [tune_bs32]="--batch_size 32"
  [tune_bs64]="--batch_size 64"
  [tune_bs128]="--batch_size 128"
  [tune_drop03]="--dropout_rate 0.3"
  [tune_drop05]="--dropout_rate 0.5"
  [tune_ad32]="--attention_dim 32"
  [tune_ad64]="--attention_dim 64"
  [tune_ad128]="--attention_dim 128"
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

record_result() {
  local exp="$1" factor="$2" val="$3"
  local best_val best_ep term
  best_val=$(grep -oP 'best val loss: \K[0-9.]+' "logs/$exp.log" | tail -1)
  best_ep=$(grep -oP 'best val epoch: \K[0-9]+' "logs/$exp.log" | tail -1)
  term=$(grep -oP 'Epoch 60, Validation Loss: \K[0-9.]+' "logs/$exp.log" | tail -1)
  [ -z "$best_val" ] && best_val="NA"
  [ -z "$best_ep" ] && best_ep="NA"
  [ -z "$term" ] && term="NA"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$exp" "$factor" "$val" "$best_val" "$best_ep" "$term" >> "$RESFILE"
  log "recorded result: $exp best_val=$best_val best_ep=$best_ep term=$term"
}

if [ ! -f "$RESFILE" ]; then
  printf 'experiment\tfactor\tvalue\tbest_val_loss\tbest_epoch_0idx\tterminal_epoch60_val_loss\n' > "$RESFILE"
fi

declare -A RESTARTS
log "sweep_queue watcher started (interval=${INTERVAL}s)"
while true; do
  all_complete=1
  for gpu in 0 1 2 3; do
    read -ra items <<< "${ORDER[$gpu]}"
    occ_idx=-1
    for i in "${!items[@]}"; do
      if is_running "${items[$i]}"; then occ_idx=$i; break; fi
    done
    if [ "$occ_idx" -ge 0 ]; then
      all_complete=0
      continue
    fi
    for i in "${!items[@]}"; do
      exp="${items[$i]}"
      if [ -f "models/$exp/COMPLETE" ]; then continue; fi
      if [ "$(epoch_of "$exp")" -ge "$MAX_EPOCHS" ]; then
        sleep 5
        record_result "$exp" "${SWEPT[$exp]%% *}" "${SWEPT[$exp]#* }"
        touch "models/$exp/COMPLETE"
        log "GPU$gpu: $exp finished ${MAX_EPOCHS} epochs; marked COMPLETE"
        continue
      fi
      key="${gpu}:${i}"
      RESTARTS[$key]=$(( ${RESTARTS[$key]:-0} + 1 ))
      if [ "${RESTARTS[$key]}" -gt 3 ]; then
        log "GPU$gpu: SKIP $exp after 3 failed restarts (manual check needed)"
        touch "models/$exp/COMPLETE"
        record_result "$exp" "${SWEPT[$exp]%% *}" "${SWEPT[$exp]#* }"
        continue
      fi
      ./hpc_sweep_launch_run.sh "$exp" "$gpu" "$exp" ${SWEPT[$exp]} >>"$LOG" 2>&1
      log "GPU$gpu slot free -> launched $exp (swept='${SWEPT[$exp]}', restart #${RESTARTS[$key]})"
      all_complete=0
      break
    done
  done
  if [ "$all_complete" -eq 1 ]; then
    log "all sweep runs complete; watcher exiting"
    break
  fi
  sleep "$INTERVAL"
done
