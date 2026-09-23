#!/bin/bash
# hpc_training_heal.sh — auto-resume the full-corpus training arms after an
# HPC reboot (or any other process-killing event). Deployed on the HPC itself
# at ~/thesis2/, invoked by cron every 10 min. Keep this file and the
# HPC-deployed copy in sync when editing (project convention, see AGENTS.md).
#
# Why this exists: 2026-08-26/27, an HPC reboot wiped /tmp/ratingnet_store
# (the corpus store lives there for training speed) and killed 3 of 4
# concurrent training runs. The store rebuilt cleanly and all 3 resumed
# correctly from their last checkpoint, but nobody was watching, so the
# runs sat idle for 4+ hours before a human/agent noticed and relaunched
# them by hand. This script is the fix: it notices and relaunches itself.
#
# Completion detection: --patience drives ReduceLROnPlateau (LR reduction),
# NOT early stopping (verified against the code, 2026-08-25 code review) - no
# cell ever exits its loop early. The only real completion signal is the
# literal "Training duration (min):" line main() prints after epoch 60. A
# run with that line in its log is done; never touch it again.

set -u
cd ~/thesis2 || exit 1

LOCK=/tmp/fm-training-heal.lock
LOG=logs/training-heal.log
STORE_DIR=/tmp/ratingnet_store
STORE_BACKUP=data/corpus_store_backup.sqlite

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

# ---- single-instance lock (stale after 20 min, matches fm-corpus-heal.sh's pattern) ----
if [ -e "$LOCK" ]; then
  lock_age=$(( $(date +%s) - $(stat -c %Y "$LOCK" 2>/dev/null || echo 0) ))
  if [ "$lock_age" -lt 1200 ]; then
    exit 0
  fi
  log "stale lock ($lock_age s old), breaking it"
fi
touch "$LOCK"
trap 'rm -f "$LOCK"' EXIT

# name : gpu : data_dir : dropout_rate : extra_args
RUNS=(
  "preflight_check_2m:1:/tmp/ratingnet_store:0.5:--lr 3e-4 --use_attention --attention_type bahdanau --attention_dim 64"
  "fullcorpus_baseline_arm1:2:/tmp/ratingnet_store:0.5:--lr 1e-4"
  "fullcorpus_attention_untuned_arm2:3:/tmp/ratingnet_store:0.5:--lr 1e-4 --use_attention --attention_type bahdanau --attention_dim 64"
  "diag_fullcorpus_lowdropout:3:/tmp/ratingnet_store:0.3:--lr 3e-4 --use_attention --attention_type bahdanau --attention_dim 64"
  "fullcorpus_deepcnn_arm5:1:/tmp/ratingnet_store:0.5:--lr 3e-4 --use_attention --attention_type bahdanau --attention_dim 64 --deeper_cnn"
  "fullcorpus_baseline_lr3e4_arm6:1:/tmp/ratingnet_store:0.5:--lr 3e-4"
)

store_restored_this_pass=0

ensure_store() {
  if [ -f "$STORE_DIR/corpus.sqlite" ]; then
    return 0
  fi
  if [ "$store_restored_this_pass" -eq 1 ]; then
    return 0  # already handled earlier in this same pass
  fi
  log "corpus store missing at $STORE_DIR - restoring from backup"
  mkdir -p "$STORE_DIR"
  if [ ! -f "$STORE_BACKUP" ]; then
    log "ABORT: no backup at $STORE_BACKUP either - cannot restore, needs a human"
    return 1
  fi
  cp "$STORE_BACKUP" "$STORE_DIR/corpus.sqlite.tmp" && mv "$STORE_DIR/corpus.sqlite.tmp" "$STORE_DIR/corpus.sqlite"
  log "store restored from backup ($(du -h "$STORE_DIR/corpus.sqlite" | cut -f1))"
  store_restored_this_pass=1
}

for entry in "${RUNS[@]}"; do
  IFS=':' read -r name gpu data_dir dropout extra <<< "$entry"

  if pgrep -f -- "--experiment $name" > /dev/null; then
    continue  # already running, nothing to do
  fi

  if grep -q "^Training duration (min):" "logs/$name.log" 2>/dev/null; then
    continue  # genuinely finished all 60 epochs, never touch again
  fi

  if [ ! -f "models/$name/latest.pth" ]; then
    log "$name: not running, no completion marker, but no checkpoint either - never started or too early to heal, skipping"
    continue
  fi

  log "$name: not running and not complete - resuming from models/$name/latest.pth"

  ensure_store || continue

  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate ratingnet2
  CUDA_VISIBLE_DEVICES="$gpu" setsid nohup python src/chess_rating_net.py \
    --train --data_dir "$data_dir" --experiment "$name" \
    --epochs 60 --batch_size 32 --val_batch_size 512 --num_workers 8 \
    --weight_decay 1e-5 --patience 5 --dropout_rate "$dropout" --split_seed 42 \
    $extra --resume "models/$name/latest.pth" \
    >> "logs/$name.log" 2>&1 < /dev/null &
  disown
  log "$name: relaunched on gpu $gpu, pid $!"
done
