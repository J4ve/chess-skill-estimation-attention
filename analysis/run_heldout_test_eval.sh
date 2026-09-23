#!/bin/bash
# Score every frozen arm's best-val + epoch-60 checkpoint on the 255k held-out
# TEST split. Run on the HPC from ~/thesis2 inside tmux; resumable
# (skips an (arm,ckpt) whose .json already exists). Writes a .DONE marker.
#
#   CUDA_VISIBLE_DEVICES=1 bash analysis/scripts/run_heldout_test_eval.sh
#
# GPU 1 only (0/2/3 reserved for v2 feature extraction). Shares GPU 1 with the
# arm6 training job - forward-only eval, fits alongside it.
set -u
cd ~/thesis2 || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ratingnet2

OUT=analysis/heldout_test_eval
mkdir -p "$OUT"
LOG="$OUT/run.log"
SCORE=analysis/scripts/score_test_split.py

# label : model-dir
ARMS=(
  "baseline:fullcorpus_baseline_arm1"
  "attn_untuned:fullcorpus_attention_untuned_arm2"
  "attn_tuned:preflight_check_2m"
  "deepcnn:fullcorpus_deepcnn_arm5"
  "lowdropout:diag_fullcorpus_lowdropout"
  "baseline_lr3e4:fullcorpus_baseline_lr3e4_arm6"
)

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

for entry in "${ARMS[@]}"; do
  IFS=':' read -r label dir <<< "$entry"
  for pair in "best:best_model.pth" "ep60:model_60.pth"; do
    IFS=':' read -r tag ckpt <<< "$pair"
    src="models/$dir/$ckpt"
    js="$OUT/${label}__${tag}.json"
    csv="$OUT/${label}__${tag}.csv"
    if [ -f "$js" ]; then
      log "SKIP $label/$tag (have $js)"
      continue
    fi
    if [ ! -f "$src" ]; then
      log "MISS $label/$tag: no $src (arm not finished?) - skipping"
      continue
    fi
    log "SCORE $label/$tag <- $src"
    python "$SCORE" \
      --checkpoint "$src" \
      --data_dir /tmp/ratingnet_store \
      --src_dir src \
      --split_seed 42 \
      --val_batch_size 512 \
      --num_workers 8 \
      --label "${label}__${tag}" \
      --out_csv "$csv" \
      --out_json "$js" >> "$LOG" 2>&1 \
      && log "OK   $label/$tag" \
      || log "FAIL $label/$tag (see $LOG)"
  done
done

log "all arms attempted"
touch "$OUT/eval.DONE"
