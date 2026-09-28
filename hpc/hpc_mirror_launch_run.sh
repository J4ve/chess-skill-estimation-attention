#!/usr/bin/env bash
# Launch one arm of the White/Black separation retrain in a named tmux session.
# Usage: hpc_mirror_launch_run.sh <session> <gpu> <arm: a|b>
#
# The two arms are the adopted configuration (attention on, lr 3e-4, batch 32,
# weight decay 1e-5, dropout 0.5, patience 5, 60 epochs, split seed 42, training
# seed 0 by default) plus one change each:
#   a  --gap_weighting                              (per-game loss weight by rating gap)
#   b  --separate_heads --diff_loss_weight 0.5      (per-side head + rating-difference term)
#
# The flag strings below must stay byte-identical to the matching RUNS entries in
# hpc_training_heal.sh. The trainer rebuilds the objective from the command line
# rather than from the checkpoint, so a resume launched with a different flag set
# would be a different experiment under the same name. The trainer refuses such a
# resume, which turns a silent mistake into a loud one, but keeping the two in
# step is still the point.
#
# Logs are appended, never truncated: the auto-resume watchdog decides a run is
# finished by finding the trainer's "Training duration (min):" line in the log,
# and a truncating launch would throw that away.
#
# Override EXPERIMENT / DATA_DIR / EPOCHS to smoke test an arm on a small subset
# without touching the real run's name, checkpoints or log.
set -euo pipefail
sess="$1"; gpu="$2"; arm="$3"
cd ~/thesis2

case "$arm" in
  a) default_exp=mirror_arm_a_gapweight
     extra=(--gap_weighting) ;;
  b) default_exp=mirror_arm_b_diffhead
     extra=(--separate_heads --diff_loss_weight 0.5) ;;
  *) echo "unknown arm '$arm' (expected a or b)" >&2; exit 2 ;;
esac

exp="${EXPERIMENT:-$default_exp}"
data_dir="${DATA_DIR:-/tmp/ratingnet_store}"
epochs="${EPOCHS:-60}"

mkdir -p logs models
if tmux has-session -t "$sess" 2>/dev/null; then
  echo "session $sess already exists, skipping"
  exit 0
fi

resume=()
if [ -f "models/$exp/latest.pth" ]; then resume=(--resume "models/$exp/latest.pth"); fi

cmd=(python -u src/chess_rating_net.py --train
     --data_dir "$data_dir" --experiment "$exp"
     --epochs "$epochs" --lr 3e-4 --batch_size 32 --val_batch_size 512
     --num_workers 8 --split_seed 42 --model_dir models
     --weight_decay 1e-5 --patience 5 --dropout_rate 0.5
     --use_attention --attention_type bahdanau --attention_dim 64
     "${extra[@]}" "${resume[@]}")
printf -v cmdline '%q ' "${cmd[@]}"
tmux new -d -s "$sess" "cd ~/thesis2 && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ratingnet2 && CUDA_VISIBLE_DEVICES=$gpu $cmdline 2>&1 | tee -a logs/$exp.log; exec bash"
echo "launched $sess (gpu=$gpu arm=$arm exp=$exp epochs=$epochs data_dir=$data_dir)"
