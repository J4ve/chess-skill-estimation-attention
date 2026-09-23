#!/bin/bash
# Run one extra-arm step with logging; guarantees a FAILED marker on a hard crash.
#   bash analysis/scripts/extra_arms_run.sh <arm_dir_name> <python args...>
# Plan: analysis/anomaly-extra-arms-plan.md. Runbook: analysis/anomaly-extra-arms-runbook.md.
ARM="$1"; shift
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ratingnet2
cd ~/thesis2
ROOT=analysis/extra_arms
case " $* " in *" --smoke "*) ROOT=analysis/extra_arms_smoke ;; esac
mkdir -p logs "$ROOT/$ARM"
LOG="logs/extra_arms_${ARM}$( [ "$ROOT" = analysis/extra_arms_smoke ] && echo _smoke ).log"
echo "START $(date -Is) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} args: $*" >> "$LOG"
python -u "$@" >> "$LOG" 2>&1
rc=$?
echo "EXIT $rc $(date -Is)" >> "$LOG"
if [ $rc -ne 0 ] && [ ! -f "$ROOT/$ARM/FAILED" ]; then
  { date -Is; echo "exit code $rc; see $LOG"; tail -50 "$LOG"; } > "$ROOT/$ARM/FAILED"
fi
exit $rc
