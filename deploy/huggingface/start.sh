#!/usr/bin/env bash
# Space entrypoint: fetch the frozen checkpoint from a private Hugging Face model
# repo (so the weights are not published with the public Space), then serve.
# Needs the Space secret HF_TOKEN; WEIGHTS_REPO and WEIGHTS_FILE can override.
set -euo pipefail

: "${WEIGHTS_REPO:=J4ve/ratingnet-weights}"
: "${WEIGHTS_FILE:=preflight_check_2m/best_model.pth}"
export WEIGHTS_REPO WEIGHTS_FILE RATINGNET_CHECKPOINT

if [ ! -f "$RATINGNET_CHECKPOINT" ]; then
  python - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id=os.environ["WEIGHTS_REPO"],
    filename=os.environ["WEIGHTS_FILE"],
    token=os.environ.get("HF_TOKEN"),
)
target = os.environ["RATINGNET_CHECKPOINT"]
os.makedirs(os.path.dirname(target), exist_ok=True)
shutil.copyfile(path, target)
print("checkpoint ready")
PY
fi

cd /home/user/app/src
exec uvicorn api:app --host 0.0.0.0 --port 7860
