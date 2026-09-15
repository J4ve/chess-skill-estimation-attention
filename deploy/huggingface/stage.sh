#!/usr/bin/env bash
# Assemble a Hugging Face Space build folder for the web prototype.
# Usage: deploy/huggingface/stage.sh <out-dir>
# The checkpoint is not staged; the Space downloads it from a private model repo
# at startup (see start.sh).
set -euo pipefail

out=${1:?usage: stage.sh <out-dir>}
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)

rm -rf "$out"
mkdir -p "$out"
cp "$here/Dockerfile" "$here/start.sh" "$out/"
cp "$here/SPACE_README.md" "$out/README.md"
# CPU torch is installed separately in the Dockerfile (tensorboard stays: the model module imports it).
grep -v -x -E 'torch' "$repo/requirements.txt" > "$out/requirements-space.txt"
rsync -a --exclude '__pycache__' --exclude '*.pyc' "$repo/src/" "$out/src/"
echo "staged Space build folder at $out"
