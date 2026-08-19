#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 CHECKPOINT OUTPUT_DIR [TAG]" >&2
  exit 2
fi

gpu=${X0LOOP_GPU:-7}
if ! nvidia-smi -i "$gpu" >/dev/null 2>&1; then
  gpu=6
fi

CUDA_VISIBLE_DEVICES=$gpu uv run python -m x0loop.eval_fid \
  --ckpt "$1" \
  --eval-config experiments/x0loop_v2/eval.yaml \
  --set gen_eval.steps=20 \
  --set "logging.out_dir=$2" \
  --tag "${3:-fid50k}"
