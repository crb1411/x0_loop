#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 CHECKPOINT OUTPUT_PREFIX" >&2
  exit 2
fi

checkpoint=$1
output_prefix=$2
gpu=${X0LOOP_GPU:-7}
if ! nvidia-smi -i "$gpu" >/dev/null 2>&1; then
  gpu=6
fi

for nfe in 4 8 20; do
  CUDA_VISIBLE_DEVICES=$gpu uv run python -m x0loop.eval_fid \
    --ckpt "$checkpoint" \
    --eval-config experiments/x0loop_v2/eval.yaml \
    --set "gen_eval.steps=$nfe" \
    --set "logging.out_dir=$output_prefix/nfe$nfe"
done
