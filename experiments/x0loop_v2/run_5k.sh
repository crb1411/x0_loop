#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 CHECKPOINT [fresh|drop|bank-fix|online]" >&2
  exit 2
fi

checkpoint=$1
branch=${2:-fresh}
gpu=${X0LOOP_GPU:-7}
if ! nvidia-smi -i "$gpu" >/dev/null 2>&1; then
  gpu=6
fi

case "$branch" in
  fresh)
    clean_enabled=false
    clean_mode=bank_fix
    ;;
  drop)
    clean_enabled=true
    clean_mode=drop
    ;;
  bank-fix)
    clean_enabled=true
    clean_mode=bank_fix
    ;;
  online)
    clean_enabled=true
    clean_mode=online
    ;;
  *)
    echo "unknown branch: $branch" >&2
    exit 2
    ;;
esac

CUDA_VISIBLE_DEVICES=$gpu uv run python -m x0loop.train \
  --config experiments/x0loop_v2/config.yaml \
  --runtime-config x0loop/configs/runtime/single_gpu.yaml \
  --set "train.resume=$checkpoint" \
  --set "clean_loop.enabled=$clean_enabled" \
  --set "clean_loop.mode=$clean_mode" \
  --set "logging.out_dir=runs/x0loop_v2/$branch"
