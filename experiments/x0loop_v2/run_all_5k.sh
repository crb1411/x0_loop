#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 CHECKPOINT" >&2
  exit 2
fi

if [[ ${SKIP_START_FID:-0} != 1 ]]; then
  experiments/x0loop_v2/eval_fid50k.sh "$1" runs/x0loop_v2/start fid50k_start
fi

for branch in fresh drop bank-fix online; do
  experiments/x0loop_v2/run_5k.sh "$1" "$branch"
done
