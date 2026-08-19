#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 CHECKPOINT" >&2
  exit 2
fi

for branch in fresh drop bank-fix online; do
  experiments/x0loop_v2/run_5k.sh "$1" "$branch"
done
