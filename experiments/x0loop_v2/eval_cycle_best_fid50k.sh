#!/usr/bin/env bash
set -euo pipefail

cycle_root=${1:-runs/x0loop_v2_from_scratch/cycle01}
output_root=${2:-$cycle_root/best_fid50k}

for branch in fresh bank-fix online; do
  branch_dir="$cycle_root/$branch"
  checkpoint=$(uv run python experiments/x0loop_v2/summarize_fid.py "$branch_dir" --samples 5000 --best-only)
  X0LOOP_GPU=${X0LOOP_GPU:-7} experiments/x0loop_v2/eval_fid50k.sh \
    "$checkpoint" "$output_root/$branch" "best_${branch}_fid50k"
done
