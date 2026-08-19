#!/usr/bin/env bash
set -euo pipefail

run_root=${1:-runs/x0loop_v2}
output_dir=${2:-runs/x0loop_v2_best_fid50k}
checkpoint=$(uv run python experiments/x0loop_v2/summarize_fid.py "$run_root" --samples 5000 --best-only)

echo "[fid-selection] checkpoint=$checkpoint"
experiments/x0loop_v2/eval_fid50k.sh "$checkpoint" "$output_dir" fid50k
