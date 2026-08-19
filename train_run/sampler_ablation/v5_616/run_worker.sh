#!/usr/bin/env bash
# Run one shard of v5_616 experiments serially on one GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"

GPU="${1:?usage: run_worker.sh GPU PARITY}"
PARITY="${2:?usage: run_worker.sh GPU PARITY}"
EXP_FILE="${ROOT}/train_run/sampler_ablation/v5_616/experiments.txt"
OUT_BASE="./runs/sampler_ablation/v5_616"

mapfile -t EXPERIMENTS < "${EXP_FILE}"

idx=0
for exp in "${EXPERIMENTS[@]}"; do
  if (( idx % 2 == PARITY )); then
    if compgen -G "${OUT_BASE}/${exp}/gen_eval_metrics_*.jsonl" > /dev/null; then
      echo "[worker:v5_616] skip existing ${exp}"
    else
      log="${OUT_BASE}/${exp}/logs/launcher.log"
      mkdir -p "$(dirname "${log}")"
      echo "[worker:v5_616] start ${exp} on GPU ${GPU}"
      CUDA_VISIBLE_DEVICES="${GPU}" bash "${ROOT}/train_run/sampler_ablation/v5_616/${exp}/run.sh" > "${log}" 2>&1
      echo "[worker:v5_616] done ${exp} on GPU ${GPU}"
    fi
  fi
  idx=$((idx + 1))
done
