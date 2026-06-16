#!/usr/bin/env bash
# Run independent single-GPU v5_616 evals on two GPUs.
# Experiments are launched in experiments.txt order, two at a time.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
EXP_FILE="${ROOT}/train_run/sampler_ablation/v5_616/experiments.txt"

if [[ ! -f "${EXP_FILE}" ]]; then
  echo "[run_all_2gpu:v5_616] missing ${EXP_FILE}; run _generate.py first" >&2
  exit 1
fi

mapfile -t EXPERIMENTS < "${EXP_FILE}"
OUT_BASE="/data/seek/aigc/x0_loop/runs/sampler_ablation/v5_616"
TODO=()
for exp in "${EXPERIMENTS[@]}"; do
  if compgen -G "${OUT_BASE}/${exp}/gen_eval_metrics_*.jsonl" > /dev/null; then
    echo "[run_all_2gpu:v5_616] skip existing ${exp}"
  else
    TODO+=("${exp}")
  fi
done
EXPERIMENTS=("${TODO[@]}")

status=0
idx=0
while (( idx < ${#EXPERIMENTS[@]} )); do
  exp0="${EXPERIMENTS[$idx]}"
  log0="${OUT_BASE}/${exp0}/logs/launcher.log"
  mkdir -p "$(dirname "${log0}")"
  echo "[run_all_2gpu:v5_616] start ${exp0} on GPU ${GPU0}"
  CUDA_VISIBLE_DEVICES="${GPU0}" bash "${ROOT}/train_run/sampler_ablation/v5_616/${exp0}/run.sh" > "${log0}" 2>&1 &
  pid0=$!

  pid1=""
  if (( idx + 1 < ${#EXPERIMENTS[@]} )); then
    exp1="${EXPERIMENTS[$((idx + 1))]}"
    log1="${OUT_BASE}/${exp1}/logs/launcher.log"
    mkdir -p "$(dirname "${log1}")"
    echo "[run_all_2gpu:v5_616] start ${exp1} on GPU ${GPU1}"
    CUDA_VISIBLE_DEVICES="${GPU1}" bash "${ROOT}/train_run/sampler_ablation/v5_616/${exp1}/run.sh" > "${log1}" 2>&1 &
    pid1=$!
  fi

  if ! wait "${pid0}"; then
    status=1
  else
    echo "[run_all_2gpu:v5_616] done ${exp0} on GPU ${GPU0}"
  fi
  if [[ -n "${pid1}" ]] && ! wait "${pid1}"; then
    status=1
  elif [[ -n "${pid1}" ]]; then
    echo "[run_all_2gpu:v5_616] done ${exp1} on GPU ${GPU1}"
  fi

  if (( status != 0 )); then
    echo "[run_all_2gpu:v5_616] stopping after failed pair" >&2
    exit "${status}"
  fi
  idx=$((idx + 2))
done

bash "${ROOT}/train_run/sampler_ablation/v5_616/summarize.sh"
