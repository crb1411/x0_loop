#!/usr/bin/env bash
# Run independent single-GPU v2_616 evals on two GPUs.
# Experiments are launched in list order, two at a time.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"

EXPERIMENTS=(
  dpmpp_2m_s20_cfg1p75
  dpmpp_2m_s20_cfg2
  dpmpp_2m_s20_cfg2p25
  dpmpp_2m_s20_cfg2p5
  dpmpp_2m_s20_cfg2p75
  dpmpp_2m_s12_cfg2
  dpmpp_2m_s16_cfg2
  dpmpp_2m_s24_cfg2
  dpmpp_2m_s30_cfg2
  dpmpp_2m_s40_cfg2
  dpmpp_2m_s16_cfg2p25
  dpmpp_2m_s24_cfg2p25
  dpmpp_2m_s30_cfg2p25
  heun_s20_cfg2
  heun_s30_cfg2
  heun_s50_cfg2
  unipc_s20_cfg2
  unipc_s50_cfg2
)

status=0
idx=0
while (( idx < ${#EXPERIMENTS[@]} )); do
  exp0="${EXPERIMENTS[$idx]}"
  echo "[run_all_2gpu:v2_616] start ${exp0} on GPU ${GPU0}"
  (
    CUDA_VISIBLE_DEVICES="${GPU0}" bash "${ROOT}/train_run/sampler_ablation/v2_616/${exp0}/run.sh"
    echo "[run_all_2gpu:v2_616] done ${exp0} on GPU ${GPU0}"
  ) &
  pid0=$!

  pid1=""
  if (( idx + 1 < ${#EXPERIMENTS[@]} )); then
    exp1="${EXPERIMENTS[$((idx + 1))]}"
    echo "[run_all_2gpu:v2_616] start ${exp1} on GPU ${GPU1}"
    (
      CUDA_VISIBLE_DEVICES="${GPU1}" bash "${ROOT}/train_run/sampler_ablation/v2_616/${exp1}/run.sh"
      echo "[run_all_2gpu:v2_616] done ${exp1} on GPU ${GPU1}"
    ) &
    pid1=$!
  fi

  if ! wait "${pid0}"; then
    status=1
  fi
  if [[ -n "${pid1}" ]] && ! wait "${pid1}"; then
    status=1
  fi

  if (( status != 0 )); then
    echo "[run_all_2gpu:v2_616] stopping after failed pair" >&2
    exit "${status}"
  fi
  idx=$((idx + 2))
done

bash "${ROOT}/train_run/sampler_ablation/v2_616/summarize.sh"
