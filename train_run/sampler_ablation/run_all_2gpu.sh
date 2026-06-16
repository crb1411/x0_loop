#!/usr/bin/env bash
# Run independent single-GPU sampler-ablation evals on two GPUs.
# Experiments are launched in list order, two at a time: one on GPU0 and one on
# GPU1. The next pair starts only after the current pair finishes.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"

EXPERIMENTS=(
  euler_s20_cfg1p5
  heun_s20_cfg1p5
  dpmpp_2m_s20_cfg1p5
  unipc_s20_cfg1p5
  unipc3_s20_cfg1p5
  unipc_s06_cfg1p5
  unipc_s10_cfg1p5
  unipc_s16_cfg1p5
  unipc_s50_cfg1p5
  dpmpp_2m_s20_cfg1
  dpmpp_2m_s20_cfg2
  dpmpp_2m_s20_cfg3
  heun_s50_cfg1p5
)

status=0
idx=0
while (( idx < ${#EXPERIMENTS[@]} )); do
  exp0="${EXPERIMENTS[$idx]}"
  echo "[run_all_2gpu] start ${exp0} on GPU ${GPU0}"
  (
    CUDA_VISIBLE_DEVICES="${GPU0}" bash "${ROOT}/train_run/sampler_ablation/${exp0}/run.sh"
    echo "[run_all_2gpu] done ${exp0} on GPU ${GPU0}"
  ) &
  pid0=$!

  pid1=""
  if (( idx + 1 < ${#EXPERIMENTS[@]} )); then
    exp1="${EXPERIMENTS[$((idx + 1))]}"
    echo "[run_all_2gpu] start ${exp1} on GPU ${GPU1}"
    (
      CUDA_VISIBLE_DEVICES="${GPU1}" bash "${ROOT}/train_run/sampler_ablation/${exp1}/run.sh"
      echo "[run_all_2gpu] done ${exp1} on GPU ${GPU1}"
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
    echo "[run_all_2gpu] stopping after failed pair" >&2
    exit "${status}"
  fi
  idx=$((idx + 2))
done

bash "${ROOT}/train_run/sampler_ablation/summarize.sh"
