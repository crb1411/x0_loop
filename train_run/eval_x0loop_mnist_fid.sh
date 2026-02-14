#!/usr/bin/env bash
set -euo pipefail
# ============================================================
# X0-Loop MNIST FID one-click evaluator
# Usage:
#   bash /mnt/seek/aigc/x0_loop/runs/eval_x0loop_mnist_fid.sh <checkpoint.pt> [num_samples] [steps] [sampler] [posterior_noise_scale]
# Example:
#   bash /mnt/seek/aigc/x0_loop/runs/eval_x0loop_mnist_fid.sh \
#     /mnt/seek/aigc/x0_loop/runs/exp_default/run_20260213_213926/checkpoints/ckpt_step_00160000.pt \
#     10000 1000 posterior 1.0
# ============================================================

WORK_DIR=/mnt/seek/aigc/x0_loop
CONDA_ENV=vl

CKPT_PATH="${1:-}"
NUM_SAMPLES="${2:-10000}"
STEPS="${3:-100}"
SAMPLER="${4:-auto}" # auto | ddim | posterior
POSTERIOR_NOISE_SCALE="${5:-}"
AUTO_INSTALL_FID_DEPS="${AUTO_INSTALL_FID_DEPS:-1}"

if [[ -z "${CKPT_PATH}" ]]; then
  echo "Usage: $0 <checkpoint.pt> [num_samples] [steps] [sampler] [posterior_noise_scale]" >&2
  exit 1
fi
if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "checkpoint not found: ${CKPT_PATH}" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found. Please install conda or adjust this script." >&2
  exit 1
fi
export CONDA_NO_PLUGINS=true
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${WORK_DIR}"

RUN_DIR="$(cd "$(dirname "${CKPT_PATH}")/.." && pwd)"
RUNTIME_EFFECTIVE_PATH="$(ls -1t "${RUN_DIR}"/runtime_*.yaml | head -n1 || true)"
if [[ -z "${RUNTIME_EFFECTIVE_PATH}" || ! -f "${RUNTIME_EFFECTIVE_PATH}" ]]; then
  echo "runtime yaml not found under run dir: ${RUN_DIR}" >&2
  exit 1
fi

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
OUT_DIR="${RUN_DIR}/fid_mnist_${TIMESTAMP}"
LOG_FILE="${OUT_DIR}/eval_fid_${TIMESTAMP}.log"
mkdir -p "${OUT_DIR}"

# Dependency check: torch-fidelity (required by evaluator).
if ! conda run -n "${CONDA_ENV}" python -c "import torch_fidelity" >/dev/null 2>&1; then
  if [[ "${AUTO_INSTALL_FID_DEPS}" == "1" ]]; then
    echo "[fid] torch-fidelity missing, installing..."
    conda run -n "${CONDA_ENV}" python -m pip install torch-fidelity
  else
    echo "[fid] missing dependency: torch-fidelity" >&2
    echo "[fid] install with: conda run -n ${CONDA_ENV} python -m pip install torch-fidelity" >&2
    exit 1
  fi
fi

EXTRA_ARGS=()
if [[ -n "${POSTERIOR_NOISE_SCALE}" ]]; then
  EXTRA_ARGS+=(--posterior-noise-scale "${POSTERIOR_NOISE_SCALE}")
fi

nohup conda run -n "${CONDA_ENV}" python -m x0loop.eval_mnist_fid \
  --config "${RUNTIME_EFFECTIVE_PATH}" \
  --runtime-config "/tmp/x0loop_runtime_none_${TIMESTAMP}.yaml" \
  --ckpt "${CKPT_PATH}" \
  --out-dir "${OUT_DIR}" \
  --num-samples "${NUM_SAMPLES}" \
  --steps "${STEPS}" \
  --sampler "${SAMPLER}" \
  "${EXTRA_ARGS[@]}" \
  > "${LOG_FILE}" 2>&1 &

echo "MNIST FID任务已在后台启动，PID: $!"
echo "checkpoint: ${CKPT_PATH}"
echo "run目录: ${RUN_DIR}"
echo "自动加载runtime: ${RUNTIME_EFFECTIVE_PATH}"
echo "输出目录: ${OUT_DIR}"
echo "日志文件: ${LOG_FILE}"
echo "查看日志: tail -f ${LOG_FILE}"
