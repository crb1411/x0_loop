#!/usr/bin/env bash
set -euo pipefail
# ============================================================
# X0-Loop one-click launcher (single node / single GPU by default)
# Usage:
#   bash /mnt/seek/aigc/x0_loop/runs/train_x0loop_oneclick.sh [resume_ckpt_path]
# ============================================================

# conda activate vl
WORK_DIR=/mnt/seek/aigc/x0_loop
CONDA_ENV=vl

# Train config
CONFIG_PATH="${WORK_DIR}/x0loop/configs/default.yaml"
RUNTIME_CONFIG_PATH="${WORK_DIR}/x0loop/configs/runtime/fsdp_checkpoint_compile.yaml"
DEFAULT_RESUME_CKPT=/mnt/seek/aigc/x0_loop/runs/exp_default/run_20260213_212657/checkpoints/ckpt_step_00148000.pt

# torchrun setup
NNODES=1
NODE_RANK=0
GPUS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-40028}"
export PYTHONPATH="${WORK_DIR}:${PYTHONPATH:-}"
export HOSTNAME="$(hostname)"

# Output directory root (each run gets its own timestamp subdir)
OUTPUT_ROOT="${WORK_DIR}/runs/exp_default"
mkdir -p "${OUTPUT_ROOT}"

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
export X0LOOP_RUN_TIMESTAMP="${TIMESTAMP}"
RUN_DIR="${OUTPUT_ROOT}/run_${TIMESTAMP}"
mkdir -p "${RUN_DIR}"
LOG_DIR="${RUN_DIR}"
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}_node${NNODES}_${NODE_RANK}.log"
RUNTIME_EFFECTIVE_PATH="${LOG_DIR}/runtime_${TIMESTAMP}.yaml"

# Resume checkpoint priority:
# 1) env RESUME_CKPT
# 2) first script argument
# 3) DEFAULT_RESUME_CKPT
RESUME_CKPT="${RESUME_CKPT:-${1:-${DEFAULT_RESUME_CKPT}}}"

# Activate conda env in non-interactive shell.
if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found. Please install conda or adjust this script." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${WORK_DIR}"

# Build full per-run config (default + runtime + run-specific overrides).
if [[ -n "${RESUME_CKPT}" && -f "${RESUME_CKPT}" ]]; then
  RESUME_VALUE="${RESUME_CKPT}"
  RESUME_STATUS="断点续训: ${RESUME_CKPT}"
else
  RESUME_VALUE="__NONE__"
  RESUME_STATUS="未找到checkpoint，改为从头训练: ${RESUME_CKPT}"
fi

python - "${CONFIG_PATH}" "${RUNTIME_CONFIG_PATH}" "${RUNTIME_EFFECTIVE_PATH}" "${RUN_DIR}" "${RESUME_VALUE}" <<'PY'
import copy
import sys
import yaml

config_path, runtime_path, output_path, run_dir, resume_value = sys.argv[1:]

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}

def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out

cfg = load_yaml(config_path)
runtime_cfg = load_yaml(runtime_path)
merged = deep_merge(cfg, runtime_cfg)

merged.setdefault("train", {})
merged.setdefault("logging", {})
merged["train"]["resume"] = None if resume_value == "__NONE__" else resume_value
merged["logging"]["out_dir"] = run_dir

with open(output_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(merged, f, sort_keys=False)
PY

nohup torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${GPUS}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m x0loop.train \
  --config "${RUNTIME_EFFECTIVE_PATH}" \
  --runtime-config "/tmp/x0loop_runtime_none_${TIMESTAMP}.yaml" \
  > "${LOG_FILE}" 2>&1 &

echo "训练任务已在后台启动，PID: $!"
echo "${RESUME_STATUS}"
echo "本次输出目录: ${RUN_DIR}"
echo "日志文件: ${LOG_FILE}"
echo "本次runtime配置: ${RUNTIME_EFFECTIVE_PATH}"
echo "查看日志: tail -f ${LOG_FILE}"
echo "终止训练: pkill -f 'x0loop.train'"
