#!/usr/bin/env bash
set -uo pipefail

CKPT="${1:-/data/seek/aigc/x0_loop/runs2/x0loop_bank_ablation/step_age1_p0p3_w0p3_tbank0p4_dt0p2/20260703_164047/checkpoints/ckpt_step_00097500.pt}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUN_ONE="$ROOT/infer/geneval/run_geneval.sh"
LOG_ROOT="$(dirname "$CKPT")/geneval_bank_cfg_ablation_20260704"
mkdir -p "$LOG_ROOT"

echo "[cfg_ablation] ckpt: $CKPT"
echo "[cfg_ablation] logs: $LOG_ROOT"

run_or_note() {
  local tag="$1"
  shift
  echo "[cfg_ablation] start $tag"
  set +e
  "$@"
  local rc=$?
  set -u
  if [ "$rc" -eq 0 ]; then
    echo "[cfg_ablation] done  $tag"
  else
    echo "[cfg_ablation] failed $tag rc=$rc"
  fi
  return 0
}

(
  echo "[cfg_ablation] GPU0 queue start"
  run_or_note heun_s20_cfg2p0 env CUDA_VISIBLE_DEVICES=0 SAMPLER=heun STEPS=20 CFG=2.0 RUN_TAG=heun_s20_cfg2p0 RUN_FOREGROUND=1 bash "$RUN_ONE" "$CKPT"
  run_or_note heun_s20_cfg2p1 env CUDA_VISIBLE_DEVICES=0 SAMPLER=heun STEPS=20 CFG=2.1 RUN_TAG=heun_s20_cfg2p1 RUN_FOREGROUND=1 bash "$RUN_ONE" "$CKPT"
  run_or_note heun_s20_cfg2p3 env CUDA_VISIBLE_DEVICES=0 SAMPLER=heun STEPS=20 CFG=2.3 RUN_TAG=heun_s20_cfg2p3 RUN_FOREGROUND=1 bash "$RUN_ONE" "$CKPT"
  run_or_note heun_s16_cfg2p2 env CUDA_VISIBLE_DEVICES=0 SAMPLER=heun STEPS=16 CFG=2.2 RUN_TAG=heun_s16_cfg2p2 RUN_FOREGROUND=1 bash "$RUN_ONE" "$CKPT"
  echo "[cfg_ablation] GPU0 queue done"
) > "$LOG_ROOT/gpu0_queue.log" 2>&1 &
pid0=$!

(
  echo "[cfg_ablation] GPU1 queue start"
  run_or_note heun_s20_cfg2p4 env CUDA_VISIBLE_DEVICES=1 SAMPLER=heun STEPS=20 CFG=2.4 RUN_TAG=heun_s20_cfg2p4 RUN_FOREGROUND=1 bash "$RUN_ONE" "$CKPT"
  run_or_note heun_s20_cfg2p5 env CUDA_VISIBLE_DEVICES=1 SAMPLER=heun STEPS=20 CFG=2.5 RUN_TAG=heun_s20_cfg2p5 RUN_FOREGROUND=1 bash "$RUN_ONE" "$CKPT"
  run_or_note heun_s24_cfg2p2 env CUDA_VISIBLE_DEVICES=1 SAMPLER=heun STEPS=24 CFG=2.2 RUN_TAG=heun_s24_cfg2p2 RUN_FOREGROUND=1 bash "$RUN_ONE" "$CKPT"
  run_or_note heun_s24_cfg2p4 env CUDA_VISIBLE_DEVICES=1 SAMPLER=heun STEPS=24 CFG=2.4 RUN_TAG=heun_s24_cfg2p4 RUN_FOREGROUND=1 bash "$RUN_ONE" "$CKPT"
  echo "[cfg_ablation] GPU1 queue done"
) > "$LOG_ROOT/gpu1_queue.log" 2>&1 &
pid1=$!

echo "$pid0" > "$LOG_ROOT/gpu0_queue.pid"
echo "$pid1" > "$LOG_ROOT/gpu1_queue.pid"
echo "[cfg_ablation] worker pids: $pid0 $pid1"
echo "[cfg_ablation] tail gpu0: tail -f $LOG_ROOT/gpu0_queue.log"
echo "[cfg_ablation] tail gpu1: tail -f $LOG_ROOT/gpu1_queue.log"

wait "$pid0" "$pid1"
echo "[cfg_ablation] all done"
