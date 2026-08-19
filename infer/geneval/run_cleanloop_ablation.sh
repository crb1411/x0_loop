#!/usr/bin/env bash

if [[ $# -ne 1 ]]; then
  echo "usage: $0 CHECKPOINT" >&2
  exit 2
fi
CKPT="$1"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUN_ONE="$ROOT/infer/geneval/run_geneval.sh"

echo "[ablation] ckpt: $CKPT"

run_or_note() {
  local tag="$1"
  shift
  echo "[ablation] start $tag"
  "$@"
  local rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "[ablation] done  $tag"
  else
    echo "[ablation] failed $tag rc=$rc"
  fi
  return 0
}

(
  echo "[ablation] GPU0 queue start"
  run_or_note cleanloop_s02_cfg1p8_rt0p5 env CUDA_VISIBLE_DEVICES=0 SAMPLER=clean_loop STEPS=2  CFG=1.8 REFINE_TIME=0.5 RUN_TAG=cleanloop_s02_cfg1p8_rt0p5 RUN_FOREGROUND=1 bash "$RUN_ONE" "$CKPT"
  run_or_note cleanloop_s03_cfg1p8_rt0p5 env CUDA_VISIBLE_DEVICES=0 SAMPLER=clean_loop STEPS=3  CFG=1.8 REFINE_TIME=0.5 RUN_TAG=cleanloop_s03_cfg1p8_rt0p5 RUN_FOREGROUND=1 bash "$RUN_ONE" "$CKPT"
  run_or_note cleanloop_s03_cfg2p2_rt0p5 env CUDA_VISIBLE_DEVICES=0 SAMPLER=clean_loop STEPS=3  CFG=2.2 REFINE_TIME=0.5 RUN_TAG=cleanloop_s03_cfg2p2_rt0p5 RUN_FOREGROUND=1 bash "$RUN_ONE" "$CKPT"
  run_or_note cleanloop_s05_cfg2p2_rt0p5 env CUDA_VISIBLE_DEVICES=0 SAMPLER=clean_loop STEPS=5  CFG=2.2 REFINE_TIME=0.5 RUN_TAG=cleanloop_s05_cfg2p2_rt0p5 RUN_FOREGROUND=1 bash "$RUN_ONE" "$CKPT"
  run_or_note cleanloop_s10_cfg2p2_rt0p5 env CUDA_VISIBLE_DEVICES=0 SAMPLER=clean_loop STEPS=10 CFG=2.2 REFINE_TIME=0.5 RUN_TAG=cleanloop_s10_cfg2p2_rt0p5 RUN_FOREGROUND=1 bash "$RUN_ONE" "$CKPT"
  echo "[ablation] GPU0 queue done"
) &
pid0=$!

(
  echo "[ablation] GPU1 queue start"
  run_or_note heun_s20_cfg2p2 env CUDA_VISIBLE_DEVICES=1 SAMPLER=heun       STEPS=20 CFG=2.2 REFINE_TIME=0.5 RUN_TAG=heun_s20_cfg2p2          RUN_FOREGROUND=1 bash "$RUN_ONE" "$CKPT"
  run_or_note heun_s30_cfg2p2 env CUDA_VISIBLE_DEVICES=1 SAMPLER=heun       STEPS=30 CFG=2.2 REFINE_TIME=0.5 RUN_TAG=heun_s30_cfg2p2          RUN_FOREGROUND=1 bash "$RUN_ONE" "$CKPT"
  run_or_note heun_s20_cfg2p5 env CUDA_VISIBLE_DEVICES=1 SAMPLER=heun       STEPS=20 CFG=2.5 REFINE_TIME=0.5 RUN_TAG=heun_s20_cfg2p5          RUN_FOREGROUND=1 bash "$RUN_ONE" "$CKPT"
  run_or_note cleanloop_s03_cfg2p5_rt0p5 env CUDA_VISIBLE_DEVICES=1 SAMPLER=clean_loop STEPS=3  CFG=2.5 REFINE_TIME=0.5 RUN_TAG=cleanloop_s03_cfg2p5_rt0p5 RUN_FOREGROUND=1 bash "$RUN_ONE" "$CKPT"
  run_or_note cleanloop_s05_cfg1p8_rt0p5 env CUDA_VISIBLE_DEVICES=1 SAMPLER=clean_loop STEPS=5  CFG=1.8 REFINE_TIME=0.5 RUN_TAG=cleanloop_s05_cfg1p8_rt0p5 RUN_FOREGROUND=1 bash "$RUN_ONE" "$CKPT"
  echo "[ablation] GPU1 queue done"
) &
pid1=$!

echo "[ablation] worker pids: $pid0 $pid1"
wait "$pid0" "$pid1"
echo "[ablation] all done"
