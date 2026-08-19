#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 fresh|bank-fix|bank-x0|online|online-x0" >&2
  exit 2
fi

branch=$1
gpu=${X0LOOP_GPU:-7}
cycle=${X0LOOP_CYCLE:-cycle01}
epochs=${X0LOOP_EPOCHS:-300}
run_steps=${X0LOOP_RUN_STEPS:-None}
resume=${X0LOOP_RESUME:-None}
fid_every=${X0LOOP_FID_EVERY:-15000}
checkpoint_every=${X0LOOP_CHECKPOINT_EVERY:-5000}
fid_samples=${X0LOOP_FID_SAMPLES:-5000}
final_fid_samples=${X0LOOP_FINAL_FID_SAMPLES:-50000}
fid_enabled=${X0LOOP_FID_ENABLED:-true}
final_fid_enabled=${X0LOOP_FINAL_FID_ENABLED:-true}
clean_warmup=${X0LOOP_CLEAN_WARMUP:-10000}
compile_enabled=${X0LOOP_COMPILE_ENABLED:-true}
compile_mode=${X0LOOP_COMPILE_MODE:-default}
compile_dynamic=${X0LOOP_COMPILE_DYNAMIC:-false}
aux_batch_ratio=${X0LOOP_AUX_BATCH_RATIO:-0.125}
aux_gradient_ratio=${X0LOOP_AUX_GRADIENT_RATIO:-0.2}
aux_target=${X0LOOP_AUX_TARGET:-}

case "$branch" in
  fresh)
    clean_enabled=false
    clean_mode=bank_fix
    branch_aux_target=velocity
    ;;
  bank-fix)
    clean_enabled=true
    clean_mode=bank_fix
    branch_aux_target=velocity
    ;;
  bank-x0)
    clean_enabled=true
    clean_mode=bank_fix
    branch_aux_target=x0
    ;;
  online)
    clean_enabled=true
    clean_mode=online
    branch_aux_target=velocity
    ;;
  online-x0)
    clean_enabled=true
    clean_mode=online
    branch_aux_target=x0
    ;;
  *)
    echo "unknown branch: $branch" >&2
    exit 2
    ;;
esac

if [[ -z "$aux_target" ]]; then
  aux_target=$branch_aux_target
fi

git_commit=$(git rev-parse HEAD)
git_dirty=false
if [[ -n "$(git status --porcelain)" ]]; then
  git_dirty=true
fi

CUDA_VISIBLE_DEVICES=$gpu uv run python -m x0loop.train \
  --config experiments/x0loop_v2/config.yaml \
  --runtime-config x0loop/configs/runtime/single_gpu.yaml \
  --set research.principles_version=v1 \
  --set "research.launch_branch=$branch" \
  --set "research.git_commit=$git_commit" \
  --set "research.git_dirty=$git_dirty" \
  --set "compile.enabled=$compile_enabled" \
  --set "compile.mode=$compile_mode" \
  --set "compile.dynamic=$compile_dynamic" \
  --set "train.resume=$resume" \
  --set "train.epochs=$epochs" \
  --set "train.run_steps=$run_steps" \
  --set train.lr=0.0003 \
  --set train.lr_scheduler.name=cosine \
  --set train.lr_scheduler.max_lr=0.0003 \
  --set train.lr_scheduler.min_lr=0.00005 \
  --set train.lr_scheduler.warmup_steps=10000 \
  --set train.lr_scheduler.cosine_steps=48500 \
  --set "clean_loop.enabled=$clean_enabled" \
  --set "clean_loop.mode=$clean_mode" \
  --set "clean_loop.warmup_steps=$clean_warmup" \
  --set "clean_loop.aux_batch_ratio=$aux_batch_ratio" \
  --set "clean_loop.aux_gradient_ratio=$aux_gradient_ratio" \
  --set "clean_loop.aux_target=$aux_target" \
  --set "gen_eval.enabled=$fid_enabled" \
  --set "gen_eval.final.enabled=$final_fid_enabled" \
  --set "gen_eval.every_steps=$fid_every" \
  --set "gen_eval.num_samples=$fid_samples" \
  --set "gen_eval.final.num_samples=$final_fid_samples" \
  --set "distributed.checkpoint.every_steps=$checkpoint_every" \
  --set logging.sample_every=0 \
  --set "logging.out_dir=runs/x0loop_v2_from_scratch/$cycle/$branch"
