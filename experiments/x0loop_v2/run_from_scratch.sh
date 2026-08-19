#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 fresh|fresh-fixed-repro|fresh-time|bank-fix|bank-x0|online|online-x0|denoise-gan|terminal-gan" >&2
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
adv_batch_ratio=${X0LOOP_ADV_BATCH_RATIO:-0.125}
adv_gradient_ratio=${X0LOOP_ADV_GRADIENT_RATIO:-0.1}
adv_start_step=${X0LOOP_ADV_START_STEP:-10000}
adv_warmup_steps=${X0LOOP_ADV_WARMUP_STEPS:-1000}

case "$branch" in
  fresh)
    clean_enabled=false
    clean_mode=bank_fix
    branch_aux_target=velocity
    adv_enabled=false
    adv_fake_space=x0_hat
    model_ignore_time=true
    time_jitter_enabled=true
    ;;
  fresh-fixed-repro)
    clean_enabled=false
    clean_mode=bank_fix
    branch_aux_target=velocity
    adv_enabled=false
    adv_fake_space=x0_hat
    model_ignore_time=true
    time_jitter_enabled=true
    ;;
  fresh-time)
    clean_enabled=false
    clean_mode=bank_fix
    branch_aux_target=velocity
    adv_enabled=false
    adv_fake_space=x0_hat
    model_ignore_time=false
    time_jitter_enabled=false
    ;;
  bank-fix)
    clean_enabled=true
    clean_mode=bank_fix
    branch_aux_target=velocity
    adv_enabled=false
    adv_fake_space=x0_hat
    model_ignore_time=true
    time_jitter_enabled=true
    ;;
  bank-x0)
    clean_enabled=true
    clean_mode=bank_fix
    branch_aux_target=x0
    adv_enabled=false
    adv_fake_space=x0_hat
    model_ignore_time=true
    time_jitter_enabled=true
    ;;
  online)
    clean_enabled=true
    clean_mode=online
    branch_aux_target=velocity
    adv_enabled=false
    adv_fake_space=x0_hat
    model_ignore_time=true
    time_jitter_enabled=true
    ;;
  online-x0)
    clean_enabled=true
    clean_mode=online
    branch_aux_target=x0
    adv_enabled=false
    adv_fake_space=x0_hat
    model_ignore_time=true
    time_jitter_enabled=true
    ;;
  denoise-gan)
    clean_enabled=false
    clean_mode=bank_fix
    branch_aux_target=velocity
    adv_enabled=true
    adv_fake_space=x0_hat
    model_ignore_time=true
    time_jitter_enabled=true
    ;;
  terminal-gan)
    clean_enabled=false
    clean_mode=bank_fix
    branch_aux_target=velocity
    adv_enabled=true
    adv_fake_space=terminal_x0
    model_ignore_time=true
    time_jitter_enabled=true
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
  --set "model_conditioning.ignore_time=$model_ignore_time" \
  --set "time_condition_jitter.enabled=$time_jitter_enabled" \
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
  --set "adversarial.enabled=$adv_enabled" \
  --set "adversarial.fake_space=$adv_fake_space" \
  --set "adversarial.batch_ratio=$adv_batch_ratio" \
  --set "adversarial.gradient_ratio=$adv_gradient_ratio" \
  --set "adversarial.start_step=$adv_start_step" \
  --set "adversarial.warmup_steps=$adv_warmup_steps" \
  --set "gen_eval.enabled=$fid_enabled" \
  --set "gen_eval.final.enabled=$final_fid_enabled" \
  --set "gen_eval.every_steps=$fid_every" \
  --set "gen_eval.num_samples=$fid_samples" \
  --set "gen_eval.final.num_samples=$final_fid_samples" \
  --set "distributed.checkpoint.every_steps=$checkpoint_every" \
  --set logging.sample_every=0 \
  --set "logging.out_dir=runs/x0loop_v2_from_scratch/$cycle/$branch"
