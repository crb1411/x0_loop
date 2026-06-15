# YAML Config Guide

This document describes the training YAML schema used by `x0loop.train`.
Runtime YAML files such as `x0loop/configs/runtime/fsdp_checkpoint_compile.yaml`
are merged on top of the experiment YAML, so runtime keys may also appear in the
resolved config.

## Minimal Shape

Most configs follow this top-level layout:

```yaml
dataset: {}
model: {}
process: {}
schedule: {}
time_sampler: {}
loss: {}
augment: {}
train: {}
logging: {}
eval: {}
sample: {}
gen_eval: {}
```

## `dataset`

```yaml
dataset:
  name: cifar10
  root: /root/data/cifar10_data
  download: false
  split: train
```

Supported `name` values:

- `cifar10`: uses `torchvision.datasets.CIFAR10`.
- `mnist`: uses `torchvision.datasets.MNIST`.
- `imagefolder`: uses `torchvision.datasets.ImageFolder`.
- `tiny-imagenet` / `tiny_imagenet`: also uses `ImageFolder`.

Fields:

- `root`: dataset root.
- `download`: used by CIFAR10/MNIST.
- `split`: used by ImageFolder/Tiny-ImageNet when `root/split` exists.

The loader transform resizes and center-crops to `model.image_size`, converts to
tensor, then maps image values to `[-1, 1]`.

## `model`

```yaml
model:
  name: dit
  image_size: 32
  in_channels: 3
  out_channels: 3
  num_classes: 10
```

Supported `name` values:

- `dit`
- `unet`
- `jit`

Common fields:

- `image_size`: square input/output size.
- `in_channels`: usually `3` for RGB, `1` for MNIST.
- `out_channels`: usually same as `in_channels`.
- `num_classes`: `0` disables class conditioning. If positive, label id
  `num_classes` is reserved as the classifier-free guidance null label.
- `cond_dim`: optional external conditioning dimension, model-dependent.
- `dropout`: model dropout.

Typical DiT/JiT fields:

- `patch_size`
- `dim`
- `depth`
- `heads`
- `mlp_ratio`
- `norm_layer` for DiT: `layernorm` / `ln` / `rmsnorm` / `rms`

Typical UNet fields:

- `base_channels`
- `channel_mult`
- `num_res_blocks`
- `time_dim_mult`
- `attention_resolutions`
- `attention_heads`

Typical JiT-specific fields:

- `bottleneck_dim`
- `in_context_len`
- `in_context_start`

## `process`

```yaml
process:
  name: flow
  output_target: x0
  sampler: heun
```

Supported `name` values:

- `diffusion`
- `flow`

`process.name` must match `schedule.mode`.

`output_target` controls what the model directly predicts:

- `eps`: model predicts the noise endpoint.
- `x0`: model predicts the clean image endpoint.
- `v`: model predicts velocity `v = eps - x0`.

Aliases accepted by code for `v` include `velocity`, `flow`,
`flow_velocity`, and `u`.

Diffusion process samplers:

- `ddim`: deterministic endpoint reconstruction.
- `posterior`: VP posterior step with optional noise.

Diffusion-only field:

```yaml
posterior_noise_scale: 1.0
```

Flow process samplers:

- `euler`
- `heun`
- `auto` / `ddim`: accepted as compatibility aliases for Euler in process config.

## `schedule`

```yaml
schedule:
  mode: flow
  num_steps: 1000
  beta_min: 0.1
  beta_max: 20.0
```

Supported `mode` values:

- `diffusion`: VP diffusion schedule.
- `flow`: linear flow schedule.

Fields:

- `num_steps`: number of discrete schedule steps for legacy/discrete sampling.
- `beta_min`, `beta_max`: diffusion schedule parameters. They are present in
  default configs and mainly matter for diffusion schedules.

## `time_sampler`

`time_sampler` controls how training time `t` is sampled.

If omitted, the code uses:

```yaml
time_sampler:
  name: legacy
```

Supported names:

### `legacy` / `schedule`

```yaml
time_sampler:
  name: legacy
```

Uses `schedule.sample_t(...)`. This is the old/default schedule-driven sampler.

### `uniform` / `uniform_continuous` / `continuous`

```yaml
time_sampler:
  name: uniform_continuous
  min_t: 1.0e-5
  max_t: 1.0
```

Samples continuous `t ~ Uniform(min_t, max_t)`.

Use this for most flow matching runs when you want continuous-time training.

### `uniform_discrete` / `discrete`

```yaml
time_sampler:
  name: uniform_discrete
  num_steps: 1000
  min_step: 1
  max_step: 1000
```

Samples integer step ids uniformly, then returns:

```text
t = step / num_steps
```

### `logit_normal` / `logitnormal`

```yaml
time_sampler:
  name: logit_normal
  mean: 0.0
  std: 1.0
  min_t: 1.0e-5
  max_t: 0.99999
```

Samples:

```text
z ~ Normal(mean, std)
t = sigmoid(z)
```

Then clamps to `[min_t, max_t]`. This concentrates samples around the middle
when `mean=0,std=1`.

### `beta`

```yaml
time_sampler:
  name: beta
  alpha: 1.0
  beta: 1.0
  min_t: 1.0e-5
  max_t: 1.0
```

Samples from a Beta distribution and maps it into `[min_t, max_t]`.

Examples:

- `alpha=1,beta=1`: uniform.
- `alpha<1,beta<1`: more endpoints.

### Flow solver-grid mixing

For flow training only, any base `time_sampler` can mix in solver-grid times:

```yaml
time_sampler:
  name: logit_normal
  mean: -0.8
  std: 0.8
  grid_mix_prob: 0.2
  grid_steps: [50, 20]
```

For each sample independently:

```text
with probability 1 - grid_mix_prob: use the base sampler above
with probability grid_mix_prob: sample from the solver grids
```

`grid_steps: [50, 20]` builds the same model-evaluation times used by flow
solver runs with 50 and 20 steps:

```text
{1/50, 2/50, ..., 1} ∪ {1/20, 2/20, ..., 1}
```

The default excludes `t=0`, matching the flow sampler behavior where the final
endpoint is reached after the last step rather than by evaluating the model at
`t=0`. Set `grid_include_t0: true` only when the objective is known to be stable
at exactly zero.
- `alpha>1,beta>1`: more middle.
- `alpha>beta`: more large `t`.
- `alpha<beta`: more small `t`.

## `loss`

The preferred schema is:

```yaml
loss:
  outer_weight: none
  terms:
    - {target: v, formula: mse, coef: 1.0}
```

Each term defines a target-space loss. The final objective is:

```text
inner = sum_i coef_i * per_space_weight_i(t) * loss_i
loss = outer_weight(t) * inner
```

If `outer_weight: none`, the objective is unweighted outside the term losses.

### Term Fields

```yaml
terms:
  - target: v
    formula: mse
    coef: 1.0
    weight: none
```

`target` options:

- `eps`: compare predicted eps against sampled eps.
- `x0`: compare predicted clean image against input image.
- `v`: compare predicted velocity against `eps - x0`.

`formula` options:

- `mse`
- `l1`
- `huber`

Additional term fields:

- `coef`: scalar multiplier for this term.
- `delta`: Huber delta, used only when `formula: huber`.
- `weight`: optional per-term t-weight. This is an escape hatch; prefer
  `outer_weight` for simpler experiments.

### Outer Weight

```yaml
loss:
  outer_weight: none
```

`outer_weight` options:

- `none`
- `triangular`
- `skew_triangular`
- `p2`
- `min_snr`
- `edm`

Weight parameters:

```yaml
outer_weight_power: 2.0
outer_weight_floor: 0.0
outer_weight_skew: 0.5
outer_weight_p2_k: 1.0
outer_weight_p2_gamma: 1.0
outer_weight_sigma_data: 0.5
min_snr_gamma: 5.0
balance_integral_steps: 2000
```

For per-term `weight`, the same parameters are read as:

```yaml
weight_power: 2.0
weight_floor: 0.0
weight_skew: 0.5
p2_k: 1.0
p2_gamma: 1.0
sigma_data: 0.5
gamma: 5.0
balance_integral_steps: 2000
```

If `weight_power` / `weight_floor` are absent, the builder falls back to
`outer_weight_power` / `outer_weight_floor`.

### Time Weights

All active loss weights are functions of `t`. Shape-based weights use:

```text
raw(t) = floor + (1 - floor) * base(t)^power
```

By default they are mean-normalized, so their average over uniform
`t in [0,1]` is approximately `1`.

`triangular`:

```text
base(t) = 1 - |2t - 1|
```

`skew_triangular`:

```text
base(t) = 1 - |2t - 1|
w(t) = normalized(base(t)^power * (1 + skew * (2t - 1)))
```

Positive `skew` makes the high-`t` side larger; negative `skew` makes the
low-`t` side larger.

Example:

```yaml
loss:
  outer_weight: skew_triangular
  outer_weight_power: 1.0
  outer_weight_floor: 0.0
  outer_weight_skew: 0.5
```

Gives:

```text
w(t) approx 2 * (1 - |2t - 1|) * (1 + 0.5 * (2t - 1))
```

because the unnormalized triangle has mean about `0.5`.

`p2`:

```text
w(t) = normalized((p2_k + snr(t))^-p2_gamma)
```

`min_snr`:

```text
w(t) = min(snr(t), gamma) / snr(t)
```

Use:

```yaml
min_snr_gamma: 5.0
```

or per-term:

```yaml
gamma: 5.0
```

`edm`:

```text
sigma = sigma(t) / alpha(t)
w(t) = normalized((sigma^2 + sigma_data^2) / (sigma * sigma_data)^2)
```

## `augment`

Current CIFAR10 configs are usually:

```yaml
augment:
  name: geom
  mode: data_only
  hflip_prob: 0.5
  max_translation: 0
  crop_min_scale: 1.0
  enable_crop_resize: false
```

Supported `name` values:

- `none`
- `geom`
- `dit`
- `dit_original`
- `strongaugment`
- `strong`

`mode` currently only supports:

- `data_only`

`geom` fields:

- `hflip_prob`: horizontal flip probability.
- `max_translation`: pixel roll shift amount.
- `crop_min_scale`: minimum crop scale.
- `enable_crop_resize`: whether random crop-resize is enabled.
- `random_crop_position`: optional; default differs by mode.

`strongaugment` adds:

- `crop_max_scale`
- `crop_min_ratio`
- `crop_max_ratio`
- `brightness`
- `contrast`
- `saturation`
- `grayscale_prob`
- `erasing_prob`
- `erase_min_scale`
- `erase_max_scale`

## `train`

```yaml
train:
  seed: 42
  deterministic: false
  epochs: 1000
  batch_size: 256
  gradient_accumulation_steps: 1
  class_dropout_prob: 0.1
  num_workers: 8
  lr: 1.0e-4
  lr_scheduler:
    name: cosine
    max_lr: 1.0e-4
    min_lr: 3.0e-5
    warmup_steps: 10000
    cosine_steps: 100000
  weight_decay: 0.04
  max_clip_grad: 10.0
  use_ema: true
  ema_decay: 0.999
  resume: null
```

Fields:

- `seed`: base random seed.
- `deterministic`: toggles deterministic backend behavior where supported.
- `epochs`: number of epochs.
- `batch_size`: per-process dataloader batch size.
- `gradient_accumulation_steps`: optimizer step every N micro-batches.
- `class_dropout_prob`: classifier-free guidance label dropout probability.
- `num_workers`: dataloader workers.
- `lr`: fallback learning rate.
- `lr_scheduler.name`: `none` / `constant` / `cosine`.
- `weight_decay`: AdamW weight decay.
- `max_clip_grad`: gradient clipping threshold.
- `use_ema`: maintain EMA copy.
- `ema_decay`: EMA decay.
- `resume`: checkpoint path or `null`.

## `logging`

```yaml
logging:
  log_every: 100
  window_size: 100
  use_tb: true
  sample_every: 10000
  sample_rank0_only: true
```

Fields:

- `log_every`: optimizer-step interval for training logs.
- `window_size`: moving window used by `MetricLogger`.
- `use_tb`: enable TensorBoard logging.
- `sample_every`: optimizer-step interval for sample images.
- `sample_rank0_only`: if true, only rank 0 writes sample images unless FSDP
  requires collective forward.
- `out_dir`: optional. If omitted, it is generated automatically.

Automatic output directory:

```text
runs/{dataset}/{process}/{model}/{output_target}target_{loss_targets}loss_{sampler}/{timestamp}
```

Example:

```text
runs/cifar10/flow/dit/x0target_vloss_heun/20260602_113637
```

## `eval`

```yaml
eval:
  enabled: true
  every_steps: 1000
  max_batches: 8
  batch_size: 256
  num_workers: 4
```

Fields:

- `enabled`: run evaluation periodically.
- `every_steps`: optimizer-step interval.
- `max_batches`: maximum eval batches; use `all` or omit for full eval loader.
- `batch_size`: eval batch size.
- `num_workers`: eval dataloader workers.

Eval logs include:

- `eval/loss_weighted`
- `eval/loss_no_weight`
- `eval/loss_outer_weight`
- `eval/loss_eps`
- `eval/loss_x0`
- `eval/loss_v`
- `eval/summary`

## `gen_eval`

`gen_eval` runs expensive generation metrics during training. It generates a
fixed number of images, computes metrics with `torch_fidelity`, then appends one
row to a separate jsonl file:

```text
<run_dir>/gen_eval_metrics_<timestamp>.jsonl
```

Default CIFAR10 flow configs use:

```yaml
gen_eval:
  enabled: true
  every_steps: 10000
  num_samples: 5000
  batch_size: 128
  steps: 20
  sampler: heun
  guidance_scale: 3.0
  input2: cifar10-train
  datasets_root: /root/data/cifar10_data
  datasets_download: false
  keep_images: false
  metrics:
    isc: true
    fid: true
    kid: true
    ppl: false
    prc: true
    mind: true
  final:
    enabled: true
    num_samples: 20000
    steps: 50
    sampler: heun
```

Fields:

- `enabled`: enable generation metrics during training.
- `every_steps`: optimizer-step interval. CIFAR10 ablation configs use `10000`.
- `num_samples`: number of fake images used for metrics.
- `batch_size`: generation batch size.
- `steps`: sampling/integration steps.
- `sampler`: sampling method. For flow, use `euler` or `heun`.
- `guidance_scale`: classifier-free guidance scale.
- `input2`: torch-fidelity real-data reference. CIFAR10 commonly uses
  `cifar10-train`.
- `datasets_root`: root for torch-fidelity registered datasets. For CIFAR10,
  point this to the same root as `dataset.root` to avoid re-downloading.
- `datasets_download`: whether torch-fidelity may download registered datasets.
  CIFAR10 ablation configs use `false`.
- `fid_statistics_file`: optional precomputed FID statistics path. Use this
  instead of `input2` when comparing against prepared reference stats.
- `keep_images`: keep generated PNGs under
  `<run_dir>/gen_eval/step_<step>/fake`. If false, remove them after metrics.
- `cache`, `cache_root`, `verbose`: forwarded to `torch_fidelity`.
- `metrics.isc`, `metrics.fid`, `metrics.kid`, `metrics.ppl`,
  `metrics.prc`, `metrics.mind`: metric switches forwarded to
  `torch_fidelity`.
- `final.enabled`: run one full generation metrics pass after training
  completes.
- `final.num_samples`: number of fake images for the final full pass. CIFAR10
  ablation configs use `20000`.
- `final.steps`: sampling steps for the final full pass. CIFAR10 ablation
  configs use `50`.
- `final.sampler`: sampler for the final full pass. CIFAR10 ablation configs
  use `heun`.

Notes:

- This is different from `eval`, which measures validation losses.
- Periodic `gen_eval` is intentionally cheaper (`5000` samples, `20` sampling
  steps). Final `gen_eval` is more complete (`20000` samples, `50` sampling
  steps, Heun).
- `torch_fidelity` must be installed in the training environment.
- PPL requires `torch_fidelity` to receive a `GenerativeModelBase` or wrapped
  generator model. x0loop `gen_eval` currently passes a generated image
  directory, so `metrics.ppl` defaults to `false`.

## `sample`

```yaml
sample:
  steps: 50
  num: 5
  sampler: heun
  guidance_scale: 1.0
  posterior_noise_scale: 1.0
  save_trace: false
```

Fields:

- `steps`: number of reverse/integration steps used for image generation.
- `num`: number of samples to generate.
- `sampler`: sampling method override.
- `guidance_scale`: classifier-free guidance scale.
- `posterior_noise_scale`: only used by diffusion posterior sampling.
- `save_trace`: whether to save the whole denoising trajectory.
- `class_labels`: optional list of labels to sample.
- `class_names`: optional list of label names used in filenames.
- `use_batch_cond`: if true, use labels from the current training batch.

Sampler options:

For diffusion:

- `auto`: use `process.sampler`.
- `ddim`
- `posterior`

For flow:

- `auto`: use `process.sampler`.
- `euler`
- `heun`

The common flow sample block:

```yaml
sample:
  steps: 50
  num: 5
  sampler: heun
  guidance_scale: 1.0
```

means:

- generate `5` images;
- integrate the flow ODE with `50` steps;
- use Heun predictor-corrector sampling;
- use no classifier-free guidance amplification because `guidance_scale=1.0`.

Classifier-free guidance behavior:

- If `model.num_classes <= 0`, conditioning is disabled.
- If `guidance_scale == 1.0`, CFG is disabled even when labels exist.
- If `guidance_scale > 1.0`, the sampler runs both conditional and null-label
  unconditional predictions and combines them.

Default labels:

- If `sample.class_labels` is omitted and class conditioning is enabled, sample
  labels cycle deterministically through class ids.
- For CIFAR10, filenames include readable labels such as `_ybird`.

## Common Examples

### Unweighted Flow `v` Loss With Heun Sampling

```yaml
process:
  name: flow
  output_target: x0
  sampler: heun

schedule:
  mode: flow

time_sampler:
  name: uniform_continuous

loss:
  outer_weight: none
  terms:
    - {target: v, formula: mse, coef: 1.0}

sample:
  steps: 50
  num: 5
  sampler: heun
  guidance_scale: 1.0
```

This trains an x0-output model using an unweighted velocity-space loss:

```text
loss = mse(v_pred, eps - x0)
```

and samples with Heun.

### Mean-Normalized Triangular Outer Weight

```yaml
loss:
  outer_weight: triangular
  outer_weight_power: 1.0
  outer_weight_floor: 0.0
  terms:
    - {target: v, formula: mse, coef: 1.0}
```

This gives:

```text
w(t) = 2 * (1 - |2t - 1|)
loss = w(t) * mse(v_pred, eps - x0)
```

### Logit-Normal Time Sampling

```yaml
time_sampler:
  name: logit_normal
  mean: 0.0
  std: 1.0
  min_t: 1.0e-5
  max_t: 0.99999
```

This samples more training points near the middle of the interval than uniform
sampling.
