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
post_eval: {}
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
- `x0`
- `eps`
- `v`
- `target`
- `snr`
- `inv_snr`
- `logsnr`
- `min_snr`
- `balance_weights`

Aliases:

- `t_x0`, `x0_t` -> `x0`
- `t_eps`, `eps_t` -> `eps`
- `t_mid`, `v_t` -> `v`
- `target_default`, `auto_target` -> `target`

Weight parameters:

```yaml
outer_weight_power: 2.0
outer_weight_floor: 0.0
min_snr_gamma: 5.0
balance_factor: 0.5
balance_time: auto
balance_integral_steps: 2000
```

For per-term `weight`, the same parameters are read as:

```yaml
weight_power: 2.0
weight_floor: 0.0
gamma: 5.0
balance_factor: 0.5
balance_time: auto
balance_integral_steps: 2000
```

If `weight_power` / `weight_floor` are absent, the builder falls back to
`outer_weight_power` / `outer_weight_floor`.

### Polynomial Weights: `x0`, `eps`, `v`

Polynomial weights use:

```text
raw(t) = floor + (1 - floor) * base(t)^power
```

By default they are mean-normalized, so their average over uniform
`t in [0,1]` is approximately `1`.

`x0` weight:

```text
base(t) = 1 - t
```

Large near `t=0`, small near `t=1`.

`eps` weight:

```text
base(t) = t
```

Small near `t=0`, large near `t=1`.

`v` weight:

```text
base(t) = 1 - |2t - 1|
```

Triangular middle-time weight. Large near `t=0.5`, small near both endpoints.

Example:

```yaml
loss:
  outer_weight: v
  outer_weight_power: 1.0
  outer_weight_floor: 0.0
```

Gives:

```text
w(t) = 2 * (1 - |2t - 1|)
```

because the unnormalized triangle has mean `0.5`.

With:

```yaml
outer_weight: x0
outer_weight_power: 2.0
outer_weight_floor: 0.0
```

the normalized weight is:

```text
w(t) = 3 * (1 - t)^2
```

### `target`

```yaml
outer_weight: target
```

When there is exactly one loss term, `target` selects the polynomial weight
matching that term target:

- term `target: x0` -> `x0` weight
- term `target: eps` -> `eps` weight
- term `target: v` -> `v` weight

When there are multiple terms, there is no single primary target, so `target`
falls back to all-ones.

### Schedule Weights

`snr`:

```text
w(t) = snr(t)
```

`inv_snr`:

```text
w(t) = 1 / snr(t)
```

`logsnr`:

```text
w(t) = log(snr(t))
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

### `balance_weights`

```yaml
outer_weight: balance_weights
balance_factor: 0.5
balance_time: auto
balance_integral_steps: 2000
```

This computes an alpha-based balancing factor:

```text
w(t) = (1 - balance_factor) + balance_factor * alpha(t) * mean(1 / alpha(t))
```

`balance_time` options:

- `auto`: infer whether batch times look discrete.
- `discrete`: average over discrete schedule steps.
- `continuous`: numerical integral over `balance_integral_steps` midpoint samples.

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
  every_steps: 5000
  num_samples: 50000
  batch_size: 64
  steps: 50
  sampler: heun
  guidance_scale: 3.0
  input2: cifar10-train
  keep_images: false
  metrics:
    isc: true
    fid: true
    kid: true
    ppl: true
    prc: true
    mind: true
```

Fields:

- `enabled`: enable generation metrics during training.
- `every_steps`: optimizer-step interval. Default for this eval path is `5000`.
- `num_samples`: number of fake images used for metrics.
- `batch_size`: generation batch size.
- `steps`: sampling/integration steps.
- `sampler`: sampling method. For flow, use `euler` or `heun`.
- `guidance_scale`: classifier-free guidance scale.
- `input2`: torch-fidelity real-data reference. CIFAR10 commonly uses
  `cifar10-train`.
- `fid_statistics_file`: optional precomputed FID statistics path. Use this
  instead of `input2` when comparing against prepared reference stats.
- `keep_images`: keep generated PNGs under
  `<run_dir>/gen_eval/step_<step>/fake`. If false, remove them after metrics.
- `cache`, `cache_root`, `verbose`: forwarded to `torch_fidelity`.
- `metrics.isc`, `metrics.fid`, `metrics.kid`, `metrics.ppl`,
  `metrics.prc`, `metrics.mind`: metric switches forwarded to
  `torch_fidelity`.

Notes:

- This is different from `eval`, which measures validation losses.
- This is different from `post_eval`, which runs once at the end and writes a
  YAML manifest.
- `torch_fidelity` must be installed in the training environment.
- PPL support depends on the torch-fidelity input mode. It is enabled by
  default here because the ablation request asks for it explicitly; if the
  installed torch-fidelity build rejects directory input for PPL, disable
  `metrics.ppl`.

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

## `post_eval`

`post_eval` runs once after training finishes successfully. It is the x0loop
equivalent of JiT's generation eval path: switch the model to eval mode, copy
EMA weights when enabled, generate images, then write a YAML manifest with the
exact sampling settings and artifact paths.

```yaml
post_eval:
  enabled: true
  steps: 50
  num: 50
  batch_size: 50
  sampler: heun
  guidance_scale: 3.0
  save_images: true
  save_grid: true
```

Fields:

- `enabled`: run generation eval after the last training epoch.
- `steps`: number of sampling/integration steps.
- `num`: number of images to generate.
- `batch_size`: generation batch size.
- `sampler`: sampling method. For flow, use `euler` or `heun`; default is
  `heun`.
- `guidance_scale`: classifier-free guidance scale. The post-train default is
  `3.0`.
- `posterior_noise_scale`: optional diffusion posterior noise scale.
- `save_images`: save individual PNG files.
- `save_grid`: save one grid PNG.
- `out_dir`: optional output directory. If omitted, uses
  `<run_dir>/post_eval`.
- `class_labels`: optional fixed labels.
- `class_names`: optional label names used in filenames.

Artifacts:

- individual images: `<run_dir>/post_eval/images/sample_000000_yairplane_x0loop.png`
- grid: `<run_dir>/post_eval/grid.png`
- manifest: `<run_dir>/post_eval/post_eval.yaml`

The manifest records the dataset/model/process/loss sections, sampling config,
generated artifact paths, and a `metrics: {}` field reserved for future FID/IS
results. Current x0loop post eval does not compute FID by default because the
CIFAR10 reference statistics path is not yet part of the config.

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

### Mean-Normalized x0 Outer Weight

```yaml
loss:
  outer_weight: x0
  outer_weight_power: 2.0
  outer_weight_floor: 0.0
  terms:
    - {target: x0, formula: mse, coef: 1.0}
```

This gives:

```text
w(t) = 3 * (1 - t)^2
loss = w(t) * mse(x0_pred, x0)
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
