# CIFAR10 Flow x0-target v-loss Ablations

This directory contains the default CIFAR10 ablation entry points.

Shared setup:

- process: `flow`
- model output: `output_target=x0`
- training loss: single `v` term, `mse`, where `v = eps - x0`
- sampler: `heun`
- epochs: `300`
- augment: horizontal flip only (`hflip_prob=0.5`)
- plots: generated automatically after successful training completion

The run directory is forced to:

```text
runs/ablations/cifar10_flow_x0_vloss/{model}/{ablation}/{timestamp}
```

## Scripts

DiT:

```bash
bash train_run/ablations/cifar10_flow_x0_vloss/train_dit_unweighted.sh
bash train_run/ablations/cifar10_flow_x0_vloss/train_dit_v_weight.sh
bash train_run/ablations/cifar10_flow_x0_vloss/train_dit_x0_weight.sh
bash train_run/ablations/cifar10_flow_x0_vloss/train_dit_logitnormal_time.sh
```

JiT:

```bash
bash train_run/ablations/cifar10_flow_x0_vloss/train_jit_unweighted.sh
bash train_run/ablations/cifar10_flow_x0_vloss/train_jit_v_weight.sh
bash train_run/ablations/cifar10_flow_x0_vloss/train_jit_x0_weight.sh
bash train_run/ablations/cifar10_flow_x0_vloss/train_jit_logitnormal_time.sh
```

All scripts accept an optional checkpoint first, then extra `x0loop.train` args:

```bash
bash train_run/ablations/cifar10_flow_x0_vloss/train_jit_unweighted.sh /path/to/ckpt.pt --set train.epochs=50
```

## Ablation Meaning

- `unweighted`: `loss.outer_weight=none`, uniform continuous time.
- `v_weight`: `loss.outer_weight=target`, with the single term target `v`, so the weight is the mean-normalized mid-time triangle.
- `x0_weight`: `loss.outer_weight=x0`, emphasizing low-noise / clean-ish times.
- `logitnormal_time`: unweighted loss, but time sampled from `logit_normal(mean=0,std=1)`.

