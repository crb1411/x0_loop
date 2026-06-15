# 615 Learnable Endpoint Ablations

This suite runs six JiT CIFAR10 flow experiments. Each experiment has its own
folder with a complete `config.yaml` and standalone `run.sh`.

Common setup:

- `process.output_target=x0`
- main loss term: `v` MSE
- `time_sampler=logit_normal(mean=0, std=1)`
- target outer weight with power `1.0`
- `time_condition_jitter mean=0.02 std=0.02`
- learnable terminal endpoint `z = (1 - beta) * mu_data + beta * eps`
- CIFAR10 standard normalization with inverse normalization before sample/FID image export

Experiments:

- `scheme2_beta0p2`: x0 head only, beta `0.2`
- `scheme2_beta0p5`: x0 head only, beta `0.5`
- `scheme2_beta0p8`: x0 head only, beta `0.8`
- `scheme1_beta0p2`: x0 head plus mudata head, beta `0.2`
- `scheme1_beta0p5`: x0 head plus mudata head, beta `0.5`
- `scheme1_beta0p8`: x0 head plus mudata head, beta `0.8`

Run all experiments, two single-GPU jobs at a time:

```bash
bash train_run/ablations/615_learnable_endpoint/run_all_2gpu.sh
```

Run one experiment manually:

```bash
CUDA_VISIBLE_DEVICES=0 bash train_run/ablations/615_learnable_endpoint/scheme2_beta0p5/run.sh
```
