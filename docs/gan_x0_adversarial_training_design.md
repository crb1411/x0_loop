# x0_hat 对抗训练设计草案

本文档设计一个训练期 GAN 增强方案，用于当前 CIFAR10 flow / JiT 的
`x0` 预测训练。核心目标不是替代现有回归目标，而是在模型已经从
`x_t` 预测出 `x0_hat` 后，用一个轻量判别器约束 `x0_hat` 落在自然图像
流形上。

## 1. 当前基线

当前 JiT ablation 中最强的完整结果是：

```text
jit / logitnormal_target_weight / no-grid
FID       10.187 @ 58500
IS        9.577
precision 0.387
recall    0.745
```

对应训练配置：

```bash
--set "loss.outer_weight=target"
--set "loss.outer_weight_power=1.0"
--set "loss.outer_weight_floor=0.0"
--set "time_sampler.name=logit_normal"
--set "time_sampler.mean=0.0"
--set "time_sampler.std=1.0"
```

这个 baseline 应作为第一轮 GAN 实验的唯一基线。不要第一轮就混入
gridmix 或 time-condition jitter，否则变量耦合后很难判断 GAN 是否有效。

## 2. 训练语义

当前 flow 语义是：

```text
x_t = (1 - t) * x0 + t * eps
t = 0: clean data endpoint
t = 1: Gaussian prior endpoint
```

现有模型路径是：

```text
fb = process.forward_sample(x0, t)
out = denoiser.forward(fb.xt, t_model)
x0_hat = process.x0_from_output(fb.xt, fb.t, out, aux={})
```

其中 `t_model` 可能是原始 `fb.t`，也可能是训练期
`time_condition_jitter` 后的条件时间。GAN 分支必须遵守以下规则：

```text
真实物理时间: fb.t
模型条件时间: t_model
```

`fb.t` 必须继续用于：

- `process.forward_sample`
- `process.x0_from_output / eps_from_output / v_from_output`
- base loss target conversion
- loss weight / outer weight
- t-bin metrics
- GAN 判别器条件时间

不要用 jitter 后的 `t_model` 做 target conversion、loss weighting 或
GAN real/fake pairing。

## 3. 判别器目标

第一版只判别 clean image manifold：

```text
real: D(x0,     t, cond)
fake: D(x0_hat, t, cond)
```

`real` 也喂同一个 batch 的 `t`。`x0` 本身不依赖 `t`，但这样能避免
conditional discriminator 把 real/fake 的条件分布差异当作捷径。

不建议第一版输入 `xt`：

```text
D(x0_hat, xt, t)
```

这会变成 endpoint-consistency critic，容易在低 `t` 时利用 `xt` 接近
`x0` 的捷径。第一版应先做 `D(x_candidate, t, cond)`。

## 4. 判别器架构

建议新增：

```text
x0loop/models/discriminator.py
```

第一版用轻量 conditional ResNet discriminator：

```text
input:  x_candidate [B, 3, 32, 32]
        t           [B]
        cond        [B] optional class label

stem:
  SNConv3x3 3 -> 16, LeakyReLU

blocks:
  ResDiscBlock 16  -> 32,  stride=2   # 16x16
  ResDiscBlock 32  -> 64,  stride=2   # 8x8
  ResDiscBlock 64  -> 128, stride=2   # 4x4
  ResDiscBlock 128 -> 128, stride=1   # 4x4

head:
  LeakyReLU
  global sum/avg pool -> [B, 128]
  linear -> scalar logit [B]
```

每个 block：

```text
SNConv3x3 -> LeakyReLU -> SNConv3x3
skip: identity or SNConv1x1
```

建议默认：

- 使用 `torch.nn.utils.spectral_norm`
- 不使用 BatchNorm
- 第一版不加 FiLM，避免判别器过强
- class-conditional 时使用 projection class embedding
- time-conditional 时使用 projection time embedding

条件 head：

```text
h = pooled_feature(x)
logit = linear(h)
logit += sum(h * t_proj(t))
logit += sum(h * y_embed(y))   # optional
```

估计开销：

```text
base_channels=16:
  params       roughly 0.4M - 0.5M
  forward MACs roughly 10M - 12M / image

每步额外包含:
  D(real)
  D(fake.detach())
  D(fake) for G adversarial loss
```

这个规模相对当前 JiT/DiT 训练可控，但仍需记录吞吐下降。

## 5. Loss 设计

第一版推荐 hinge loss：

```text
L_D_real = mean(relu(1 - D(x0, t, cond)))
L_D_fake = mean(relu(1 + D(x0_hat.detach(), t, cond)))
L_D      = L_D_real + L_D_fake + R1

L_G_adv  = -mean(D(x0_hat, t, cond))
L_G      = L_base + lambda_adv(step) * E_t[w_G(t) * L_G_adv_per_sample]
```

可选 logistic non-saturating：

```text
L_D = softplus(-D_real) + softplus(D_fake)
L_G_adv = softplus(-D_fake_for_G)
```

首轮建议用 hinge；如果 D 很快饱和，再切 logistic。

## 6. t-aware 权重

核心问题：`t` 大时 `x_t` 接近噪声，`x0_hat` 很容易不像自然图像。
如果让这些样本强训 GAN，判别器会很快学会容易的 fake，给生成器的梯度
变成“从几乎纯噪声硬 hallucinate 真实图像”，反而破坏 flow 回归目标。

因此 adversarial loss 应主要作用在 `x0_hat` 已经有图像结构的区域。

推荐第一版使用分段权重：

```text
t in [0.00, 0.05): 0.25
t in [0.05, 0.35): 1.00
t in [0.35, 0.65): 0.50
t in [0.65, 1.00]: 0.05
```

或者使用 smooth gate：

```text
g_clean(t)    = (1 - t)^p
g_low_cut(t)  = sigmoid((t - t_min) / tau)
g_high_cut(t) = sigmoid((t_max - t) / tau)
w_G(t)        = normalize(g_clean * g_low_cut * g_high_cut)
```

建议初值：

```text
t_min = 0.02 - 0.05
t_max = 0.55 - 0.75
p     = 1.0 - 2.0
tau   = 0.03 - 0.08
```

第一版更建议分段权重，容易解释和调试。D 和 G 可以共用同一个
`w_adv(t)`；如果 D 过强，再让 D 的高 `t` 权重更低。

## 7. 更新与 detach 策略

D step：

```text
with torch.no_grad() or detach:
    x0_hat = process.x0_from_output(fb.xt, fb.t, out, aux={})

D_real = D(x0, fb.t, cond)
D_fake = D(x0_hat.detach(), fb.t, cond)
backward L_D
step d_optimizer
```

G step：

```text
x0_hat = process.x0_from_output(fb.xt, fb.t, out, aux={})
freeze D params
D_fake_for_G = D(x0_hat, fb.t, cond)
L_G = L_base + lambda_adv * weighted_adv_loss
backward L_G
step generator optimizer
```

实现上应注意：

- D 更新时 fake 必须 detach。
- G adversarial 更新时不要 detach `x0_hat`。
- G adversarial 前向时冻结 D 参数，但保留从 D 输出到 `x0_hat` 的梯度。
- GAN 不应并入 `CompositeLoss`，因为它需要 D、detach、独立 optimizer、
  独立 checkpoint。

建议默认：

```text
D:G = 1:1
start_step = 10000
lambda_adv warmup_steps = 10000
R1 lazy interval = 16
```

## 8. 配置草案

建议新增配置段：

```yaml
adversarial:
  enabled: false
  fake_space: x0_hat
  loss: hinge
  weight: 0.02
  start_step: 10000
  warmup_steps: 10000
  update_every: 1
  d_steps: 1
  condition_on_t: true
  condition_on_class: true
  clamp_fake_for_d: false

  t_weight:
    name: piecewise
    bins:
      - [0.00, 0.05, 0.25]
      - [0.05, 0.35, 1.00]
      - [0.35, 0.65, 0.50]
      - [0.65, 1.00, 0.05]

  r1:
    gamma: 1.0
    interval: 16

discriminator:
  name: x0_resnet
  base_channels: 16
  spectral_norm: true
  time_projection: true
  class_projection: true
  lr: 2.0e-4
  weight_decay: 0.0
  betas: [0.0, 0.99]
```

## 9. 实现边界

建议修改模块：

```text
x0loop/models/discriminator.py
  X0DiscriminatorConfig
  X0Discriminator

x0loop/losses/adversarial.py
  hinge / logistic adversarial losses
  t_weight helpers

x0loop/training/factories.py
  build_discriminator
  build_adversarial_config

x0loop/training/engine.py
  build D and d_optimizer
  D update
  G adversarial branch
  logging

x0loop/training/checkpointing.py
x0loop/utils/checkpoint.py
  save/load discriminator and d_optimizer
```

不要修改的边界：

- `BaseProcess` 不应该知道 GAN。
- `CompositeLoss` 不应该接收 discriminator。
- `Denoiser` 只负责 `xt,t -> out` 和已有 target conversion。

GAN fake 统一从训练层取：

```python
x0_hat = process.x0_from_output(batch.fb.xt, batch.fb.t, batch.out, aux={})
```

## 10. 首轮 5 个实验

全部基于当前最强 baseline：

```text
jit / logitnormal_target_weight / no-grid
```

共同配置：

```bash
--set "loss.outer_weight=target"
--set "loss.outer_weight_power=1.0"
--set "loss.outer_weight_floor=0.0"
--set "time_sampler.name=logit_normal"
--set "time_sampler.mean=0.0"
--set "time_sampler.std=1.0"
```

实验矩阵：

| ID | adv weight | t weighting | start / warmup | R1 | 目的 |
| --- | ---: | --- | --- | ---: | --- |
| gan_w0p01_piecewise | 0.01 | piecewise default | 10k / 10k | 1.0 lazy16 | 最保守 smoke-quality run |
| gan_w0p02_piecewise | 0.02 | piecewise default | 10k / 10k | 1.0 lazy16 | 推荐主实验 |
| gan_w0p05_piecewise | 0.05 | piecewise default | 10k / 10k | 1.0 lazy16 | 测试 adv 上限 |
| gan_w0p02_lowt | 0.02 | only `[0.05,0.35)` weight=1 | 10k / 10k | 1.0 lazy16 | 验证只打低 t 是否更稳 |
| gan_w0p02_nohigh | 0.02 | `[0.05,0.65]`, high t=0 | 10k / 10k | 1.0 lazy16 | 检查高 t 小权重是否仍有害 |

判断标准：

```text
明确有效:
  final FID < 10.0
  KID 改善
  recall 不低于 baseline 太多
  precision 不靠明显牺牲 recall 换来

可保留:
  FID 10.0 - 10.3
  precision/recall 平衡更好
  sample grid 主观更稳

放弃:
  FID > 10.5
  recall 明显下降
  D acc 长期 > 0.90
  base eval loss 明显恶化
```

## 11. 必须记录的指标

训练日志新增：

```text
gan/d_loss
gan/g_adv_loss
gan/d_real_loss
gan/d_fake_loss
gan/d_real_logit_mean
gan/d_fake_logit_mean
gan/d_real_prob_mean
gan/d_fake_prob_mean
gan/d_acc_real
gan/d_acc_fake
gan/d_acc_total
gan/g_weight
gan/d_lr
gan/d_grad_norm
gan/r1_penalty
gan/enabled_t_fraction
```

建议 t-bin 指标：

```text
gan/tbin_count
gan/tbin_d_acc
gan/tbin_d_real_prob
gan/tbin_d_fake_prob
gan/tbin_g_adv_loss
```

继续保留并对照：

```text
loss_x0 / loss_v / loss_eps
eval/loss_x0 / eval/loss_v / eval/loss_eps
FID / KID / precision / recall / IS
img_s / iter_s / gpu_mem_gb
```

## 12. 失败信号

硬失败：

- 任意 GAN loss/logit/grad 出现 NaN 或 Inf。
- `grad_norm`、`gan/d_grad_norm` 持续爆炸。
- `img_s` 比 baseline 下降超过 40%-50%，且短验证无收益。
- sample grid 连续两个 eval 周期出现大面积噪声、全黑/全白、严重色偏或重复模板。

判别器过强：

- `gan/d_acc_total > 0.90` 持续多个 log 窗口。
- `gan/d_fake_prob_mean < 0.05` 且 `gan/d_real_prob_mean > 0.95` 持续。
- 高 `t` bin D acc 接近 1，但低/中 `t` 没有收益。
- `gan/g_adv_loss` 上升，同时 base `loss_x0/loss_v/loss_eps` 停滞或恶化。

生成质量风险：

- FID 降但 recall 明显下降：模式收缩。
- precision 升、recall 降，并且 sample grid 出现重复类别或纹理。
- GAN loss 变好但 eval denoising loss 变差：adversarial objective 偏离主任务。

## 13. 最小验证流程

1. 先实现 `adversarial.enabled=false` 时完全不改变现有训练。
2. 跑 100-500 step smoke test，确认 D/G loss、D acc、R1、t-bin 指标有数。
3. 跑一个 10k-20k step 便宜验证，检查 D 是否饱和。
4. 只对未饱和配置跑完整 58.5k。
5. 最终报告必须列：

```text
FID, KID, precision, recall, IS
img_s drop ratio
gpu_mem_gb increase
D acc by t-bin
base eval loss change
fixed-seed sample grid
```

第一轮成功标准不应该只看 FID。GAN 可能提升 precision 但伤 recall；如果 recall
明显下降，即使 FID 略降也应谨慎。
