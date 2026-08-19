# x0_loop

`x0_loop` 是一个用于图像生成实验的训练代码仓，主要围绕 CIFAR-10 上的 flow / diffusion 过程、JiT / DiT / UNet 模型、不同时间采样、loss weight、learnable endpoint、geneval 评估和 clean loop 训练分支展开。

当前 README 同时适用于 `master` 和 `cleanLoop` 两个分支。两者共享主体训练、评估、采样和配置系统；`cleanLoop` 分支额外包含 bank 训练逻辑。

## 代码结构

```text
x0loop/
  train.py                    # 训练入口: uv run python -m x0loop.train
  eval_fid.py                 # checkpoint standalone geneval 入口
  core/                       # schedule、process 基类、time sampler、配置工具
  processes/                  # flow / diffusion 过程和采样器
  models/                     # DiT / JiT / UNet / discriminator
  losses/                     # atomic loss、loss weight、组合 loss
  training/                   # 训练主循环、eval、geneval、checkpoint、clean loop
  tests/                      # pytest 测试

train_run/                    # 主要实验脚本、sampler ablation、gen eval 配置
train_run2/                   # 当前常用 base / clean loop 训练入口
infer/                        # 推理和 standalone geneval 脚本
docs/                         # 配置和设计说明
runs/, runs2/                 # 实验输出目录
```

## 环境

项目使用仓库根目录下的 uv 环境，并锁定与当前 CUDA 12.8 驱动兼容的 PyTorch 构建：

```bash
uv sync
```

常用依赖包括：

```text
torch
torchvision
pyyaml
tensorboard
pytest
```

环境同时包含训练曲线所需的 `matplotlib` 和 geneval 指标所需的 `torch-fidelity`。

在仓库根目录运行命令，不需要设置机器相关的 `PYTHONPATH`：

```bash
uv run python -m x0loop.train --help
```

## 配置系统

训练入口为：

```bash
uv run python -m x0loop.train \
  --config path/to/config.yaml \
  --runtime-config x0loop/configs/runtime/ddp_checkpoint_compile.yaml \
  --set logging.out_dir=runs/my_experiment
```

配置会合并：

1. 实验 YAML，例如 `train_run2/base/config.yaml`
2. runtime YAML，例如 `x0loop/configs/runtime/ddp_checkpoint_compile.yaml`
3. 命令行 `--set key=value`

常见顶层字段：

```yaml
dataset: {}
model: {}
process: {}
schedule: {}
time_sampler: {}
time_condition_jitter: {}
model_conditioning: {}
clean_loop: {}
loss: {}
augment: {}
train: {}
logging: {}
eval: {}
sample: {}
gen_eval: {}
distributed: {}
compile: {}
```

更完整的 YAML 说明见 [docs/yaml_config_guide.md](docs/yaml_config_guide.md)。

## 训练

推荐优先使用已有脚本启动，避免手动拼接 runtime 和输出目录。

CIFAR-10 数据固定放在 `/mnt/data/crb/data`；其余配置、checkpoint、日志和图片输出均使用仓库根目录下的相对路径。

首次准备环境后，可运行两步真实训练 smoke test：

```bash
uv sync
uv run python -m x0loop.train \
  --config x0loop/configs/cifar10_train_smoke.yaml \
  --runtime-config x0loop/configs/runtime/debug_runtime.yaml
```

### base 训练

```bash
bash train_run2/base/run.sh
```

### ignore time 训练

```bash
bash train_run2/base_ignore_time/run.sh
```

对应配置中会包含：

```yaml
model_conditioning:
  ignore_time: true
  time_constant: 0.5
```

表示模型收到的时间条件固定为 `0.5`，但训练构造 `xt = (1-t)x0 + t z` 的真实 `t` 仍然照常采样。

### clean loop 训练

`cleanLoop` 分支中可使用：

```bash
bash train_run2/cleanloopv1_500ep/run.sh
```

关键配置：

```yaml
clean_loop:
  enabled: true
  bank_size: 4096
  bank_prob: 0.25
  warmup_steps: 10000
  loss_bank_weight: 0.3
  time_constant: 0.5
```

含义：

- `warmup_steps` 前只训练 fresh 样本，但持续向 FIFO bank 写入 `(x0_hat, x0, y)`。
- `warmup_steps` 后，按 `bank_prob` 从 bank 中采样旧的 `x0_hat` 作为模型输入。
- fresh 分支保持原 loss，例如 v loss 和 t-bin 日志。
- bank 分支使用恒定权重的 x0 MSE：

```text
loss = fresh_scale * weighted_fresh_loss
     + bank_scale * loss_bank_weight * bank_x0_mse
```

## 输出目录

脚本通常会写入：

```text
runs2/ablations/...
```

每次启动会生成一个时间戳目录，常见内容：

```text
resolved_config.yaml          # 最终合并后的配置
launch_config.yaml            # 启动配置快照
metrics_*.jsonl               # 训练/eval 标量
gen_eval_metrics_*.jsonl      # geneval 指标
logs/log.txt                  # 主训练日志
logs/launcher.log             # shell/torchrun 启动日志
checkpoints/                  # ckpt_step_*.pt
samples/                      # 训练中采样图
gen_eval/                     # geneval fake images 和临时结果
```

## 日志说明

训练日志终端预览是一行，核心结构如下：

```text
[step i/N] (loss) ... | (tbin) ... | lr=... iter_s=... img_s=... | (diag) ... | (clean) ... | (progress) ... | (meta) ... | (summary) ...
```

其中：

- `(loss)` 是真正 backward 的总 loss 公式。
- `(tbin)` 默认摘出 `[0.50,0.55)` 这个常看的 t 区间。
- `(diag)` 是诊断 MSE，例如 `z_mse / x0_mse / v_mse`，不一定直接参与训练目标。
- `(clean)` 是 clean loop 的 bank 状态和混合比例。
- `(progress)` 中 `progress_pct = current_step / total_steps * 100`。
- `(summary)` 是完整 t-bin 统计。

完整原始指标仍写入 `metrics_*.jsonl`，TensorBoard 也会保留数值字段。

## Geneval / FID 评估

### 训练中 geneval

在训练配置中设置：

```yaml
gen_eval:
  enabled: true
  num_samples: 50000
  batch_size: 256
  steps: 20
  sampler: heun
  guidance_scale: 2.2
  guidance_schedule: null
```

训练会在对应 step 或 final 阶段执行 geneval，并写入 `gen_eval_metrics_*.jsonl`。

### Standalone geneval

最简单入口：

```bash
bash infer/geneval/run_geneval.sh /path/to/ckpt_step_00058500.pt
```

`x0loop.eval_fid` 会从 checkpoint 路径向上查找 `launch_config.yaml`，自动恢复训练时的高层配置，例如：

```yaml
model_conditioning:
  ignore_time: true
  time_constant: 0.5
```

因此 standalone geneval 不需要手动重复指定这些训练时配置。

## 推理

已有 CIFAR-10 推理脚本：

```bash
bash infer/infer_flow/infer_cifar10.sh
```

该脚本通常基于某个 run 的 `resolved_config.yaml` 和 checkpoint 生成小批量样例图。

## 测试

常用测试：

```bash
pytest -q x0loop/tests/test_clean_loop.py \
          x0loop/tests/test_time_sampling.py \
          x0loop/tests/test_time_condition_jitter.py \
          x0loop/tests/test_flow_sampling.py
```

全部测试：

```bash
pytest -q x0loop/tests
```

## 常见实验点

### time sampler

常用配置：

```yaml
time_sampler:
  name: logit_normal
  mean: 0.0
  std: 1.0
  min_t: 1.0e-3
  max_t: 0.999
```

也支持 uniform、beta、grid mix 等配置。启动日志会打印 t 采样密度形状。

### time condition jitter

```yaml
time_condition_jitter:
  enabled: true
  mean: 0.02
  std: 0.02
  prob: 1.0
  min_t: 1.0e-3
  max_t: 0.999
```

它只改变送入模型的时间条件，不改变构造 `xt` 使用的真实 `t`。

### loss weight

当前常见 weight function：

```text
none
triangular
skew_triangular
p2
min_snr
edm
```

更多说明见 [docs/loss_time_weight_functions.md](docs/loss_time_weight_functions.md)。

## 分支说明

### master

主线训练、采样、eval、loss weight、learnable endpoint 等逻辑。

### cleanLoop

在主线基础上增加 clean loop bank 训练：

- FIFO bank 保存 `(x0_hat, x0, y)`。
- fresh 和 bank 样本拼成一个 batch，只执行一次模型 forward。
- fresh 分支保持原 loss。
- bank 分支使用 x0 MSE。
- geneval 和普通采样不依赖 bank。

## Git 注意事项

实验输出目录 `runs/`、`runs2/` 通常不应提交。配置、脚本、源码和文档可以提交。

提交前建议检查：

```bash
git status --short
pytest -q x0loop/tests/test_clean_loop.py x0loop/tests/test_time_sampling.py
```
