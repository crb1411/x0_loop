# 扩散 / Flow 训练 Loss 设计与解耦

这个文档用于固定 `x0_loop` 当前训练 loss 的设计边界：哪些东西属于过程定义，哪些属于时间采样，哪些属于模型直接预测目标，哪些属于 loss 聚合策略。核心目标是避免把 DDPM、Flow、target parameterization、loss weight 混在一起。

## 1. 四层解耦

当前最合适的抽象是四层：

```text
TimeSchedule        : t 的采样空间、alpha(t)、sigma(t)、SNR(t)
BaseProcess         : x_t = alpha(t) * x0 + sigma(t) * eps，以及 step / sample
OutputTarget        : model_out 直接代表 eps / x0 / v
LossConfig          : 在哪些预测空间算 loss，以及如何对不同 t 加权
```

已有代码中，`BaseProcess` 已经把 `output_target` 和 `x0_from_output / eps_from_output / v_from_output` 解耦得比较好。也就是说，模型可以直接输出 `eps/x0/v` 任意一种，但 loss 不必只能在这个空间计算；训练侧可以统一从 `model_out` 还原出 `x0_hat / eps_hat / v_hat`，再按配置组合 loss。

因此后续实现不应该把 `loss_type == eps` 写死到 process 内部，也不应该让 DDPM/Flow 分支各自维护一套 loss。

## 2. DDPM 与 Flow 的过程差异

二者统一写成：

```text
x_t = alpha(t) * x0 + sigma(t) * eps
```

但二者的系数几何不同。

### 2.1 DDPM / VP diffusion

DDPM 近似处在单位圆参数化上：

```text
alpha(t)^2 + sigma(t)^2 = 1
```

其中 `alpha(t)` 从 1 下降到 0，`sigma(t)` 从 0 上升到 1。这个空间里 `x0` 与 `eps` 是正交混合，常见 `v` 定义为：

```text
v = alpha(t) * eps - sigma(t) * x0
```

因为 `alpha^2 + sigma^2 = 1`，从 `v` 反解 `x0/eps` 时数值上比较自然。

### 2.2 Flow / Rectified Flow

Flow 更接近线性插值：

```text
alpha(t) = 1 - t
sigma(t) = t
alpha(t) + sigma(t) = 1
```

此时一般不满足：

```text
alpha(t)^2 + sigma(t)^2 = 1
```

所以 Flow 与 DDPM 不能只理解成同一个 `t` 换个名字。二者至少有三点差别：

1. `t` 的几何含义不同：DDPM 是圆弧 / SNR 轨迹，Flow 是线段插值。
2. `v` 的尺度不同：沿用 diffusion 的 `v = alpha eps - sigma x0` 时，反解必须保留 `alpha^2 + sigma^2` 分母。
3. loss weight 的自然定义空间不同：DDPM 常用 SNR/logSNR 直觉，Flow 常用 t 或路径速度直觉。

已有 `BaseProcess` 中从 `v` 反解时使用了 `alpha^2 + sigma^2`，这一点对 Flow 是必要的，不能省略。

## 3. t 的采样空间

训练时存在两个容易混淆的概念：

```text
外部采样索引: 1..1000
归一化连续时间: 0..1
```

建议内部全部使用 `t in (0, 1]`，并把 `1..num_steps` 只作为 diffusion 离散采样的外部实现细节。

### 3.1 推荐规则

```text
diffusion:
    t_idx ~ Uniform{1, ..., num_steps}
    t = t_idx / num_steps

flow:
    t ~ Uniform(0, 1)
```

也就是说：

- `TimeSchedule.sample_t` 负责采样 `t`。
- `BaseProcess.forward_sample` 只消费归一化后的 `t`。
- model 永远只看到统一语义的 `t`，即 float tensor in `(0, 1]`。
- 如果后续需要 timestep embedding 使用整数 index，应由 embedding 层或 adapter 从 `t` 派生，不要污染 process/loss。

### 3.2 反向采样网格

采样 loop 也应使用归一化时间网格：

```text
t: 1 -> 0
```

至于是 1000 step、50 step、20 step，属于 sampler 离散化策略，不属于训练 target 定义。

## 4. model 输出 target

模型直接输出目标由 `output_target` 控制：

```text
output_target = eps | x0 | v
```

它只表示 `model_out` 的直接语义，不表示最终 loss 只能在这个空间算。

推荐保持以下接口语义：

```text
x0_hat  = process.x0_from_output(xt, t, model_out, aux)
eps_hat = process.eps_from_output(xt, t, model_out, aux)
v_hat   = process.v_from_output(xt, t, model_out, aux)
```

对应 target：

```text
x0_target  = x0
eps_target = eps
v_target   = alpha(t) * eps - sigma(t) * x0
```

这使得训练目标可以写成外层组合：

```text
loss_inner(t) = c_eps * L(eps_hat, eps)
              + c_x0  * L(x0_hat,  x0)
              + c_v   * L(v_hat,   v)
```

其中 `c_eps/c_x0/c_v` 是不同预测空间的全局系数，不建议把它们和 `weight(t)` 混成一个东西。

## 5. t-dependent difficulty 与 loss weight

不同 `t` 下，`x0/eps/v` 的预测难度不同。

直观上：

```text
t -> 0: xt 接近 x0，x0 容易，eps 困难
t -> 1: xt 接近 eps，eps 容易，x0 困难
v:      往往在中间区间更均衡，但仍依赖 schedule 几何
```

因此可以引入时间权重：

```text
weight(t, space)
```

其中 `space` 可取：

```text
eps | x0 | v | shared
```

但实现上建议先采用最外层 shared 权重，而不是把权重塞进每个 loss 项内部。

## 6. 推荐的 loss 定义空间

推荐主定义：

```text
loss_t = weight(t) * (
    c_eps * loss_eps
  + c_x0  * loss_x0
  + c_v   * loss_v
)
```

也就是用户倾向的最外层空间。

这样做有三个好处：

1. `weight(t)` 表示这个 timestep 样本整体的重要性。
2. `c_eps/c_x0/c_v` 表示不同预测空间的全局训练偏好。
3. 不会出现 `eps` 空间和 `x0` 空间各自偷偷定义一套 t weighting，导致实验不可解释。

如果后续确实需要 per-space weight，也应显式写成二级结构，而不是隐含在 loss 函数里：

```text
loss_t = weight_shared(t) * (
    c_eps * weight_eps(t) * loss_eps
  + c_x0  * weight_x0(t)  * loss_x0
  + c_v   * weight_v(t)   * loss_v
)
```

但第一阶段不建议启用 `weight_eps/weight_x0/weight_v`，否则变量过多。

## 7. LossConfig 建议

建议新增或整理一个训练侧配置：

```python
@dataclass
class LossConfig:
    spaces: tuple[str, ...] = ("eps",)        # eps/x0/v 可组合
    coef_eps: float = 1.0
    coef_x0: float = 0.0
    coef_v: float = 0.0

    reduction: str = "mean"                  # mean over non-batch dims
    weight_type: str = "none"                # none/snr/min_snr/logsnr/custom
    weight_space: str = "outer"              # outer/per_space
    min_snr_gamma: float | None = None
    eps: float = 1e-8
```

其中 `weight_space="outer"` 是默认推荐。

训练伪代码：

```python
fb = process.forward_sample(x0, t)
model_out = model(fb.xt, fb.t, cond=cond)

loss_terms = {}

if coef_eps != 0:
    eps_hat = process.eps_from_output(fb.xt, fb.t, model_out, fb.aux)
    loss_terms["eps"] = mse_per_sample(eps_hat, process.eps_target(fb))

if coef_x0 != 0:
    x0_hat = process.x0_from_output(fb.xt, fb.t, model_out, fb.aux)
    loss_terms["x0"] = mse_per_sample(x0_hat, process.x0_target(fb))

if coef_v != 0:
    v_hat = process.v_from_output(fb.xt, fb.t, model_out, fb.aux)
    loss_terms["v"] = mse_per_sample(v_hat, process.v_target(fb))

inner = coef_eps * loss_terms.get("eps", 0) \
      + coef_x0  * loss_terms.get("x0",  0) \
      + coef_v   * loss_terms.get("v",   0)

w = loss_weight(fb.t, process.schedule, config)
loss = (w * inner).mean()
```

注意：`mse_per_sample` 应先对非 batch 维求平均，返回 `[B]`，然后再乘 `weight(t)`。

## 8. weight(t) 的候选定义

初始建议只保留少量可解释选项。

### 8.1 none

```text
weight(t) = 1
```

这是 baseline。

### 8.2 SNR based

```text
snr(t) = alpha(t)^2 / sigma(t)^2
```

可用于 diffusion，也可用于 flow，但 flow 下的语义是由线性插值诱导出来的 SNR，不应和 VP diffusion 的 SNR 完全等同。

### 8.3 min-SNR

常见思想是限制高 SNR 区域权重，避免小 t 区间主导训练：

```text
weight(t) = min(snr(t), gamma) / snr(t)
```

这更适合 `eps` 或 diffusion 场景。若用于多空间组合 loss，需要在实验命名中明确。

### 8.4 custom callable

研究阶段建议允许传入函数：

```python
weight = fn(t=t, alpha=alpha, sigma=sigma, snr=snr, process_type=mode)
```

但配置文件中必须记录 `weight_type=custom:<name>`，保证可复现。

## 9. 推荐实验矩阵

为了先确定主设计，不建议一开始全组合搜索。优先做以下矩阵：

```text
Process:
  diffusion / flow

OutputTarget:
  eps / x0 / v

LossSpace:
  same_as_output
  eps+x0
  v+x0

Weight:
  none
  outer:min_snr
  outer:custom_simple
```

其中最关键的对照是：

```text
output_target = eps, loss_space = eps
output_target = eps, loss_space = eps+x0
output_target = v,   loss_space = v
output_target = v,   loss_space = v+x0
```

这样可以区分：

1. 模型直接预测什么更稳定。
2. loss 监督在哪个空间更有效。
3. t 权重是否只是修正难度分布，还是改变了主优化目标。

## 10. 实现边界

建议保持如下边界：

```text
TimeSchedule:
    sample_t / alpha / sigma / snr

BaseProcess:
    forward_sample / step / x0_from_output / eps_from_output / v_from_output / target converters

LossComputer:
    调用 process converters
    计算 per-space loss
    在最外层乘 weight(t)
    返回 loss 和 log dict
```

不要做：

```text
- 不要在 model 内部处理 DDPM/Flow 差异
- 不要在 process 内部写死训练 loss
- 不要让 output_target 决定 loss_space
- 不要把 t_idx=1..1000 泄漏到 loss 逻辑
- 不要把 weight(t) 隐藏在 eps/x0/v 的 target converter 里
```

## 11. 当前结论

当前最合理的版本是：

```text
1. 过程统一为 x_t = alpha(t) x0 + sigma(t) eps。
2. DDPM 与 Flow 只通过 TimeSchedule 的 alpha/sigma 几何区分。
3. 训练内部统一使用 t in (0, 1]。
4. output_target 只控制 model_out 的直接含义。
5. loss_space 独立配置，可在 eps/x0/v 空间组合监督。
6. weight(t) 默认定义在最外层：

   loss_t = weight(t) * (loss_eps * c_eps + loss_x0 * c_x0 + loss_v * c_v)

7. per-space t weight 暂不作为默认设计，只保留为后续扩展。
```

这个设计的关键是：`process` 负责物理/几何一致性，`target converter` 负责参数化一致性，`loss` 负责实验目标。三者不互相偷逻辑。
