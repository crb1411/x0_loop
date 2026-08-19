# x0loop v2 从零训练与采样分析

> 本文记录具体实验与证据。所有后续设计和结论必须遵循
> [`x0loop_research_principles.md`](x0loop_research_principles.md)；若二者冲突，先修订原则
> 文档并记录理由，再继续实验。
>
> 当前遵循原则版本：v1。每个新 run 启动前必须在本文件或 run manifest 中完成该文档
> 第 10 节的启动清单。

## 最终目标

最终目标是稳定降低 CIFAR-10 生成 FID。权威协议固定为 EMA、Heun-20、CFG
2.2、50,000 张样本、seed 20260819。5k FID 用于训练过程中的 checkpoint
选择；最终 50k FID 是最重要的结果参考，KID、precision、recall、不同 NFE 和
trajectory defect 用于判断改善的稳定性、代价和机制，不能被辅助 loss 替代。

所有支撑正式结论的比较实验从随机初始化开始；共享 checkpoint 分叉只用于低成本机制
筛选，不能冒充独立从零结果。每完成三次训练，必须先复盘训练设计和采样过程，再修改
下一轮方案。未经三次训练复盘，不扫描 bank size。

## 固定训练协议

- 数据：`/mnt/data/crb/data` 中的 CIFAR-10。
- 模型：JiT，dim 512、depth 12、8 heads。
- 主训练：300 epochs，batch size 256，共 58,500 optimizer steps。
- 优化器：AdamW；10k warmup 后 cosine，LR 3e-4 到 5e-5。
- EMA decay：0.996。
- 中途评估：每 5,000 step 保存 checkpoint；只在 15k/30k/45k 计算固定噪声
  Heun-20/CFG-2.2 的 5k FID，最终 checkpoint 直接进行 50k FID。中途 FID 是
  决策门而非训练组成部分。
- 最终评估：50k FID；随后对最有解释价值的 checkpoint 补 NFE 4/8/20 的
  FID、KID、precision、recall。
- 训练和 FID 随机流隔离；不同方法的 fresh batch 在相同 seed 下尽量保持一致。

## 每组三次训练的复盘规则

每组三次训练结束后必须回答：

1. 最低 5k FID、最终 50k FID 是否优于 FRESH，差异是否超过评估波动？
2. fresh exposure、学习率、EMA、训练步数和数据顺序是否严格可比？
3. 辅助梯度实际比例是否保持在 fresh 输出梯度的 10%–30%？
4. rollout occupancy 是否覆盖完整 Heun grid，而非集中在高噪声区？
5. replay age、depth 和 solver index 是否与性能变化相关？
6. 训练 transition、评估 transition、CFG、endpoint 和时间条件是否完全一致？
7. NFE 4/8/20 的收益方向是否一致？若只改善低 NFE，则将方法定位为少步纠错。
8. precision/recall 哪一侧变化？FID 下降是否来自牺牲多样性？
9. 从同一 root noise 跟踪每个 Heun step：状态范数、速度范数、局部截断误差、
   x0 预测漂移和最终样本是否出现过平滑、过饱和或类条件漂移？
10. 若 trajectory defect 改善但最终 FID 不变，停止优化 replay，转向终点分布损失。

## Cycle 01：核心假设首次从零检验

### Run 01 — FRESH

- 状态：训练完成（step 58,500；最终 50k FID 已完成）
- GPU：6
- 作用：建立相同代码、数据、优化器和最终采样协议下的从零基线。
- clean loop：关闭。
- 输出：`runs/x0loop_v2_from_scratch/cycle01/fresh`

### Run 02 — BANK-FIX

- 状态：step 15,005 提前终止（15k 固定 FID 劣于同 step FRESH，且轨迹同向退化）
- GPU：7
- 作用：检验对齐 endpoint、Heun-20、CFG-2.2 和 EMA teacher 后，replay 是否提供
  超出 FRESH 的 FID 信号。
- fresh batch：完整保留。
- auxiliary batch ratio：0.125；目标输出梯度比例：0.2。
- replay：按 solver index 分层，记录 depth、root noise ID、producer step。
- 输出：`runs/x0loop_v2_from_scratch/cycle01/bank-fix`

### Run 03 — ONLINE

- 状态：在 step 15,102 提前终止（固定 FID 与 sampler trace 共同证伪）
- GPU：7
- 作用：与 BANK-FIX 对比，判断 replay/off-policy 是否是主要瓶颈。
- trajectory：当前 EMA 在线生成实际 Heun-20/CFG-2.2 occupancy。
- fresh batch 与辅助梯度约束同 BANK-FIX。
- 输出：`runs/x0loop_v2_from_scratch/cycle01/online`

### Cycle 01 结果表

| Run | 最低 5k FID | 对应 step | 最终 50k FID | NFE 4 | NFE 8 | NFE 20 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FRESH | 10.5959 | 45,000 | 6.2143 | — | — | 6.2143 | 0.7454 | 0.4454 |
| BANK-FIX | 19.3422 | 15,000 | — | — | — | — | — | — |
| ONLINE | 22.1847 | 10,000（warmup） | — | — | — | — | — | — |

### Cycle 01 采样轨迹分析

三次训练已保存同一组 root noise 和 label 在三个模型上的 Heun trace，并逐 step 比较
状态、velocity、x0_hat、局部 Euler/Heun defect 以及最终图像。
分析工具固定使用 `experiments/x0loop_v2/analyze_sampling_trajectory.py`，输出完整 trace、
最终图像网格、逐步 JSON 指标和 Markdown 摘要。

三模型 step-15k 的 64 个固定 root 对照已经完成：

| 指标 | FRESH | BANK-FIX | ONLINE |
|---|---:|---:|---:|
| 5k FID | 16.7631 | 19.3422 | 26.4748 |
| 最终样本 RMS | 0.9784 | 0.8984 | 0.5219 |
| t=0.10 x0 drift RMS | 0.0133 | 0.0253 | 0.1367 |
| t=0.10 相对 Heun correction | 0.1458 | 0.3618 | 0.6264 |

BANK-FIX 位于 FRESH 与 ONLINE 之间：replay 减轻了在线 rollout 的严重动态范围坍缩，
但低噪声末段仍出现同方向的 correction 和 x0 drift 放大。原始三模型 trace 位于
`runs/x0loop_v2_from_scratch/cycle01/trajectory_analysis_step15000_threeway`。

ONLINE-15k 出现 FID 大幅退化后，先做了同 step、同 root 的 FRESH/ONLINE 中途诊断：

| 指标 | FRESH-15k | ONLINE-15k |
|---|---:|---:|
| 最终样本 RMS | 0.9582 | 0.5118 |
| 平均 Heun correction RMS | 0.00192 | 0.00700 |
| 最大相对 Heun correction | 0.142 | 0.640 |
| t=0.10 的 x0 drift RMS | 0.0125 | 0.1347 |

同 root 图像保留了大致语义结构，但 ONLINE 明显发灰、低对比度。异常集中在低噪声
末段，并最终把样本动态范围压缩约一半；这与 FID 从 22.18 退化至 26.47 一致，故不是
单纯的评估噪声。原始产物位于
`runs/x0loop_v2_from_scratch/cycle01/trajectory_analysis_step15000`。

### Cycle 01 设计结论与下一轮修改

三次训练的共同证据不支持继续增加 bank size。模型原生输出 x0，但辅助损失先把输出
转换为 velocity 再回归 teacher velocity；在线性 flow 中
`v=(x_t-x0)/t`，因此对 x0 输出的 Jacobian 为 `-1/t`。ONLINE 的严重坍缩和
BANK-FIX 的较轻末段失稳构成同一连续谱。

Cycle 02 保留完整 fresh loss、相同 EMA/Heun-20/CFG-2.2 occupancy 和分层 replay，
只把辅助监督改为由 EMA teacher velocity 代数重建的 x0，并把输出梯度比例从 0.2
降到 0.1。
同时使用 GPU tensor-ring bank；不扫描容量。

## Cycle 02：去除 velocity 参数化奇异性

三个分支共享 Cycle 01 FRESH step-10k checkpoint，等价于共享同一从零训练前缀；各自
继续到 step 15k 后做固定 5k FID 和三模型同-root Heun trace。这一轮是低成本机制筛选，
不是三条独立的从零正式结果；任何胜出定义仍须重新从随机初始化训练 300 epochs：

1. `FRESH-ACCEL`：仅验证 compile 后的数值基线。
2. `BANK-X0`：GPU replay，EMA teacher 等价 x0，aux batch 0.125，梯度比例 0.1。
3. `ONLINE-X0`：当前 EMA 在线 Heun occupancy，teacher 等价 x0，梯度比例 0.1。

15k 决策规则：任何辅助分支若未优于同轮 FRESH，不继续到 300 epochs；若改善则继续到
30k/45k，最后才做 50k FID。

### Cycle 02 筛选结果

| Run | step-15k 固定 5k FID | 相对 FRESH | 决策 |
|---|---:|---:|---|
| FRESH-ACCEL | 16.0143 | — | 编译基线有效 |
| BANK-X0 | 24.9222 | +8.9079 | 停止，不做 50k FID |
| ONLINE-X0 | 24.0846 | +8.0704 | 停止，不做 50k FID |

FRESH-ACCEL 与 BANK-X0 都从同一 Cycle 01 FRESH step-10k checkpoint 恢复、使用相同
compile、数据随机流、EMA、Heun-20/CFG-2.2 和 5k FID seed。BANK-X0 完整保留 256
fresh batch，另加 32 个辅助状态，日志中的输出梯度比例稳定为 0.1；因此退化不能归因于
fresh exposure 被替换。

FRESH/BANK-X0 的 64 个固定 root、step-15k Heun 轨迹对照如下：

| 指标 | FRESH-ACCEL | BANK-X0 |
|---|---:|---:|
| endpoint RMS | 0.8172 | 0.8175 |
| 最终样本 RMS | 1.0063 | 0.8584 |
| 平均 Heun correction RMS | 0.001925 | 0.001999 |
| 最大相对 Heun correction | 0.1184 | 0.2317 |
| t=0.10 x0 drift RMS | 0.0124 | 0.0156 |
| 最终样本同 root RMS 距离 | — | 0.3938 |

与 Cycle 01 的 velocity BANK 相比，BANK-X0 的低噪声 x0 drift 明显减小，但 FID 和最终
动态范围反而更差。这已经否定“主要退化仅由 velocity 的 `1/t` Jacobian 引起”：去掉该
奇异性修复了部分局部 defect，却没有恢复生成分布。BANK-X0 沿整条轨迹逐步收缩，最终
RMS 比 FRESH 低约 15%，与 FID `+8.91` 同向。原始轨迹位于
`runs/x0loop_v2_from_scratch/cycle02/trajectory_analysis_step15000_fresh_bankx0`。

审计还发现，运行时 replay 虽覆盖 20 个 solver level，但 `aux_n=32` 的余数固定分给
index 0–11，使采样的 solver-index 均值恒为 8.0，而严格均匀 occupancy 应为 9.5。
代码已改为每次随机排列 level、长期均匀分配余数，并加入统计回归测试。该偏置必须在
结果边界中记录，但其幅度不足以把 `+8.91` 的退化解释为有效 x0loop 信号；不会因此先
扫描 bank size 或把 BANK-X0 延长到 300 epochs。

Cycle 02 启动版本用 `x_t - t*v_teacher` 重建 x0；在线性 flow 中它与 teacher 直接 x0
输出代数等价，但经过 BF16 velocity conversion，严格说不应称为 native x0。后续代码已
让 trajectory 同时保存 velocity 和 EMA 直接 x0 输出，`*-x0` 分支直接监督后者；本轮
结果保留为“等价 x0 一致性”证据，不夸大为 direct-native 实验。

ONLINE-X0 完成后的三模型固定 root、step-15k Heun-20 轨迹如下。这里重新运行了全部
三个模型，避免把修改训练目标前的 Cycle 01 trace 当作当前采样证据：

| 指标 | FRESH-ACCEL | BANK-X0 | ONLINE-X0 |
|---|---:|---:|---:|
| 5k FID | 16.0143 | 24.9222 | 24.0846 |
| endpoint RMS | 0.8172 | 0.8175 | 0.8179 |
| 最终样本 RMS | 1.0063 | 0.8584 | 0.7957 |
| 平均 Heun correction RMS | 0.001925 | 0.001999 | 0.002524 |
| 最大相对 Heun correction | 0.1184 | 0.2317 | 0.4920 |
| t=0.10 x0 drift RMS | 0.0124 | 0.0156 | 0.0261 |
| t=0.05 x0 drift RMS | 0.0183 | 0.0305 | 0.0537 |

ONLINE-X0 与 BANK-X0 的终点同-root RMS 距离只有 0.1555，小于它们各自与 FRESH 的
0.4348 和 0.3938；两种状态来源产生的是同类收缩。在线 occupancy 的 solver-index
均值稳定在约 9.5，且完全没有 replay age，因此 ONLINE 仍退化直接说明 replay/off-policy
不是主要瓶颈。完整结果位于
`runs/x0loop_v2_from_scratch/cycle02/trajectory_analysis_step15000_threeway`。

### Cycle 02 三次筛选训练复盘

1. **FID 门槛**：两个辅助分支相对 FRESH 分别退化 `+8.91` 和 `+8.07`，远大于 5k
   评估的常见微小回摆。二者都未通过预注册的 15k 门槛，因此不消耗 50k FID；这不是
   最终 300-epoch 方法比较，也不声称获得正式从零结论。
2. **可比性**：三者共享完全相同的随机初始化到 step-10k FRESH 前缀，并继续相同 5k
   optimizer steps；batch、LR、EMA、数据顺序、checkpoint 和 5k FID seed 相同。两个
   辅助分支均完整保留 256 个 fresh 样本，并另加 32 个辅助状态，没有 fresh 稀释。
3. **梯度约束**：BANK-X0 和 ONLINE-X0 的辅助输出梯度比均稳定为 0.1，符合原则中的
   10%–30% 范围。失败不是旧版 `loss_bank_weight=2` 的无约束放大。
4. **occupancy**：ONLINE-X0 覆盖完整 Heun grid，solver-index 均值约 9.5。BANK-X0
   覆盖全部 20 层但存在已修复的余数分配偏置，均值为 8.0；该边界不足以解释两个独立
   辅助分支同方向、约 8–9 FID 的退化。
5. **off-policy/replay**：ONLINE 不含 replay age 且状态由当前 EMA 在线生成，仍只比
   BANK 改善 0.84 FID；主要矛盾不是 FIFO 容量、陈旧度或 off-policy，不再扫描 bank
   size。ONLINE 成本还明显更高。
6. **训练/推理闭环**：状态构造与评估都使用正确 learnable endpoint、EMA、实际
   Heun-20 grid、CFG 2.2 和相同标签规则；时间条件均为训练配置的 fixed 0.5。剩余错位
   不在 transition，而在监督目标：同状态回归 EMA x0 是自蒸馏一致性，并不恢复终点
   生成分布。
7. **不同 NFE**：两个候选未通过主 NFE-20 的低成本 FID 门槛，按停止规则不追加
   NFE 4/8/20 评估。当前没有证据把该定义定位成少步纠错方法。
8. **precision/recall**：因候选被 5k FID 明确支配而未运行昂贵的 50k distribution
   metrics；轨迹中的 RMS 收缩提示可能同时伤害 precision/recall 平衡，但不能用这一
   机制推断冒充正式指标。
9. **采样过程**：BANK 和 ONLINE 从相同 endpoint 出发，沿轨迹逐步降低动态范围；
   ONLINE 最终 RMS 比 FRESH 低 20.9%，最大相对 correction 为 FRESH 的 4.15 倍，低
   噪声 x0 drift 也持续放大。局部 x0 一致性 loss 没有形成分布恢复力，反而把 sampler
   推向更平滑的 EMA 条件均值。
10. **研究决策**：去除 velocity 的 `1/t` Jacobian 修复了 Cycle 01 的一部分局部
    defect，却没有改善 FID；online 又排除了 replay/off-policy 主因。按照研究原则，
    停止继续调 FIFO、bank size 和同状态 paired/EMA MSE，下一轮转向实际采样终点的
    分布级目标。

### Cycle 03 预注册方向：终点分布匹配

Cycle 03 在启动任何大实验前先完成实现与梯度/显存 smoke。其可证伪假设是：**完整保留
fresh 回归时，对真实 Heun-20/CFG-2.2 采样终点施加小权重、类别条件的分布损失，能够
抵消条件均值收缩，并在 step-15k 固定 5k FID 上优于同前缀 FRESH；若做不到，则当前
x0loop rollout 不提供有用的 FID 优化方向。**

低成本三分支仍共享同一可信 step-10k FRESH 前缀：

1. `FRESH`：同代码与预算基线；
2. `DENOISE-GAN`：现有 fresh-noise `x0_hat` GAN，隔离“加入判别器”本身的影响；
3. `TERMINAL-GAN`：用当前 EMA 生成真实 inference occupancy，只在末端截断反传，判别
   真实 CIFAR-10 与最终生成 x0；不再添加 paired/teacher MSE，不使用 replay。

三者均保留完整 fresh loss、相同 5k steps、EMA、Heun-20/CFG-2.2 和固定 5k FID。
GAN 从 step-10k 后启用，首轮只用一个保守权重并做 warmup，不进行权重扫描；判别器必须
按类别条件化，real/fake 使用相同 label occupancy。TERMINAL-GAN 的首版只允许最后一个
采样区间保留梯度，其余 rollout `no_grad`，以控制显存和 wall time。进入训练前必须验证：
终端 fake 确实来自固定 inference kernel、D/G detach 正确、完整 fresh 梯度不变、无 GAN
时数值回归、方法级 MFU/显存可接受。只有 15k FID 优于 FRESH 才进入独立从零 300-epoch
训练和 50k FID；否则停止继续堆叠 consistency/replay 变体。

### Cycle 03 精确执行协议

实现与 smoke 通过后，三条筛选分支冻结如下，结果出来前不改变：

| Run | GPU | 训练定义 | 预期稳定 s/step | 峰值显存 | 方法级 MFU |
|---|---:|---|---:|---:|---:|
| FRESH | 6 | adversarial/clean-loop 均关闭 | 0.087 | 6.93 GiB | 5.97% |
| DENOISE-GAN | 7 | fresh `x0_hat` distribution control | 0.109 | 6.94 GiB | 4.78% |
| TERMINAL-GAN | 7（顺序运行） | EMA Heun prefix + student final Euler | 0.253 | 8.64 GiB | 9.09% |

- 共享前缀：`cycle01/fresh/checkpoints/ckpt_step_00010000.pt`；三者从 global step 10k
  各继续 5k steps，到 step 15k。它们是机制筛选，不作为独立从零训练。
- fresh：每步完整 256 样本；GAN 另取同 batch 前 32 个真实图像/标签，绝不替换 fresh。
- D：类别条件 base-16 spectral-normalized ResNet、hinge、AdamW 2e-4、R1 gamma 1 lazy-16；
  real/fake 标签 occupancy 严格相同。
- G 辅助梯度：从 step 10k 启用，在 1,000 steps 内从 0 线性升到 fresh 输出梯度的 0.10；
  每步实测并记录比例，scale 上限 10。不使用未经测量的固定 GAN loss 系数。
- DENOISE-GAN：只使用训练 batch 中已有的 `x0_hat`，判别时间仍为对应 fresh t，用于
  隔离判别器本身是否有益；GAN batch 同样为 32。
- TERMINAL-GAN：从正确 learnable endpoint 独立采 root noise，EMA 在 `no_grad` 下执行
  Heun-20 的前 19 个区间，严格使用 CFG 2.2、固定模型时间 0.5 和真实 label/null-label；
  只对 sampler 本来就采用 Euler 的最后 `t=0.05 -> 0` 区间使用学生并保留梯度。D 的
  real/fake 均以 terminal t=0 条件化；不使用 replay、paired GT 或 teacher MSE。
- RNG：terminal rollout 使用由 train seed、global step 和 rank 决定的独立 forked stream；
  smoke 中两个 GAN 分支的 fresh loss、t-bin 和诊断指标逐项完全一致。
- 评估：step 15k 固定 seed 20260819、5k samples、EMA Heun-20/CFG-2.2 FID；训练前不做
  额外中途 FID。任一 GAN 分支未优于同轮 FRESH 即停止，不做 50k；若改善，先做三模型
  固定 root 全 Heun trace，再决定是否进入随机初始化 300-epoch 正式训练。
- 预计 wall time：FRESH 纯训练约 7 分钟，DENOISE-GAN 约 9 分钟，TERMINAL-GAN 约
  21 分钟；各自固定 5k FID 另约 1.5–2 分钟。方法级 MFU 计入 JiT teacher/student 和
  已测 D forward/backward FLOPs，lazy R1 的二阶项未被 PyTorch counter 计入。

正确性 smoke 已验证：72 个全量单元测试通过；terminal helper 与手算 Heun transition
逐元素一致；prefix 无梯度、最后一步可反传；EMA 权重交换不会碰触已有 autograd graph；
D/G detach 与 checkpoint 保存成功；两个分支输出梯度比均精确为 0.100。eager 单步
TERMINAL fake RMS 为 1.0066，没有复现 Cycle 02 的即时动态范围坍缩。

### Cycle 03 筛选结果

三个分支均在实现提交 `03c72bf` 上从同一个 Cycle 01 FRESH step-10k checkpoint 恢复，
继续恰好 5,000 个 optimizer steps，并以同一 EMA、seed、label 顺序和
Heun-20/CFG-2.2 协议评估。实际结果为：

| Run | step-15k 固定 5k FID | 相对 FRESH | 纯训练 wall time | 稳定 s/step | img/s | 峰值显存 | 方法级 MFU | 决策 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| FRESH | 15.9306 | — | 5m31s | 0.0653 | 3922.7 | 6.92 GiB | 7.98% | 同预算基线 |
| DENOISE-GAN | 16.4591 | +0.5285 | 9m15s | 0.1152 | 2221.8 | 6.94 GiB | 4.53% | 停止，不做 50k |
| TERMINAL-GAN | 17.9438 | +2.0133 | 21m54s | 0.2526 | 1013.4 | 8.64 GiB | 9.11% | 停止，不做 50k |

这里的 wall time 只计训练；每条 5k FID 单独计时。FRESH 比预注册 smoke 进一步快，说明
此前 12/21 小时的外推已经不适用于当前向量化、编译后的实现。TERMINAL-GAN 的主训练
模型 MFU 仅 2.06%，但把 19 段 EMA/CFG rollout 与 D 计算纳入后，方法级 MFU 为 9.11%、
每 step 22.772 TFLOP；较高的方法级 MFU 不代表更高计算效率，因为最终 FID 更差且每步
耗时仍是 FRESH 的 3.87 倍。

DENOISE-GAN 最后 100 steps 的 fresh loss 为 0.268933，接近 FRESH 的 0.268151；输出
梯度比精确维持 0.10。D accuracy 为 0.771，real/fake logit 分别为 0.549/-0.612，说明
该 control 的判别器确实提供了可分方向，但它仍未改善最终生成 FID。

TERMINAL-GAN 最后 100 steps 的 fresh loss 为 0.273897，已高于 FRESH；输出梯度比虽然
同样精确维持 0.10，但 D accuracy 只有 0.500，real/fake logit 分别为 -0.233696 和
-0.233427，几乎完全重合。归一化器仍把极弱且不稳定的 adversarial 输出梯度放大到目标
比例，G scale 在 0.1647–2.4759 间变化。terminal fake RMS 本身稳定在约 0.998，因此
退化不是 rollout 当场数值坍缩，而是训练方向缺少可靠的密度比信息。

### Cycle 03 修改训练方法后的采样分析

三模型 step-15k 已用 64 个固定 root、seed 20260819、完整 Heun-20/CFG-2.2 重新生成
trace；没有沿用 Cycle 02 的轨迹。结果位于
`runs/x0loop_v2_from_scratch/cycle03/trajectory_analysis_step15000_threeway`：

| 指标 | FRESH | DENOISE-GAN | TERMINAL-GAN |
|---|---:|---:|---:|
| endpoint RMS | 0.817177 | 0.817147 | 0.817394 |
| 最终样本 RMS | 0.999436 | 0.999858 | 0.954461 |
| 平均 Heun correction RMS | 0.001956 | 0.001989 | 0.003245 |
| 最大相对 Heun correction | 0.128534 | 0.140660 | 0.456355 |
| t=0.10 x0 drift RMS | 0.012658 | 0.013446 | 0.034447 |
| t=0.05 x0 drift RMS | 0.020143 | 0.021703 | 0.080205 |

FRESH 与 DENOISE-GAN 的最终同-root RMS 距离为 0.2865；FRESH 与 TERMINAL-GAN 为
0.3067。TERMINAL-GAN 的异常从 `t<=0.20` 开始增长：相对 correction 在
`t=0.20/0.15/0.10` 分别约为 0.0469/0.1244/0.4564，而 FRESH 约为
0.0260/0.0443/0.1285。固定图像网格中语义仍大致一致，但 TERMINAL-GAN 对比度和纹理
稍弱；这与最终 RMS 降低 4.5% 及 FID 退化同向。

训练只对最后一个采样区间保留梯度，推理却在所有时间复用同一个忽略时间条件的共享
backbone。因此“只训练最后一步”并不等于参数影响只局限于最后一步：更新从低噪声末段
反向污染了整条共享 vector field，采样 trace 已直接显示这种传播。未来若再引入终点
损失，必须显式限制其作用路径（例如 solver-index gated correction），不能只靠截断
rollout graph 声称局部化。

### Cycle 03 三次筛选训练复盘

1. **FID 门槛**：DENOISE-GAN 与 TERMINAL-GAN 分别比同轮 FRESH 差 0.5285 和
   2.0133；均未通过启动前冻结的 15k 门槛，因此不运行 50k FID，也不进入从零 300
   epochs。当前结论只否定这两个分布损失实现，不否定历史 x0 或终点匹配假设。
2. **可比性**：三者共享相同 step-10k 前缀、5k continuation、fresh batch、LR、EMA、
   数据顺序和 evaluator。GAN 另加 32 样本，不替换 256 个 fresh 样本；fresh exposure
   完整相同。
3. **梯度约束**：两个 GAN 分支在 1,000-step warmup 后均实测保持 0.10 输出梯度比。
   但 TERMINAL 的 D 准确率约 0.5，证明“范数受控”不等于“方向可信”；下一轮必须新增
   辅助方向 readiness 门槛。
4. **occupancy**：TERMINAL-GAN 的 no-grad EMA prefix 覆盖实际 Heun-20 全网格，学生只
   对 sampler 本来采用 Euler 的最后区间反传。没有 Cycle 01 bank 集中在高噪声的缺陷。
5. **off-policy/replay**：本轮不使用 replay；每步由当前 EMA 在线生成 terminal state。
   因此退化不能归因于 FIFO age、depth 或容量，继续扫描 bank size 没有依据。
6. **训练/推理闭环**：endpoint、EMA prefix、Heun grid、CFG 2.2、label/null-label、固定
   时间条件和最后 Euler transition 均与评估 kernel 对齐。剩余问题是 adversarial 方向
   质量与共享参数作用域，不是 transition 错位。
7. **不同 NFE**：候选在主 NFE-20 筛选即被支配，按停止规则不追加 NFE 4/8/20。
   因而尚无证据把终点 GAN 定位为少步纠错。
8. **precision/recall**：没有为被支配候选运行昂贵的 50k distribution metrics。终点
   RMS 和图像网格提示轻微动态范围/纹理损失，但不把视觉判断冒充 precision/recall。
9. **采样过程**：DENOISE-GAN 基本跟随 FRESH；TERMINAL-GAN 从 t=0.2 后逐步偏离，
   t=0.05 x0 drift 为 FRESH 的 3.98 倍，最大相对 correction 为 3.55 倍。修改终点训练
   目标确实改变了完整 sampler，而且方向与 FID 退化一致。
10. **研究决策**：停止 naive terminal GAN、GAN 权重扫描和判别器容量盲扫。下一轮先
    验证 frozen-generator 下 critic 是否能在 held-out 样本上可靠区分 real/fake，并测量
    辅助参数梯度与 fresh 梯度的方向关系；readiness 未通过前，不启动第四轮生成器训练。

### Cycle 04 启动前诊断门槛

Cycle 04 暂不预注册为“更强 GAN”。其候选方向是严格的两分布 score-difference 目标：
冻结的高质量 real-score teacher 与在当前生成分布上训练的 fake-score model 共同给出
方向，并采用 two-time-scale 更新。这个方向来自 DMD 对真实/生成 score 差的定义，以及
DMD2 对 fake critic 不准确会导致训练不稳定的修正，而不是把普通判别器损失改名为 DMD：
[DMD](https://arxiv.org/abs/2311.18828)、[DMD2](https://arxiv.org/abs/2405.14867)、
[官方 DMD2 实现](https://github.com/tianweiy/DMD2)。它也继续遵循 Self Forcing 的实际
inference occupancy 与截断反传思想：[Self Forcing](https://arxiv.org/abs/2506.08009)。

在冻结 Cycle 04 三分支之前，必须先完成不更新生成器的诊断：

1. 固定 terminal 样本集，分别训练/验证 critic，报告 held-out accuracy/AUC、real/fake
   logit margin 和过拟合差；若验证端仍接近随机，当前 critic 不具备 generator 信号资格。
2. 在固定 fresh/terminal batch 上分别计算参数梯度，报告 cosine、范数和分层贡献；即使
   输出梯度范数是 0.10，若共享 backbone 梯度与 fresh 长期负相关，也不能直接施加。
3. 若 score critic 通过 readiness，再比较“全共享 backbone”与“只在声明 solver index
   生效的 correction path”；训练 sampler 必须同步启用相同路径。
4. 只有诊断通过后，才冻结 FRESH、score-difference、gated-ablation 三次训练的假设、
   预算和停止规则。否则更换目标或 critic 定义，不消耗三次训练周期。

#### 冻结生成器 critic readiness 结果

诊断固定 TERMINAL-GAN step-15k EMA，不更新生成器，使用实际 Heun-20/CFG-2.2 kernel
生成 8,192 个训练 fake 与 2,048 个完全隔离的 held-out fake；real 来自 CIFAR-10 train
的无重叠样本，并逐项使用相同 class label。固定数据生成耗时 174.1 秒；同 Cycle 03
架构的 fresh critic 训练 2,000 steps 耗时约 73 秒。阈值无关的 held-out AUROC 为主判断：

| critic | held-out AUROC | sign accuracy | logit margin | 解释 |
|---|---:|---:|---:|---|
| random init | 0.4882 | 0.5000 | -0.0042 | 随机基线正常 |
| Cycle 03 共训练 critic | 0.5230 | 0.5000 | 0.0032 | 几乎没有分布方向 |
| frozen-G fresh critic | 0.7824 | 0.5317 | 0.4010 | 同架构具备可分容量 |

fresh critic 的 train/held-out AUROC 差只有 0.0126。sign accuracy 较低是 hinge/R1 后 logit
整体平移造成的校准现象，不能覆盖 AUROC 与 margin 证据。结论是：Cycle 03 失败的首要
问题不是 base-16 critic 容量，而是 generator/critic 同时变化时，critic 未能跟上当前
生成分布。该结果支持 two-time-scale 和 generator-gradient readiness gate，不支持简单
加宽 D 或增大 GAN loss。原始结果位于
`runs/x0loop_v2_from_scratch/cycle04/readiness/terminal_step15000/critic_readiness.*`。

#### 输出梯度控制到参数梯度的映射

随后在 4 个固定 batch、每 batch 256（terminal auxiliary 为 32）上，不执行 optimizer
step，分别计算 fresh 与 terminal generator loss 对 59.48M 参数的梯度。两种 critic 的
输出端比例都被精确设为 0.10，但映射到共享参数后为：

| critic | 输出梯度比 | 参数梯度比 | 参数 cosine | fresh/combined cosine | 负 dot 参数量占比 |
|---|---:|---:|---:|---:|---:|
| Cycle 03 共训练 critic | 0.1000 | 0.2228 | -0.4964 | 0.9767 | 0.7834 |
| frozen-G fresh critic | 0.1000 | 0.3018 | -0.0380 | 0.9561 | 0.9840 |

Cycle 03 的弱 critic 不仅没有可靠 margin，其 generator 方向还与 fresh 明显反向；输出端
10% 实际成为参数端 22.3%。fresh critic 虽能区分分布，但参数方向整体近乎正交，实际
比例达到 30.2%。分层上，共训练 critic 在最后四个 blocks 的比值为 39.7%、cosine
-0.538；fresh critic 的 conditioning 比值为 42.2%、cosine -0.408。因而 Cycle 03 的
“梯度已控制为 10%”只在输出张量成立，不能证明共享 backbone 更新温和。

这两项诊断共同改变 Cycle 04 设计：

1. 普通 real/fake classifier 只证明分布可分，不能作为 score-difference 的替代品；不把
   frozen-G fresh critic 直接接回 generator。
2. 新辅助目标必须控制**实际可训练参数**梯度，而非只控制最后输出；同时报告全局与分层
   cosine/比例。
3. score-difference 的 fake score 先在固定 terminal 分布上独立训练并通过 held-out
   denoising readiness；正式训练采用 fake-score 多步更新、generator 较慢更新。
4. terminal 辅助反传只进入 solver-index gated correction path；完整 fresh loss 继续
   更新 base 与 correction。inference 的同一末段必须启用该 correction，保证闭环。
5. 为隔离 correction architecture 与 score 目标，下一组三次筛选改为 `FRESH`、
   `GATED-FRESH`、`GATED-DMD`，而不是缺少架构对照的 shared-DMD/gated-DMD。三者仍共享
   step-10k 前缀；只有筛选胜出后才从随机初始化运行 300 epochs。

官方 DMD2 实现训练 fake score 时对生成样本重新加噪并回归生成 x0，generator 每若干
fake-score 更新才接收一次 normalized real/fake score difference；本项目只移植这一数学
结构，不照搬其 SD/EDM 噪声参数化，并继续使用本项目正确的 learnable endpoint、条件
标签和 Heun occupancy。下一门槛是先证明该 x0/flow 版本的 fake score 在 held-out fake
上优于未适配的 real teacher，再冻结三分支精确配置。

fake-score readiness 在看到结果前冻结为：复用上述 8,192/2,048 fixed terminal split，
从同一 step-15k EMA 初始化 real teacher 与 fake score；real teacher 全程冻结，fake score
只更新 JiT backbone，endpoint `mu_data` 固定。训练 1,000 steps、batch 64、AdamW
`lr=1e-4, betas=(0.9,0.95), weight_decay=0`，时间按原 logit-normal sampler 抽取，使用
项目原生 v-target composite loss；不使用 classifier、GAN 或 generator gradient。每 100
steps 在固定 uniform-t、固定 endpoint noise 的 2,048 held-out fake 上比较两者 x0/v MSE。
通过门槛预先定义为：step-1000 fake-score 的 held-out x0 与 v MSE 都至少比 frozen real
teacher 低 10%，且 10 个 time bin 中没有任何一 bin 比 teacher 高 10% 以上。未通过则不
启动 `GATED-DMD`，先修正 fake-score 参数化或时间权重。

fake-score readiness v1 已按上述定义完成并失败：1,000 steps 耗时 64.5 秒，稳定
0.0592 秒/step、峰值 3.54 GiB；teacher/fake 的 held-out x0 MSE 为
0.120435/0.120697（比值 1.0022），v MSE 为 3.34291/3.05321（比值 0.9133）。虽然所有
time bin 都没有恶化超过 10%，但 aggregate 双 10% 门槛不通过。分箱显示改善几乎全来自
`t<0.1`：该处 v MSE 因 `1/t` 映射被放大，native v-target 把大部分优化能力投入低噪声；
`t>0.5` 的 x0 MSE 只改善约 0%–1%，最高噪声 bin 还退化 1.05%。因此不把 v1 fake score
接入 generator。

readiness v2 在结果前冻结为仅修改被证据指向的两项：fake score 直接回归其原生模型
输出 x0，score query/train/eval 都使用 uniform `t∈[0.05,0.95]`，避开 DMD normalized
score 在两个 endpoint 的病态区间。数据 split、EMA 初始化、1,000 steps、batch、LR、
优化器和双 10%/逐 bin 门槛均保持不变。若 v2 仍失败，停止实现 Cycle 04 generator
训练，先重新评估当前 time-agnostic JiT 是否有能力同时表示 real/fake score。

readiness v2 同样按门槛失败：teacher/fake 的 held-out x0 MSE 为
0.093464/0.094190（比值 1.0078），v MSE 为 0.252497/0.254194（比值 1.0067）；所有
bin 虽都未超过 10% 退化线，但没有任何总体适配收益。step100–1000 在同一范围振荡，
不能解释为预算刚好不够。因为 terminal 样本本来就是这个 EMA 的 Heun-20 生成分布，
real teacher 已对其近似自洽，当前 fake score 无法识别出稳定的 real/fake score gap。
按预注册停止 DMD generator 实现，不通过降低门槛或选择最好中途 step 强行启动训练。

### Cycle 04 实际方向：先验证显式时间条件

结构审计确认 Cycle 01–03 的 JiT 在训练与每个 Heun solver step 都收到固定模型时间
`0.5`；真实 path t 只参与 noising/解析变换，不进入 backbone。它可以解释两类共同现象：
末段辅助更新会污染全部 solver step，以及同一 backbone 很难同时表示 real/fake score。
仓库中的历史 time-aware 配置同时改变过 CFG 等因素且没有留存可审计结果，不能作为证据。

下一组三次训练冻结为新的、从随机初始化开始的 prerequisite 周期：

1. `FRESH-FIXED-REPRO`：当前代码和固定时间定义，从零复现 300 epochs，消除 Cycle 01
   之后编译/性能改动的版本差异；GPU 6。
2. `FRESH-TIME`：除 `model_conditioning.ignore_time=false`、关闭在 fixed-time 下原本无效的
   time jitter 外，其余完全相同；从零 300 epochs。它先回答显式时间是否改善基础 FID，
   不把 baseline architecture 收益冒充 x0loop。
3. `ONLINE-X0-TIME`：只有 `FRESH-TIME` 在 step-15k 不劣于 matched FRESH 才启动；同样
   从随机初始化，完整 fresh + 32 online Heun states，EMA native x0 target，并将 auxiliary
   控制改为实际参数梯度 0.10，而非输出梯度 0.10。训练/评估均使用 time-aware
   Heun-20/CFG-2.2。

前两支共享 seed、初始化规则、数据、58,500-step/300-epoch 预算、LR/EMA 和评估协议，
但不共享 checkpoint。15k 固定 5k FID 是 prerequisite 门：若 `FRESH-TIME` 明显更差，
它按预注册提前停止，不启动第三支，并以另一个 fixed-time 机制对照补足本周期第三次训练；
若不劣，继续 30k/45k，胜出候选才做 50k。第三支相对 matched `FRESH-TIME` 的 15k FID
不改善即停止。每条方法修改后都重新生成同-root Heun 全轨迹。

### Cycle 04 run 1：FRESH-FIXED-REPRO 完整结果

`FRESH-FIXED-REPRO` 已从随机初始化完成 300 epochs/58,500 steps。运行使用 GPU 6、
commit `4410c0e`，启动时 `git_dirty=false`；最终 checkpoint、resolved config 和所有指标
均已落盘。固定 EMA/Heun-20/CFG-2.2/seed-20260819 的筛选曲线为：

| step | 样本数 | FID | 评测耗时 |
|---:|---:|---:|---:|
| 15,000 | 5,000 | 16.3331 | 73.3 s |
| 30,000 | 5,000 | 12.0041 | 70.0 s |
| 45,000 | 5,000 | 10.7707 | 70.1 s |
| 58,500 | 50,000 | **6.16413** | 690.8 s |

Cycle 01 旧 fixed-time 最佳 checkpoint 的权威 FID 为 6.21432；新结果只低 0.05019，
不足以脱离单次评估/训练波动，结论是当前代码成功匹配复现 fixed-time baseline，而不是
发现了新的 FID 改善。相同 step 的旧/新 5k FID 在 30k 为 11.7266/12.0041，在 45k
为 10.5959/10.7707，也支持两条曲线接近但并非逐点相同。

最后 1,000 个日志记录的中位训练性能为 0.0667 s/step、3,840.6 image/s、7.15 GiB；
按每步 5.154 TFLOP 和 H800 dense BF16 989 TFLOP/s 计算，主训练/方法 MFU 均为
7.82%。训练时 20 秒硬件采样的 GPU busy 平均约 52%，功耗通常 370–389 W，进程平均
使用约 1.36 个 CPU 核且 I/O wait 为 0；最终 Heun 采样则持续 99%–100% GPU busy、
约 675–700 W。证据指向 backward/optimizer/逐步 launch 的小 kernel 与同步间隙，而非
DataLoader、显存容量或硬件降频。下一步按预注册启动 `FRESH-TIME`，不因这条 matched
baseline 的轻微数值优势修改实验定义。

### Cycle 04 run 2：FRESH-TIME 完整结果

`FRESH-TIME` 同样从随机初始化完成 300 epochs/58,500 steps，运行使用 GPU 6、commit
`fc1918a`、`git_dirty=false`。除 `model_conditioning.ignore_time=false` 和关闭无效的
fixed-time jitter 外，训练预算、seed、模型、优化器、EMA 与 evaluator 均与 matched
fixed baseline 相同。结果为：

| step | 样本数 | fixed FID | time-aware FID | time-aware 改善 |
|---:|---:|---:|---:|---:|
| 15,000 | 5,000 | 16.3331 | 14.9119 | 1.4212（8.70%） |
| 30,000 | 5,000 | 12.0041 | 10.9770 | 1.0271（8.56%） |
| 45,000 | 5,000 | 10.7707 | 10.2560 | 0.5148（4.78%） |
| 58,500 | 50,000 | 6.16413 | **5.47036** | **0.69377（11.26%）** |

三个稀疏筛选点与权威 50k FID 全部同向；显式时间条件应成为后续 x0loop 的 matched
FRESH 基线。这个结果修复的是此前 time-agnostic backbone 的结构性 prerequisite，不能
冒充 x0loop 收益。最后 1,000 个稳定记录的 time-aware 性能为 0.0664 s/step、
3,854.8 image/s、7.15 GiB 和 7.85% counted MFU；与 fixed 的 0.0667 s/step、7.82%
MFU 等价。完成训练和三次中途 FID 用时 1 小时 9 分 59 秒，最终 50k FID 用时
683.1 秒。

训练定义修改后，使用 256 个相同 root noise/label、Heun-20/CFG-2.2 对两个最终 EMA
checkpoint 做完整 solver-grid trace。time-aware 最终样本 RMS 为 0.9973，fixed 为
1.0261；在低噪声 `t=0.1`，相对 Heun correction 由 8.23% 降至 6.78%，最后 x0 drift
由 0.01399 降至 0.01173（低 16.1%）。time-aware 在高噪声首步 correction 更大
（19.80% 对 11.96%），但在末段更稳定，符合“早期按 t 重整、末段减少误差修正”的
解释。两者最终样本 RMS 距离为 0.4200，说明改善不是微小数值扰动。原始逐步统计与固定
root grid 保存在 `runs/x0loop_v2_from_scratch/cycle04/trajectory_fixed_vs_time/`。

因此 Cycle 04 第三条训练继续采用预注册 `ONLINE-X0-TIME`：它必须与新 time-aware
FRESH 比较，不能再以旧 fixed baseline 为对手。辅助 batch 为 32、完整保留 fresh，
训练/评估均使用 time-aware Heun-20/CFG-2.2；辅助强度按参数梯度范数目标 0.10 控制。

实现加入 `clean_loop.aux_gradient_space=parameter`，用全体可训练参数上的 fresh/aux VJP
范数逐 step 设置辅助 scale，不写入 `.grad`，随后只对组合 loss 做 optimizer backward。
完整测试为 81 passed；从成熟 time-aware step-10k checkpoint 的 one-step smoke 得到
fresh/aux 原始参数梯度范数 0.50636/0.10259、scale 0.49356，实际比例精确为 0.10000。
随机初始化时 EMA 与 student 相同导致 native-x0 auxiliary 近零、scale cap 暂时无法达到
目标，因此正式分支保持预注册 10k warmup，不能从 step 0 开辅助。

动态 compile 会因可变 active batch 触发大量重编译，且早期 exact-VJP smoke 暴露 retained
graph 与 AOTAutograd donated buffer 冲突，故正式分支锁定 `compile.dynamic=false`；这不是
数值早停而是启动前排除无效性能路径。成熟 checkpoint 的 100-step static benchmark
后 8 个稳定日志点为 0.5265 s/step、486.2 image/s、13.94 GiB、参数梯度比 0.10000。
计入 fresh/aux 两次测量 VJP、online teacher 和组合 backward 后，每步 19.736 TFLOP，
主训练 MFU 0.99%、方法级 MFU 3.79%。预计 15k 门槛约 57–60 分钟；若通过并完整运行，
总训练约 7.2 小时，另加稀疏/最终 FID。15k 门槛固定为必须优于同 step
`FRESH-TIME=14.9119`，否则停止并进入三次训练周期复盘。

正式 `ONLINE-X0-TIME` 首次启动使用 commit `d2a9324`、GPU 6、随机初始化和 10k
warmup。它正常完成 step 10,000 并保存 checkpoint，但在同一进程首次切换到参数梯度
VJP 时、任何 auxiliary optimizer step 发生之前退出。根因是此前普通 compiled backward
已按单次消费启用 functorch donated-buffer，而参数梯度控制需要两次
`retain_graph=True` VJP 后再做组合 backward。根据研究原则，这属于代码故障，不计作
Cycle 04 第三次有效训练；step-10k checkpoint 是未受辅助项影响的完整 warmup 状态，可
用于故障修复后的原 run continuation。

修复在模型首次 compile 前，仅对 `compile + clean-loop-v2 + parameter-gradient` 组合关闭
`functorch.donated_buffer`，不改变 FRESH、BANK 或输出梯度路径。两项针对性验证均通过：
从随机初始化运行 4 steps、在 step 2→3 跨越 warmup 边界后正常结束；从正式 run 自己的
step-10k checkpoint 运行成熟 auxiliary step，得到 `aux_scale=0.51951`、实际参数梯度比
`0.10000`。全套测试为 83 passed。正式第三支随后从同一 step-10k checkpoint 续跑，
保持原预注册 15k FID gate；修复前没有 auxiliary 更新，因此不存在回滚或挑选 checkpoint。

### Cycle 04 run 3：ONLINE-X0-TIME 在 15k 门槛证伪

修复后，正式 run 从它自己的完整 step-10k warmup checkpoint 继续；commit `5ed8a7f`、
GPU 6、`git_dirty=false`，并保持预注册的完整 batch-256 fresh loss、batch-32 online
Heun state、moving EMA native-x0 target 和实际参数梯度比 0.10。step-10k warmup 的补充
固定 5k FID 为 19.5863；matched `FRESH-TIME` step-10k 为 19.3437，仅差 0.2426，排除
了“独立 warmup 起点已经失效”的主要混杂。5k auxiliary 更新后的结果为：

| run | step-10k FID | step-15k FID | 10k→15k 变化 | 相对 matched FRESH@15k |
|---|---:|---:|---:|---:|
| FRESH-TIME | 19.3437 | **14.9119** | -4.4318（改善 22.9%） | — |
| ONLINE-X0-TIME | 19.5863 | **44.8095** | +25.2232（恶化 128.8%） | +29.8976（约 3.00 倍） |

因此该分支按启动前冻结的 gate 提前停止，不运行 30k/45k 或 50k FID，计作一次有效证伪
训练。中断发生在同步 evaluator 返回后；权威结果和轨迹均使用完整 step-15k checkpoint。
训练后来得及继续约 420 steps，紧急 checkpoint 又被中断信号打断，产生的 85 MiB 不完整
文件已删除，避免误加载；完整 step-15k checkpoint 与全部评估 JSON 保留。

step 14.5k–15k 的 fresh loss 与 matched FRESH 几乎相同：均值 0.26351 对 0.26368；
x0/v 诊断也略低而非变坏。auxiliary loss 已降至约 0.00108，实际参数梯度比每个记录点
都为 0.10000。这说明 pixel/teacher consistency loss、fresh loss 和梯度范数控制均无法
预警分布退化；10% 的 self-referential 方向连续积累 5k steps 仍足以改变生成分布。

用 256 个相同 root noise/label 做 time-aware Heun-20/CFG-2.2 trace 后，ONLINE 的最终
样本 RMS 为 1.09814，FRESH 为 0.98294（高 11.7%），pairwise RMS 距离为 0.38748。
`|x|>1.5` 像素占比从 13.75% 升至 20.18%，`|x|>2` 从 2.06% 升至 5.23%；低噪声最后
x0 drift 从 0.01199 增至 0.01523。与此同时，ONLINE 的平均 Heun correction 反而略低
（0.002034 对 0.002090），最大相对 correction 也略低（0.2037 对 0.2212）。因此当前
失败不是 solver 数值不自洽，而是 moving-EMA 自蒸馏沿一条错误但更自洽的轨迹放大动态
范围；trajectory defect 改善不能替代 FID。原始 trace、逐步 JSON 和图像位于
`runs/x0loop_v2_from_scratch/cycle04/trajectory_time_vs_online_step15000/`。

稳定 500-record 窗口性能为 0.5400 s/step、474.1 image/s、13.94 GiB；主训练 MFU
0.97%，计入 rollout、两次参数 VJP 和组合 backward 的方法级 MFU 3.70%，每步
19.736 TFLOP。从 continuation 启动到 step 15k 训练约 46 分钟，5k FID 另用 90 秒。

### Cycle 04 三次训练全局复盘

1. **核心 FID**：FRESH-FIXED-REPRO 的权威 50k FID 为 6.1641；显式时间条件在所有
   筛选点同向改善，并把权威 FID 降到 5.4704。ONLINE-X0-TIME 在 15k 即以 44.8095
   被 time-aware FRESH 14.9119 明显支配。Cycle 04 的真实进展是建立更强的 time-aware
   baseline，不是验证到 x0loop 收益。
2. **公平性**：前两支均从随机初始化训练完整 300 epochs；第三支也从随机初始化开始，
   10k warmup 后因代码故障从自身 checkpoint 无损续跑，并按预注册 15k gate 停止。
   三者 fresh exposure、数据、LR、EMA 和 evaluator 一致；第三支另加辅助 batch，不替换
   fresh。
3. **闭环与 occupancy**：ONLINE 状态由实际 time-aware EMA Heun-20/CFG-2.2 在线生成，
   solver index 均值约 9.5，覆盖完整 0–19 网格；endpoint、label/null-label 与最终 FID
   sampler 对齐。失败不能归因于旧 bank endpoint、FIFO age 或高噪声 occupancy 集中。
4. **梯度与监督**：实际全参数梯度比精确为 0.10，但 moving EMA 提供的是自身滞后
   prediction，而非真实分布锚点。范数受控不代表累计方向安全；下一轮必须记录 fresh/aux
   参数 cosine，并区分 moving-teacher feedback 与 paired-x0 目标本身。
5. **采样过程**：time-aware FRESH 相比 fixed baseline 在末段 drift 更低并改善 FID；
   ONLINE 又进一步降低部分中段 correction，却显著扩大最终动态范围并使 FID 崩溃。
   因而后续不以 correction/consistency 单独作为成功条件。
6. **计算效率**：time-aware FRESH 约 0.0664 s/step、7.85% MFU；ONLINE 为
   0.5400 s/step、方法级 3.70% MFU，单 step 慢约 8.1 倍。除非 FID 先证明信号有效，
   不再为当前 moving-EMA native-x0 目标继续做性能优化。
7. **结论边界**：本轮否定“moving EMA + online Heun occupancy + native-x0 MSE +
   10% 参数梯度”这一实现，不否定历史 x0 或 inference-state training。证据支持的是：
   自洽性监督缺乏真实分布锚点时，可把 sampler 推向错误的稳定轨迹。

### Cycle 05 启动前机制筛选

在消耗下一组从零 300-epoch 训练前，先用 Cycle 04 ONLINE 自己的 step-10k checkpoint
做不计正式训练次数的共享前缀筛选，精确隔离 moving teacher feedback：

1. `PREFIX-FRESH`：关闭 auxiliary，续训 5k steps；
2. `PREFIX-MOVING`：复用 Cycle 04 已完成结果；
3. `PREFIX-FROZEN`：在 step 10k 冻结 EMA teacher，其他 online state、native-x0、
   batch、参数梯度 0.10 和 evaluator 完全不变，续训 5k steps。

三者必须使用同一个 step-10k 模型/optimizer/EMA 前缀。实现同时记录 fresh/aux 全参数
cosine 和 combined/fresh cosine。`PREFIX-FROZEN` 若不能把 15k FID 恢复到 16.40 以下
（matched FRESH 的 10% 范围），则停止所有 paired-x0 self-distillation，下一方向必须
引入真实分布锚点或 solver-index gated correction；若恢复但未优于 14.9119，只证明
moving feedback 是主要故障，仍不进入完整 300-epoch；只有低于 14.9119 且动态范围不
比 FRESH 偏离超过 3%，才有资格进入下一组三分支从零正式周期。该筛选不能冒充从零结论。

筛选实现新增 `clean_loop.teacher_mode=moving|frozen`。frozen teacher 在 warmup 边界从
当前 EMA 快照一次，之后不再更新；moving evaluation EMA 仍照常更新并用于 FID。两者
分别写入 checkpoint，post-warmup resume 若缺少 frozen teacher state 会拒绝运行，避免
静默改变实验定义。参数 VJP 同时报告 fresh/aux cosine 和 combined/fresh cosine。全套
测试为 86 passed；step-10k 两步 smoke 的 aux cosine 为 0.5300/0.2595，combined/fresh
cosine 为 0.9968/0.9959，实际梯度比均为 0.10000。额外从包含 frozen state 的 checkpoint
恢复一步，日志确认 teacher 正确恢复而非重新快照。上述 smoke 只验证实现，不参与 FID
判断。

#### Cycle 05 共享前缀机制筛选结果

三支均从 Cycle 04 ONLINE 自己的同一个 step-10k model/optimizer/EMA checkpoint 开始，
各续训 5k steps；moving 复用既有结果，另两支由 commit `a070e6c` 在 GPU 6 运行。固定
EMA/Heun-20/CFG-2.2/5k FID 为：

| shared-prefix branch | teacher | step-15k FID | 相对 PREFIX-FRESH |
|---|---|---:|---:|
| PREFIX-FRESH | 无 auxiliary | **14.9085** | — |
| PREFIX-MOVING | 当前 moving EMA | 44.8095 | +29.9011 |
| PREFIX-FROZEN | step-10k frozen EMA | 20.4158 | +5.5073 |

PREFIX-FRESH 与独立从零 FRESH-TIME step-15k 的 14.9119 只差 0.00345，证明共同前缀和
continuation 没有隐藏质量问题。冻结 teacher 相比 moving 恢复 24.3937 FID，确认 moving
teacher/student 正反馈是动态范围崩溃的主要来源；但 frozen 仍未达到预注册 16.40 恢复
线，更未优于 FRESH。因此不启动 paired-x0 的从零 300-epoch 正式周期，也不通过降低
梯度比例继续扫描。

256 个同 root 三模型 trace 显示，PREFIX-FRESH/FROZEN/MOVING 最终 RMS 分别为
1.00118/1.00443/1.09814；frozen 相对 fresh 仅高 0.32%，`|x|>1.5` 占比也为
14.85%/14.92%，已经消除 moving 的尺度膨胀。frozen 的平均 Heun correction 更低
（0.001966 对 fresh 0.002119），多数中低噪声 x0 drift 也更低，但最终 pairwise RMS
仍为 0.22376，FID 明显更差。它说明 frozen step-10k teacher 把 student 约束在旧的、
FID 19.5863 的 inference field 附近，阻止了无辅助 FRESH 在 10k→15k 改善到 14.9085；
“更直/更自洽”依旧不是“更接近真实分布”。原始结果在
`runs/x0loop_v2_from_scratch/cycle05-screen/trajectory_prefix_three_step15000/`。

frozen 全窗口 aux 参数 cosine 约 0.49–0.52，combined/fresh cosine 约 0.9967–0.9969；
aux loss 从约 0.00335 降至 0.00280，实际梯度比始终 0.10000。即使单步全局方向与 fresh
近乎同向，持续的子空间偏置仍足以损伤生成语义。由此停止 moving/frozen EMA
paired-x0 **self-distillation**。下一项只允许测试一个不计正式训练次数的强锚点诊断：
固定使用已收敛 FRESH-TIME step-58.5k EMA（权威 FID 5.4704）生成相同 Heun occupancy
和 native-x0 target，student 仍从共同 step-10k 前缀续训 5k。该分支是外部 teacher
distillation readiness，不是 x0loop 改善，也不能冒充从零结果；它必须把 15k FID 降到
13.42 以下（相对 PREFIX-FRESH 至少改善 10%）且最终 RMS 偏差不超过 3%，才证明
inference-state paired target 在 teacher 有真实数据质量锚点时值得保留。否则彻底停止
paired target，转向显式终点分布匹配和 solver-index gated correction。

强 teacher 诊断实现允许 frozen mode 从显式 workspace-relative checkpoint 读取 EMA，并
校验 time-conditioning 和全部参数 key；路径写入 resolved config，teacher shadow 随 student
checkpoint 继续保存。全套测试为 89 passed。用 FRESH-TIME step-58.5k EMA 做 one-step
smoke，日志确认加载 step 58,500，aux 参数 cosine 为 0.1916、combined/fresh cosine 为
0.9954、实际比例为 0.10000；实现门槛通过，数值结果仍必须由预注册 5k FID 判断。

### Cycle 05 强 teacher 诊断结论：停止 paired-x0 target

强 teacher 分支从与 PREFIX-FRESH 完全相同的 step-10k checkpoint 继续 5k steps，固定
aux batch ratio 0.125、参数梯度范数比 0.10、EMA/Heun-20/CFG-2.2、seed 20260819。
step-15k 的固定 5k FID 为 **16.4361**，未通过预注册的 13.42 门槛，并且比同前缀
PREFIX-FRESH 的 14.9085 恶化 1.5276（+10.25%）。因此强 teacher 没有证明当前
inference-occupancy paired-x0 target 值得进入从零 300-epoch 正式实验；本结果只属于
外部知识蒸馏 readiness，不能解释为 x0loop 收益。

训练过程的辅助参数梯度范数比始终为 0.10000。除含首次编译的第一个 1k block 外，后四个
1k block 的 aux/fresh 参数 cosine 为 0.4056、0.4086、0.4050 左右，combined/fresh
cosine 均约 0.9963；稳定 step 为约 0.528--0.532 秒。辅助方向既没有超出预算，也不是
全局反向冲突，但持续的小幅子空间偏置仍损害 FID。

固定 root 的 256-sample 轨迹分析位于
`runs/x0loop_v2_from_scratch/cycle05-screen-strong-teacher/trajectory_step15000/`：

| model | final RMS | mean Heun correction | max relative correction |
|---|---:|---:|---:|
| PREFIX-FRESH | 1.001185 | 0.002119 | 0.219728 |
| STRONG-STUDENT | 0.993560 | 0.001888 | 0.192242 |
| STRONG-TEACHER | 0.997317 | 0.002267 | 0.197959 |

STRONG-STUDENT 相对 PREFIX-FRESH 的 final RMS 仅低 0.76%，满足预注册的 3% 尺度条件，
且 Heun correction 更小；但两者最终样本 RMS 距离为 0.23403，FID 反而显著变差。
student 与 strong teacher 的最终样本距离仍为 0.31007。至此可以排除“只要 teacher 足够强，
paired-x0 就会把 inference state 拉向更好分布”这一工作假设；更低 trajectory defect、合理
动态范围、正梯度 cosine 都不能替代真实分布锚点。

按预注册规则，后续彻底停止 moving/frozen/external-teacher paired-x0 target，不进行
aux ratio、teacher checkpoint 或 bank size 扫描。下一设计必须同时满足：

1. 完整保留 fresh loss；分布辅助项仍以参数梯度范数控制在 fresh 的 10% 起步。
2. 直接用真实 CIFAR-10 与实际 Heun-20/CFG-2.2 终点样本构造分布目标，不使用 paired
   ancestral GT、EMA x0 或 teacher x0 作为生成质量代理。
3. 分布梯度只进入显式的 solver-index gated correction path，初始主干函数与 FRESH-TIME
   完全等价；这样可以隔离“分布目标是否有效”和“污染完整 vector field”两个问题。
4. 在启动训练前先通过 frozen-generator held-out critic readiness 与固定 batch 参数梯度
   检查；禁止重复 Cycle 03 中 co-trained critic AUC 接近 0.5 时仍更新 generator 的错误。
5. 下一组三分支定义为同初始化、从零 300 epochs 的 `FRESH-TIME`、`GATED-CONTROL`
   （有 correction path 但无分布梯度）和 `GATED-DIST`。训练和 FID sampler 都必须加载
   同一 correction path；每完成三支后再做一次全局复盘。

在实现 `GATED-DIST` 前，先对新的 FRESH-TIME step-58.5k EMA 运行一次不更新 generator
的终点 critic readiness。门槛在看结果前冻结为：class-matched fixed split 上 held-out
AUROC ≥ 0.70、train/held-out AUROC gap ≤ 0.05、real-minus-fake logit margin > 0。三项同时
通过才允许使用对抗分布梯度；任一失败则不训练 GAN generator，改用非对抗、可微分的
固定特征分布距离。这个 readiness 不计入三次正式训练。

## 时间与吞吐分析

300 epochs 在 CIFAR-10、batch 256 下是 58,500 optimizer steps（每 epoch 约
195 steps）。此前估计的 12–13 小时 ONLINE 与 21–25 小时 BANK-FIX 并非 epoch
配置过大，而是旧辅助实现把单 step 分别从 FRESH 的约 0.107 秒放大到约
0.87–1.0 秒和 1.5–1.6 秒。

旧 BANK-FIX 的主要瓶颈是按 solver index 分组后，针对很小的 parent batch 串行执行
EMA teacher 和 CFG forward；平均 depth 约 9 时会产生几十到上百次小 kernel，实测
GPU 利用率仅约 24%。优化保持训练和采样的 Heun-20/CFG-2.2 数学定义不变：

1. 不同 solver time 的 parent state 合并成一个异构时间 batch，一次完成 batched
   Heun 更新；BANK 稳态由约 1.5–1.6 降至约 0.276 秒/step。
2. CFG conditional/unconditional 合并成一次双倍 batch forward；BANK 进一步降至
   约 0.210 秒/step。
3. `torch.compile(mode=default)` 修复 eager/compile checkpoint 与 EMA key 兼容后，
   早期短 smoke 中 FRESH 曾达到约 0.050 秒/step；Cycle 02 长窗口的权威中位数为
   0.0849 秒/step，BANK-X0 为 0.1572 秒/step。
   `reduce-overhead` 会与连续 CFG CUDA Graph 输出复用冲突，明确不采用。
4. 测试过每 4 step 集中刷新 128 条 bank state，但稳态反而约 0.27 秒/step，原因是
   CPU bank 写入突发；默认保留每 step 刷新。
5. bank 改为 GPU tensor ring 后，compiled BANK 稳态中位数从约 0.191 降到约
   0.157 秒/step，方法级 MFU 从约 4.0% 升至 4.84%；显存峰值约 8.68 GiB。
6. ONLINE 的可变 active batch 让静态 compile 反复重编译；Cycle 02 最终稳定长窗口约
   0.463 秒/step、方法级 MFU 3.37%、显存 13.50 GiB。独立 100-step `dynamic=true`
   smoke 的首步编译从 104 秒增至 200 秒，但稳定窗口降至 0.387 秒/step、方法级 MFU
   4.04%、显存 8.61 GiB。与 static 相同 28 个记录点的 loss 平均相对差为 0.026%，
   fresh loss 为 0.021%，属于编译数值差异。只有后续 FID 证据仍需要 online rollout
   时才采用该优化。

MFU 使用 `torch.utils.flop_counter.FlopCounterMode` 对当前 59.48M JiT 实测：batch-256
主模型 forward+backward 为 5.154 TFLOP。方法级计算再计入 CFG 双 batch、EMA teacher
Heun forward 和辅助 backward，并除以本机 700 W H800 的 dense BF16 989 TFLOP/s。
稳定窗口可由 `experiments/x0loop_v2/analyze_training_efficiency.py` 复算：

| Run | 中位 s/step | img/s | 峰值 GiB | 主训练 MFU | 方法级 MFU | 方法 TF/step |
|---|---:|---:|---:|---:|---:|---:|
| FRESH-ACCEL | 0.0849 | 3060 | 6.93 | 6.14% | 6.14% | 5.154 |
| BANK-X0 | 0.1572 | 1640 | 8.68 | 3.32% | 4.84% | 7.516 |
| ONLINE-X0 static | 0.4629 | 554 | 13.50 | 1.13% | 3.37% | 15.441 |
| ONLINE-X0 dynamic smoke | 0.3867 | 666 | 8.61 | 1.35% | 4.04% | 15.441 |
| Cycle 04 FRESH-FIXED-REPRO | 0.0667 | 3841 | 7.15 | 7.82% | 7.82% | 5.154 |
| Cycle 04 FRESH-TIME | 0.0664 | 3855 | 7.15 | 7.85% | 7.85% | 5.154 |
| Cycle 04 ONLINE-X0-TIME | 0.5400 | 474 | 13.94 | 0.97% | 3.70% | 19.736 |

据此，58,500-step/300-epoch 的当前长窗口预算为：FRESH 稳定 step 的纯训练外推约
1 小时 5 分；Cycle 04 实际从启动到完成训练和三次中途 FID 为 1 小时 8 分 48 秒；
向量化、编译后的 BANK 约 2 小时 33 分；ONLINE static 约 7 小时 31 分，dynamic
约 6 小时 17 分（另加约 3 分 20 秒首次编译）。中途 5k FID 只在
15k/30k/45k 三次运行，Cycle 04 实测共 3 分 33 秒；最终 50k FID 实测 11 分 31 秒。
实际 wall time 还包含依模式而异的首次编译约
1–4 分钟和 checkpoint 写盘。不能仅为提高利用率增大全局 batch：这会减少 300 epochs 内的
optimizer steps 并改变 FID 对照定义。

## 运行记录

### 2026-08-19：Cycle 01 启动

- 全尺寸 smoke：FRESH 约 2.7 s/step，ONLINE 含 rollout 约 16.2 s/step；峰值显存
  分别约 10.45 GB 与 13.00 GB。
- 正式 FRESH 已在 GPU 6 启动；稳定训练阶段约 0.11–0.14 s/step（不含 FID）。
- 正式 ONLINE 已在 GPU 7 启动。step 10,000 前关闭辅助项，以保证 teacher 已形成有效
  EMA lag，并与 FRESH 保持相同 warmup；step 10,000 后才计入在线 rollout 成本和信号。
- ONLINE 在 step 15,000 的固定 FID 与同 root sampler trace 共同确认低噪声末段坍缩，
  因最终 FID 是主目标，不再花约 11 小时继续该被支配定义；中断时额外保存了
  step-15,102 checkpoint。
- BANK-FIX 随后在 GPU 7 从共享 FRESH step-5,000 随机初始化前缀启动，继续 53,500
  steps。这样三个方法在启用辅助项前共享完全相同的模型/优化器前缀。
- step 5,000 的固定噪声 5k FID：FRESH 与尚未启用辅助项的 ONLINE 都为
  `40.02581897388575`。两者共同日志窗口的 loss、fresh loss、x0/v MSE、LR 和
  grad norm 也逐项完全一致，验证了 warmup 可比性及 FID 随机流隔离。
- 首次 5k FID 单卡耗时约 109–115 秒；生成临时图在计算后清理，指标已写入各 run 的
  `gen_eval_metrics_*.jsonl`。
- 两条进程首次到 step 10,000 时，训练内的可视化采样读取不存在的 `sample` 配置并以
  `KeyError` 退出，发生在 optimizer step 完成之后、checkpoint 保存之前。已修复为缺省
  配置安全读取；正式实验关闭重复的训练内 trace（保留 5k FID 和训练后三模型固定-root
  trace）。两条分支统一从 step 5,000 checkpoint 恢复，并限制 continuation 为 53,500
  steps，确保最终仍恰好为 global step 58,500。
- 恢复后 step 10,000 checkpoint 与 5k FID 均成功，FRESH/ONLINE 同为
  `22.184701185475376`。ONLINE 随后启用 aux：batch 32，首批窗口 solver index 均值
  约 9.48（目标为 0–19 均匀 occupancy），输出梯度比为 0.20，稳定 step 时间约
  0.87 秒。按该实测速度保留原始在线设计，不使用冷启动 smoke 的 16.2 秒外推。
- FRESH step 15,000 的 5k FID 为 `16.76314923002724`，曲线继续正常下降。
- FRESH step 20,000 的 5k FID 为 `14.01027652750281`。相邻 5k 区间的改善量从
  17.84、5.42 降到 2.75，但尚未形成平台，因此继续完整基线训练。
- FRESH step 25,000 的 5k FID 为 `12.300025363403847`，相对 20k 再改善 1.71。
- FRESH step 30,000 的 5k FID 为 `11.72661705146561`，相对 25k 仅改善 0.57，
  首次接近平台区；至少继续观察 35k/40k 后再判断最佳 checkpoint 是否前移。
- FRESH step 35,000 的 5k FID 为 `11.476269195858038`，相对 30k 仅改善 0.25；
  平台趋势增强，但 35k 仍是当前最佳 checkpoint。
- FRESH step 40,000 的 5k FID 为 `11.120722019962102`，相对 35k 改善 0.36，
  继续刷新最佳；30k 后是放缓而不是完全停止。
- FRESH step 45,000 的 5k FID 为 `10.595904527588118`，相对 40k 改善 0.52，
  再次刷新最佳。
- FRESH step 50,000 的 5k FID 为 `10.647890983544585`，比 45k 略差 0.052；
  属于平台内小幅回摆，当前最佳 checkpoint 保持 45k。
- FRESH step 55,000 的 5k FID 为 `10.615361312725327`，仍比 45k 略差 0.019；
  45k 基本确认为 FRESH 的固定 5k checkpoint 选择点。
- FRESH 完成 58,500 steps；最终 step 的权威 50k FID 为
  `6.3991110891795415`。step-45k（5k FID 选择点）的独立 50k FID 正在计算，完成后
  决定 FRESH 的最终报告 checkpoint。
- 首次独立 step-45k 评测暴露 evaluator 重建错误：`eval_fid` 未传入训练配置中的
  `model_conditioning.ignore_time=true`，导致模型看到真实 t 而不是固定 t=0.5；所得
  50k FID 12.4329 和诊断 5k FID 16.5109 均作废。已修复并加入回归测试，修复后先以
  5k 精确复现训练内 10.5959，再重跑 50k。
- 修复后的 step-45k 独立 5k FID 精确复现 `10.595904527588118`；权威 50k 结果为
  FID `6.214322283416891`、KID `0.0017544317245483398`、precision `0.74544`、
  recall `0.44536`。它优于最终 step-58.5k 的 50k FID `6.3991110891795415`，因此
  FRESH 最终报告 checkpoint 确定为 step 45,000。
- ONLINE step 15,000 的 5k FID 为 `26.47479848787509`。它比同 step FRESH
  `16.76314923002724` 差 9.71，也比自身 step-10k warmup 退化 4.29，是明显负向信号。
  同 root sampler trace 又确认低噪声末段动态范围坍缩，因此在 step 15,102 提前终止，
  避免继续消耗约 11 小时；BANK-FIX 仍按原定义运行到 15k 决策门后再统一复盘。

## 运行命令

```bash
X0LOOP_GPU=6 experiments/x0loop_v2/run_from_scratch.sh fresh
X0LOOP_GPU=6 experiments/x0loop_v2/run_from_scratch.sh bank-fix
X0LOOP_GPU=7 experiments/x0loop_v2/run_from_scratch.sh online
```

三次训练完成后的固定 root 采样分析：

```bash
CUDA_VISIBLE_DEVICES=7 uv run python -m experiments.x0loop_v2.analyze_sampling_trajectory \
  --checkpoint fresh=runs/x0loop_v2_from_scratch/cycle01/fresh/checkpoints/ckpt_step_00058500.pt \
  --checkpoint bank_fix=runs/x0loop_v2_from_scratch/cycle01/bank-fix/checkpoints/ckpt_step_00058500.pt \
  --checkpoint online=runs/x0loop_v2_from_scratch/cycle01/online/checkpoints/ckpt_step_00058500.pt \
  --out runs/x0loop_v2_from_scratch/cycle01/trajectory_analysis
```

每个分支按固定 5k FID 选择自身最佳 checkpoint，再分别运行权威 50k FID：

```bash
X0LOOP_GPU=7 experiments/x0loop_v2/eval_cycle_best_fid50k.sh \
  runs/x0loop_v2_from_scratch/cycle01
```
