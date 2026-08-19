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

据此，58,500-step/300-epoch 的当前长窗口预算为：FRESH 纯训练约 1 小时 23 分；
向量化、编译后的 BANK 约 2 小时 33 分；ONLINE static 约 7 小时 31 分，dynamic
约 6 小时 17 分（另加约 3 分 20 秒首次编译）。中途 5k FID 只在
15k/30k/45k 三次运行，共约
5 分钟；最终 50k FID 约 15 分钟。实际 wall time 还包含依模式而异的首次编译约
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
