# x0loop v2 从零训练与采样分析

## 最终目标

唯一主指标是 CIFAR-10 最终生成 FID。权威协议固定为 EMA、Heun-20、CFG
2.2、50,000 张样本、seed 20260819。5k FID 用于训练过程中的 checkpoint
选择；KID、precision、recall、trajectory defect 和辅助损失只解释结果，不能替代
最终 50k FID。

所有比较实验从随机初始化开始，不再从历史 checkpoint 分叉。每完成三次训练，必须
先复盘训练设计和采样过程，再修改下一轮方案。未经三次训练复盘，不扫描 bank size。

## 固定训练协议

- 数据：`/mnt/data/crb/data` 中的 CIFAR-10。
- 模型：JiT，dim 512、depth 12、8 heads。
- 主训练：300 epochs，batch size 256，共 58,500 optimizer steps。
- 优化器：AdamW；10k warmup 后 cosine，LR 3e-4 到 5e-5。
- EMA decay：0.996。
- 中途评估：每 5,000 step 保存 checkpoint，并计算固定噪声 Heun-20/CFG-2.2
  的 5k FID。
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

- 状态：运行中（2026-08-19 11:25 CST 启动）
- GPU：6
- 作用：建立相同代码、数据、优化器和最终采样协议下的从零基线。
- clean loop：关闭。
- 输出：`runs/x0loop_v2_from_scratch/cycle01/fresh`

### Run 02 — BANK-FIX

- 状态：待完成
- GPU：6（FRESH 完成后启动）
- 作用：检验对齐 endpoint、Heun-20、CFG-2.2 和 EMA teacher 后，replay 是否提供
  超出 FRESH 的 FID 信号。
- fresh batch：完整保留。
- auxiliary batch ratio：0.125；目标输出梯度比例：0.2。
- replay：按 solver index 分层，记录 depth、root noise ID、producer step。
- 输出：`runs/x0loop_v2_from_scratch/cycle01/bank-fix`

### Run 03 — ONLINE

- 状态：运行中（2026-08-19 11:25 CST 启动；前 10k step 为统一 warmup）
- GPU：7
- 作用：与 BANK-FIX 对比，判断 replay/off-policy 是否是主要瓶颈。
- trajectory：当前 EMA 在线生成实际 Heun-20/CFG-2.2 occupancy。
- fresh batch 与辅助梯度约束同 BANK-FIX。
- 输出：`runs/x0loop_v2_from_scratch/cycle01/online`

### Cycle 01 结果表

| Run | 最低 5k FID | 对应 step | 最终 50k FID | NFE 4 | NFE 8 | NFE 20 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FRESH | 11.1207 | 40,000（当前） | — | — | — | — | — | — |
| BANK-FIX | — | — | — | — | — | — | — | — |
| ONLINE | 22.1847 | 10,000（warmup） | — | — | — | — | — | — |

### Cycle 01 采样轨迹分析

待三次训练完成后填写。至少保存同一组 root noise 和 label 在三个模型上的 Heun
trace，并逐 step 比较状态、velocity、x0_hat、局部 Euler/Heun defect 以及最终图像。
分析工具固定使用 `experiments/x0loop_v2/analyze_sampling_trajectory.py`，输出完整 trace、
最终图像网格、逐步 JSON 指标和 Markdown 摘要。

### Cycle 01 设计结论与下一轮修改

待三次训练完成后填写。修改必须由最终 FID 和采样轨迹共同支持。

## 运行记录

### 2026-08-19：Cycle 01 启动

- 全尺寸 smoke：FRESH 约 2.7 s/step，ONLINE 含 rollout 约 16.2 s/step；峰值显存
  分别约 10.45 GB 与 13.00 GB。
- 正式 FRESH 已在 GPU 6 启动；稳定训练阶段约 0.11–0.14 s/step（不含 FID）。
- 正式 ONLINE 已在 GPU 7 启动。step 10,000 前关闭辅助项，以保证 teacher 已形成有效
  EMA lag，并与 FRESH 保持相同 warmup；step 10,000 后才计入在线 rollout 成本和信号。
- BANK-FIX 将在 FRESH 完成后使用 GPU 6 启动。
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
