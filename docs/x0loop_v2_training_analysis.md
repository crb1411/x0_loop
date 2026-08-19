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
| FRESH | 40.0258 | 5,000（当前） | — | — | — | — | — | — |
| BANK-FIX | — | — | — | — | — | — | — | — |
| ONLINE | 40.0258 | 5,000（warmup） | — | — | — | — | — | — |

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
