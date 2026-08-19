# x0loop 长期研究指导原则

> 状态：生效中
>
> 适用范围：本仓库内后续全部 x0loop 设计、实现、训练、采样、评估和性能优化
>
> 原则版本：v1（建立长期研究约束）

本文是本项目后续所有 x0loop 训练、采样、评估和性能优化的最高层研究约束。
具体实验方案可以迭代，但不得静默偏离本文。若证据要求修改原则，必须先在本文记录
修改原因、证据和影响，再开展新的实验。

## 1. 研究目标与结论边界

最终目标是稳定降低真实生成协议下的 CIFAR-10 FID，并确认改善来自 x0loop 方法本身，
而不是评估波动、训练预算差异或实现错误。FID 是最重要的结果指标和实验决策依据之一，
但不是孤立的唯一证据：KID、precision、recall、不同 NFE 的结果和采样轨迹用于判断
FID 改善是否稳定、是否牺牲多样性，以及改善发生在采样过程的哪个环节。

一次具体实现失败，只能否定该实现和对应实验条件，不能自动否定“历史 x0 有效”这一
更宽泛的假设。尤其是 endpoint、训练状态分布、监督目标或 sampler 未闭环时，结论必须
表述为“当前实现没有验证到 x0loop 的核心假设”。

## 2. 训练与推理必须形成同一个闭环

每种方法必须先定义唯一的目标 inference kernel，再据此生成训练中的 rollout state。
以下要素必须在训练、轨迹分析和 FID 采样之间一致，并写入配置或实验记录：

- endpoint 分布及其可学习参数；
- solver、时间网格、步数和状态 transition；
- CFG scale、conditional/unconditional 组合方式和 label 规则；
- 噪声耦合、root noise、时间条件和模型输出参数化；
- online、EMA teacher 和 replay producer 的具体版本。

修改训练方法后，必须同步分析对应采样过程。不能只观察 loss 或终点 FID；至少要在固定
root noise 上比较逐步状态范数、x0 漂移、velocity、Euler/Heun correction、局部 defect
和最终样本动态范围。训练 transition 与最终 sampler 不一致时，不进入正式结论。

## 3. 公平、从零和可归因

用于支持主要结论的分支必须从随机初始化训练到固定的 300 epochs，并共享数据、模型、
优化器、学习率、EMA、batch、随机种子规则和 optimizer-step 预算。共享早期 checkpoint
只允许用于低成本筛选或定位问题，不能替代从零复现实验，也不能被描述为独立的从零结果。

每轮优先只改变一个具有明确因果假设的因素。若必须同时修改多个耦合项，实验记录必须
说明为何无法拆分，并补做必要消融。任何方法不得通过减少 fresh exposure、增加总训练
步数或改变最佳 checkpoint 选择规则获得隐性优势。

fresh loss 默认完整保留。x0loop 信号首先作为受控辅助目标引入，记录实际输出梯度范数
比例；初始目标范围为 fresh 梯度的 10%–30%。是否扩大比例由 FID 和轨迹证据决定，不能
仅凭辅助 loss 下降决定。

## 4. 固定 FID 协议与指标解释

权威比较采用 EMA、固定 sampler/CFG、固定 CIFAR-10 reference statistics、固定预处理
和 50,000 张生成样本。seed、label 顺序、checkpoint 选择规则和实现版本必须留档。
主报告同时给出训练结束 checkpoint 与按预先声明规则选出的最佳 checkpoint，禁止看到
最终结果后修改选择规则。

训练中使用较少的固定 5k FID 作为早停和选点信号，默认只在 15k、30k、45k steps 评估；
最终候选才运行 50k FID。5k FID 只用于筛选，不能替代权威 50k FID。接近或优于基线的
结果必须检查重复评估或多 seed 稳定性，并报告差值而非只报告最好一次。

对有希望的最终候选补充 NFE 4/8/20 的 FID、KID、precision 和 recall：

- 只有低 NFE 改善时，将方法定位为少步采样或误差自纠正；
- FID 改善但 recall 明显下降时，必须标记潜在的多样性代价；
- trajectory defect 改善但分布指标不变时，停止继续调 replay，转向终点分布匹配；
- loss 改善而 FID 与采样轨迹不改善时，不视为研究进展。

## 5. 三次训练为一个研究周期

每完成三次训练，必须暂停新增大实验，复盘整个周期后再设计下一轮。三次训练应尽量包含：

1. 同预算 FRESH 基线或可信的基线复现；
2. 当前最小 x0loop 改动；
3. 用于区分主要机制的对照或消融。

周期复盘必须同时回答：最终/中途 FID 是否超过波动，fresh exposure 是否相同，辅助梯度
是否受控，rollout occupancy 是否覆盖目标 sampler，全链路配置是否一致，replay age/depth
是否造成 off-policy，逐步采样误差如何变化，以及下一轮修改由哪条证据直接支持。

在一次周期复盘之前，不进行无方向的大规模超参数扫描，尤其不优先扫描 bank size。
优先级固定为：先修 endpoint 和轨迹对齐，再验证 online/off-policy，再考虑 replay 容量或
更复杂的终点分布损失。

## 6. 计算效率服务于同一实验定义

持续测量 step time、数据等待、forward/backward、rollout、bank 操作、评估、显存、GPU
利用率和方法级 MFU。优先消除串行小 kernel、CPU/GPU 同步、重复 CFG forward 和低效
bank 搬运，让 GPU 尽量满载。

性能优化必须做数值等价 smoke 和短程曲线对照。不能为了提高 MFU 而改变有效 batch、
optimizer steps、sampler 数学定义或比较预算；若确需改变，必须建立新的同预算基线。
wall time 与 GPU-hours 都要报告，FID 评估时间与纯训练时间分开记录。

## 7. 证据、复现与停止规则

每次实验保存完整配置、git commit、命令、环境、seed、日志、checkpoint、指标 JSON 和
固定 root trace。实验分析写入 `docs/x0loop_v2_training_analysis.md`，原始产物放在对应
run 目录；失败和被证伪结果同样保留，禁止只记录成功结果。

满足以下任一条件时应提前停止分支并记录原因：明显数值失稳；固定 FID 持续显著劣于
同 step FRESH 且轨迹给出一致退化证据；实验实现已偏离目标 inference kernel。节省的
算力用于更能区分假设的实验，而不是在已被支配的配置上补足形式上的 300 epochs。

任何“改善”至少需要通过实现正确性、同预算对照、权威 FID 和采样轨迹四道检查。
最终研究判断以可复现的总体证据为准，不以单次最好数字、训练 loss 或视觉样例为准。

## 8. 指标优先级与实验决策顺序

所有实验按以下顺序判断，前一层未通过时，不得用后一层的好结果掩盖问题：

1. **正确性门槛**：endpoint、训练 transition、inference kernel、CFG、EMA、时间条件、
   数据和 evaluator 均与声明一致；不通过时 FID 数字无效。
2. **核心结果**：固定权威协议下的 50k FID。FID 是最重要的结果参考指标；正式声称
   方法改善时，必须以它为主要依据。
3. **稳健性与代价**：KID、precision、recall、不同 seed 和 NFE 4/8/20 用于判断
   FID 改善是否稳定，以及是否来自 mode dropping、过平滑或只对某个 sampler 有效。
4. **机制证据**：同 root trajectory、occupancy、defect、梯度比例和 replay 元数据解释
   改善或退化为何发生。机制指标不能替代 FID，但决定下一步该修改什么。
5. **计算效率**：在结果等价时，优先 wall time、GPU-hours 更低且 MFU/吞吐更高的方案；
   计算优势不能补偿生成质量退化。

禁止在看到结果后更换 reference statistics、seed 集合、样本数、sampler、CFG、EMA 或
checkpoint 选择规则。若两个 50k FID 的差距接近评估波动，结论写为“未分出胜负”，
并用预先固定的额外 seed 重复评估或训练；不得只挑更好的 seed。

## 9. 一次训练和三次训练周期的定义

“一次训练”指一个在启动前已经冻结配置、假设、预算和停止规则，并产生独立 run 目录、
日志与可审计结果的分支。以下情况分别处理：

- 达到预定 300 epochs 的 run，计为一次完整正式训练；
- 按启动前声明的 FID/数值稳定性/轨迹门槛提前停止的 run，计为一次有效证伪训练，
  但不得冒充 300-epoch 结果；
- 因代码错误、机器故障或 evaluator 错配而退出的 run 不计数，修复后必须重跑；
- 从共享 checkpoint 分叉的短程 run 只计为筛选训练，其结论用于选择下一轮，不作为
  独立从零证据。

每累计三个有效训练，无论完整结束还是按规则提前证伪，必须先更新实验分析文档并完成
一次周期复盘。若周期内修改过训练目标、rollout、endpoint、teacher、状态构造或 loss
权重，复盘中必须重新分析采样过程；不能沿用修改前 sampler trace 的结论。

## 10. 每次实验的强制执行清单

启动前在 `docs/x0loop_v2_training_analysis.md` 或对应 run manifest 中冻结：

- 单句可证伪假设，以及相对哪个 FRESH/当前最佳方法比较；
- 唯一 inference kernel 和与之闭环的训练状态构造；
- 训练预算、fresh exposure、辅助梯度目标、GPU、seed 和停止规则；
- 5k 筛选点、50k 权威评估条件、checkpoint 选择规则；
- 预期 step time、显存、MFU/吞吐，以及发生退化时要检查的 trajectory 指标。

训练中记录 step time 分解、GPU 利用率、显存、loss、梯度比例、rollout/replay 统计和固定
5k FID。性能数据只采用 warmup/compile 完成后的稳定窗口，并同时报告：样本吞吐、
optimizer-step 吞吐、训练主模型 MFU 和包含 teacher/rollout 的方法级有效 MFU。MFU 的
计算口径、GPU 型号、精度和 batch 必须一致，否则只能做定性比较。

结束后先校验 checkpoint、配置与 evaluator，再写入最终指标和同 root trace。任何训练
方法修改都必须检查 Heun 全时间网格，重点报告低噪声末段的 x0 drift、相对 correction、
样本 RMS/动态范围和类条件行为。最后明确写出：支持/不支持哪个假设、能否进入正式
50k FID、下一项修改由哪条证据触发。

## 11. 工程与运行约束

除唯一允许的外部数据根目录 `/mnt/data/crb/data` 外，代码、配置、日志、checkpoint、
评估产物和脚本引用均使用基于仓库或当前脚本位置的相对路径，不依赖某台机器的工作区
绝对路径。项目命令统一通过仓库内 uv 环境执行，以 `uv run ...` 作为可复现入口。

正式实验优先使用 GPU 7，GPU 7 不可用时使用 GPU 6。启动前确认卡上没有冲突进程，
并以稳定阶段的显存、利用率、功耗和 profiler 时间分解判断是否真正吃满算力。允许做
向量化、编译、融合 CFG forward 和 GPU-resident replay 等等价优化，但必须先通过数值
smoke；任何改变训练数学定义的“提速”都视为新方法，必须建立新基线。

## 12. 原则变更治理

本文优先于具体实验 README、临时计划和口头假设。后续每次 x0loop 迭代开始前必须先读
本文并在实验记录中注明遵循的原则版本。若新证据要求偏离本文，先在下方追加变更记录，
写明证据、修改内容、影响到的历史结论和生效实验；不得先运行后补理由。

### 变更记录

- v1：确立 FID 核心地位、训练/推理闭环、300-epoch 公平对照、三次训练复盘、采样
  强制分析、MFU/吞吐约束和实验可复现要求。
