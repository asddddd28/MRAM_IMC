# MRAM-IMC 项目进度与后续任务

更新时间：2026-09-15

## 最新：固定RAM恢复地址交换（2026-09-15）

新增MacroStateRAM功能模型，将交换并入状态写回/恢复，不改输入和权重。保留现有位面抵消，第一层T64/16样本：不交换最大33；全16每16步固定恢复地址交换最大19（各位面19/18/16/12/14），活跃8固定交换24。原8样本每步交换最大24，未优于16步周期。

关闭全部位面转发和抵消，原8样本仅地址交换使66→33，从7bit降至6bit，未达5bit。每次地址恢复逐位匹配直接交换，所有逐步总膜电位整数误差0。

按每步全部16状态槽读写的基准，各方案逻辑读写记录数相同；未证明实际能耗相同，空闲槽跳过访问及本地RAM路由需另评估。78项测试通过。报告IMC_ResNet/checkpoints/macro_ram_restore_assessment.md，汇总macro_ram_restore_summary.json。正式后端未替换。

## 最新：Macro膜电位成对交换（2026-09-15）

固定权重和输入映射，交换同一神经元上下文内的五MR+五SCU；第一层逐步验证全局整数和，所有方案误差0。原8张诊断T64，原分配峰值33：每8步活跃8个固定交换→24，全16固定交换→19；按近期频率条件交换分别31/22。16×18均摊布局全部Macro有输入，固定交换仍有30→21收益。

新增8张测试：不交换30，活跃固定25，全16固定16。原8张将全16固定交换放宽至每16逻辑步，峰值仍19，交换数减少57.30%。16步周期尚未在新增8张复查。推荐下一步固定配对低频交换的硬件搬运与大样本验证；正式后端未替换。

全11层16张输入活动统计：排除空Macro并按有效行归一化，平均Macro最高/最低接收率1.00～1.83倍，第一层1.25倍；第一层同位置8步窗口最大/均值平均1.805，27.7%窗口超过2。局部冷热差异比全层均值明显。交换不会改变输入频率本身，只改变历史状态归属。

76项测试通过。报告：IMC_ResNet/checkpoints/macro_state_swap_assessment.md；汇总macro_state_swap_summary.json；输入频率macro_input_activity.json。结果仅为整数模型及有限样本，未完成全网交换、模拟非理想和PPA验证。

## 最新：回到独立位面 MR（2026-09-14）

沿用高编号位面转发+b0/b1/b2抵消，完成三套权重×两种第一层输入分配，共6组回放。原始/kd_mr/mt_mr_high在8×36分配下最大MR为33/36/40；在16×18均摊下为30/30/29，均无>31观测。原始权重均摊方案新增8张测试峰值25，也无>31。该结果仅覆盖第一层16张测试，不能宣称整网5bit保证，输入路由尚未物理验证。

新增独立MR精确前向+STE代理，并完成32训练样本的有/无MR约束配对试验。从共享kd_mr出发，两组峰值均停留36，超限观测均17；有约束T64验证94.14%，起点94.53%，不替换权重。训练抽样峰值仅30、超限为0，后续应从训练集挖掘尾部样本/位置。

完成测试索引64、115的全11层实际回放：第一层33，其余层均≤15；局部整数误差0，T16/T64预测与数字参考一致。原8张全网运行中断未计入完成。新增IndependentCancelMR实际累加后端、复现入口和两项测试；联合73项通过。详见 IMC_ResNet/checkpoints/independent_mr_exploration.md 和 independent_mr_results.json。正式部署后端未替换。

## 最新组合试验：8组对照+1次延续（2026-09-14）

统一64训练样本/1epoch，独立256验证；比较多时间窗、MR梯度强度、蒸馏、stem解冻和通道窗口预算。
推荐 **多时间窗+蒸馏+MR约束（kd_mr）**：1024张独立测试T16 **72.3633%→73.8281%**，T64 **92.1875%→92.8711%**。
8风险样本全11层共享MR范围 **[-363,220]→[-332,212]**；9-bit越界 **2560→1351**，仍需 **10 bit**。
128样本延续仅将负峰值-332→-331，验证略退步，不采用。所有9次试验均未达到保持Acc并缩到9bit。
推荐检查点 SNN_ResNet10/checkpoints/shared_mr_sweep_v2/kd_mr.pt，未替换正式模型。
逐Macro误差0、预测分歧0、联合测试 **71 passed**。完整对照见 [组合试验报告](IMC_ResNet/checkpoints/shared_mr_sweep_assessment.md)。

## 最新微调：共享 MR 有符号尾部 QAT（2026-09-14）

新增 --shared-bits 9，以[-248,247]裕量目标训练共享C；原硬复位权重、32训练样本、1epoch，仅layer1.conv1。
与同配置零惩罚对照相比，8风险样本全11层C范围 **[-372,220]→[-329,220]**，负向幅度下降11.56%，仍需10 bit。
128数字测试T64均90.625%；T16对照81.25%，候选75.78125%，短窗口损失明显，暂不采用。
两候选全网共享bank逐Macro误差0、数字预测分歧0；联合测试 **69 passed**。
通道贡献预算、多窗口蒸馏等建议与局限见 [训练评估](IMC_ResNet/checkpoints/shared_mr_training_assessment.md)。
检查点在 SNN_ResNet10/checkpoints/shared_mr_qat_pilot/，未替换正式模型。

## 最新理论与全网验证：每 Macro 一个有符号共享 MR（2026-09-14）

保留5×4-bit SCU，C += c0+2c1+4c2+8c3−16c4。全11映射卷积层/8风险样本/T64，
C范围 **[-363,220]**，普通二补码最小统一 **10 bit**；各层7～10 bit。逐Macro误差0，T16/T64数字预测分歧0。
通用T64无复位界：单fold **13 bit**；本网络最大4fold **15 bit**。没有无时间上界的有限精确保证。
对第一层先前5×6-bit MR，共享10-bit MR将含SCU的每Macro状态从50降至30 bit（40%）。
联合测试 **68 passed**。正式后端未替换。见 [共享MR位宽评估](IMC_ResNet/checkpoints/shared_macro_mr_assessment.md)。

## 最新验证：向 LSB 转发补接 MR0_b5（2026-09-14）

只增加 MR4[1]↔MR0[5]，放在原 b1 仲裁末尾，其余规则不变。
第一层/8样本/T64：MR0..MR4 峰值 **[114,61,24,5,25]**，整体 **123→114**，MR4 **28→25**。
新增连接命中 **129,913** 次；MR≥48 **607,398→107,648**，MR≥64 **71,929→4,650**。
转发次数不变、旧对照完全复现，逐 Macro 误差0，联合测试 **65 passed**。
尾部改善但峰值仍高于原方向33。见 [补接评估](IMC_ResNet/checkpoints/mr_lsb_mr0b5_assessment.md)。

## 最新验证：反向向 LSB 转发（2026-09-14）

保持 b0/b1/b2 抵消，等权正位转发改为 MR3→MR0，优先最低编号空位。
同第一层/8 样本/T64：MR0..MR4 峰值 **[123,61,24,5,28]**，整体 **33→123**，MR4 **25→28**。
转发数 3,504,812→9,348,053，总抵消单位 9,626,937→9,371,054；低位面产生堆积。
旧对照完全复现、逐 Macro 误差 0；联合测试 **63 passed**。当前固定方向策略不采用。
见 [LSB 方向转发评估](IMC_ResNet/checkpoints/mr_lsb_forward_assessment.md)。

## 最新估计：两位合并减法机会

X=MR4[1:0]，Y=MR3[2:1]/MR2[3:2]/MR1[4:3]，仅在原轨迹观察，不启用新减法。
累加后/原抵消前，Y>X 条件比例 **4.4096%**，Y>=X 为 **10.1469%**；
原抵消后尚有 **1,871,314** 个 X=01/Y=10 的借位机会检查。不是去重拦截事件数或 I/(I+E)。
额外 MR0[5:4] 只新增 2 次允许相等的联合匹配。原统计全部复现。
见 [两位减法机会评估](IMC_ResNet/checkpoints/mr_two_bit_subtraction_assessment.md)。

## 最新统计：三位逃逸乘积（2026-09-14）

b2 严格逃逸 60,228 次、抵消 27,823 次，E2/(I2+E2)=68.4013%。
沿用两位主口径，三位乘积 **3.72469%**；I0 含全部 b0 成功配对时为 **2.97514%**。
原轨迹全部复现；该乘积不等同于实际联合路径概率。
见 [三位逃逸统计](IMC_ResNet/checkpoints/mr_three_bit_escape_assessment.md)。

## 最新统计：两位逃逸乘积

补测 b1 向上进位 440,094 次，其中严格无正位伙伴逃逸 323,348 次；b1 成功配对 1,009,468 次。
按 E0×E1/[(I0+E0)(I1+E1)]，沿用严格逃逸且 I0 仅进位前拦截：**5.44535%**；
I0 包含全部 b0 成功配对：**4.34953%**。这是指定乘积指标，不是逐路径两位联合概率。
见 [两位逃逸统计](IMC_ResNet/checkpoints/mr_two_bit_escape_assessment.md)。

## 最新修正：b0 进位逃逸事件，而非等待未命中

实际 b0→b1 进位 2,572,593 次；其中 b0 存活期间各检查点正位一直全 0 的严格事件 1,637,649 次。
严格事件 / 两类 b0 成功抵消 = **21.8449%**；若分母仅进位前拦截，为 **28.9413%**。
此前 94.0433% 是状态未命中率，不能作为用户定义的进位逃逸比值。
轨迹全部复现，联合测试 **60 passed**。见 [逃逸事件统计](IMC_ResNet/checkpoints/mr_b0_escape_assessment.md)。

## 最新统计：MR b0–b4 翻转与抵消机会

当前转发+b0/b1/b2 方案，第一层/8 样本/T64，每位 102,760,448 个有效 Macro-时间步观测。
更新净翻转、上下沿、IF 复位和置 1 比例分开统计。b0 进位单位拦截率 **37.744%**；
残余 b0、b1、b2 状态条件匹配率约 **5.957%、4.451%、0.292%**。
b3 与 MR3_b4 匹配 **0/887,979**，b4 在正位 b0–b4 范围内没有等权伙伴。
原轨迹统计复现、逐 Macro 误差 0；联合测试 **57 passed**。
见 [位活动及配对统计](IMC_ResNet/checkpoints/mr_bit_activity_assessment.md)。

## 最新验证：b2 抵消仅连接 MR3/MR2

新增 MR4[2] ↔ MR3[3]/MR2[4]，在 b1 后检查，其他机制不变。
原转发下，第一层/8 张风险样本/T64 整体峰值 **37→33**，MR3 **37→25**，MR4 **25→25**；
MR0..MR4 峰值 **[28,31,33,25,25]**。新增 b2 配对 27,823 次，额外抵消 111,292 个 MR4 单位。
原转发及 b1 配对次数不变；无转发对照 b2 命中为 0。旧对照统计全部复现，逐 Macro 误差为 0。
联合测试 **54 passed**。MR2=33 仍超过 5-bit 上限；正式后端未改。
见 [b2 抵消评估](IMC_ResNet/checkpoints/mr_b2_cancel_assessment.md)。

## 最新验证：只扩展 b1 负向抵消，转发不变

新增 MR4[1] ↔ MR3[2]/MR2[3]/MR1[4]，旧更新后最多配对清零一路，不含 MR0[5]。
保持原正向转发，第一层/8 张风险样本/T64 整体峰值 **57→37**，MR≥48 **116→0**；
MR0..MR4 峰值 **[28,31,33,37,25]**。新增配对 1,009,468 次，转发数仍为 3,504,812。
无转发时新增组整体峰值仍为 56。所有旧对照统计复现一致，逐 Macro 误差为 0，联合测试 **51 passed**。
当前最高值 37 仍需 6 bit；未接入正式后端或完成全网络验证。
见 [b1 抵消扩展评估](IMC_ResNet/checkpoints/mr_b1_cancel_assessment.md)。

## 最新验证：正位面跨位面发放

按 MR0→MR3 方向，在四个等权位的新 0→1 事件上转发，优先最高编号的空目标。
第一层/同 8 张风险样本/T64，组合负向抵消后，MR0..MR4 峰值为 **[28,31,35,57,42]**；
相对仅负向抵消，整体峰值 **56→57**，MR≥48 次数 **200→116**，MSB 峰值 **25→42**。
单独正向转发整体峰值为 **78**。逐 Macro 膜电位守恒，联合测试 **48 passed**。
当前固定优先级把瓶颈转移到 MR3，未改善整体峰值，不接入正式后端。
见 [正位面转发评估](IMC_ResNet/checkpoints/mr_positive_forward_assessment.md)。

## 最新验证：用户提出的 MR 等权位配对抵消

已验证 MR4[0] 与 MR3[1]/MR2[2]/MR1[3]/MR0[4] 一对一清零守恒；进一步比较了负向进位前抵消。
不改权重与原映射，原模型的第一层/8张风险样本/T64 数值对照：MSB MR 峰值 **64→25**，
所有 MR 峰值 **66→56**，`MR>=48` **10342→200**，`MR>=64` **10→0**，逐 Macro 粗状态误差为0。
剩余峰值在正位平面 MR0；这不是对负膜电位进行有损钳位。独立模块和联合回归 **45 passed**。
详细条件和时序见 [配对抵消评估](IMC_ResNet/checkpoints/mr_pair_cancel_assessment.md)。
原部署后端未改；还未完成全层/128张或物理硬件验证。

## 最新评估：优先调整映射，再继续 QAT

详细评估见 [MR 方法评估](IMC_ResNet/checkpoints/mr_strategy_assessment.md)。
对第一层进行独立数值对照：原 8×36 输入布局改为 16×18，使用原来补零的 8 个 Macro，
输入和权重一致移动，原脉冲/IF reset 轨迹不变。原硬复位峰值 **66→46**，当前 QAT 模型 **62→48**，
每步有符号整数累加误差均为 0。仅测试 8 张风险样本的第一层全空间/T64，未接入全网新映射或验证物理路由。
最大计数下降但平均活动位宽上升，不能声称能耗或总存储同比下降。
建议先确认输入分发约束并完成新映射全网/128 张回归，再围绕新映射做 QAT；5-bit 目标仍要求峰值≤31。

## 最新进展：首次 MR-aware QAT 试验

**本轮完成结果：**梯度平衡 MR 候选（`mr_qat_t64_balanced/alpha_10.pt`）在 8 张风险诊断集上，
相对同组零惩罚对照 max MR **65→62**、`MR>=64` **2→0**、`MR>=48` **22936→18477**。
原硬复位在同集为 max MR=66、10 次超限。全 11 层 Cluster 与 int5 数字参考 T=16/T=64 预测均一致。
前 128 张数字测试 T=64 **89.84375%→90.625%**，T=16 **89.0625%→85.9375%**，存在短窗口精度损失。
训练仅使用前 64 张、1 epoch；全空间实际 Cluster 仅验证上述 8 张，尚未证明 128 张或全测试集安全。

已实现并运行 `SNN_ResNet10/scripts/finetune_mr_qat.py`：融合 BN 后的 int5 权重、
16×36×5 二补码位平面、SCU 基数 16、跨时间无符号计数及 IF 发放清零。
前向与实际累加器一致，反向使用类别分布位平面 STE。首轮只训练历史超限层
`layer1.conv1.weight`，固定 BN，保留完整 T=64，空间位置采样。
包含相同种子/数据顺序的零惩罚对照，损失含 overflow、peak、quiet。

**关键修复：**旧 LeakageIFNode 只重写充电函数，但 SpikingJelly 的 eval 快捷路径绕过该函数。
旧文档中 leakage 模型推理成绩不能解释为“启用泄漏”的成绩。已修复旧训练入口和共享节点，
统一数字/Cluster 加载器恢复 leakage、阈值和 QAT 前向权重；新检查点保存融合拓扑和阈值。

结果与限制见 [MR-QAT pilot 报告](IMC_ResNet/checkpoints/mr_qat_pilot_report.md)。
训练记录：`SNN_ResNet10/checkpoints/mr_qat_t64/`、`mr_qat_t64_lr2e4/`、`mr_qat_t64_lr1e3/`、
`mr_qat_t64_stable/`、`mr_qat_t64_balanced/`。最后一组使用 IF detach-reset 梯度和显式 CE/MR 梯度范数平衡；
前向硬复位与 MR 清零不变。早期实验发现 CE 梯度约 1e8，断开复位梯度后仍约 1e4，MR 项被淹没，
详情见 `IMC_ResNet/checkpoints/mr_qat_gradient_diagnostic.json` 和各组 history。
新增 MR/泄漏/重载测试与现有联合回归共 **40 passed**。
全 11 层 Cluster 诊断使用索引 `[0,1,2,3,4,5,64,115]`，覆盖历史全部 7-bit 风险样本。
原硬复位诊断复现 max MR=66、10 次超限观测；leakage 部署初始化为 max MR=68、60 次超限。
**诊断集已无 6-bit 超限，但完整安全目标仍待更大范围验证；不把训练采样峰值当作实际 Cluster 峰值，也不把诊断集精度当作泛化精度。**

以下第 1–7 节保留本轮开始前的历史计划与结果；其中“尚未实现 MR-aware QAT”已被本节更新取代。

## 1. 当前目标

在现有 FashionMNIST SNN-ResNet10 和实际 Cluster 流程上，完成 **Leakage + MR-aware Quantization-Aware Training（QAT）**，在不修改 Cluster 内部硬件结构的前提下，通过权重训练、膜电位泄漏和调度/映射约束降低 MR 峰值，同时保持分类精度和硬复位模型的信息表达能力。

## 2. 已完成工作

### 2.1 基础模型

- 已在 `SNN_ResNet10` 中建立显式时间维度的 SNN ResNet10。
- 使用 FashionMNIST 完成 ANN、SNN 转换和硬复位微调流程。
- 已验证 T=16 与 T=64 的推理差异。
- 原硬复位微调模型：`SNN_ResNet10/checkpoints/hard_reset_finetuned.pt`。
- 原模型前 2000 张测试样本：T=16 为 74.55%，T=64 为 91.75%。

### 2.2 实际 Cluster

- 已将隐藏卷积映射到实际 Cluster/MR/SCU 后端。
- 已完成 int5 数字参考与实际 Cluster 的回归，前 128 张、T=64 为 117/128=91.41%。
- Cluster 与 int5 数字参考逐样本预测一致。
- 已完成 MR 条件监督、Macro/bit-plane 追踪和位宽分布统计。

### 2.3 MR 高位宽分析

- 原始实际 Cluster 最高 MR=66，需要 7-bit；6-bit 上限为 63。
- 所有 7-bit 事件发生在 T=62~64。
- 高 MR 神经元在 64 步内没有发放，MR 未被清空。
- 最大案例中负向权重贡献明显大于正向贡献，说明根因是无符号 MR 长时间累积，而不是简单的正负 Macro 之间跨 Macro 抵消。
- 高 MR 既可能在同一 Macro 内多位共存，也可能分散于不同 Macro；同 Macro 抵消是主要来源。
- 完整报告：
  - `IMC_ResNet/checkpoints/mr_conditions_t64_n128.md`
  - `IMC_ResNet/checkpoints/mr_width_distribution_t64_n128.md`

### 2.4 Leakage + 权重量化感知微调

已新增：

```text
SNN_ResNet10/scripts/finetune_leakage_qat.py
```

实现内容：

- `LeakageIFNode`：`v(t)=lambda*v(t-1)+x(t)`，发放后仍硬复位到 0。
- `FakeQuantConv2d`：逐输出通道对称 int5 fake quant，使用 STE。
- 支持从 ANN checkpoint 或已有硬复位微调 checkpoint 初始化。
- 保存每个 leakage 候选的权重和实验 JSON。

从已有硬复位模型继续微调的结果：

| leakage | T=16 ACC | T=64 ACC |
|---:|---:|---:|
| 1.00 原模型 | 74.55% | 91.75% |
| 0.99 + QAT | **85.80%** | **87.30%** |
| 0.97 + QAT | 84.90% | 84.95% |

说明：这些是前 5000 张训练样本、1 epoch、前 2000 张测试样本的历史记录。已发现 eval 绕过泄漏的问题，表中数字不能作为启用 leakage 的推理成绩；新结果见顶部报告。

结果目录：

```text
SNN_ResNet10/checkpoints/leakage_qat_from_hard/
```

## 3. 当前明确结论

### 已经做了什么

```text
Leakage-aware + int5 weight-QAT
```

### 还没有做什么

```text
MR-aware QAT
```

训练阶段尚未加入：

- MR 状态的跨时间累积；
- 正/负 bit-plane 和 Macro 映射；
- MR 清空与 IF 硬复位同步；
- MR 饱和/溢出代理；
- `MR>=48`、`MR>=64` 或峰值位宽惩罚。

因此目前 MR 位宽仍只在实际 Cluster 推理阶段统计。

## 4. 下一阶段任务：MR-aware QAT

### 4.1 第一阶段：建立可微分 MR 代理

新增一个训练侧简化 Cluster/MR 模块，要求与实际后端保持以下语义：

1. 输入为时间步脉冲；
2. 权重先经过与部署一致的 int5 fake quant；
3. 将权重分解为正负 bit-plane 事件；
4. MR 为无符号计数，并跨时间步累积；
5. 在输出神经元发放后清空对应 MR；
6. 使用 STE 或 softplus 近似处理不可导计数/溢出操作。

不要用单纯的 `clamp(conv_output)` 替代 MR 代理，因为这无法表达 MR 无符号累积、SCU 抵消和发放时序。

### 4.2 第二阶段：定义训练损失

建议初始损失：

```text
L = L_cls
  + alpha * L_overflow
  + beta  * L_peak
  + gamma * L_quiet
```

其中：

- `L_cls`：分类交叉熵；
- `L_overflow`：对 `MR-56` 使用 softplus 惩罚，逐事件累计；
- `L_peak`：对 batch 内 MR 峰值超过 56 的部分惩罚；
- `L_quiet`：对长时间不发放且 MR 继续增长的状态进行轻度惩罚。

训练时先使用 `MR_target=56`，最终检查真实 6-bit 上限 `MR<=63`。alpha/beta/gamma 需要小范围网格搜索，避免分类精度被 MR 约束压垮。

### 4.3 第三阶段：Leakage 与硬复位的联合处理

重点不是简单增大发放率。硬复位会丢弃超阈值残差，因此需要：

- 优先使用轻微 leakage，如 0.99 或 0.995；
- 保留硬复位，不改成软复位；
- 统计 reset 前膜电位和 overshoot；
- 只惩罚异常大的 overshoot，不强制所有神经元提高发放率；
- 让 MR-aware loss 优先压制“长期不发放”的高 MR状态。

### 4.4 第四阶段：实际 Cluster 验证

对每个候选 checkpoint 运行：

- T=16、T=64 数字模型；
- int5 数字参考；
- 实际 Cluster；
- MR 全量或前 128 张监督统计。

必须输出：

- ACC；
- max MR；
- 平均有效位宽和峰值有效位宽分布；
- `MR>=48` 数量；
- `MR>=64` 数量；
- firing rate 与 zero-firing ratio；
- Cluster/int5 逐样本预测差异；
- 与原始硬复位模型的对照表。

## 5. 推荐执行顺序

1. 以 `leakage=0.99` checkpoint 为初始化；
2. 先实现 MR 代理，但用单层/小张量单元测试验证数值等价；
3. 加入不带惩罚的 MR 代理，检查数字输出与现有 Cluster 参考；
4. 加入 `MR_target=56` 的 soft overflow loss；
5. 先小数据、1 epoch 扫描 alpha/beta；
6. 选出候选后运行 T=64；
7. 最后接入实际 Cluster，确认真实 MR 峰值下降且 ACC 可接受；
8. 同步更新 `SNN_ResNet10/README.md` 和 `IMC_ResNet/README.md`。

## 6. 复现入口

```powershell
# Leakage + 权重量化感知微调
python MRAM_IMC/SNN_ResNet10/scripts/finetune_leakage_qat.py `
  --leakages 0.99 0.97 `
  --init-checkpoint MRAM_IMC/SNN_ResNet10/checkpoints/hard_reset_finetuned.pt `
  --epochs 1 --steps 16 --train-size 5000 --test-size 2000

# 现有测试
python -m pytest -q MRAM_IMC/IMC_ResNet/tests MRAM_IMC/model_imc_cluster_simple/tests MRAM_IMC/SNN_ResNet10/tests
```

当前联合测试基线为 **30 passed**。

## 7. 交付判断标准

MR-aware QAT 阶段至少应满足：

- T=64 ACC 相比实际 Cluster 基线下降不超过可接受范围；
- max MR 从 66 降到 63 以下，最好低于 56；
- 7-bit 事件清零或显著减少；
- `MR>=48` 事件明显下降；
- Cluster 与 int5 数字参考仍保持可解释的一致性；
- 所有新增 MR 代理、损失和统计均有单元测试。
