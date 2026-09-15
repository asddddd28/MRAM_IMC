# IMC_ResNet — 实际 Cluster 算术与 ResNet 混合推理

## 2026-09-13 更新：MR-QAT 验证

统一加载器恢复 checkpoint 的 leakage、内置 IF 阈值和融合拓扑，并物化旧 QAT 前向权重。
修复 eval 模式绕过泄漏充电的错误。MR-QAT 代理与实际后端通过多 fold、跨时间、选择性清零和泄漏整网对照。

`run_finetuned_cluster.py` 新增 `--test-indices`，结果显式保存全部样本索引；
MR 统计新增 `mr_ge48`/`mr_ge64`，评估新增逐 IF 发放率和零发放比例。

本轮全 11 层 Cluster 使用风险诊断集 `[0,1,2,3,4,5,64,115]`，覆盖历史全部 7-bit 风险样本。
采用 6-bit 上限、wide_reference 观察未经饱和的原始溢出，并与同 checkpoint int5 数字参考逐样本比较。
详细结果见 [MR-QAT pilot 报告](checkpoints/mr_qat_pilot_report.md)。不能替代前 128 张或完整测试集验证。

最终梯度平衡候选相对同组零惩罚对照：max MR **65→62**，`MR>=64` **2→0**，
`MR>=48` **22936→18477**；T=16/T=64 均无 Cluster/int5 预测差异。
报告包括所有层位宽分布、发放率和零发放比例；8 张诊断集无超限不构成全输入保证。

```powershell
python MRAM_IMC/IMC_ResNet/scripts/run_finetuned_cluster.py `
  --checkpoint MRAM_IMC/SNN_ResNet10/checkpoints/mr_qat_t64_balanced/alpha_10.pt `
  --test-indices 0 1 2 3 4 5 64 115 --steps 16 64 --mr-bits 6 `
  --output MRAM_IMC/IMC_ResNet/checkpoints/mr_qat_repeat_stress8.json
```

## 本版与旧版的区别

旧版 `ClusterStats` 只统计 IF 输出，**并未模拟 MR**。旧文件
`checkpoints/cluster_t64.json` 的 91.85% 是 2000 张样本上的浮点 SNN 结果，
不是实际 Cluster 推理结果。其中 `cluster_cycles` 实际是输出元素数量，
`tdp_pulses = events * 5` 也不是有效的 TDP 脉冲数/周期模型，不能用于硬件评估。
旧脚本保留为 `scripts/run_event_accounting.py`，仅用于回溯。

新版 `scripts/run_cluster_inference.py` 使用**真实 SCU/MR 算术计算卷积输出**，
不再报告这些无物理依据的周期/脉冲估算。

## 执行结构

```text
时间步优先：for t in 1..T
  数字 stem 卷积 + IF
  每个残差块：
    输入：IF 的 0/1 脉冲 × 已知 VoltageScaler 比例
    conv1 → [Cluster 位平面 MAC / SCU / MR] → 数字 bias/scale → IF
    conv2 → [Cluster 位平面 MAC / SCU / MR] ─┐
    shortcut（需要投影时也用 Cluster）───────┤ 数字残差相加 → IF
  数字池化 / FC → 累计 logits
窗口读出：累计 logits 的 argmax（等价于时间平均后 argmax）
```

### 实际映射

- 已替换 **11 个隐藏卷积**：8 个残差主分支卷积和 3 个 shortcut 投影卷积。
- 每个 `(样本, 输出通道, 输出空间位置)` 对应一个逻辑 Cluster 状态。
  这是逻辑映射，不意味着芯片必须实例化这么多物理 Cluster。
- 每个 Cluster：16 个 Macro，每 Macro 36 输入，5 个二补码权重位平面。
- 融合 BN 后的权重按输出通道做 int5 PTQ；比例取 `max(abs(weight))/15`，
  本轮实际使用对称 -15..15，内核也支持二补码 -16。
- `fan_in = Cin × Kh × Kw`，每 fold 最多 576 个输入，不足的补零。
  layer3.conv2 / layer4.conv1 使用 2 folds，layer4.conv2 使用 4 folds。
- 每 fold 执行实际位平面 PMAC。浮点 bmm 仅用于 0/1 点积，每项结果为 0..36
  的可精确表示整数；随后转换为整数，不用普通浮点卷积替代结果。
- 每 Macro / bit-plane 保留独立状态：

```python
z = scu + pmac_count
scu = z % 16
mr = mr + z // 16
u = sum((scu + 16 * mr) * [1, 2, 4, 8, -16])
```

- SCU 固定为 4 bit，与现有简化模型的 `16 * MR` 规约一致。
- 所有 folds 更新同一组 MR/SCU，然后才进行当前时间步的 IF 判决。
- 状态跨时间步保留；对应 IF 发放后，同一输出神经元所有 Macro/bit-plane
  的 MR/SCU 一起硬清零。conv2 和 shortcut 共享 block 输出 IF 的清零信号。
- 独立 batch 前重置状态，统计数据不清零；发放清零前的峰值也被保留。

### 数值对接及当前边界（重要）

本版是 **Cluster 算术内核 + 数字外围**，不是全网络全整数硬件模型：

1. 归一化后的连续输入包含负数，stem 暂时仍在数字端计算，未伪装成 0/1 输入。
2. bias、VoltageScaler、残差相加、IF 比较/膜电位及最终池化/FC 仍在数字端。
3. Cluster 返回由 SCU/MR 恢复的累计值，再与上一步累计值做差，作为本步
   数字 IF 的输入增量，防止重复积分；这一步也在数字端。硬清零时相应历史值一起清零。
4. 全部 IF 使用 `v_reset=0` 硬复位，匹配 SimpleCluster 发放清零语义。
   官方默认软复位的 91.85% 不能直接作为本模型不损失精度的承诺。
5. 阈值沿用官方校准后的 IF 阈值和缩放，并在数字域比较，**不直接套用 theta=560**。
6. TDP 使用精确的脉冲积分等价值（SCU 数值）参与规约，不逐 slot 展开整个网络。
   测试检查 eigen_train 脉冲积分与 SCU 数值一致；没有模拟模拟电压、时序或能耗。
7. 内核是 SimpleCluster 的批量化实现，而不是为每个输出点调用慢速 Python
   标量循环。测试逐步比较标量参考的 SCU、MR、规约值、发放和清零后状态。

## MR 位宽统计

默认 `--mr-bits 8 --overflow wide_reference`：阈值为 255，但不截断实际累积值，
所以超出 8 bit 的峰值不会被回绕/饱和隐藏。这不等于实际只用 8 bit 能无损运行。
支持 `error`（超限报错）、`wrap`、`saturate`，用于后续有限位宽实验。
内部整数存储也有范围检查。

对**每一个 fold、清零前**的全部 MR 记录：

| 字段 | 含义 |
|---|---|
| `max_mr` | 本层观察到的最大计数 |
| `max_required_bits` | 最大计数的无符号有效位宽 |
| `mean_observed_bits` | 全部 fold 观测点的有效位宽均值 |
| `mean_logical_peak_bits` | 每个逻辑 MR 先取整段序列峰值，再平均其有效位宽 |
| `over_limit_fold_observations` | 超过配置位宽上限的 MR×fold 观测次数，不是首次溢出次数 |
| `logical_mrs_over_limit` | 本轮峰值曾超限的逻辑 MR 数量 |
| `peak_by_channel_macro_bit` | `[输出通道,16,5]` 峰值表，空间位置/样本取最大 |

零的有效位宽约定为 0；统计包括 80 路物理格式中的补零/空闲位平面，
因此平均有效位宽可以小于 1。它不代表硬件能实现小于 1 bit 的寄存器。
选硬件位宽应看峰值，而非均值。统计窗口为本次运行的最大 T；
同一次运行较短 T 的 ACC 由累计 logits 截取，**不能把最终 MR 统计误当成各个 T 的独立结果**。

## 运行与可复现对照

环境：CPU、已安装的 SpikingJelly 0.0.0.0.14。默认沿用训练集前 1280 张校准，
FashionMNIST 固定前 N 张测试。所有对照使用同一权重、校准、样本和输入归一化。

```powershell
cd D:\Projects\ai
python -m pytest -q MRAM_IMC/IMC_ResNet/tests

# 实际 Cluster 逐 fold 计算并统计所有 MR，远慢于普通 Conv2d；先做小样本调试。
python MRAM_IMC/IMC_ResNet/scripts/run_cluster_inference.py `
  --test-size 32 --batch-size 2 --threads 4 `
  --steps 8 16 32 64 --mr-bits 8 --overflow wide_reference `
  --output MRAM_IMC/IMC_ResNet/checkpoints/actual_t64_n32.json

python MRAM_IMC/IMC_ResNet/scripts/summarize_results.py `
  MRAM_IMC/IMC_ResNet/checkpoints/actual_t64_n32.json
```

脚本依次计算：

1. 同子集 ANN 基线。
2. 浮点官方 SNN，软复位。
3. 浮点 SNN，硬复位：隔离复位规则影响。
4. int5 反量化权重 + 普通 Conv2d + 硬复位：隔离量化影响。
5. int5 + 实际 Cluster + 硬复位：隔离 Cluster 算术映射影响。

保存检查点 SHA256、映射表、各时间窗 ACC、逐样本预测、运行耗时和完整 MR 峰值图。
对照完成会保存中间结果，只有 `status=complete` 才表示实际 Cluster 推理也完成。

## 目录结构

```text
IMC_ResNet/
├── models/
│   ├── cluster_backend.py       # 位平面权重、分块 PMAC、SCU/MR、统计
│   ├── imc_resnet.py            # FX 图替换与残差清零路由
│   └── cluster_runtime.py       # 旧事件计数器，不用于新推理
├── scripts/
│   ├── run_cluster_inference.py # 新版网络与同子集对照
│   ├── summarize_results.py     # 自动生成对照与 MR 表
│   └── run_event_accounting.py  # 旧版，仅用于回溯
├── tests/
│   ├── test_actual_cluster.py   # 标量一致性、溢出、分块卷积、残差网络
│   └── test_cluster.py          # 旧计数器测试
├── checkpoints/
├── requirements.txt
└── README.md
```

## 本轮实测（2026-09-12）

结果文件：`checkpoints/actual_t64_n32.json`；自动生成的详细表：
`checkpoints/actual_t64_n32.md`；全局加权位宽：`checkpoints/actual_t64_n32_summary.json`。

配置：固定前 32 张测试图、batch=2、CPU 4 threads、1280 张训练图校准，
每个 batch 时间步优先连续运行到 T=64，截取四个读出窗口。
这是小子集调试，不能和旧版 2000 张的 ACC 直接作转换损失比较。

| 模型 | T=8 | T=16 | T=32 | T=64 |
|---|---:|---:|---:|---:|
| 浮点官方 SNN，软复位 | 25.00% | 59.38% | 81.25% | 90.62% |
| 浮点 SNN，硬复位 | 9.38% | 15.62% | 50.00% | 59.38% |
| int5 数字参考，硬复位 | 9.38% | 18.75% | 46.88% | 59.38% |
| 实际 Cluster，硬复位 | 9.38% | 18.75% | 46.88% | 59.38% |

同子集 ANN ACC 为 87.50%。小样本上 SNN 可能高于 ANN，不能据此宣称转换提高了泛化精度。
四个 T 下，Cluster 与 int5 数字参考的逐样本分类预测不一致数均为 0。
这与独立单元测试中的 SCU/MR/卷积数值对照共同支持本轮映射正确性，
但仅预测相同并不意味着浮点 logits 逐 bit 相同。

MR：本轮所有 fold 均未观察到超过 255 的计数；全局最大 MR=40，所需有效位宽 6 bit。
全局平均观测有效位宽 0.5755 bit；平均逻辑 MR 峰值位宽 0.9170 bit。
均值包含零和补零 lanes；它们不是推荐硬件位宽。
**只能说本次样本/窗口未超限，不能证明其他图片、更长窗口和其他权重不会溢出。**

实际 Cluster 计算及全量 MR 统计耗时 338.18 s（不含校准和数字对照）。
已通过 IMC 项目 12 项测试及原简化模型 4 项测试。

本轮精度下降首先应调查硬清零：在不量化权重时，改成硬清零已出现明显下降。
后续应针对硬复位校准阈值/缩放或训练，而不是先假定是 MR 溢出。
若为了恢复官方软复位精度而保留阈上余量，就需要明确修改 Cluster 的复位语义，
不能仅替换 IF 配置却继续宣称等价于当前 SimpleCluster 硬清零模型。

## 接入微调后的显式硬复位 SNN

入口：`scripts/run_finetuned_cluster.py`；适配层：`models/finetuned_snn.py`。
它直接加载 `SNN_ResNet10/checkpoints/hard_reset_finetuned.pt`，**不经过 ANN
Converter、不重新训练，也不重新估计 IF 阈值**。旧检查点没有保存 IF 阈值属性，
因此同时加载训练时使用的 `calibrated.json`，校验 9 个阈值完整性，并记录两个来源文件的 SHA256。

映射顺序：

1. 严格加载微调后的 Conv/BN/FC 参数及 BN running statistics，恢复硬复位 IF。
2. 在 eval 模式下融合隐藏层的 11 组 Conv-BN，保留原 IF 阈值与残差拓扑。
3. 对融合后的卷积权重按输出通道量化为 int5。数字参考使用同一套反量化权重。
4. 将 11 个隐藏卷积替换为 `ClusterConv2d`，输入为未经 VoltageScaler 缩放的 0/1 脉冲（input_scale=1）。
5. 每个 Macro/权重位平面通过真实 PMAC → SCU → MR → 有符号归约计算输出。
   每 fold 共用同一组 16×5 MR；卷积输出采用累计值差分，防止数字 IF 重复积分。
6. conv1 由所在 block 的 lif1 发放清零；conv2 与 projection shortcut
   由同一个 lif_out 同时清零。无 projection 的 identity 分支保留数字残差连接。

连续输入 stem（含 BN）、FC、融合后的 bias、残差相加及 IF 比较仍为数字浮点边界。
这是接入实际 Cluster 累加器的混合模型，不是全网络纯整数或逐时隙电路仿真。
默认 MR 配置为 8 bit，使用 `wide_reference` 不截断并统计每次超过 255 的情况；
只有确认没有超限，才能说本次数据不需要触发 8-bit 溢出处理。

运行（工作区根目录）：

```powershell
python MRAM_IMC/IMC_ResNet/scripts/run_finetuned_cluster.py `
  --test-size 128 --batch-size 2 --steps 16 64 --threads 4 `
  --mr-bits 8 --overflow wide_reference `
  --output MRAM_IMC/IMC_ResNet/checkpoints/finetuned_t64_n128.json

python MRAM_IMC/IMC_ResNet/scripts/summarize_results.py `
  MRAM_IMC/IMC_ResNet/checkpoints/finetuned_t64_n128.json
```

四组对照均使用同一测试子集、batch、输入归一化和时间窗口：原浮点模型、
融合后的浮点模型、int5 数字参考、实际 Cluster。结果记录逐样本分类预测、
融合前后不一致数、Cluster/int5 参考不一致数及各层完整 MR 统计。
`status=complete` 才表示完整评估结束；`running_controls` / `running_cluster`
不应解释为已经得到 Cluster ACC。该入口不重新计算 ANN 基线。

### 微调模型实际 Cluster 实测（2026-09-12）

已完成 `checkpoints/finetuned_t64_n128.json`（status=complete）。固定测试集前 128 张、
T=64、batch=2、CPU 4 threads；使用此前 T=16/1 epoch/5000 张训练得到的检查点，未重新训练。
检查点 SHA256、全部 IF 阈值和输入归一化已与 2000 张数字 SNN 评估记录逐项核对一致。

| 模型（同一 128 张子集） | T=16 | T=64 |
|---|---:|---:|
| 原始微调浮点硬复位 SNN | 75.78% | 91.41% |
| 隐藏 Conv-BN 融合后的浮点 SNN | 75.78% | 91.41% |
| int5 数字参考 | 80.47% | 91.41% |
| 实际 Cluster | 80.47% | **91.41%** |

实际 Cluster 在 T=64 正确分类 117/128 张；T=16 为 103/128 张。
两个时间窗下 Cluster 与 int5 数字参考的逐样本预测不一致数均为 0，
融合前后浮点模型的预测不一致数也均为 0。这不等同于保证浮点 logits 逐 bit 一致。
T=16 的小子集上量化后 ACC 较高，不能据此推断量化普遍提高准确率。
此前 91.75% 是 2000 张数字 SNN 子集结果，不应拿它与本轮 128 张结果直接计算映射损失。

| 层 | 最大 MR | 最大有效位宽 | 平均观测有效位宽 |
|---|---:|---:|---:|
| layer1.conv1 | **66** | **7** | 0.6965 |
| layer1.conv2 | 35 | 6 | 0.3667 |
| layer2.conv1 | 29 | 5 | 0.4340 |
| layer2.conv2 | 14 | 4 | 0.3594 |
| layer2.downsample.0 | 15 | 4 | 0.0548 |
| layer3.conv1 | 13 | 4 | 0.2873 |
| layer3.conv2 | 14 | 4 | 0.3717 |
| layer3.downsample.0 | 8 | 4 | 0.0343 |
| layer4.conv1 | 15 | 4 | 0.3326 |
| layer4.conv2 | 18 | 5 | 0.5709 |
| layer4.downsample.0 | 6 | 3 | 0.0309 |

**本轮 MR 仍配置为 8 bit，全程没有超过 255；但峰值 66 已超过 6-bit 上限 63。**
因此不能再沿用旧模型/32 张实验中“观察到的峰值只需 6 bit”的结论。
本轮观察到的峰值需要 7 bit，不构成其他样本/时间窗口使用 7 bit 均不会溢出的保证。
MR 仍以 int32 参考存储实现；由于每个 fold 均未越过 8-bit 上限，
本次没有需要执行 wrap/saturate/error 溢出处理的情况。

全局平均观测有效位宽 **0.3970 bit**；平均逻辑 MR 峰值位宽 **0.8012 bit**。
这些均值按相应 MR 观测/逻辑状态数量加权，包含零和补零 lanes，零计为 0 个有效位，
不是物理寄存器的平均配置位宽，也不能用来决定单个 MR 的安全位宽。
超过 63 的 fold 观测共 10 次，涉及 7 个逻辑 MR（包含样本/空间坐标维度）；
按通道/Macro/bit-plane 归并后的大于 63 峰值只有三项，均位于 layer1.conv1：
零基索引 (channel=3, macro=5, bit=0):64；(9,3,1):66；(22,5,4):64。
所有通道/Macro/bit-plane 峰值完整保存在结果 JSON 的 `peak_by_channel_macro_bit`。

实际 Cluster 推理及全量统计耗时 **1607.60 秒**（约 26 分 48 秒），
含三组数字对照的总运行耗时 **1683.51 秒**（约 28 分 4 秒）。
新增 BN 融合、残差清零路由、int5 数值对照、阈值恢复和新结果格式测试；
IMC、简化 Cluster 和 SNN 联合测试共 **23 passed**。

自动生成的详细表：`checkpoints/finetuned_t64_n128.md`；
全局汇总：`checkpoints/finetuned_t64_n128_summary.json`。

## 未做负载均衡的 MR 同时状态监督与位宽分布

完成 `checkpoints/mr_conditions_t64_n128.json`，状态 `complete`。
沿用上述 128 张、T=64、batch=2、相同权重/阈值/映射，未做负载均衡或其他数值改动。
新增 `models/mr_observer.py` 只读监控，默认关闭。全网络监控耗时 1373.91 秒；
再对 6 张包含目标神经元的图片从 t=1 追踪，耗时 62.15 秒；总计 1436.13 秒。

- T16 ACC **80.46875%**，T64 ACC **91.40625%**，与原实验逐样本预测完全一致。
- **所有层完整 MR 统计（包括两种直方图和完整峰值图）与原实验完全一致**。
- 捕获全部 10 次 MR≥64 观测，对应 7 个逻辑 MR；没有以跨时间归并的峰值代替同时状态。
- 逐 Macro / bit-plane 的逻辑峰值直方图之和，均与各层整体直方图一致。
- 6 张图片中的 9 个目标神经元共追踪 576 个时间步快照；真正按权重正负分解的
  累计整数 MAC 与 SCU/MR 有符号归约误差均为 **0**，追踪重跑预测也完全一致。

### 大 MR 的实测成因

**关键不是膜电位很大而没发放，而是无符号计数一直增长，但有符号膜电位朝负方向积累。**

1. `layer1.conv1` 中 MR≥32 的神经元-fold 观测平均已经 **58.16 步未复位**，
   相比该层全体神经元-step 的平均 19.07 步明显更长。98.2144% 的这些观测充电膜电位小于 0，
   仅 0.0264% 在当步发放。这是条件关联统计，不是单因素消融实验。
2. 所有 7-bit 事件都在 **t=62–64**；7 个目标神经元从 t=1 到64均没有发放，
   实际负权重累计贡献都大于正权重贡献。它们的融合 bias 均为正，所以这些极值并非负 bias 压制产生。
3. 最大值示例：sample=2、layer1.conv1、channel=9、position=266（28×28中的row=9,col=14），
   t=64，Macro3 的五个 MR 为 **[40, 66, 35, 38, 50]**。
   最热lane对应23个权重位1；64步累计1065个有效输入事件，`MR=1065//16=66`、`SCU=9`。
   平均每步16.6406次，有效输入活动比例72.35%。
4. 此时按补码位平面归约（含SCU）：P=49570、N=59952，净值 **-10382**。
   按真正权重正负分别统计则为 **8002−18384=-10382**，两种分解完全一致。
   数字膜电位约为 `-10382×0.016343771 + 64×0.708483934 = -124.338`，
   远低于阈值2.689975。正bias不足以补偿负向驱动，因此不发放也不清MR。

这不是“正负恰好抵消到接近0”，而是**负向驱动占主导、长时间不复位**。
低4位与符号位之间较高的抵消比例，还包含二补码表示内部的抵消，不能当作真正正负权重的抵消比例。

### 正负部分是否都很大？在同一个 Macro 还是不同 Macro？

这里“正/负部分”先指系数 `[1,2,4,8,-16]` 的位平面；MR本身均无符号。

- **并非两边都必须到7bit**：10次≥64事件中，没有一次低4位和符号位MR同时≥64。
  低位峰值≥64时，符号位最大值为46–55；唯一符号位达到64的事件中，低4位最大值为60。
- 在 `layer1.conv1` 的 MR≥32 条件下，38.710%的神经元-fold观测正负两侧都至少有一个MR≥32。
  36.994%在同一Macro内同时出现，16.500%存在不同Macro的正负高MR组合。
  **这两个条件可以重叠，不能相加解释为互斥分类。**
- 最大值示例中，Macro0、3、4都有≥32的MR；Macro3内部既有66，也有符号位50。
  因此大MR既可在单个Macro内多位共存，也可分散到多个Macro。
- 就补码加权抵消量而言，≥32观测中约 **97.076%** 可在同Macro内完成，
  余下约2.924%是跨Macro额外抵消；≥64观测中同Macro占比约99.613%（这些汇总采用MR-only）。
  最大66的上述具体快照，计入SCU后同Macro抵消占100%，8个有效Macro的净贡献全部为负，
  **不是某个Macro巨大正值与另一个Macro巨大负值之间相互抵消**。

### 有效位宽分布

位宽定义 `MR.bit_length()`，零计0bit；实际寄存器仍配置8bit。

| 位宽 | 数值范围 | 占全部逐fold观测 | 占非零逐fold观测 | 占全部逻辑MR峰值 |
|---:|---:|---:|---:|---:|
| 0 | 0 | 77.871462% | — | 61.936848% |
| 1 | 1 | 10.643096% | 48.096696% | 13.717960% |
| 2 | 2–3 | 7.095551% | 32.065160% | 12.671983% |
| 3 | 4–7 | 2.999606% | 13.555373% | 7.196315% |
| 4 | 8–15 | 1.092919% | 4.938958% | 2.982101% |
| 5 | 16–31 | 0.291166% | 1.315792% | 1.425461% |
| 6 | 32–63 | 0.006200389% | 0.028019876% | 0.069331733% |
| 7 | 64–127（实测最高66） | 10次，1.078e-8% | 4.871e-8% | 7个，5.749e-7% |

- 逐fold观测包含重复观察，总数92,778,004,480；非零20,530,415,634。
  平均有效位宽：含零 **0.396977**，非零 **1.793960**。
- 逻辑峰值：每个(样本、通道、空间位置、Macro、bit-plane)整个窗口只计一次，
  总数1,217,658,880；非零463,479,345。平均位宽：含零 **0.801226**，非零 **2.104990**。
- **这些计数都不等于芯片物理MR数量**。补零lane也计入；排除零观测不等于静态剔除所有补零容量。
- 7bit事件虽然极少，仍意味着直接统一改成6bit会遇到超限。此次未做缩位宽ACC实验；
  128张/T=64的观测也不是其他样本或时间窗的安全位宽保证。

### 复现与文件

```powershell
# 重新监督同一128张并自动追踪所有7-bit事件及两层的top案例。
python MRAM_IMC/IMC_ResNet/scripts/analyze_mr_conditions.py

# 仅从已完成结果生成分析、核验和图，不重复跑网络。
python MRAM_IMC/IMC_ResNet/scripts/report_mr_conditions.py `
  MRAM_IMC/IMC_ResNet/checkpoints/mr_conditions_t64_n128.json --plot
python MRAM_IMC/IMC_ResNet/scripts/report_mr_distribution.py `
  MRAM_IMC/IMC_ResNet/checkpoints/mr_conditions_t64_n128.json `
  --output-prefix MRAM_IMC/IMC_ResNet/checkpoints/mr_width_distribution_t64_n128 --plot

python -m pytest -q MRAM_IMC/IMC_ResNet/tests `
  MRAM_IMC/model_imc_cluster_simple/tests MRAM_IMC/SNN_ResNet10/tests
```

联合测试 **30 passed**。新增测试包括同/跨Macro抵消、监控前后数值一致、发放前快照、
复位年龄、多fold有符号追踪、7bit事件完整覆盖、零值分母、只读报告和完整小网络回归。

- `checkpoints/mr_conditions_t64_n128.md`：条件统计、所有7bit事件、完整16×5快照和时间轨迹。
- `checkpoints/mr_conditions_t64_n128.json`：原始同时状态、各层/各Macro/各bit-plane峰值分布。
- `checkpoints/mr_conditions_t64_n128_checks.json`：事件覆盖与分组直方图守恒核验。
- `checkpoints/mr_conditions_t64_n128.png`：最大MR对应神经元的时间轨迹。
- `checkpoints/mr_width_distribution_t64_n128.md/.json/.png`：两种口径的完整位宽分布、数量、占比和图。

## 下一阶段：MR-aware QAT 计划（2026-09-13）

当前实际 Cluster 基线已经完成 MR 条件监督和位宽分布统计。原始硬复位微调模型在前 128 张测试样本、T=64 下的最高 MR 为 **66**，因此需要 7-bit 表示；6-bit 无符号 MR 的上限为 63。大 MR 主要来自长时间不发放、负向权重驱动占主导且没有发生硬复位的神经元。

### 当前 Leakage + QAT 状态

已在 `SNN_ResNet10` 中完成 LeakageIF 和 int5 权重量化感知微调，并从 `hard_reset_finetuned.pt` 继续训练得到 `leakage=0.99`、`0.97` 两组候选。当前结果显示 0.99 更适合作为后续 MR-aware QAT 的初始化，但这些结果只约束了权重，并未约束 MR 位宽。

因此，当前实际 Cluster 接入前必须重新验证：

- Leakage 数字模型与 Cluster 数字状态更新是否一致；
- 重新 int5 量化后的权重是否与 fake quant 前向一致；
- T=64 ACC 是否保持在原实际 Cluster 基线附近；
- MR 最大值、平均有效位宽、`MR>=48` 和 `MR>=64` 事件是否下降。

### MR-aware QAT 的设计目标

在不改变 Cluster 内部硬件结构的前提下，在训练图中加入简化但语义一致的 MR 代理：

1. 输入脉冲与 int5 权重分解为正/负 bit-plane 事件；
2. 按现有 Macro 和 lane 映射累计无符号 MR 状态；
3. 在发放/硬复位时清空对应 MR 状态；
4. 用平滑的 overflow penalty 替代不可导的硬截断；
5. 分类损失、MR 峰值惩罚和发放/长期不复位约束联合优化。

建议优先使用安全目标 `MR<=56` 进行软惩罚，为 6-bit 上限 63 留出余量；最终仍以真实 Cluster 的 `MR<=63` 和 ACC 结果为准。不能简单对卷积输出做 clamp，因为 MR 是无符号、跨时间累积，正负抵消发生在 SCU/后级，不等价于对有符号卷积输出截断。

### 计划实验

- leakage：优先 `0.99`，再比较 `1.00/0.995/0.98/0.97`；
- T：至少评估 T=16 和 T=64；
- MR 目标：`63`（硬上限）、`56`（软安全上限）；
- 数据规模：先用前 128 张做 Cluster 快速回归，再用前 2000 张做数字模型评估；
- 指标：ACC、发放率、zero-firing 比例、最大 MR、平均/峰值有效位宽、`MR>=48`、`MR>=64` 事件数、数字参考与实际 Cluster 的逐样本预测差异。

### 当前已知限制

Leakage 会改变膜电位状态，而现有 Cluster 后端需要显式同步该更新；如果只在训练模型中加入 leakage、部署时仍使用无 leakage 的 IF，则两者不能直接进行严格的 MR/ACC 对比。因此 MR-aware QAT 完成后必须扩展实际 Cluster 推理入口，使数字 IF 和 MR 清空时序保持一致。
