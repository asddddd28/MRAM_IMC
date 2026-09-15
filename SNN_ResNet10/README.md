# SNN_ResNet10

## 2026-09-13 更新：MR-aware QAT pilot

新增 `models/mr_qat.py` 和 `scripts/finetune_mr_qat.py`：BN 融合后 int5 QAT，
训练侧精确前向的 Macro/二补码 bit-plane/SCU/MR 状态代理，发放清零，overflow + peak + quiet 损失。
首轮只训练 `layer1.conv1.weight`，空间采样、完整 T=64。

修复旧 leakage 节点在 eval 模式下绕过泄漏充电的问题；下文旧 leakage 精度表仅是历史记录，
不能作为启用泄漏的推理结果。新检查点内置 IF 阈值、leakage 和融合标记，使用统一加载器。

从工作区根目录复现（使用新输出目录保留已有结果）：

```powershell
python MRAM_IMC/SNN_ResNet10/scripts/finetune_mr_qat.py `
  --output-dir MRAM_IMC/SNN_ResNet10/checkpoints/mr_qat_repeat `
  --train-size 64 --test-size 128 --steps 64 --alphas 0 10 `
  --lr 0.001 --beta 1 --gamma 0.1 --positions 64 --batch-size 2 --threads 4 --balance-mr-gradient
```

最新记录：`checkpoints/mr_qat_t64_balanced/`；日志为 `checkpoints/mr_qat_t64_balanced_final_run.log`。
此前对照保存在 `mr_qat_t64/`、`mr_qat_t64_lr2e4/`、`mr_qat_t64_lr1e3/`、`mr_qat_t64_stable/`。
默认断开 IF reset 的反向梯度，前向硬复位不变；`--balance-mr-gradient` 用 detached 范数比平衡
CE/MR 梯度（系数上限 1e5），逐 batch 保存原始梯度范数、系数和损失。关闭时使用固定损失系数。
数字精度、实际 MR 与限制见 [本轮报告](../IMC_ResNet/checkpoints/mr_qat_pilot_report.md)。
最终 `mr_qat_t64_balanced/alpha_10.pt`：前 128 张 T=64 为 90.625%、T=16 为 85.9375%；
同组对照分别为 89.84375%、89.0625%。风险诊断集的实际 MR 峰值由对照 65 降至 62，超限观测 2→0。
训练数据仅 64 张；这不是完整训练或统一 6-bit 安全证明。下文为此前阶段记录。

基于 **SpikingJelly 0.0.0.0.14** 的 FashionMNIST SNN ResNet-10，用于验证 ANN→SNN 权重迁移、激活校准和脉冲发放率。

## 本轮新增：激活校准与发放率统计

运行：

```powershell
cd D:\Projects\ai\MRAM_IMC\SNN_ResNet10
python scripts/calibrate_and_evaluate.py `
  --ann-checkpoint ..\ANN_ResNet10\checkpoints\resnet10_fashionmnist_5ep.pt `
  --steps 8 --calibration-batches 10 --test-size 2000
```

脚本流程：

1. 使用训练集前 10 个 batch 作为校准集；
2. 注册 ANN ReLU hook，统计每个激活层的 99.9 percentile；
3. 将统计出的激活范围映射为 SNN IFNode threshold；
4. 迁移 ANN 的 74 个权重张量；
5. 在测试集上推理；
6. 通过 IFNode hook 统计各层 firing rate。

当前映射是用于验证的简化方案：ANN 的 5 个 ReLU 激活统计值依次映射到 SNN 的 IF 节点，超出的 IF 节点沿用最后一个阈值。它不是最终的逐层解析转换，因此结果应作为校准实验而不是最终模型。

## 当前结果（CPU，2026-09-11）

ANN 5 epoch 基线测试 ACC：**92.13%**。

| 方法 | T | 测试样本 | ACC | 相对直接迁移 |
|---|---:|---:|---:|---:|
| 直接权重迁移 + rate | 8 | 2000 | 9.70% | 基线 |
| ANN percentile threshold 校准 | 8 | 2000 | **36.90%** | +27.20 个百分点 |

校准结果：

```text
stem IF threshold:   1.6348
layer1 IF threshold:  2.6900
layer2 IF threshold:  3.6110
layer3 IF threshold:  4.0806
layer4 IF threshold:  5.6521
```

各 IF 节点的 firing rate 已写入：

```text
D:\Projects\ai\MRAM_IMC\SNN_ResNet10\checkpoints\calibrated.json
```

观测到的 firing rate 大约为：

```text
IF0: 14.47%
IF1:  5.30%
IF2:  3.97%
IF3:  1.47%
IF4:  1.28%
IF5:  0.52%
IF6:  0.88%
IF7:  0.40%
IF8:  2.60%
```

后面层发放率明显下降，说明残差分支经过多层卷积后存在较强的脉冲稀疏/尺度衰减。这也是 ACC 仍然低于 ANN 的主要线索之一。


### 时间窗口扩展实验（官方 ANN2SNN）

在相同设置下（`99.9%` 激活校准、Conv-BN fusion、FashionMNIST 测试集前 2000 张、CPU），增大仿真时间步后结果如下：

| 时间步 T | ACC | 相对 T=8 | 平均 IF 发放率 |
|---:|---:|---:|---:|
| 8 | 25.00% | — | 6.89% |
| 16 | 56.40% | +31.40 个百分点 | 7.61% |
| 32 | 82.60% | +57.60 个百分点 | 8.26% |
| 64 | 91.85% | +66.85 个百分点 | 8.50% |
| 128 | 92.30% | +67.30 个百分点 | 8.73% |

结果文件分别为：

```text
checkpoints/official_ann2snn.json
checkpoints/official_t16.json
checkpoints/official_t32.json
checkpoints/official_t64.json
checkpoints/official_t128.json
```

结论：当前转换模型在 `T=8` 时脉冲累计明显不足；将窗口增加到 `T=64` 后，ACC 已达到 **91.85%**，接近 ANN 的 **92.13%** 基线（差 0.28 个百分点）；继续增加到 `T=128` 仅提升 0.45 个百分点，说明准确率在约 `T=64` 附近趋于饱和。对于后续 Cluster/TDP 调试，建议先使用 `T=64` 作为精度与仿真开销的折中点，`T=128` 作为高精度参考点。

注意：这些 ACC 是在固定的 2000 张测试子集上测得，适用于模型对比与调试，不应直接当作完整测试集最终成绩。
## 目录结构

```text
SNN_ResNet10/
├── models/snn_resnet10.py
├── scripts/evaluate_transfer.py
├── scripts/calibrate_and_evaluate.py
├── tests/test_snn.py
├── checkpoints/
├── data/
├── requirements.txt
└── README.md
```

## 下一步建议

1. 将目前的近似 threshold 映射改成按具体 SNN IF 层逐层校准；
2. 对 residual add 前后的尺度分别统计，避免 shortcut 和 residual 分支量纲不一致；
3. 测试 T=16、32，观察后层 firing rate 是否提升；
4. 用校准集重新估计 BN running mean/variance；
5. 必要时只微调 IF threshold、BN affine 参数或最后一层，而不是重新训练整个 SNN；
6. 最终再增加发放事件数、平均 firing rate、推理时间和 TDP 脉冲负载估算。

## SpikingJelly 官方 ANN2SNN 对照实验

SpikingJelly 自带 `spikingjelly.activation_based.ann2snn.Converter`。与当前自定义 SNN 不同，官方 Converter 会基于校准数据统计 ReLU 电压范围，并插入 `VoltageScaler + IFNode + VoltageScaler`；同时可以执行 Conv-BN fusion。官方实现支持 `max`、`99.9%` 和比例缩放等模式。相关实现见 SpikingJelly 官方仓库的 `activation_based/ann2snn/converter.py`。citeturn0view1

本项目新增：

```text
scripts/official_ann2snn.py
```

运行：

```powershell
python scripts/official_ann2snn.py `
  --ann-checkpoint ..\ANN_ResNet10\checkpoints\resnet10_fashionmnist_5ep.pt `
  --steps 8 --calibration-batches 10 --test-size 2000
```

实测结果（CPU，2026-09-11）：

```text
官方 Converter，99.9% 校准，Conv-BN fusion，T=8：25.00%
```

对应文件：

```text
checkpoints/official_ann2snn.json
```

这个结果低于当前自定义 threshold 映射的 36.90%，但它是更严格、可复现的官方转换基线。不能简单将两者视为同一模型：官方 Converter 生成的是 FX GraphModule，并要求逐时间步输入 4D 图像；当前自定义模型使用显式 `[T,B,C,H,W]` 接口。官方 Converter 的优势是转换规则完整、包含 VoltageScaler、校准和 Conv-BN fusion，后续应优先围绕它调整输入归一化、时间步数和残差结构。

### 为什么官方结果仍然只有 25%

当前 ANN 是为连续值输入训练的，而转换后 SNN 使用 IF 脉冲输出；此外 ResNet 的 shortcut/residual add 和最后 Linear 读出会放大误差。官方 Converter 解决了激活尺度问题，但不会自动解决训练拓扑与 SNN 时间读出之间的所有差异。下一步建议：

- 对比 `mode=max`、`mode=99.9%` 和比例模式；
- 评估 T=16、32、64；
- 将 ANN 输入归一化和 rate coding 规范统一；
- 针对 residual add 做专门的 shortcut scale；
- 用官方转换后的模型统计每个 VoltageScaler/IFNode 的 firing rate；
- 必要时对 IF threshold、BN affine 和最后分类层做少量校准微调。


## 硬复位微调权重：T=64 推理验证（2026-09-12）

本实验使用显式 `SNNResNet10`，不是官方 Converter FX 模型，也没有接入 Cluster。
权重来自 `hard_reset_finetuned.pt`：T=16、前 5000 张训练样本、1 epoch 微调。
此次不重新训练，只延长推理窗口；同一个检查点测试 FashionMNIST 前 2000 张，
batch=64，输入归一化 mean=0.2860 / std=0.3530，所有 IF 为硬复位 `v_reset=0`。

| 推理时间窗口 | 正确数 / 样本数 | ACC |
|---|---:|---:|
| T=16（重载复核） | 1491 / 2000 | 74.55% |
| T=64 | 1835 / 2000 | **91.75%** |

延长时间窗口提升 **17.20 个百分点**。T=16 结果与微调结束时一致。
两种窗口在同一条 T=64 时间序列中分别读取前缀平均 logits，合计 CPU 耗时约 120.89 秒。
这个耗时不是两次独立评估的耗时之和，也不是纯 T=64 单张延迟。

旧检查点未保存 IF 阈值，因此评估脚本显式恢复 `calibrated.json` 的全部 9 个阈值，
并在结果 JSON 记录阈值、来源、归一化配置和检查点 SHA256；不要仅加载 state_dict 后使用默认阈值。
本结果为 2000 张测试子集结果，不能视为完整测试集成绩，也不能视为 int5 / IMC 精度。

从工作区根目录复现：

```powershell
python MRAM_IMC/SNN_ResNet10/scripts/evaluate_finetuned.py --steps 16 64 --test-size 2000 --batch-size 64 --threads 8
```

结果：`checkpoints/hard_reset_finetuned_eval_t64.json`。
新增时序一致性测试验证连续 T=1 调用与一次 T 步 forward 的输出完全一致，
并验证 reset 后结果可复现。SNN 测试结果：**2 passed**。

### 微调权重接入实际 Cluster 的后续验证

实际映射入口在 `../IMC_ResNet/scripts/run_finetuned_cluster.py`，保留本模型的 IF 阈值，
融合隐藏 Conv-BN 后进行 int5 量化，映射 11 个隐藏卷积到实际 SCU/MR 累加器。
2026-09-12 已完成同一前 128 张测试子集的 T=64 对照：浮点 SNN、融合浮点 SNN、
int5 数字参考及实际 Cluster 均为 **117/128 = 91.41%**，Cluster 与 int5 参考逐样本预测一致。
这是 128 张子集实验，不替代上文 2000 张数字 SNN 的 91.75% 成绩。
MR 峰值 **66，需要 7 bit**，没有超过 8-bit 上限 255。
完整统计与边界说明见 `../IMC_ResNet/README.md` 及
`../IMC_ResNet/checkpoints/finetuned_t64_n128.md`。

## Leakage + 权重量化感知微调（2026-09-13）

当前已完成第一阶段的 Leakage + int5 权重量化感知微调，但需要明确：**此阶段尚未对 MR 累加器位宽进行量化感知约束**。

### 已实现

- `LeakageIFNode`：每次充电前执行 `v <- leakage * v + x`；发放后仍采用硬复位 `v_reset=0`。
- `FakeQuantConv2d`：逐输出通道对称 int5 fake quantization，量化范围为 `[-15, 15]`，使用 STE 反向传播。
- 支持从 `hard_reset_finetuned.pt` 初始化，继续进行 Leakage + QAT 微调。
- 支持扫描多个 leakage 系数并保存独立 checkpoint 和 JSON 结果。

### 当前结果

实验使用 FashionMNIST 前 5000 张训练样本、前 2000 张测试样本、1 epoch 微调。结果不是完整测试集成绩。

| 初始化方式 | leakage | T=16 ACC | T=64 ACC |
|---|---:|---:|---:|
| 原硬复位微调模型 | 1.00 | 74.55% | 91.75% |
| 硬复位模型继续 Leakage+QAT | 0.99 | 85.80% | 87.30% |
| 硬复位模型继续 Leakage+QAT | 0.97 | 84.90% | 84.95% |

当前以 `leakage=0.99` 作为后续 MR-aware QAT 的优先初始化点：泄漏较温和，T=16 精度最高，且比 0.97 更好地保留长时间积分信息。

结果目录：

```text
checkpoints/leakage_qat_from_hard/
```

训练入口：

```powershell
python scripts/finetune_leakage_qat.py `
  --leakages 0.99 0.97 `
  --init-checkpoint checkpoints/hard_reset_finetuned.pt `
  --epochs 1 --steps 16 --train-size 5000 --test-size 2000
```

### 与 MR-aware QAT 的边界

目前的损失仍只有分类损失；训练时没有模拟 Cluster 的无符号 MR 跨时间累积、正负 bit-plane、Macro 映射、SCU 抵消和硬件溢出。因此当前版本属于：

```text
Leakage-aware + weight-QAT
```

还不是：

```text
Leakage-aware + weight-QAT + MR-bitwidth-aware QAT
```

下一阶段将在不修改 Cluster 硬件结构的前提下，加入可微分的简化 MR 状态仿真和位宽惩罚，重点约束 6-bit 上限 `MR<=63`，并通过实际 Cluster 回归确认精度和 MR 峰值变化。
