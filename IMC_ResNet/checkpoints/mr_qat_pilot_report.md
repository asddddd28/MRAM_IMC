# MR-aware QAT pilot — 2026-09-13

## 本轮最终对照结论

最后一组：detach-reset + 梯度平衡，lr=0.001，alpha=10，beta=1.0，gamma=0.1。
前 128 张数字测试：MR 候选 T=16 为 85.94%、T=64 为 90.62%；零惩罚对照为 89.06%、89.84%。
候选相对对照改变 1176 个 int5 权重、2379 个 bit-plane。
8 张风险诊断集：max MR 65 → 62；MR≥48 22936 → 18477；MR≥64 2 → 0。
候选与 int5 数字参考预测差异：{'16': 0, '64': 0}。
诊断集结果不构成 128 张或全测试集位宽保证，当前只完成小规模 pilot。

联合回归：40 passed；日志 `SNN_ResNet10/checkpoints/mr_qat_tests.log`。

## 实现与语义修复

- MR 代理使用 16 Macro × 36 输入 × 5 个二补码 bit-plane；SCU 基数 16。
- 每 fold 观察清零前 MR；无符号状态跨时间累积，输出 IF 发放后同步清零。MR 本身不泄漏。
- 前向 bit-plane 和计数精确；反向使用高斯类别分布 STE、floor STE 和 surrogate spike reset。
- 隐藏 Conv-BN 先融合再进行 int5 QAT；BN 固定，仅训练 layer1.conv1.weight。
- 修复 IFNode.eval() 快捷路径绕过泄漏函数的问题；旧 leakage 推理成绩不可作为启用泄漏的成绩。
- 统一加载器恢复 leakage、阈值、融合拓扑，并物化旧 QAT 的实际前向权重。

## 数字模型：前 128 张测试样本

每组从相同 leakage=0.99 检查点初始化，前 64 张训练样本、1 epoch、完整 T=64。
同组所有 alpha 使用相同随机种子、数据顺序和空间采样序列。测试前缀仅作探索性评估。

| 实验 | alpha | T=16 ACC | T=64 ACC | 训练采样最大 MR |
|---|---:|---:|---:|---:|
| 部署一致初始化 | — | 88.28% | 89.06% | — |
| lr=2e-05, positions=32, beta=0.05, gamma=0.01, detach_reset=False, balance=False | 0 | 89.06% | 89.06% | 63 |
| lr=2e-05, positions=32, beta=0.05, gamma=0.01, detach_reset=False, balance=False | 0.1 | 89.06% | 89.06% | 63 |
| lr=2e-05, positions=32, beta=0.05, gamma=0.01, detach_reset=False, balance=False | 1 | 89.06% | 89.06% | 63 |
| lr=0.0002, positions=64, beta=1.0, gamma=0.1, detach_reset=False, balance=False | 0 | 89.84% | 89.06% | 65 |
| lr=0.0002, positions=64, beta=1.0, gamma=0.1, detach_reset=False, balance=False | 10 | 89.84% | 89.06% | 65 |
| lr=0.001, positions=64, beta=10.0, gamma=1.0, detach_reset=False, balance=False | 0 | 83.59% | 90.62% | 63 |
| lr=0.001, positions=64, beta=10.0, gamma=1.0, detach_reset=False, balance=False | 100 | 83.59% | 90.62% | 63 |
| lr=0.0002, positions=64, beta=1.0, gamma=0.1, detach_reset=True, balance=False | 0 | 88.28% | 89.06% | 65 |
| lr=0.0002, positions=64, beta=1.0, gamma=0.1, detach_reset=True, balance=False | 10 | 88.28% | 89.06% | 65 |
| lr=0.001, positions=64, beta=1.0, gamma=0.1, detach_reset=True, balance=True | 0 | 89.06% | 89.84% | 63 |
| lr=0.001, positions=64, beta=1.0, gamma=0.1, detach_reset=True, balance=True | 10 | 85.94% | 90.62% | 63 |

训练采样峰值不能与全空间 Cluster 峰值直接比较，也不是位宽安全证明。

损失：CE + alpha×mean(softplus(max_lane(MR−56)))/56 + beta×mean(relu(max_all(MR−56)))/56 + gamma×quiet。
quiet 是按未发放年龄加权的 overflow 项；alpha=0 时 beta/gamma 同时为 0。
balance 轮按 CE/MR 梯度范数比对整个 MR 损失乘以 detached 系数（上限 1e5），逐 batch 保存实际系数与两种梯度范数。

导出模型与同组零惩罚对照的整数权重变化：

| 实验 | alpha | 全模型张量一致 | int5 权重改变数 | bit-plane 改变数 |
|---|---:|---|---:|---:|
| mr_qat_t64 | 0.1 | True | 0 | 0 |
| mr_qat_t64 | 1 | True | 0 | 0 |
| mr_qat_t64_lr2e4 | 10 | True | 0 | 0 |
| mr_qat_t64_lr1e3 | 100 | True | 0 | 0 |
| mr_qat_t64_stable | 10 | True | 0 | 0 |
| mr_qat_t64_balanced | 10 | False | 1176 | 2379 |

## 实际 Cluster：8 张已知风险诊断样本

固定索引 [0,1,2,3,4,5,64,115]，覆盖历史 128 张中全部 4 张出现 7-bit 事件的样本。
映射全部 11 个隐藏卷积，T=64，所有空间位置和补零 lanes；6-bit 上限，wide_reference 记录原始溢出而不饱和。
此集合按历史风险选择，其 ACC 不代表无偏测试精度。

| 模型 | T=64 ACC | max MR | MR≥48 | MR≥64 | 超限逻辑 MR | int5 预测差异 T16/T64 |
|---|---:|---:|---:|---:|---:|---|
| 原硬复位 | 100.00% | 66 | 10,342 | 10 | 7 | 0/0 |
| leakage 部署初始化 | 100.00% | 68 | 28,232 | 60 | 28 | 0/0 |
| lr=2e-4 零惩罚对照 | 100.00% | 68 | 28,227 | 60 | 28 | 0/0 |
| lr=1e-3 零惩罚对照 | 100.00% | 64 | 24,758 | 1 | 1 | 0/0 |
| detach-reset 零惩罚对照 | 100.00% | 68 | 28,241 | 60 | 28 | 0/0 |
| 梯度平衡组零惩罚对照 | 100.00% | 65 | 22,936 | 2 | 1 | 0/0 |
| 梯度平衡 MR 惩罚候选 | 100.00% | 62 | 18,477 | 0 | 0 | 0/0 |

完整平均有效位宽、逻辑峰值位宽直方图、逐层 firing rate 和 zero-firing ratio 见各 stress8 JSON。

## 边界与后续

- 这是首次 MR-aware QAT 小规模对照，不是全层、全数据训练，也未证明统一 6-bit MR 安全。
- 新候选尚未完成前 128 张或完整测试集的全空间实际 Cluster 回归。
- 历史原始模型 128 张结果（max MR=66、ACC=91.41%）保持原文件，不与上述 8 张 ACC 混用。
- 未断开 IF reset 梯度的前三轮各组内，MR 候选与零惩罚对照导出模型相同。发现分类梯度范数约 1e8，MR 梯度被淹没。
- stable 轮只断开 IF reset 的反向梯度，前向硬复位和 MR 清零保持不变；不把早期结果归因于 MR 惩罚。
- stable 轮 CE 梯度仍约 1e4；balanced 轮显式平衡分类/MR 梯度，额外反向计算用于记录与缩放。
- 下一阶段应在训练集上挖掘高 MR 的空间位置、增加样本和更新步数，并用独立验证集选择候选。
- 继续保留零惩罚对照，先确认实际 bit-plane 与 MR 超限事件改善，再扩大实际 Cluster 回归。
