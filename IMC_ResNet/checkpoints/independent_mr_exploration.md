# 独立位面 MR 再探索（2026-09-14）

## 实验口径

每个 Macro 仍保存五个无符号 MR 和五个 4-bit SCU。沿用已有 +16 等权跨位面转发，以及 b0 进位拦截、b1/b2 状态抵消；不增加新的抵消通路。

先回放三套权重（原始、共享 MR 的 kd_mr、mt_mr_high），比较第一层原始 8×36 与 16×18 均摊输入。所有位置、T64、测试样本 0,1,2,3,4,5,64,115，逐步验证 Macro 内有符号整数和守恒。这里只报告抵消后、IF 复位前峰值；不能将有限样本最大值等同于硬件最坏情况保证。16×18 分配需要输入路由支持，未做物理验证。

## 独立 MR 训练

从 shared_mr_sweep_v2/kd_mr.pt 出发，配对训练 kd_control 与 kd_mr；同样的训练集前 32 张、1 epoch、随机种子 20260914、batch 2、lr=0.00025，只更新 layer1.conv1.weight。损失保留 T16/T64 CE 与蒸馏，MR 目标设 29，梯度范数比 rho=0.3；对照 rho=0。验证使用训练集 59000:59256，与训练样本分离。

MRProxy 的独立模式在前向执行精确整数转发/抵消，反向对原始连续 MR 使用恒等 STE，忽略离散路由选择的导数。这是训练近似，不能保证优化一定有效。逐步进位/复位对照和有限梯度测试已通过。

独立硬件回放使用 IndependentCancelMR，验证每个 fold 的局部守恒，并从实际 SCU/MR 重构输出。正式部署后端及原始权重均未替换。

## 复现入口

- IMC_ResNet/scripts/evaluate_independent_mr_transfer.py：三套权重 × 两种分配。
- SNN_ResNet10/scripts/sweep_shared_mr_qat.py --independent-cancel --cases kd_control kd_mr --train-size 32 --init-checkpoint MRAM_IMC/SNN_ResNet10/checkpoints/shared_mr_sweep_v2/kd_mr.pt --lr 0.00025 --output-dir MRAM_IMC/SNN_ResNet10/checkpoints/independent_mr_sweep
- IMC_ResNet/scripts/evaluate_shared_macro_mr.py --independent-cancel --output <独立结果路径>：全部 11 个卷积层，原始分配。

## 已完成结果

| 权重 | 原始分配最大 MR | 16×18 分配最大 MR | 原始分配 >31 观测次数 |
|---|---:|---:|---:|
| 原始 | 33 | 30 | 3 |
| shared kd_mr | 36 | 30 | 17 |
| shared mt_mr_high | 40 | 29 | 25 |

三组均摊方案在原 8 张诊断样本中均无 >31 观测。原始权重均摊方案在额外测试索引 128:136 上最大 MR=25，无 >31 观测。总计 16 张仍不足以证明 5-bit 最坏情况安全。

配对训练结果（验证集 256 张）：起点 T16/T64=72.2656%/94.5313%，独立 MR 最大 36、超限观测 17；无 MR 约束组=74.2188%/92.9688%，MR=36/17；有 MR 约束组=73.4375%/94.1406%，MR=36/17。本轮新权重均不推荐替换起点。

两个训练过程的抽样 MR 最大值均仅 30，>31 事件为 0，说明本次训练样本/空间抽样未覆盖诊断尾部。后续应从训练集挖掘高 MR 样本与位置；不能直接反复用诊断测试样本训练后再称独立验证。

73 项测试通过。当前证据支持优先探索输入均摊，同时改进训练集尾部覆盖，不支持宣称已实现整网 5-bit MR。


## 全网络实际回放

原先8张整轮回放中断，未写出最终统计，不能算完成。重新完成测试索引64、115的全11层实际回放：第一层最大33，其余10层最大15；各层每fold局部整数误差均0，T16/T64预测均与数字参考一致。结果见 independent_full_original_2.json。该检查使用原始8×36分配，没有把均摊直接接入整网，不宣称整网5bit已经验证。

推荐顺序：第一层均摊输入的更大样本验证及输入路由评估；从训练集进行高MR样本/位置挖掘；再做针对独立抵消后MR的训练。此次精确抵消规则未扩展，训练权重不替换。
