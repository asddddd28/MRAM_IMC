# 简化版 Python 推理/调试模型方案

## 结论

建议先做一个**离散、可观测、确定性的 Cluster 简化模型**，而不是直接把完整模拟链路改成逐纳秒/逐晶体管仿真。它用于 Python 推理、算法调试和 RTL 对照；现有 `cluster_model` 保留为规范行为模型，简化模型作为旁路 reference backend。

## 先澄清 TDP 的驱动量

附带 SNPU 论文中的 eigen-train/计算编码主要是把**输入激活的多级值或 spike train**映射成固定步长的脉冲序列：对 `m` 位值，第 `n` 位在 `2^m` 个 slot 中产生 `2^n` 个脉冲，周期为 `2^m/2^n`。它不是“每个静态权重都单独产生一个周期”的通用规则。

对本项目的五位有符号权重，权重仍应作为 MRAM 中的静态 int5 位平面。更合理的简化 TDP 驱动链是：

```text
输入 spike × 权重位平面
        ↓
  PMAC 得到每个 bit plane 的计数
        ↓
  SCU/MR 保存跨步数字历史
        ↓
  TDP 根据当前 SCU 余数 a 和符号分解 P/N
        ↓
  固定 step 循环生成离散方波/脉冲事件
        ↓
  累加到简化膜电位或直接按等价整数规则比较
```

因此，不建议简单地对 `abs(weight)` 调用 TDP；那会把静态权重存储编码和时间域输入/部分和编码混在一起。只有在要专门验证“权重串行读取/脉冲宽度调制”的硬件方案时，才另建 weight-driven TDP 模式。

## 建议的三层模型

### 1. `SimpleCluster`：默认推理内核

目标是快、可读、易定位错误：

- 固定 `16 × 36 × 5` 结构；
- 使用 Python int 保存每个 Macro/bit plane 的 `scu` 和 `mr`；
- 每个逻辑步执行全部输入折；
- 用 `z = scu + pmac_count`、`carry = z // L`、`scu = z % L`、`mr += carry`；
- 所有折完成后只比较一次；
- `U >= theta` 发放，发放后 80 路状态硬清零；
- 返回完整 trace：每折 count/carry、候选状态、`S/G/U`、spike 和提交状态；
- 不计算电压、RC、电荷共享或随机抖动。

它应与现有 `mode=integer_reference, backend=scalar` 逐步严格一致，作为推理和调试的黄金 reference。

### 2. `SimpleTDP`：固定步长脉冲观察器

目标是观察“循环/计数器如何生成序列”，不改变默认推理结果：

- 输入 `P/N` 或带符号 `scu`；
- `step` 是一个离散时间步；
- `window_slots` 默认 32，对应五位编码；
- 每个循环输出一个 `PulseEvent(slot, polarity, amplitude, source_bit)`；
- 支持逐 bit plane 查看，也支持叠加后的 `TW/SG`；
- 输出 `pulse_count`、首末 slot、周期和边沿列表；
- 所有波形由整数循环产生，便于和 Verilog/波形日志逐项比较。

建议把现在的 `eigen_train()` 作为通用无符号 eigen-train；再增加一个 `scu_to_tdp()`，明确它是从 Cluster 的 SCU 余数/正负分支生成波形。不要让 TDP 波形对象偷偷参与默认整数判决。

### 3. `Trace/Oracle`：调试比较层

提供：

- `SimpleCluster` vs 现有 `Cluster(mode="integer_reference")`；
- `SimpleCluster` vs 独立矩阵 IF oracle；
- 逐步比较 `candidate.scu/mr`、`state.scu/mr`、`S/G/U` 和 spike；
- 第一个差异报告到 `step/fold/macro/bit`；
- 可选导出 JSONL 和 CSV，供波形/RTL 对照。

## 推荐 API

```python
from cluster_model.simple import SimpleCluster

model = SimpleCluster(theta=560, scu_bits=4, mr_bits=8)
result = model.step(inputs, weights)
print(result.spike)
print(result.trace["folds"])

# 仅生成当前 SCU 的 TDP 观察波形
wave = model.tdp_waveform()
print(wave.events)
```

`SimpleCluster` 不应复制完整 `Cluster` 的模拟代码；它可以复用 `encoding.py`、`digital.py` 和 `tdp.py` 的小函数，但必须拥有独立、短小的事务执行路径。

## 实现顺序

1. 先实现 `simple.py` 的纯整数单步模型和 `SimpleStepResult`。
2. 用现有 T01–T09 测试向量做逐步等价测试，尤其是跨折抵消、MR 溢出和误发放清零。
3. 增加 `scu_to_tdp()` 和逐 slot 测试，验证周期、脉冲数、正负分支。
4. 增加 `simple` CLI：输出短 trace，不保存模拟设备参数。
5. 最后再考虑将 TDP 脉冲的 `TW/SG` 连接到一个可选的简化电荷模型；默认仍以整数结果为准。

## 不建议现在做的事情

- 不要让 TDP 直接以 `abs(weight)` 替代 PMAC/SCU；
- 不要在简化模型中引入 float、随机噪声和 RC 参数；
- 不要把静态权重位平面、激活 spike train、SCU 余数三种编码混成一个接口；
- 不要用波形的浮点面积反推整数 `U`；
- 不要改变当前完整模型已经通过的事务边界和状态所有权。

## 验收标准

- 随机输入/权重至少 100 个时间步与整数参考模型逐步一致；
- 3-bit/4-bit SCU、单折/多折、正负权重和边界阈值一致；
- 发生错误时状态不改变；
- `commit=False` 可重复且不推进状态；
- TDP 波形在固定 seed 下完全确定，并能定位到 bit plane、slot 和 polarity；
- 简化模型速度明显高于完整非理想模型，但不宣称硬件 PPA。
