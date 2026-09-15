# model_imc_cluster_simple

独立于完整 `model_imc_cluster` 的简化 Python 推理/调试模型。目标是快速验证算法、打印逐折 trace，并为 RTL 的 SCU/MR/TDP 时序提供整数 reference。

## 运行

```powershell
cd D:\Projects\ai\MRAM_IMC\model_imc_cluster_simple
$env:PYTHONPATH="D:\Projects\ai\MRAM_IMC\model_imc_cluster_simple;D:\Projects\ai\MRAM_IMC\model_imc_cluster"
python -m pytest -q
```

## 使用

```python
import numpy as np
from model_imc_cluster_simple import SimpleCluster, SimpleConfig

m = SimpleCluster(SimpleConfig(theta=560))
x = np.zeros((16, 36), dtype=int); x[0, 0] = 1
w = np.zeros((16, 36), dtype=int); w[0, 0] = 3
r = m.step(x, w)
print(r.spike, r.trace["total_u"])
print(r.trace["folds"][0]["carry"])
```

## TDP 调试

```python
waveforms = m.tdp_waveform()  # (16,5)，每一路 SCU 一个固定 32-slot 波形
```

`eigen_train(value, bits=5)` 按论文式规则生成固定窗口方波；`scu_to_tdp()` 显式按 bit plane 展开 SCU，MSB 作为负 TDP 分支。波形只用于观察循环计数和 slot，不替代 `SimpleCluster` 的精确整数判决。

## 设计边界

- 不包含模拟电压、RC、电荷共享、随机失配和 PPA 估算。
- 仍保留每个 Macro/bit plane 独立 SCU/MR、跨折累积、单次比较、硬清零和 `commit=False` 预览。
- `SimpleCluster` 应与完整模型 `mode="integer_reference", backend="scalar"` 逐步一致；测试中使用随机输入和权重进行对照。
