# 五位补码 MRAM–IMC Cluster Python 模型

依据 **`Reference.md` v1.0（2026-09-09）** 实现。默认结构是 **16 个 Macro × 36 个输入 × 5 个位平面**，每个 Macro 保留五路独立 SCU/MR；一个 Cluster 每个逻辑时间步只产生一个 spike。

**验收结果：136 项测试通过，0 失败、0 错误、0 跳过。** 详见 [测试报告](docs/TEST_REPORT.md) 和 `reports/acceptance/summary.json`。

## 目录导航

| 路径 | 职责 |
|---|---|
| `cluster_model/` | 模型源码；按数字、模拟、执行器、存储、映射等职责划分 |
| `profiles/` | 整数参考、理想与假设非理想配置 |
| `tests/acceptance/` | Reference T01–T13 验收 |
| `tests/api/` | API、精度、事务、映射、项目兼容性测试 |
| `tests/integration/` | CLI 与输出产物测试 |
| `scripts/` | 测试执行及报告汇总脚本 |
| `docs/` | 架构/维护指南与核验报告 |
| `reports/` | 可重跑的测试与示例输出 |

简化版 Python 推理/调试模型的设计方案见 [SIMPLIFIED_MODEL_PLAN.md](docs/SIMPLIFIED_MODEL_PLAN.md)。`tdp.py`\n目前用于固定步长波形观察，不替代默认整数判决。\n\n完整目录树、模块阅读顺序、状态所有权及维护约定见
[架构与维护指南](docs/ARCHITECTURE.md)。原始 `Reference.md`、图片与 PDF 保留原路径，
根目录 `run_model.py` / `run_tests.py` 和 `cluster_model.model` 兼容导入保持可用。

TDP/eigen-train 离散方波参考模型见 `cluster_model/tdp.py` 与 `docs/ARCHITECTURE.md`；它按固定步长循环生成周期序列，正负权重分别选择 TDP 分支。

## 1. 快速使用

在 PowerShell 中进入本目录：

```powershell
cd D:\Projects\ai\MRAM_IMC\model_imc_cluster

# 本次已在 Python 3.13.13 / NumPy 2.5.1 / pytest 9.1.1 环境运行。
# 其他环境首次使用时安装依赖：
python -m pip install -r requirements.txt

# 完整测试；保留 JUnit XML、结构化汇总和原始输出
python run_tests.py

# 精确整数参考，Python 任意精度计数，报告实验 MR 容量超限
python run_model.py reference --config profiles/integer_reference.json

# 理想 TDP → 电荷 → 共享 → MR参考 → 比较器
python run_model.py simulate

# 切换 3-bit SCU；阈值 100 的示例能观察到发放和硬复位
python run_model.py simulate --scu-bits 3 --theta 100 --output reports/spiking

# 假设失配/抖动参数，不是工艺实测参数
python run_model.py simulate --analog-config profiles/nonideal_assumptions.json --output reports/nonideal

# 8→4→2 的两层小型 IF SNN，逐步与独立矩阵 IF oracle 比较
python run_model.py infer --steps 8
```

也支持 `python -m cluster_model simulate`。在仓库根目录可执行：

```powershell
python D:\Projects\ai\MRAM_IMC\model_imc_cluster\run_tests.py
python D:\Projects\ai\MRAM_IMC\model_imc_cluster\run_model.py infer
```

`profiles/*.json` 是推荐的配置入口。`config.yaml` 保留同一默认配置，内容采用 JSON 格式（合法的 YAML 子集），可以直接传给 `--config`；不需要 PyYAML。

## 2. Python API

以下示例从本目录运行：

```python
import numpy as np
from cluster_model import Cluster, ClusterConfig

cluster = Cluster(ClusterConfig(scu_bits=4, mr_bits=8, theta_count=10))
x = np.zeros((16, 36), dtype=int)
w = np.zeros((16, 36), dtype=int)
x[0, 0] = 1
w[0, 0] = 3

for step in range(4):
    result = cluster.step(x, w)
    print(step, result.diagnostics["total_u"], int(result.spike))
# 0 3 0
# 1 6 0
# 2 9 0
# 3 12 1

assert not np.any(cluster.state.scu)
assert not np.any(cluster.state.mr)
```

- `result.candidate`：最终判决前的候选五路状态。
- `result.state`：实际 spike 决定提交或硬复位后的状态，返回副本。
- `result.diagnostics`：每折 PMAC、进位、SCU/MR、局部/全局归约、溢出、操作计数。
- `result.analog`：边沿、原始/有效 TW、符号、局部电荷、保持时间、共享电压、参考和 margin；纯数字模式为 `None`。
- `result.events`：含时间、资源和依赖的功能事件列表。
- `cluster.step(..., commit=False)`：无提交预览；同一事件 ID 的噪声复现。
- `cluster.snapshot()/restore()`：数字状态、逻辑步、参考电压与物理时间快照。

### 输入折叠

```python
from cluster_model import map_fc

# 一个 800 项输入可以展开成两折；仍然只判决一次。
result = cluster.step_tiles(map_fc(np.ones(800, dtype=int), np.zeros(800, dtype=int)))
```

输入必须是整数二值数据，权重必须是 `[-16, 15]` 整数。不会把 `0.5`、越界整数或浮点权重悄悄截断。也可通过 `weight_bits=` 传入形状 `(16,36,5)` 的二值读出结果。

### 上下文与映射

- `LogicalMemory` 分开管理权重行和状态行；读写均复制，不共享可变数组。
- `FCLayer` 支持 `(batch, inputs)`，每个样本/输出使用独立上下文；所有输出在同一个物理 Cluster 模型上顺序复用。
- `iter_conv_tiles` 接受 CHW 输入和 OIHW 权重，按 `(channel, kernel_y, kernel_x)` 展开，支持整数 stride/padding。
- `flat_address` / `unflatten_address` 是可逆映射，尾部零填充。
- `state_capacity_bits` 可显式限制逻辑后备容量。默认 `None` 代表**物理容量未知**，不是无限硬件存储。

## 3. 核心语义

1. 补码位权为 `(1,2,4,8,-16)`；不是符号—幅度编码。
2. 原始状态始终是 `(16,5)` 的 SCU 和 `(16,5)` 的 MR。`U/S/G` 是派生量，不会写回替代原五路状态。
3. 每折分别更新 `z=a+k; carry=z//L; a=z%L; h+=carry`。严格模式在**每次实际 MR 更新**检查溢出，不能被全局抵消或最终复位掩盖。
4. 每个逻辑时间步完成全部输入折后才比较一次；异常不会提交一半的状态。
5. 实际比较器的 `fire` 决定当前上下文全部 80 路 SCU 和 80 路 MR 的清零。不会清除其他上下文或权重。
6. 默认无神经元泄漏、无偏置、硬清零，是 IF 工作点；电容泄漏与神经元泄漏是不同概念。
7. 模拟电容每次转换预充电，不重复累加已经包含在数字状态中的历史。
8. 电荷按 `q=sign*I*TW` 产生；共享电压来自总电荷除以所有接入电容，零贡献电容不会被移除。
9. 参考由 `theta_eff=theta-L*G` 得到，可以为负；正比较器 offset 抬高门槛；等号发放。

## 4. 模式、精度与错误处理

| 配置 | 行为 |
|---|---|
| `mode=integer_reference` | 直接用精确整数候选 `U>=theta` 判决 |
| `mode=finite_digital` | 同样纯数字判决，配合明确的有限位宽策略 |
| `mode=ideal_behavioral` | 理想归一化电路转换链；拒绝偷偷开启非理想参数 |
| `mode=nonideal_behavioral` | 参数化边沿、充电、共享、参考和比较器模型 |
| `backend=scalar` | 独立 Python 循环参考，计数可任意精度增长 |
| `backend=numpy` | CPU NumPy 有界计数运算；归约仍使用 Python 整数 |

NumPy 后端在固定宽度更新前检查 `int64` 容量，超出报 `int64_capacity_exceeded`，不会静默回卷或假装无限精度。模拟链为 float64；本次理想等价性验收使用归一化 profile。

MR 策略：

- `error`（默认）：报 `MROverflowError`，保留溢出位置与原始候选状态，不提交。
- `wide_reference`：保留宽计数，并记录实验 MR 容量超限。
- `wrap`：显式回卷故障分析。
- `saturate`：显式饱和变体，不称为默认硬件。

`ModelError.code` / `.diagnostics` 和 `cluster.last_error` 提供可定位诊断。CLI 失败退出码为 2，并写 `error.json` 和 `status=failed` 的 `summary.json`。

## 5. 非理想模型边界

已提供的**归一化假设**包括：逐位平面增益、固定芯片失配、公共/差分边沿抖动、最小有效脉宽、符号死区/未稳、局部恒流 RC 充电与保持、注入电荷、寄生电容、一阶共享建立、参考量化/范围/建立、比较器 offset/noise/超时，以及受控 `margin_injection`。

- `chip_seed` 只用于生成固定设备参数；`event_seed` 根据稳定的芯片/上下文/逻辑步/操作 ID 采样。改变执行顺序不改变同一事件的样本。
- `DeviceInstance.save/load` 可保存和重载同一设备。固定失配不会每个时间步重抽。
- `Timing(share_at=..., compare_at=...)` 是相对候选就绪时刻的显式时间；非法时序会报错，不会自动“修好”。
- 当前采用共同起点、持续边沿及同步共享的声明假设，不实现任意错开窄脉冲 XOR 或任意开关网络。
- 模型假定非零脉宽充电前符号已经有效；未满足时停止，不虚构亚稳态概率。
- 数字读写延迟未给出，事件只记录功能顺序（零占用时长）；模拟时间是参数化值。事件调度器可检测资源冲突，但**没有标定的真实资源表或完整控制器**。

## 6. 本次完成范围

| Reference 阶段 | 本次状态 |
|---|---|
| P0 | 完成：配置、编码、精确整数/有限计数、状态事务、trace |
| P1 | 完成：默认归一化理想链与对应行为验收 |
| P2 | 部分完成：参数化非理想链、T12 列举的诊断/复现、功能事件和资源检查；未实现完整存储故障和真实开关/控制器 |
| P3 | 完成 CPU 范围：标量/NumPy 一致、FC/小卷积映射、小型两层网络推理、首次状态分歧和逻辑容量诊断；未实现 GPU 执行器 |
| P4 / T14 | 未实现 PyTorch 梯度封装与训练，本次不开放 `train` 命令 |
| P5 | 未实现：缺少 SPICE/实测标定、实际存储布局及三模式控制协议 |

**本项目是已测试的行为模型，不是已标定芯片仿真器。** 不报告真实芯片面积、能量、能效、吞吐或延迟精度；日志中的电压/电荷也不代表添加了硬件 ADC。

## 7. 产物

- `profiles/`：整数参考、理想 profile 与带明确假设的非理想参数。
- `reports/acceptance/`：测试汇总、JUnit XML、原始 pytest 输出。
- `reports/reference/`、`reports/simulate/`、`reports/simulate_scu3/`、`reports/nonideal/`、`reports/infer/`：固定种子示例。
- 每个示例保存 `manifest.json`、`inputs.json`、`trace.jsonl`、`summary.json`；单 Cluster 示例另存设备参数。
- 原始 `Reference.md`、示意图和 PDF 未修改。


