# Cluster 项目结构与维护指南

## 1. 目录职责

所有命令和下文的相对路径均以 `model_imc_cluster/` 为项目根。
原始规范 [Reference.md](../Reference.md)、图片及 PDF 保持原路径与原内容；
本文是代码导航，不覆盖硬件规范。

```text
model_imc_cluster/
├── README.md                       # 使用说明与入口导航
├── Reference.md                    # 权威规范（保留原位置）
├── TEST_REPORT.md                  # 兼容旧链接，指向 docs/TEST_REPORT.md
├── config.yaml                    # 兼容默认配置（JSON 格式的 YAML 子集）
├── requirements.txt               # 运行/测试依赖
├── pytest.ini                     # 默认测试发现范围
├── run_model.py                   # 兼容 CLI 入口
├── run_tests.py                   # 兼容测试入口，只转发 main
├── cluster_model/                 # 模型源码；公共 API 路径保持不变
│   ├── __init__.py                # 稳定公共 API
│   ├── __main__.py                # python -m cluster_model
│   ├── config.py                  # 结构、模式、位宽与模拟参数校验
│   ├── encoding.py                # int5 编解码、输入校验
│   ├── digital.py                 # PMAC、SCU/MR 与整数归约
│   ├── analog.py                  # 设备、事件噪声与模拟转换链
│   ├── tdp.py                     # 固定步长 TDP/eigen-train 方波序列
│   ├── events.py                  # 功能事件、依赖与显式时序
│   ├── cluster.py                 # 单 Cluster 事务执行器
│   ├── memory.py                  # 权重行/上下文状态后备存储
│   ├── mapping.py                 # FC/卷积折叠及 FC 层复用
│   ├── trace.py                   # JSON/JSONL 与首次状态分歧
│   ├── cli.py                     # reference/simulate/infer
│   └── model.py                   # 旧导入兼容层，无独立计算实现
├── profiles/                      # 整数参考、理想、非理想实验配置
├── scripts/
│   ├── __init__.py
│   └── run_tests.py               # pytest 调用与机器可读报告汇总
├── tests/
│   ├── conftest.py                # 统一导入路径
│   ├── acceptance/
│   │   └── test_reference.py      # Reference T01–T13 行为验收
│   ├── api/
│   │   ├── test_interfaces.py     # 参数/精度/事务/映射/存储 API
│   │   └── test_project_structure.py # 兼容入口与报告脚本回归
│   └── integration/
│       └── test_cli.py            # 命令行、成功/失败产物、外部目录启动
├── docs/
│   ├── ARCHITECTURE.md            # 本文
│   └── TEST_REPORT.md             # 核验结果、规范对应关系与限制
└── reports/                       # 可复现实验及测试输出，不放模型实现
    └── acceptance/                # summary.json、junit.xml、pytest-output.txt
```

源码继续采用小型平铺包，不为现有十余个模块额外引入多层子包或路径兼容副本。
新增计算逻辑应放入相应职责模块，而非根入口、测试脚本或 `model.py`。

## 2. 建议阅读顺序

1. `config.py` / `encoding.py`：固定 16×36×5 结构、输入范围及补码位权。
2. `digital.py`：每个神经元的 `(16,5)` SCU/MR 历史及进位守恒。
3. `analog.py`：TDP→TW/SG→充电/保持→共享→MR 参考→比较器。
4. `cluster.py`：将单折/多折执行串联成一次判决、一次提交。
5. `memory.py` / `mapping.py`：同一个物理 Cluster 上复用多个逻辑上下文。
6. `tests/acceptance/test_reference.py`：逐项对照规范；再看 API 和 CLI 回归。

## 3. 执行数据流与状态所有权

```text
输入/权重 → 编码 → 每折 PMAC → 候选 SCU/MR 更新（逐折检查）
                                      ↓ 所有折完成
                            S/G/U 精确归约
                              ├─ 数字：U >= theta
                              └─ 模拟：S 的转换链与 theta-L*G 参考比较
                                      ↓ 唯一实际 spike
                             硬清零或保留候选 → 原子提交
```

- 每路 `a + L*h` 保存历史，正常进位满足新值等于旧值加当前计数。
  跨 Macro/位平面的有符号归约不能替代局部历史，否则长期抵消时的 MR 溢出会被隐藏。
- `candidate` 为判决前状态，`state` 为拟提交/已提交结果；返回数组不与工作槽共享。
- `commit=False` 返回预览结果，不修改数字工作状态、逻辑步、物理时间及参考电压。
  `ModelError` 仍可更新诊断 `last_error`；诊断不是快照回滚的一部分。
- 一次 `step_tiles` 完成后才比较。迭代失败、逐折溢出或模拟错误不能提交半成品。
  原子性限于单次逻辑步，`run` 的全序列和 `FCLayer.forward` 的整层不是一个事务。
- 神经元数字状态和逻辑步属于 `ContextRecord`。物理时间、参考建立电压属于复用设备，
  正常上下文切换不把它们清零。显式 `Cluster.reset()` 才重新开始整场仿真。
- 模拟电容每次转换预充电；不要把上一轮电荷再次加到已有数字历史上。
- 实际比较器的 spike 决定硬清零，包括非理想误发放，不能用理想 oracle 代替提交决定。

## 4. 精度、随机性和物理边界

- 标量路径采用 Python 任意精度整数；NumPy 只用于有界 PMAC/计数更新，转换前检查 int64。
  加权归约和 MR 门槛修正始终使用 Python 整数。模拟端使用浮点，不应反向污染整数 oracle。
- `chip_seed` 决定固定设备失配；`event_seed + chip_id + context + step + operation`
  经稳定 SHA256 事件键决定动态噪声。重复预览/改变调用顺序不应改变同一事件样本。
- `theta_eff` 可为负；正 comparator offset 抬高门槛，`margin >= 0` 发放。
- 未提供数字读写时长时，只记录零时长功能事件及依赖；模拟时序为归一化假设。
- P4/T14 训练、GPU、任意开关网络、完整存储故障、真实三模式控制器和 P5 标定未实现。
  本整理不增加芯片 PPA 或实测精度声明。

## 6. TDP 方波序列模型

`cluster_model.tdp` 独立提供论文式的离散 TDP/eigen-train：对 `m` 位无符号值，
第 `n` 位产生 `2^n` 个脉冲，脉冲周期为 `2^m/2^n` 个步长，窗口为 `2^m` 个步长。
`eigen_train()` 返回固定步长的 `PulseSequence`，`signed_eigen_train()` 用正负极性选择
Positive/Negative TDP 分支。多位叠加保留为离散多电平幅值，便于观察同一 slot 的重叠 bit 脉冲；若要模拟独立物理线，应分别查看各 bit plane。

这部分是 TDP 的**数字时序参考模型**：循环/计数器每经过一个步长推进一个 slot，
不使用随机脉冲，也不把权重大小错误地解释成单个连续模拟延迟。当前 Cluster 的既有
`analog.convert()` 仍保留归一化边沿—电荷—比较器模型；两者分别用于时序波形分析和
现有行为验收，避免未经论文电路细节确认就改变已验证的积分语义。

## 7. 运行与扩展约定

```powershell
# 项目根：默认发现全部三组测试
python -m pytest -q
python run_tests.py

# 可按职责回归；这些命令不更新 acceptance 的统一报告
python -m pytest tests/acceptance -q
python -m pytest tests/api -q
python -m pytest tests/integration -q

# 新脚本位置与旧入口执行相同的测试汇总实现
python scripts/run_tests.py
python -m scripts.run_tests

# 旧 CLI / 公共导入路径保持有效
python run_model.py simulate
python -m cluster_model infer
```

从项目外调用时，使用根脚本绝对路径；配置与 `--output` 的显式相对路径按当前工作目录解释。
CLI 未显式指定 output 时仍写项目内 `reports/<command>/`。
测试脚本在新运行前移除旧 JUnit，防止 pytest 启动失败被旧成功计数掩盖。
新增功能应先补相应组的测试；规范条款变更需核对 `Reference.md`，再更新代码与核验文档。

