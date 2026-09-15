# ANN_ResNet10

FashionMNIST 上的基础 ResNet-10 实现，用于先验证 ANN 推理、训练和后续权重映射。

## 结构

- 输入：`1×28×28`
- Stem：3×3 卷积
- 四个 residual stage，每个 stage 1 个 BasicBlock（ResNet-10）
- 通道数：32 → 64 → 128 → 256
- 自适应平均池化 + 10 类全连接层
- 默认不使用预训练权重，训练结果保存到 `checkpoints/`

## 运行

```powershell
cd D:\Projects\ai\MRAM_IMC\ANN_ResNet10
python -m pip install -r requirements.txt
python -m pytest -q tests
python scripts/train.py --epochs 5
```

快速冒烟训练：

```powershell
python scripts/train.py --epochs 1 --train-size 10000 --test-size 2000 --output checkpoints/smoke.pt
```

评估已有权重：

```powershell
python scripts/train.py --eval-only --output checkpoints/smoke.pt --test-size 10000
```

运行时会自动下载 FashionMNIST；若网络不可用，需要把数据集放入 `data/`。
本目录先提供 ANN baseline，不将该模型误称为 SNN 或 MRAM-IMC 等价模型。
