"""Evaluate ANN-to-SNN weight transfer on FashionMNIST.

The script deliberately reports a *direct-transfer baseline*: ANN weights are
copied without retraining or calibration, so the result is useful for finding
scale/threshold mismatches rather than claiming a production conversion.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import SNNResNet10, convert_ann_weights, reset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ann-checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--test-size", type=int)
    p.add_argument("--encoding", choices=["rate", "bernoulli"], default="rate")
    p.add_argument("--threshold", type=float, default=1.0)
    p.add_argument("--output", type=Path, default=Path("checkpoints/ann_to_snn.json"))
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # IF 神经元接收非负输入电流；这里不使用 ANN 训练时的 Normalize，
    # 避免把像素变成负值。像素强度本身作为发放率/电流幅值。
    ds = datasets.FashionMNIST(a.data, train=False, download=True,
                               transform=transforms.ToTensor())
    if a.test_size:
        ds = Subset(ds, range(min(a.test_size, len(ds))))
    loader = DataLoader(ds, a.batch_size, shuffle=False, num_workers=0)

    model = SNNResNet10(v_threshold=a.threshold).to(device)
    info = convert_ann_weights(model, a.ann_checkpoint)
    model.eval()
    correct = total = 0
    output_abs_sum = 0.0
    forward_seconds = 0.0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            reset(model)
            if a.encoding == "rate":
                # 确定性 rate coding：每个时间步输入同一幅图像电流。
                sequence = x.unsqueeze(0).repeat(a.steps, 1, 1, 1, 1)
            else:
                # 伯努利编码：像素强度是每个时间步产生脉冲的概率。
                sequence = (torch.rand((a.steps,) + tuple(x.shape), device=device) < x.unsqueeze(0)).to(x.dtype)
            start = time.perf_counter()
            logits = model(sequence)
            forward_seconds += time.perf_counter() - start
            pred = logits.mean(0).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
            output_abs_sum += float(logits.abs().sum())

    result = {
        "accuracy": correct / total,
        "correct": correct,
        "samples": total,
        "steps": a.steps,
        "encoding": a.encoding,
        "threshold": a.threshold,
        "mean_abs_output": output_abs_sum / (total * a.steps),
        "forward_seconds": forward_seconds,
        "samples_per_second": total / forward_seconds if forward_seconds else 0.0,
        "weight_transfer": {
            "copied_tensors": len(info["copied"]),
            "source_tensors": info["source_tensors"],
            "tensor_ratio": len(info["copied"]) / info["source_tensors"],
        },
        "device": str(device),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


