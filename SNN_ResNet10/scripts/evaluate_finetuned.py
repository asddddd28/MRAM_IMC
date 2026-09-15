"""Evaluate a fine-tuned explicit hard-reset SNN, without further training.

Legacy checkpoints do not store IF thresholds. Restore them explicitly from
calibrated.json and record the values/source alongside the evaluation result.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from spikingjelly.activation_based import neuron

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from SNN_ResNet10.models.snn_resnet10 import reset
from IMC_ResNet.models.finetuned_snn import load_finetuned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'checkpoints/hard_reset_finetuned.pt')
    parser.add_argument('--thresholds', type=Path, default=ROOT/'checkpoints/calibrated.json')
    parser.add_argument('--data', type=Path, default=ROOT/'data')
    parser.add_argument('--output', type=Path, default=ROOT/'checkpoints/hard_reset_finetuned_eval_t64.json')
    parser.add_argument('--steps', type=int, nargs='+', default=[16, 64])
    parser.add_argument('--test-size', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    if min(args.steps + [args.test_size, args.batch_size, args.threads]) < 1:
        parser.error('steps, test-size, batch-size and threads must be positive')
    torch.set_num_threads(args.threads)
    model, metadata = load_finetuned(args.checkpoint, args.thresholds)
    nodes = [(name, node) for name, node in model.named_modules() if isinstance(node, neuron.IFNode)]
    model.eval()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.2860,), (.3530,))])
    dataset = datasets.FashionMNIST(args.data, train=False, download=False, transform=transform)
    loader = DataLoader(Subset(dataset, range(min(args.test_size, len(dataset)))), batch_size=args.batch_size, shuffle=False)
    steps = sorted(set(args.steps))
    correct = {step: 0 for step in steps}
    total = 0
    start = time.perf_counter()
    with torch.inference_mode():
        for batch, (images, targets) in enumerate(loader, 1):
            reset(model)
            # A single temporal rollout provides each requested prefix result.
            # Calling forward with T=1 preserves IF state between time steps.
            logits_sum = None
            for step in range(1, steps[-1] + 1):
                logits = model(images.unsqueeze(0))[0]
                logits_sum = logits.clone() if logits_sum is None else logits_sum + logits
                if step in correct:
                    correct[step] += int(((logits_sum / step).argmax(1) == targets).sum())
            total += targets.numel()
            print(f'batch {batch}/{len(loader)} samples={total} ' + ' '.join(f'T={t}: {correct[t]/total:.4%}' for t in steps), flush=True)
    reset(model)
    result = {
        'status': 'complete', 'model': 'explicit SNNResNet10 (not official FX/IMC)',
        **metadata, 'hard_reset': True,
        'if_thresholds': {name: node.v_threshold for name, node in nodes},
        'normalization': {'mean': [.2860], 'std': [.3530]},
        'dataset': 'FashionMNIST', 'split': 'test', 'subset': 'first N, unshuffled',
        'samples': total, 'batch_size': args.batch_size, 'device': 'cpu',
        'results': [{'steps': t, 'correct': correct[t], 'accuracy': correct[t]/total} for t in steps],
        'seconds': time.perf_counter() - start,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
