"""Evaluate the fine-tuned hard-reset SNN through real Cluster accumulators."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from IMC_ResNet.models.cluster_backend import ClusterConv2d
from IMC_ResNet.models.finetuned_snn import (
    load_finetuned, fuse_explicit_hidden, make_explicit_quantized_reference,
    make_explicit_cluster, SingleStepExplicit,
)
from IMC_ResNet.scripts.run_cluster_inference import evaluate
from IMC_ResNet.scripts.summarize_results import summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    snn_root = ROOT.parent / 'SNN_ResNet10'
    parser.add_argument('--checkpoint', type=Path, default=snn_root/'checkpoints/hard_reset_finetuned.pt')
    parser.add_argument('--thresholds', type=Path, default=snn_root/'checkpoints/calibrated.json')
    parser.add_argument('--data', type=Path, default=snn_root/'data')
    parser.add_argument('--steps', type=int, nargs='+', default=[16, 64])
    parser.add_argument('--test-size', type=int, default=128)
    parser.add_argument('--test-indices', type=int, nargs='+', help='Explicit diagnostic subset; overrides test-size')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--mr-bits', type=int, default=8)
    parser.add_argument('--overflow', choices=['wide_reference', 'error', 'wrap', 'saturate'], default='wide_reference')
    parser.add_argument('--output', type=Path, default=ROOT/'checkpoints/finetuned_t64_n128.json')
    args = parser.parse_args()
    if min(args.steps + [args.test_size, args.batch_size, args.threads]) < 1:
        parser.error('steps and sizes must be positive')
    torch.set_num_threads(args.threads)
    torch.manual_seed(20260912)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.2860,), (.3530,))])
    dataset = datasets.FashionMNIST(args.data, train=False, download=False, transform=transform)
    indices = args.test_indices if args.test_indices is not None else list(range(min(args.test_size, len(dataset))))
    if len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= len(dataset):
        parser.error('test-indices must be unique valid dataset indices')
    loader = DataLoader(Subset(dataset, indices), args.batch_size, shuffle=False)
    original, metadata = load_finetuned(args.checkpoint, args.thresholds)
    fused = fuse_explicit_hidden(original)
    reference = make_explicit_quantized_reference(fused)
    actual, mapping = make_explicit_cluster(fused, args.mr_bits, args.overflow)
    result = {
        'schema': 'actual_cluster_v1', 'model_source': 'fine-tuned explicit SNNResNet10',
        **metadata, 'device': 'cpu', 'threads': args.threads, 'batch_size': args.batch_size,
        'samples': len(loader.dataset), 'test_indices': indices,
        'steps': sorted(set(args.steps)), 'hard_reset': True, 'configured_mr_bits': args.mr_bits,
        'normalization': {'mean': [.2860], 'std': [.3530]},
        'overflow_policy': args.overflow, 'mapping': mapping,
        'boundary': 'float stem/FC/folded bias/residual add; fused int5 hidden convolutions; original digital hard-reset IF thresholds',
        'mr_statistics_window': f'whole sequence through T={max(args.steps)}, every fold BEFORE reset/overflow policy; all padded lanes included',
        'controls': {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save(status):
        result['status'] = status
        # A distinct partial file avoids exposing a half-written JSON to readers.
        temp = args.output.with_suffix('.tmp')
        temp.write_text(json.dumps(result, indent=2), encoding='utf8')
        temp.replace(args.output)
    start = time.perf_counter()
    save('running_controls')
    try:
        for name, model in [('float_hard_reset', original), ('float_fused_hard_reset', fused),
                            ('int5_hard_reset_digital', reference)]:
            result['controls'][name] = evaluate(SingleStepExplicit(model), loader, result['steps'], name)
            save('running_controls')
        save('running_cluster')
        result['cluster'] = evaluate(SingleStepExplicit(actual), loader, result['steps'], 'actual_finetuned_cluster', cluster=True)
        result['mr_by_layer'] = {name: module.bank.stats.report() for name, module in actual.named_modules()
                                 if isinstance(module, ClusterConv2d)}
        def disagreements(first, second):
            return {t: sum(a != b for a, b in zip(pred, second['predictions_by_steps'][t]))
                    for t, pred in first['predictions_by_steps'].items()}
        result['prediction_disagreements_vs_int5_digital'] = disagreements(result['cluster'], result['controls']['int5_hard_reset_digital'])
        result['fusion_prediction_disagreements'] = disagreements(result['controls']['float_hard_reset'], result['controls']['float_fused_hard_reset'])
        result['total_seconds'] = time.perf_counter() - start
        save('complete')
        result['mr_summary'] = summary(result)
        save('complete')
        print(json.dumps({'accuracies': {name: report['accuracy_by_steps'] for name, report in result['controls'].items()},
                          'cluster': result['cluster']['accuracy_by_steps'],
                          'disagreements': result['prediction_disagreements_vs_int5_digital'],
                          'mr_summary': result['mr_summary']}, indent=2), flush=True)
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
        save('failed')
        raise


if __name__ == '__main__':
    main()
