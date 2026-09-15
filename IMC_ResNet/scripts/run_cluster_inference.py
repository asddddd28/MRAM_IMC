"""Evaluate real SCU/MR Cluster inference and same-subset digital controls."""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from spikingjelly.activation_based import ann2snn, functional, neuron

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from ANN_ResNet10.models.resnet10 import ResNet10
from IMC_ResNet.models.cluster_backend import ClusterConv2d
from IMC_ResNet.models.imc_resnet import (
    make_cluster_model, make_quantized_digital_reference, set_hard_reset,
    reset_cluster_model,
)


def evaluate(model, loader, steps, name, cluster=False):
    correct = {t: 0 for t in steps}
    predictions = {t: [] for t in steps}
    total = 0
    start = time.perf_counter()
    firing, seen, hooks = {}, {}, []
    for node_name, node in model.named_modules():
        if isinstance(node, neuron.IFNode):
            firing[node_name] = {'spikes': 0, 'slots': 0, 'silent_neurons': 0, 'neurons': 0}
            def observe(_node, _inputs, output, key=node_name):
                active = output.bool()
                row = firing[key]
                row['spikes'] += int(active.sum())
                row['slots'] += active.numel()
                seen[key] = active if key not in seen else seen[key] | active
            hooks.append(node.register_forward_hook(observe))
    with torch.no_grad():
        for batch, (x, y) in enumerate(loader):
            if cluster:
                reset_cluster_model(model)
            else:
                functional.reset_net(model)
            logits_sum = None
            for t in range(1, max(steps) + 1):
                logits = model(x)
                logits_sum = logits if logits_sum is None else logits_sum + logits
                if t in correct:
                    pred = logits_sum.argmax(1)
                    correct[t] += int((pred == y).sum())
                    predictions[t].extend(pred.tolist())
            total += len(y)
            for key, active in seen.items():
                firing[key]['silent_neurons'] += int((~active).sum())
                firing[key]['neurons'] += active.numel()
            seen.clear()
            print(f'{name}: {total}/{len(loader.dataset)} samples, '
                  f'T={max(steps)} ACC={correct[max(steps)] / total:.4f}, '
                  f'elapsed={time.perf_counter()-start:.1f}s', flush=True)
    if cluster:
        reset_cluster_model(model)  # Flush the last batch peak histogram.
    for hook in hooks:
        hook.remove()
    for row in firing.values():
        row['firing_rate'] = row['spikes'] / max(1, row['slots'])
        row['zero_firing_ratio'] = row['silent_neurons'] / max(1, row['neurons'])
    return {'samples': total, 'seconds': time.perf_counter() - start,
            'firing_by_layer': firing,
            'accuracy_by_steps': {str(t): correct[t] / total for t in steps},
            'predictions_by_steps': {str(t): predictions[t] for t in steps}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ann-checkpoint', type=Path, default=ROOT.parent/'ANN_ResNet10/checkpoints/resnet10_fashionmnist_5ep.pt')
    parser.add_argument('--data', type=Path, default=ROOT/'data')
    parser.add_argument('--steps', type=int, nargs='+', default=[8, 16, 32, 64])
    parser.add_argument('--calibration-batches', type=int, default=10)
    parser.add_argument('--test-size', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--mr-bits', type=int, default=8)
    parser.add_argument('--overflow', choices=['wide_reference', 'error', 'wrap', 'saturate'], default='wide_reference')
    parser.add_argument('--output', type=Path, default=ROOT/'checkpoints/actual_cluster.json')
    a = parser.parse_args()
    if min(a.steps + [a.test_size, a.batch_size, a.calibration_batches, a.threads]) < 1:
        parser.error('steps and sizes must be positive')
    torch.set_num_threads(a.threads)
    torch.manual_seed(20260912)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.2860,), (.3530,))])
    train = datasets.FashionMNIST(a.data, train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(a.data, train=False, download=True, transform=tf)
    cal = DataLoader(Subset(train, range(min(a.calibration_batches*128, len(train)))), 128, shuffle=False)
    loader = DataLoader(Subset(test, range(min(a.test_size, len(test)))), a.batch_size, shuffle=False)
    ann = ResNet10().eval()
    ck = torch.load(a.ann_checkpoint, map_location='cpu', weights_only=False)
    ann.load_state_dict(ck['model'] if 'model' in ck else ck)
    with torch.no_grad():
        ann_correct = sum(int((ann(x).argmax(1) == y).sum()) for x, y in loader)
        # Converter may fuse/mutate modules, so do not reuse ann as a control.
        snn = ann2snn.Converter(cal, device='cpu', mode='99.9%', fuse_flag=True)(copy.deepcopy(ann)).eval()
    model, mapping = make_cluster_model(snn, a.mr_bits, a.overflow)
    result = {
        'schema': 'actual_cluster_v1', 'device': 'cpu', 'threads': a.threads,
        'samples': len(loader.dataset), 'test_indices': [0, len(loader.dataset)-1],
        'steps': sorted(set(a.steps)), 'calibration_samples': len(cal.dataset),
        'mode': '99.9%', 'checkpoint': str(a.ann_checkpoint.resolve()),
        'checkpoint_sha256': hashlib.sha256(a.ann_checkpoint.read_bytes()).hexdigest(),
        'overflow_policy': a.overflow, 'ann_accuracy': ann_correct / len(loader.dataset),
        'mapping': mapping,
        'boundary': 'float stem/FC/bias/scalers/residual add; int5 Cluster hidden convolutions; digital hard-reset IF',
        'mr_statistics_window': f'whole sequence through T={max(a.steps)}, every fold BEFORE reset, includes all padded MR lanes',
        'controls': {},
    }
    # Save completed stages so interruption never masquerades as a full result.
    a.output.parent.mkdir(parents=True, exist_ok=True)
    def save(status):
        result['status'] = status
        a.output.write_text(json.dumps(result, indent=2), encoding='utf8')
    save('running_controls')
    for name, control in [('float_soft_reset', snn),
                           ('float_hard_reset', set_hard_reset(copy.deepcopy(snn))),
                           ('int5_hard_reset_digital', make_quantized_digital_reference(snn))]:
        result['controls'][name] = evaluate(control, loader, result['steps'], name)
        save('running_controls')
    result['cluster'] = evaluate(model, loader, result['steps'], 'actual_cluster', cluster=True)
    result['mr_by_layer'] = {name: module.bank.stats.report() for name, module in model.named_modules()
                             if isinstance(module, ClusterConv2d)}
    qpred = result['controls']['int5_hard_reset_digital']['predictions_by_steps']
    result['prediction_disagreements_vs_int5_digital'] = {
        t: sum(x != y for x, y in zip(pred, qpred[t]))
        for t, pred in result['cluster']['predictions_by_steps'].items()}
    save('complete')
    print(json.dumps({'ann_accuracy': result['ann_accuracy'],
                      'cluster_accuracy': result['cluster']['accuracy_by_steps'],
                      'prediction_disagreements': result['prediction_disagreements_vs_int5_digital'],
                      'mr': {k: {s: v[s] for s in ('max_mr', 'max_required_bits', 'mean_observed_bits', 'mean_logical_peak_bits', 'over_limit_fold_observations')}
                             for k,v in result['mr_by_layer'].items()}}, indent=2))


if __name__ == '__main__':
    main()
