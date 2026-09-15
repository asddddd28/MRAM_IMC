"""Replay the unmodified T64 Cluster with read-only simultaneous-state monitors.

The first pass measures all lanes and saves high-MR snapshots. A second pass
replays only samples containing selected peaks, tracing exact signed-weight
contributions from t=1. Neither pass changes mapping, weights or reset policy.
"""
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
from IMC_ResNet.models.finetuned_snn import load_finetuned, fuse_explicit_hidden, make_explicit_cluster, SingleStepExplicit
from IMC_ResNet.models.imc_resnet import reset_cluster_model
from IMC_ResNet.models.mr_observer import MRObserver


def attach(model, mapping, trace_targets=None):
    observers = {}
    for entry in mapping:
        name, sink = entry['layer'], entry['reset_if']
        observer = MRObserver(name, sink)
        if trace_targets:
            observer.trace_targets = set(trace_targets.get(name, []))
        model.get_submodule(name).observer = observer
        node = model.get_submodule(sink)
        node.register_forward_pre_hook(observer.before_if)
        node.register_forward_hook(observer.after_if)
        observers[name] = observer
    return observers


def run(model, observers, dataset, indices, batch_size, steps, progress):
    loader = DataLoader(Subset(dataset, indices), batch_size, shuffle=False)
    wrapper = SingleStepExplicit(model)
    correct = {t: 0 for t in steps}
    predictions = {str(t): [] for t in steps}
    count = 0
    start = time.perf_counter()
    with torch.no_grad():
        for x, y in loader:
            reset_cluster_model(model)
            ids = indices[count:count+len(y)]
            for observer in observers.values():
                observer.begin_batch(ids)
            summed = None
            for t in range(1, max(steps)+1):
                for observer in observers.values():
                    observer.step = t
                logits = wrapper(x)
                summed = logits if summed is None else summed + logits
                if t in correct:
                    pred = summed.argmax(1)
                    correct[t] += int((pred == y).sum())
                    predictions[str(t)].extend(pred.tolist())
            for name, observer in observers.items():
                observer.finish_batch(model.get_submodule(name))
            count += len(y)
            progress(count, len(indices), time.perf_counter()-start)
    reset_cluster_model(model)  # Flush final peak histogram exactly once.
    return {'samples': count, 'seconds': time.perf_counter()-start,
            'accuracy_by_steps': {str(t): correct[t]/count for t in steps},
            'predictions_by_steps': predictions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=ROOT/'checkpoints/finetuned_t64_n128.json')
    parser.add_argument('--output', type=Path, default=ROOT/'checkpoints/mr_conditions_t64_n128.json')
    parser.add_argument('--test-size', type=int, default=128)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--skip-trace', action='store_true')
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding='utf8'))
    if baseline['status'] != 'complete' or not 0 < args.test_size <= baseline['samples']:
        parser.error('requires complete baseline and matching prefix subset')
    torch.set_num_threads(args.threads)
    torch.manual_seed(20260912)
    original, metadata = load_finetuned(Path(baseline['checkpoint']), Path(baseline['threshold_source']))
    for key in ('checkpoint_sha256', 'threshold_sha256', 'if_thresholds'):
        if metadata[key] != baseline[key]:
            raise ValueError(f'baseline mismatch: {key}')
    fused = fuse_explicit_hidden(original)
    actual, mapping = make_explicit_cluster(fused, 8, 'wide_reference')
    observers = attach(actual, mapping)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.2860,), (.3530,))])
    dataset = datasets.FashionMNIST(ROOT.parent/'SNN_ResNet10/data', train=False, download=False, transform=tf)
    result = {'schema': 'mr_conditions_v1', **metadata, 'baseline': str(args.baseline.resolve()),
              'samples': args.test_size, 'steps': baseline['steps'], 'batch_size': baseline['batch_size'],
              'threads': args.threads, 'mapping': mapping, 'overflow_policy': 'wide_reference',
              'semantics': 'same-time post-fold/pre-reset unsigned MR; IF voltage is end-of-step; padding included; bitplane signs are not weight signs'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save(status):
        result['status'] = status
        temp = args.output.with_suffix('.tmp')
        temp.write_text(json.dumps(result, indent=2), encoding='utf8')
        temp.replace(args.output)
    def progress(n, total, seconds):
        result['progress'] = {'samples_done': n, 'total': total, 'seconds': seconds}
        print(f'{result["status"]}: {n}/{total}, {seconds:.1f}s', flush=True)
        # Detailed reports are persisted at the end of each pass.
        save(result['status'])
    start = time.perf_counter()
    save('running_monitor')
    try:
        result['cluster'] = run(actual, observers, dataset, list(range(args.test_size)), baseline['batch_size'], baseline['steps'], progress)
        result['mr_by_layer'] = {name: actual.get_submodule(name).bank.stats.report() for name in observers}
        result['conditions_by_layer'] = {name: obs.report() for name, obs in observers.items()}
        result['prediction_disagreements_vs_baseline'] = {
            t: sum(a != b for a, b in zip(pred, baseline['cluster']['predictions_by_steps'][t][:args.test_size]))
            for t, pred in result['cluster']['predictions_by_steps'].items()}
        if any(result['prediction_disagreements_vs_baseline'].values()):
            raise AssertionError('monitor changed baseline predictions')
        if args.test_size == baseline['samples']:
            result['all_mr_statistics_identical_to_baseline'] = result['mr_by_layer'] == baseline['mr_by_layer']
            if not result['all_mr_statistics_identical_to_baseline']:
                raise AssertionError('monitor changed MR statistics')
        save('monitor_complete')
        if not args.skip_trace:
            targets = {}
            for name, report in result['conditions_by_layer'].items():
                cases = report['events_7bit'] + report['top_cases'][:2]
                targets[name] = sorted({(c['sample'], c['channel'], c['position']) for c in cases})
            ids = sorted({sample for entries in targets.values() for sample, _, _ in entries})
            result['trace_targets'] = targets
            if ids:
                traced, mapping = make_explicit_cluster(fused, 8, 'wide_reference')
                trace_observers = attach(traced, mapping, targets)
                save('running_trace')
                result['trace_run'] = run(traced, trace_observers, dataset, ids, baseline['batch_size'], baseline['steps'], progress)
                result['traces_by_layer'] = {name: obs.trace for name, obs in trace_observers.items() if obs.trace}
                result['trace_test_indices'] = ids
                errors = [s['integer_decomposition_error'] for rows in result['traces_by_layer'].values() for s in rows]
                result['trace_max_integer_decomposition_error'] = max(map(abs, errors), default=0)
                result['trace_prediction_disagreements'] = {t: sum(p != baseline['cluster']['predictions_by_steps'][t][i] for p, i in zip(pred, ids)) for t, pred in result['trace_run']['predictions_by_steps'].items()}
                if any(errors) or any(result['trace_prediction_disagreements'].values()):
                    raise AssertionError('trace decomposition/prediction mismatch')
        result['total_seconds'] = time.perf_counter()-start
        save('complete')
    except Exception as exc:
        result['error'] = repr(exc)
        save('failed')
        raise
    print(json.dumps({k: result[k] for k in ('status', 'cluster', 'prediction_disagreements_vs_baseline', 'total_seconds')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
