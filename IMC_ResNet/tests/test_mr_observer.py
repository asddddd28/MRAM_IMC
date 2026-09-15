"""Read-only monitoring, simultaneous cancellation and reset-aware traces."""
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from spikingjelly.activation_based import neuron
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from IMC_ResNet.models.cluster_backend import ClusterConv2d
from IMC_ResNet.models.mr_observer import MRObserver, contribution_metrics


def test_local_and_cross_macro_cancellation():
    mr = np.zeros((2, 16, 5), dtype=np.int32)
    mr[0, 0, 3], mr[0, 0, 4] = 64, 32  # Same weighted magnitude.
    mr[1, 0, 3], mr[1, 1, 4] = 64, 32
    m = contribution_metrics(mr)
    assert m['net'].tolist() == [0, 0]
    assert m['local_cancelled'].tolist() == [8192, 0]
    assert m['cross_macro_cancelled'].tolist() == [0, 8192]
    assert m['cancellation_ratio'].tolist() == [1., 1.]


def test_observer_preserves_math_and_captures_pre_reset_state():
    conv = nn.Conv2d(36, 1, 1, bias=True)
    with torch.no_grad():
        conv.weight.fill_(-1)
        conv.bias.zero_()
    plain = ClusterConv2d(conv, 1.)
    watched = ClusterConv2d(conv, 1.)
    sink = neuron.IFNode(v_threshold=1., v_reset=0.)
    obs = MRObserver('conv', 'sink', thresholds=(1, 2, 64))
    obs.trace_targets = {(9, 0, 0)}
    watched.observer = obs
    sink.register_forward_hook(lambda m, a, out: watched.fire(out))
    sink.register_forward_pre_hook(obs.before_if)
    sink.register_forward_hook(obs.after_if)
    obs.begin_batch([9])
    x = torch.ones(1, 36, 1, 1)
    with torch.no_grad():
        for t in range(1, 5):
            obs.step = t
            expected, actual = plain(x), watched(x)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            # Force reset after accumulating a large negative voltage.
            spike = sink(actual + (1000 if t == 3 else 0))
            plain.fire(spike)
    assert all(s['integer_decomposition_error'] == 0 for s in obs.trace)
    assert [s['age_since_reset'] for s in obs.trace] == [1, 2, 3, 1]
    assert [s['fired'] for s in obs.trace] == [False, False, True, False]
    assert obs.trace[2]['max_mr'] == 6  # Copied before the firing hook clears it.
    assert obs.trace[3]['true_weight_negative_since_reset'] == 540
    np.testing.assert_array_equal(plain.bank.mr.numpy(), watched.bank.mr.numpy())
    obs.finish_batch(watched)
    plain.reset()
    watched.reset()
    assert plain.bank.stats.report() == watched.bank.stats.report()
    report = obs.report()
    assert sum(sum(h.values()) for h in report['peak_bits_by_macro']) == 80
    assert sum(sum(h.values()) for h in report['peak_bits_by_plane']) == 80


def test_multifold_trace_accumulates_before_single_if_decision():
    # 600 inputs require two folds sharing the SAME 16x5 bank.
    conv = nn.Conv2d(600, 1, 1, bias=False)
    with torch.no_grad():
        conv.weight[:, :300].fill_(1)
        conv.weight[:, 300:].fill_(-1)
    watched = ClusterConv2d(conv, 1.)
    sink = neuron.IFNode(v_threshold=1., v_reset=0.)
    obs = MRObserver('conv', 'sink', thresholds=(1, 32, 64))
    obs.trace_targets = {(0, 0, 0)}
    watched.observer = obs
    sink.register_forward_hook(lambda m, a, out: watched.fire(out))
    sink.register_forward_pre_hook(obs.before_if)
    sink.register_forward_hook(obs.after_if)
    obs.begin_batch([0])
    with torch.no_grad():
        for t in range(1, 4):
            obs.step = t
            sink(watched(torch.ones(1, 600, 1, 1)))
    assert len(obs.trace) == 6
    assert all(s['integer_decomposition_error'] == 0 for s in obs.trace)
    assert all(not s['fired'] for s in obs.trace)
    last = obs.trace[-1]
    assert last['true_weight_positive_since_reset'] == 3*300*15
    assert last['true_weight_negative_since_reset'] == 3*300*15
    assert last['v_charged'] == 0
    # Per-fold snapshots share the end-of-step IF voltage, not an imagined
    # intermediate threshold comparison after the first fold.
    assert obs.trace[-2]['with_scu']['net'] > 0
    assert obs.trace[-2]['v_charged'] == 0


def test_capture_every_7bit_lane_and_repeated_observations():
    conv = nn.Conv2d(36, 1, 1, bias=False)
    with torch.no_grad():
        conv.weight.fill_(-1)
    watched = ClusterConv2d(conv, 1.)
    sink = neuron.IFNode(v_threshold=1., v_reset=0.)
    obs = MRObserver('conv', 'sink')
    watched.observer = obs
    sink.register_forward_pre_hook(obs.before_if)
    sink.register_forward_hook(obs.after_if)
    obs.begin_batch([0])
    with torch.no_grad():
        for t in range(1, 33):
            obs.step = t
            sink(watched(torch.ones(1, 36, 1, 1)))
    obs.finish_batch(watched)
    watched.reset()
    events = obs.events_7bit
    assert [s['step'] for s in events] == [29, 30, 31, 32]
    lanes = sum(int((np.asarray(s['mr']) >= 64).sum()) for s in events)
    assert lanes == watched.bank.stats.report()['observed_bits_histogram']['7'] == 8
    assert obs.summaries['64']['same_macro_both_high'] == 4
    assert obs.summaries['64']['different_macros_both_high'] == 0
    assert all(s['v_charged'] < 0 and not s['fired'] for s in events)


def test_distribution_denominators_and_zero_handling():
    from IMC_ResNet.scripts.report_mr_distribution import distribution
    d = distribution({'0': 50, '1': 25, '3': 25})
    assert d['total'] == 100 and d['nonzero'] == 50
    assert d['mean_bits_including_zero'] == 1
    assert d['mean_bits_nonzero'] == 2
    assert d['rows'][0]['percent_nonzero'] is None
    assert d['rows'][2]['count'] == 0
    assert sum(r['percent_all'] for r in d['rows']) == 100
    assert sum(r['percent_nonzero'] or 0 for r in d['rows']) == 100


def test_report_is_safe_before_first_high_event():
    obs = MRObserver('conv', 'sink')
    assert obs.report()['threshold_conditions']['32']['age_min'] is None
    assert obs.summaries['32']['age_min'] == 10**9


def test_explicit_network_observers_do_not_change_inference_or_mr():
    from SNN_ResNet10.models.snn_resnet10 import SNNResNet10
    from IMC_ResNet.models.finetuned_snn import fuse_explicit_hidden, make_explicit_cluster, SingleStepExplicit
    from IMC_ResNet.models.imc_resnet import reset_cluster_model
    from IMC_ResNet.scripts.analyze_mr_conditions import attach
    from IMC_ResNet.scripts.report_mr_conditions import evidence_checks
    torch.set_num_threads(2)
    torch.manual_seed(72)
    fused = fuse_explicit_hidden(SNNResNet10(base_channels=2, v_threshold=.3).eval())
    plain, _ = make_explicit_cluster(fused)
    watched, mapping = make_explicit_cluster(fused)
    observers = attach(watched, mapping)
    for obs in observers.values():
        obs.begin_batch([0, 1])
    x = torch.randn(2, 1, 8, 8)
    with torch.no_grad():
        for t in range(1, 5):
            for obs in observers.values():
                obs.step = t
            torch.testing.assert_close(SingleStepExplicit(plain)(x), SingleStepExplicit(watched)(x), atol=0, rtol=0)
            for name in observers:
                torch.testing.assert_close(plain.get_submodule(name).bank.mr, watched.get_submodule(name).bank.mr, atol=0, rtol=0)
    for name, obs in observers.items():
        obs.finish_batch(watched.get_submodule(name))
    reset_cluster_model(plain)
    reset_cluster_model(watched)
    stats = {name: watched.get_submodule(name).bank.stats.report() for name in observers}
    assert stats == {name: plain.get_submodule(name).bank.stats.report() for name in observers}
    evidence_checks({'conditions_by_layer': {name: obs.report() for name, obs in observers.items()}, 'mr_by_layer': stats})
