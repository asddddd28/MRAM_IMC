"""Arithmetic fidelity, overflow/reset behavior, and FX network wiring tests."""
import sys
from pathlib import Path
import copy
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from spikingjelly.activation_based import ann2snn, functional, neuron

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model_imc_cluster_simple import SimpleCluster, SimpleConfig, eigen_train
from model_imc_cluster_simple.model import encode_weights
from IMC_ResNet.models.cluster_backend import ClusterAccumulator, ClusterConv2d
from IMC_ResNet.models.imc_resnet import make_cluster_model, make_quantized_digital_reference, reset_cluster_model

torch.set_num_threads(2)


def test_scalar_equivalence_multifold_temporal_and_tdp():
    rng = np.random.default_rng(17)
    bank = ClusterAccumulator()
    scalar = SimpleCluster(SimpleConfig(theta=87, overflow_policy='wide_reference'))
    for _ in range(16):
        tiles = []
        for fold in range(3):
            x = rng.integers(0, 2, (16, 36))
            w = rng.integers(-16, 16, (16, 36))
            tiles.append((x, w))
            counts = (x[..., None] * encode_weights(w)).sum(1)
            bank.add_counts(torch.tensor(counts, dtype=torch.int32).reshape(1, 1, 1, 16, 5))
        ref = scalar.step_tiles(tiles)
        np.testing.assert_array_equal(bank.scu.reshape(16, 5).numpy(), np.asarray(ref.candidate.scu, dtype=np.int32))
        np.testing.assert_array_equal(bank.mr.reshape(16, 5).numpy(), np.asarray(ref.candidate.mr, dtype=np.int32))
        assert bank.reduce().item() == ref.trace['total_u']
        for count in bank.scu.flatten().tolist():
            assert eigen_train(count).pulse_count == count
        bank.fire(bank.reduce() >= 87)
        np.testing.assert_array_equal(bank.mr.reshape(16, 5).numpy(), np.asarray(ref.state.mr, dtype=np.int32))
    bank.reset()
    assert bank.stats.report()['fold_observations'] == 48


@pytest.mark.parametrize('policy,expected', [('wide_reference', 256), ('wrap', 0), ('saturate', 255)])
def test_overflow_observed_before_reset(policy, expected):
    bank = ClusterAccumulator(8, policy)
    counts = torch.full((1, 1, 1, 16, 5), 16, dtype=torch.int32)
    for _ in range(256):
        bank.add_counts(counts)
    assert bank.mr.max().item() == expected
    bank.fire(torch.ones((1, 1, 1), dtype=torch.bool))
    assert bank.mr.max() == 0
    bank.reset()
    report = bank.stats.report()
    assert report['max_mr'] == 256
    assert report['max_required_bits'] == 9
    assert report['mean_logical_peak_bits'] == 9
    assert report['over_limit_fold_observations'] == 80
    assert report['logical_mrs_over_limit'] == 80


def test_error_overflow():
    bank = ClusterAccumulator(1, 'error')
    with pytest.raises(OverflowError, match='mr_overflow'):
        bank.add_counts(torch.full((1, 1, 1, 16, 5), 36, dtype=torch.int32))


@pytest.mark.parametrize('in_channels,kernel,pad', [(3, 3, 1), (65, 3, 1), (7, 1, 0)])
def test_mapped_conv_equals_int5_conv_with_selective_reset(in_channels, kernel, pad):
    torch.manual_seed(3)
    conv = nn.Conv2d(in_channels, 2, kernel, padding=pad, stride=2)
    mapped = ClusterConv2d(conv, 1.7)
    for _ in range(4):
        spikes = torch.randint(0, 2, (2, in_channels, 5, 5)).float()
        actual = mapped(spikes * 1.7)
        weight = mapped.qweight.float() * mapped.weight_scale[:, None, None, None]
        expected = F.conv2d(spikes * 1.7, weight, mapped.bias, stride=2, padding=pad)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        mapped.fire((actual > 0).float())
    mapped.reset()
    assert mapped.bank.stats.report()['fold_observations'] == 4 * mapped.folds
    with pytest.raises(ValueError, match='not scaled binary'):
        mapped(torch.full((1, in_channels, 5, 5), .4))


class TinyResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 2, 3, padding=1), nn.ReLU())
        self.conv1 = nn.Conv2d(2, 2, 3, padding=1)
        self.r1 = nn.ReLU()
        self.conv2 = nn.Conv2d(2, 2, 3, padding=1)
        self.skip = nn.Conv2d(2, 2, 1)
        self.r2 = nn.ReLU()
    def forward(self, x):
        x = self.stem(x)
        return self.r2(self.conv2(self.r1(self.conv1(x))) + self.skip(x))


def test_fx_residual_reset_routing_and_network_equivalence():
    torch.manual_seed(123)
    inputs = torch.rand(4, 1, 4, 4)
    loader = DataLoader(TensorDataset(inputs, torch.zeros(4)), batch_size=2)
    snn = ann2snn.Converter(loader, device='cpu', mode='max', fuse_flag=True)(TinyResidual().eval()).eval()
    actual, mapping = make_cluster_model(snn)
    reference = make_quantized_digital_reference(snn)
    assert len(mapping) == 3
    by_name = {x['layer']: x['reset_if'] for x in mapping}
    assert by_name['conv2'] == by_name['skip'] != by_name['conv1']
    with torch.no_grad():
        for _ in range(2):
            functional.reset_net(reference)
            reset_cluster_model(actual)
            for _ in range(16):
                torch.testing.assert_close(actual(inputs), reference(inputs), atol=1e-5, rtol=1e-5)
    reset_cluster_model(actual)



def test_zero_mr_statistics_and_batch_finalization():
    bank = ClusterAccumulator()
    zeros = torch.zeros((2, 3, 4, 16, 5), dtype=torch.int32)
    bank.add_counts(zeros)
    bank.reset()
    bank.reset()  # No double counting when reset is repeated between batches.
    report = bank.stats.report()
    assert report['max_mr'] == report['max_required_bits'] == 0
    assert report['mean_observed_bits'] == report['mean_logical_peak_bits'] == 0
    assert report['logical_peak_bits_histogram'] == {'0': 2 * 3 * 4 * 16 * 5}
    assert np.shape(report['peak_by_channel_macro_bit']) == (3, 16, 5)


def test_result_summary_rejects_legacy_and_incomplete():
    from IMC_ResNet.scripts.summarize_results import summary
    with pytest.raises(ValueError, match='completed'):
        summary({'accuracy': .9185, 'cluster_model': 'event accounting'})
    with pytest.raises(ValueError, match='completed'):
        summary({'schema': 'actual_cluster_v1', 'status': 'running_controls'})
