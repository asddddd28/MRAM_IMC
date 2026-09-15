"""Explicit fine-tuned topology: BN fusion, quantization and reset routing."""
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from spikingjelly.activation_based import functional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from SNN_ResNet10.models.snn_resnet10 import SNNResNet10
from IMC_ResNet.models.cluster_backend import ClusterConv2d
from IMC_ResNet.models.imc_resnet import reset_cluster_model
from IMC_ResNet.models.finetuned_snn import (
    fuse_explicit_hidden, make_explicit_quantized_reference, make_explicit_cluster,
)

torch.set_num_threads(2)


def small_model():
    torch.manual_seed(53)
    model = SNNResNet10(base_channels=2, v_threshold=.7).eval()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            with torch.no_grad():
                module.running_mean.uniform_(-.3, .3)
                module.running_var.uniform_(.5, 1.5)
                module.weight.uniform_(.7, 1.3)
                module.bias.uniform_(-.2, .2)
    return model


def test_explicit_bn_fusion_preserves_float_sequence():
    original = small_model()
    fused = fuse_explicit_hidden(original)
    assert isinstance(original.layer1.bn1, nn.BatchNorm2d)
    assert isinstance(fused.layer1.bn1, nn.Identity)
    images = torch.randn(8, 2, 1, 8, 8)
    with torch.no_grad():
        torch.testing.assert_close(original(images), fused(images), atol=1e-5, rtol=1e-5)
    assert all(m.v_reset == 0 for m in fused.modules() if hasattr(m, 'v_reset'))


def test_explicit_cluster_matches_quantized_reference_and_resets():
    fused = fuse_explicit_hidden(small_model())
    reference = make_explicit_quantized_reference(fused)
    actual, mapping = make_explicit_cluster(fused, mr_bits=8, overflow='error')
    assert len(mapping) == 11
    links = {entry['layer']: entry['reset_if'] for entry in mapping}
    assert links['layer2.conv2'] == links['layer2.downsample.0'] == 'layer2.lif_out'
    assert links['layer2.conv1'] == 'layer2.lif1'
    assert not isinstance(actual.stem[0], ClusterConv2d)
    images = torch.randn(8, 2, 1, 8, 8)
    with torch.no_grad():
        for _ in range(2):
            functional.reset_net(reference)
            reset_cluster_model(actual)
            torch.testing.assert_close(actual(images), reference(images), atol=1e-5, rtol=1e-5)
    reset_cluster_model(actual)
    for module in actual.modules():
        if isinstance(module, ClusterConv2d):
            assert module.bank.mr is None
            assert module.previous_total is None
            assert module.bank.stats.report()['logical_peak_bits_histogram']
    with torch.no_grad():
        actual(images[:1])
        # Triggering one output IF resets both corresponding residual branches.
        actual.layer2.lif_out(torch.full((2, 4, 4, 4), 1e6))
    assert actual.layer2.conv2.bank.mr.count_nonzero() == 0
    assert actual.layer2.downsample[0].bank.mr.count_nonzero() == 0
    reset_cluster_model(actual)


def test_unfused_explicit_mapping_is_rejected():
    with pytest.raises(ValueError, match='Fold BatchNorm'):
        make_explicit_cluster(small_model())


def test_legacy_finetuned_checkpoint_restores_all_thresholds(tmp_path):
    import json
    from IMC_ResNet.models.finetuned_snn import load_finetuned
    from spikingjelly.activation_based import neuron

    source = SNNResNet10().eval()
    checkpoint = tmp_path / 'weights.pt'
    calibration = tmp_path / 'thresholds.json'
    torch.save({'model': source.state_dict(), 'steps': 16, 'hard_reset': True}, checkpoint)
    thresholds = [float(i + 1) / 2 for i in range(9)]
    calibration.write_text(json.dumps({'snn_if_thresholds': thresholds}), encoding='utf8')
    loaded, metadata = load_finetuned(checkpoint, calibration)
    assert [m.v_threshold for m in loaded.modules() if isinstance(m, neuron.IFNode)] == thresholds
    assert all(m.v_reset == 0 for m in loaded.modules() if isinstance(m, neuron.IFNode))
    assert metadata['training_steps'] == 16
    for key, value in source.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
    calibration.write_text(json.dumps({'snn_if_thresholds': thresholds[:-1]}), encoding='utf8')
    with pytest.raises(ValueError, match='exactly one threshold'):
        load_finetuned(checkpoint, calibration)


def test_summary_cli_handles_finetuned_result_without_ann(tmp_path, monkeypatch):
    import json
    from IMC_ResNet.scripts import summarize_results

    path = tmp_path / 'result.json'
    stats = {
        'max_mr': 40, 'max_required_bits': 6, 'mean_observed_bits': 3.,
        'mean_logical_peak_bits': 6., 'over_limit_fold_observations': 0,
        'logical_mrs_over_limit': 0, 'observed_bits_histogram': {'0': 1, '6': 1},
        'logical_peak_bits_histogram': {'6': 1},
    }
    path.write_text(json.dumps({
        'status': 'complete', 'schema': 'actual_cluster_v1',
        'samples': 1, 'steps': [64], 'model_source': 'fine-tuned explicit SNNResNet10',
        'controls': {'int5_hard_reset_digital': {'accuracy_by_steps': {'64': 1.}}},
        'cluster': {'accuracy_by_steps': {'64': 1.}, 'seconds': 1.},
        'mr_by_layer': {'layer1.conv1': stats},
        'prediction_disagreements_vs_int5_digital': {'64': 0},
    }), encoding='utf8')
    monkeypatch.setattr(sys, 'argv', ['summarize_results.py', str(path)])
    summarize_results.main()
    markdown = path.with_suffix('.md').read_text(encoding='utf8')
    assert 'fine-tuned explicit SNNResNet10' in markdown
    assert '100.00%' in markdown
    assert 'ANN ACC' not in markdown
    assert json.loads(path.with_name('result_summary.json').read_text())['max_mr'] == 40
