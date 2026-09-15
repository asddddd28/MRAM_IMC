"""Map the fine-tuned explicit SNN to the existing real SCU/MR backend.

No ANN2SNN conversion or new threshold calibration is performed here. The
trained IF thresholds and BatchNorm running statistics are retained. Hidden
Conv-BN pairs are folded in eval mode before int5 quantization. The continuous
stem and final readout remain digital, as in the original Cluster runner.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import fuse_conv_bn_eval
from spikingjelly.activation_based import neuron

from SNN_ResNet10.models.snn_resnet10 import SNNResNet10
from SNN_ResNet10.models.mr_qat import LeakageIFNode
from .cluster_backend import ClusterConv2d, quantize_int5


def load_finetuned(checkpoint_path, threshold_path):
    """Restore legacy state_dict AND the non-serialized IF threshold attributes."""
    checkpoint_path, threshold_path = Path(checkpoint_path), Path(threshold_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint.get('hard_reset') is not True:
        raise ValueError('Expected a hard-reset fine-tuned SNN checkpoint')
    model = SNNResNet10().eval()
    leakage = float(checkpoint.get('leakage', 1.0))
    for name, node in list(model.named_modules()):
        if isinstance(node, neuron.IFNode):
            model.set_submodule(name, LeakageIFNode(v_reset=0., leakage=leakage))
    if checkpoint.get('hidden_bn_fused'):
        model = fuse_explicit_hidden(model)
    model.load_state_dict(checkpoint['model'], strict=True)
    # Legacy weight-QAT saves latent weights, including the digital stem.
    # Materialize its actual forward weights before any BN folding.
    if checkpoint.get('quantization') == 'per-output-channel symmetric int5 STE':
        with torch.no_grad():
            for conv in model.modules():
                if isinstance(conv, nn.Conv2d):
                    scale = conv.weight.flatten(1).abs().amax(1).clamp_min(1e-8) / 15
                    scale = scale[:, None, None, None]
                    conv.weight.copy_((conv.weight / scale).round().clamp(-15, 15) * scale)
    nodes = [(name, node) for name, node in model.named_modules() if isinstance(node, neuron.IFNode)]
    thresholds = checkpoint.get('if_thresholds')
    if thresholds is None:
        thresholds = json.loads(threshold_path.read_text(encoding='utf8'))['snn_if_thresholds']
    if len(thresholds) != len(nodes):
        raise ValueError('Need exactly one threshold per IF node')
    for (_, node), threshold in zip(nodes, thresholds):
        if not math.isfinite(float(threshold)):
            raise ValueError('Thresholds must be finite')
        node.v_threshold = max(float(threshold), 1e-3)  # Match original training.
        node.v_reset = 0.0
    metadata = {
        'checkpoint': str(checkpoint_path.resolve()),
        'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        'training_steps': checkpoint.get('steps'),
        'leakage': leakage,
        'hidden_bn_fused': bool(checkpoint.get('hidden_bn_fused')),
        'threshold_source': 'checkpoint' if 'if_thresholds' in checkpoint else str(threshold_path.resolve()),
        'threshold_sha256': None if 'if_thresholds' in checkpoint else hashlib.sha256(threshold_path.read_bytes()).hexdigest(),
        'if_thresholds': {name: node.v_threshold for name, node in nodes},
    }
    return model.eval(), metadata


def explicit_targets(model):
    """Each residual conv2 and projection share the block's output IF reset."""
    if not isinstance(model, SNNResNet10):
        raise TypeError('Only the explicit SNNResNet10 topology is supported')
    targets = []
    for stage in range(1, 5):
        prefix = f'layer{stage}'
        block = model.get_submodule(prefix)
        targets.extend([(f'{prefix}.conv1', f'{prefix}.lif1', f'{prefix}.bn1'),
                        (f'{prefix}.conv2', f'{prefix}.lif_out', f'{prefix}.bn2')])
        if isinstance(block.downsample, nn.Sequential):
            targets.append((f'{prefix}.downsample.0', f'{prefix}.lif_out', f'{prefix}.downsample.1'))
    return targets


def fuse_explicit_hidden(model):
    """Return a fresh eval model with hidden Conv-BN folded, not recalibrated."""
    fused = copy.deepcopy(model).eval()
    for module in fused.modules():
        if isinstance(module, neuron.IFNode):
            if module.v_reset != 0.0:
                raise ValueError('Cluster mapping requires hard-reset IF neurons')
            module.reset()
    for conv_name, _, bn_name in explicit_targets(fused):
        conv, bn = fused.get_submodule(conv_name), fused.get_submodule(bn_name)
        if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.Identity):
            continue
        if not isinstance(conv, nn.Conv2d) or not isinstance(bn, nn.BatchNorm2d):
            raise ValueError('Expected an unfused Conv2d/BatchNorm2d pair')
        fused.set_submodule(conv_name, fuse_conv_bn_eval(conv, bn))
        fused.set_submodule(bn_name, nn.Identity())
    return fused


def make_explicit_quantized_reference(fused):
    """Ordinary Conv2d using exactly the same dequantized int5 weights."""
    model = copy.deepcopy(fused).eval()
    for name, _, bn_name in explicit_targets(model):
        if not isinstance(model.get_submodule(bn_name), nn.Identity):
            raise ValueError('Fold BatchNorm before quantization')
        conv = model.get_submodule(name)
        q, scale = quantize_int5(conv.weight)
        with torch.no_grad():
            conv.weight.copy_(q.float() * scale[:, None, None, None])
    return model


def make_explicit_cluster(fused, mr_bits=8, overflow='wide_reference'):
    """Replace 11 hidden convolutions; post-IF inputs are unscaled binary spikes."""
    model = copy.deepcopy(fused).eval()
    mapping = []
    for name, sink, bn_name in explicit_targets(model):
        if not isinstance(model.get_submodule(bn_name), nn.Identity):
            raise ValueError('Fold BatchNorm before Cluster mapping')
        conv = model.get_submodule(name)
        replacement = ClusterConv2d(conv, input_scale=1.0, mr_bits=mr_bits, overflow=overflow)
        model.set_submodule(name, replacement)
        def clear_on_spike(_module, _inputs, output, bank=replacement):
            bank.fire(output)
        model.get_submodule(sink).register_forward_hook(clear_on_spike)
        mapping.append({'layer': name, 'reset_if': sink, 'input_scale': 1.0,
                        'folded_bn': bn_name, 'fan_in': replacement.fan_in,
                        'folds': replacement.folds, 'out_channels': conv.out_channels})
    return model, mapping


class SingleStepExplicit(nn.Module):
    """Adapt the explicit [T,B,C,H,W] API to the existing time-first evaluator."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        return self.model(images.unsqueeze(0))[0]
