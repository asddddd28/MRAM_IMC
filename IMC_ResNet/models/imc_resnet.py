"""Wire real Cluster convolutions into the official ANN2SNN FX graph."""
import copy
from torch import nn
from spikingjelly.activation_based import neuron
from spikingjelly.activation_based.ann2snn.modules import VoltageScaler
from .cluster_backend import ClusterConv2d, quantize_int5


def set_hard_reset(model):
    for module in model.modules():
        if isinstance(module, neuron.IFNode):
            module.v_reset = 0.0
    return model


def cluster_targets(model):
    """Reject unsupported input encodings; skip only the continuous stem."""
    targets = []
    for node in model.graph.nodes:
        if node.op != 'call_module':
            continue
        conv = model.get_submodule(node.target)
        if not isinstance(conv, nn.Conv2d):
            continue
        parent = node.args[0]
        if parent.op == 'placeholder':
            continue
        source = model.get_submodule(parent.target) if parent.op == 'call_module' else None
        if not isinstance(source, VoltageScaler):
            raise ValueError(f'{node.target}: expected post-IF VoltageScaler')
        before = parent.args[0]
        if before.op != 'call_module' or not isinstance(model.get_submodule(before.target), neuron.IFNode):
            raise ValueError(f'{node.target}: source does not encode binary spikes')
        # Follow only to the FIRST IF decision. Residual conv2/shortcut share it.
        pending, seen, sinks = list(node.users), set(), set()
        while pending:
            next_node = pending.pop()
            if next_node in seen:
                continue
            seen.add(next_node)
            mod = model.get_submodule(next_node.target) if next_node.op == 'call_module' else None
            if isinstance(mod, neuron.IFNode):
                sinks.add(next_node.target)
            else:
                pending.extend(next_node.users)
        if len(sinks) != 1:
            raise ValueError(f'{node.target}: expected one hard-reset IF consumer, got {sinks}')
        targets.append((node.target, float(source.scale), next(iter(sinks))))
    return targets


def make_cluster_model(snn, mr_bits=8, overflow='wide_reference'):
    model = set_hard_reset(copy.deepcopy(snn))
    mapping = []
    for name, scale, reset_name in cluster_targets(model):
        conv = model.get_submodule(name)
        replacement = ClusterConv2d(conv, scale, mr_bits, overflow)
        model.set_submodule(name, replacement)
        # Default argument binds the correct bank, not the last loop iteration.
        def reset_on_spike(_module, _inputs, output, bank=replacement):
            bank.fire(output)
        model.get_submodule(reset_name).register_forward_hook(reset_on_spike)
        mapping.append({'layer': name, 'reset_if': reset_name, 'input_scale': scale,
                        'fan_in': replacement.fan_in, 'folds': replacement.folds,
                        'out_channels': conv.out_channels})
    return model.eval(), mapping


def make_quantized_digital_reference(snn):
    """Same int5 weights/reset, but ordinary Conv2d: isolate mapping errors."""
    model = set_hard_reset(copy.deepcopy(snn))
    for name, _, _ in cluster_targets(model):
        conv = model.get_submodule(name)
        q, scale = quantize_int5(conv.weight)
        conv.weight.data.copy_(q.float() * scale[:, None, None, None])
    return model.eval()


def reset_cluster_model(model):
    # Explicit handling avoids third-party reset warnings for our nn.Modules.
    for module in model.modules():
        if isinstance(module, (neuron.IFNode, ClusterConv2d)):
            module.reset()
