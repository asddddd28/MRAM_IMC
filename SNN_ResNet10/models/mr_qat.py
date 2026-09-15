"""Exact-forward sampled MR observer with a differentiable bit-plane surrogate.

The physical bank is 16 macros x 5 two's-complement planes, 36 inputs per
macro, SCU radix 16. Temporal counts do NOT leak: only the digital IF leaks.
Sampling selects spatial positions once per sequence, never individual times.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from spikingjelly.activation_based import neuron


class LeakageIFNode(neuron.IFNode):
    def __init__(self, *args, leakage=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 < leakage <= 1:
            raise ValueError('leakage must be in (0, 1]')
        self.leakage = float(leakage)

    def neuronal_charge(self, x):
        self.v = self.leakage * self.v + x

    def single_step_forward(self, x):
        # IFNode's eval JIT bypasses neuronal_charge and silently drops leakage.
        return neuron.BaseNode.single_step_forward(self, x)


def quantized_weight(weight):
    scale = weight.detach().flatten(1).abs().amax(1).clamp_min(1e-12) / 15
    normalized = weight / scale[:, None, None, None]
    hard = normalized.round().clamp(-16, 15)
    q = normalized + (hard - normalized).detach()
    return q, scale


class QATConv2d(nn.Conv2d):
    def forward(self, x):
        q, scale = quantized_weight(self.weight)
        return F.conv2d(x, q * scale[:, None, None, None], self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


def bit_planes(weight, temperature=0.5):
    """Hard int5 bits forward; Gaussian categorical relaxation backward.

    This is a surrogate gradient, not a derivative of integer bit extraction.
    It handles sign transitions using the same two's-complement encoding as
    ClusterConv2d, including nonmonotonic low-bit changes.
    """
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    q, scale = quantized_weight(weight)
    normalized = weight / scale[:, None, None, None]
    bins = torch.arange(-16, 16, device=weight.device)
    shifts = torch.arange(5, device=weight.device)
    table = (((bins.long() & 31)[:, None] >> shifts) & 1).to(weight.dtype)
    probs = (-(normalized[..., None] - bins).square() / temperature).softmax(-1)
    soft = probs @ table
    hard = (((q.detach().long() & 31)[..., None] >> shifts) & 1).to(weight.dtype)
    return hard + (soft - soft.detach())


class MRProxy:
    """Read-only forward hook; loss gradients affect weights and spike history.

    Store total unsigned counts so floor(total/16) exactly gives MR and
    remainder(total,16) gives SCU. This avoids cancelling temporal gradients
    through a surrogate modulo. Losses observe each fold before IF reset.
    """
    def __init__(self, conv, target=56, positions=32, temperature=0.5, shared_bits=None, independent_cancel=False):
        if conv.groups != 1 or conv.dilation != (1, 1) or conv.padding_mode != 'zeros':
            raise ValueError('ordinary zero-padded convolutions required')
        if target <= 0 or positions < 0 or temperature <= 0:
            raise ValueError('invalid proxy configuration')
        self.conv = conv
        self.target, self.positions, self.temperature = target, positions, temperature
        self.independent_cancel = independent_cancel
        if independent_cancel and shared_bits is not None:
            raise ValueError("independent cancellation and shared MR are mutually exclusive")
        self.shared_bits = shared_bits
        if shared_bits is not None and shared_bits < 6:
            raise ValueError('shared_bits must be >=6 for the 8-unit margin')
        self.reset()

    def reset(self):
        self.total = self.mr = self.indices = self.packed = None
        self.encoded = self.previous_raw = None
        self.outside_5bit = 0
        self.overflow_losses, self.peak_losses, self.quiet_losses = [], [], []
        self.max_mr = self.ge48 = self.ge64 = self.observations = 0
        self.spikes = self.spike_slots = 0
        self.ever_fired = self.age = None
        self.shared_min = self.shared_max = self.shared_outside = 0

    def observe(self, _module, inputs, output):
        x = inputs[0]
        c = self.conv
        patches = F.unfold(x, c.kernel_size, padding=c.padding, stride=c.stride)
        b, k, p = patches.shape
        if self.indices is None:
            self.indices = (torch.arange(p, device=x.device) if not self.positions or self.positions >= p
                            else torch.randperm(p, device=x.device)[:self.positions].sort().values)
        patches = patches.index_select(-1, self.indices)
        p = len(self.indices)
        folds = math.ceil(k / 576)
        patches = F.pad(patches, (0, 0, 0, folds * 576 - k)).reshape(b, folds, 16, 36, p)
        if self.packed is None:
            planes = bit_planes(c.weight, self.temperature).reshape(c.out_channels, k, 5)
            planes = F.pad(planes, (0, 0, 0, folds * 576 - k))
            self.packed = planes.reshape(c.out_channels, folds, 16, 36, 5).permute(1, 2, 3, 0, 4).reshape(folds, 16, 36, -1)
        for fold in range(folds):
            inp = patches[:, fold].permute(1, 0, 3, 2).reshape(16, b * p, 36)
            counts = torch.bmm(inp, self.packed[fold]).reshape(16, b, p, c.out_channels, 5).permute(1, 3, 2, 0, 4)
            self.total = counts if self.total is None else self.total + counts
            continuous = self.total / 16
            self.mr = continuous + (continuous.floor() - continuous).detach()
            if self.independent_cancel:
                self.mr = self.encode_independent(continuous)
            if self.shared_bits is not None:
                beta = self.mr.new_tensor([1, 2, 4, 8, -16])
                shared = (self.mr * beta).sum(-1)
                limit = 1 << (self.shared_bits - 1)
                excess = torch.maximum(shared-(limit-1-8), (-limit+8)-shared)
                tail = excess.amax(-1)
                value = shared.detach()
                self.shared_min = min(self.shared_min, int(value.min()))
                self.shared_max = max(self.shared_max, int(value.max()))
                self.shared_outside += int(((value < -limit) | (value > limit-1)).sum())
            else:
                excess = self.mr - self.target
                # Tail max per neuron prevents rare unsafe lanes being diluted by padding.
                tail = excess.flatten(-2).amax(-1)
            self.overflow_losses.append(F.softplus(tail).mean() / self.target)
            self.peak_losses.append(F.relu(excess.amax()) / self.target)
            if self.age is not None:
                self.quiet_losses.append((F.softplus(tail) * self.age.clamp_max(64) / 64).mean() / self.target)
            detached = self.mr.detach()
            self.max_mr = max(self.max_mr, int(detached.max()))
            self.outside_5bit += int((detached > 31).sum())
            self.ge48 += int((detached >= 48).sum())
            self.ge64 += int((detached >= 64).sum())
            self.observations += detached.numel()

    def encode_independent(self, continuous):
        """Exact integer routing forward; identity STE through raw MR backward.

        Routing choices are discrete and their derivatives are ignored. This
        is a surrogate gradient, not a differentiable hardware simulation.
        """
        from IMC_ResNet.models.mr_pair_cancel import (
            add_carries_with_positive_forward, clear_one_b1_pair, clear_one_b2_pair)
        raw = continuous.detach().floor().to(torch.int32)
        if self.previous_raw is None:
            self.previous_raw = torch.zeros_like(raw)
            self.encoded = torch.zeros_like(raw)
        carry = raw - self.previous_raw
        self.encoded, _, _ = add_carries_with_positive_forward(self.encoded, carry)
        self.encoded, _ = clear_one_b1_pair(self.encoded)
        self.encoded, _ = clear_one_b2_pair(self.encoded)
        self.previous_raw = raw
        return continuous + (self.encoded.to(continuous.dtype) - continuous).detach()

    def fire(self, _module, _inputs, spike):
        selected = spike.flatten(2).index_select(-1, self.indices)
        self.total = self.total * (1 - selected[..., None, None])
        active = selected.detach().bool()
        if self.independent_cancel:
            self.encoded.masked_fill_(active[..., None, None], 0)
            self.previous_raw.masked_fill_(active[..., None, None], 0)
        self.ever_fired = active if self.ever_fired is None else self.ever_fired | active
        self.age = ((torch.zeros_like(selected) if self.age is None else self.age) + 1) * (~active)
        self.spikes += int(active.sum())
        self.spike_slots += active.numel()

    def losses(self):
        zero = self.conv.weight.sum() * 0
        mean = lambda values: torch.stack(values).mean() if values else zero
        return {'overflow': mean(self.overflow_losses), 'peak': mean(self.peak_losses),
                'quiet': mean(self.quiet_losses)}

    def report(self):
        return {'independent_cancel': self.independent_cancel, 'outside_5bit': self.outside_5bit, 'max_mr': self.max_mr, 'mr_ge48': self.ge48, 'mr_ge64': self.ge64,
                'observations': self.observations, 'sampled_positions': len(self.indices),
                'firing_rate': self.spikes / max(1, self.spike_slots),
                'zero_firing_ratio': float((~self.ever_fired).float().mean()),
                'shared_bits': self.shared_bits, 'shared_min': self.shared_min,
                'shared_max': self.shared_max, 'shared_outside': self.shared_outside}
