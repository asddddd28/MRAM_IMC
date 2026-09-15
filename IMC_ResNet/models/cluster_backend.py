"""Batched 16-macro x 36-input x int5 Cluster arithmetic (CPU reference).

Unlike the old event counter, outputs are reconstructed from SCU/MR state.
SCU=4 bits is intentional: the existing SimpleCluster reduction uses 16*MR.
TDP is evaluated by its exact pulse integral; no analog/timing claim is made.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

BETA = (1, 2, 4, 8, -16)


def bit_histogram(value_hist):
    """Zero occupies 0 significant bits; an implemented register needs >=1 bit."""
    result = {}
    for value, count in enumerate(value_hist):
        if count:
            bits = str(value.bit_length())
            result[bits] = result.get(bits, 0) + int(count)
    return result


def mean_bits(hist):
    return sum(int(k) * v for k, v in hist.items()) / max(1, sum(hist.values()))


class MRStats:
    """Observe EVERY fold before overflow handling and before firing reset.

    Histograms include all 80 counters, including zero/padded lanes. Peak maps
    are [output_channel, macro, weight_bit]; spatial/sample axes are reduced.
    A logical MR is one (sample, output_channel, spatial_position, macro, bit).
    """
    def __init__(self, mr_bits=8):
        if mr_bits < 1 or mr_bits > 30:
            raise ValueError('mr_bits must be in [1,30]')
        self.limit = (1 << mr_bits) - 1
        self.mr_bits = mr_bits
        self.hist = np.zeros(1, dtype=np.int64)
        self.peak_hist = np.zeros(1, dtype=np.int64)
        self.peak_map = None
        self.over_limit_observations = 0
        self.fold_observations = 0

    @staticmethod
    def merge(a, b):
        if len(a) < len(b):
            a = np.pad(a, (0, len(b) - len(a)))
        a[:len(b)] += b
        return a

    def observe(self, mr):
        # numpy bincount is fast for CPU integer tensors; no approximate sampling.
        arr = mr.numpy()
        h = np.bincount(arr.reshape(-1))
        self.hist = self.merge(self.hist, h)
        self.over_limit_observations += int(h[self.limit + 1:].sum())
        self.fold_observations += 1

    def finish_batch(self, peak):
        if peak is None:
            return
        arr = peak.numpy()
        self.peak_hist = self.merge(self.peak_hist, np.bincount(arr.reshape(-1)))
        channel_peak = arr.max(axis=(0, 2))  # [B,O,P,16,5] -> [O,16,5]
        self.peak_map = (channel_peak.copy() if self.peak_map is None
                         else np.maximum(self.peak_map, channel_peak))

    def report(self):
        bits = bit_histogram(self.hist)
        peak_bits = bit_histogram(self.peak_hist)
        maximum = len(self.hist) - 1
        return {
            'configured_mr_bits': self.mr_bits,
            'max_mr': maximum,
            'mr_ge48': int(self.hist[48:].sum()),
            'mr_ge64': int(self.hist[64:].sum()),
            'max_required_bits': maximum.bit_length(),
            'mean_observed_bits': mean_bits(bits),
            'mean_logical_peak_bits': mean_bits(peak_bits),
            'observed_bits_histogram': bits,
            'logical_peak_bits_histogram': peak_bits,
            'over_limit_fold_observations': self.over_limit_observations,
            'logical_mrs_over_limit': int(self.peak_hist[self.limit + 1:].sum()),
            'fold_observations': self.fold_observations,
            'peak_by_channel_macro_bit': None if self.peak_map is None else self.peak_map.tolist(),
        }


class ClusterAccumulator:
    """Vectorized SimpleCluster SCU/MR core, with explicit commit/reset.

    Counts have shape [batch, output_channel, position, 16, 5]. All input
    folds update the SAME bank before one downstream neuron decision.
    int32 storage is guarded against overflow even in wide_reference mode.
    """
    def __init__(self, mr_bits=8, overflow='wide_reference'):
        if overflow not in ('wide_reference', 'error', 'wrap', 'saturate'):
            raise ValueError('invalid overflow policy')
        self.stats = MRStats(mr_bits)
        self.overflow = overflow
        self.scu = self.mr = self.peak = None

    def add_counts(self, counts):
        if counts.dtype != torch.int32 or counts.shape[-2:] != (16, 5):
            raise ValueError('expected int32 counts [...,16,5]')
        if counts.min() < 0 or counts.max() > 36:
            raise ValueError('each macro PMAC count must be in [0,36]')
        if self.scu is None:
            self.scu = torch.zeros_like(counts)
            self.mr = torch.zeros_like(counts)
            self.peak = torch.zeros_like(counts)
        if counts.shape != self.scu.shape:
            raise ValueError('reset bank before changing batch/shape')
        if int(self.mr.max()) > 2**31 - 4:
            raise OverflowError('int32 reference storage exhausted')
        z = self.scu + counts
        candidate = self.mr + torch.div(z, 16, rounding_mode='floor')
        self.stats.observe(candidate)
        self.peak = torch.maximum(self.peak, candidate)
        if self.overflow == 'error' and int(candidate.max()) > self.stats.limit:
            raise OverflowError('mr_overflow')
        self.scu = z.remainder(16)
        if self.overflow == 'wrap':
            candidate = candidate.remainder(self.stats.limit + 1)
        elif self.overflow == 'saturate':
            candidate = candidate.clamp_max(self.stats.limit)
        self.mr = candidate

    def reduce(self):
        # Equivalent to summing the signed TDP pulse integrals plus SG carry.
        beta = torch.tensor(BETA, dtype=torch.int64)
        return ((self.scu.to(torch.int64) + 16 * self.mr.to(torch.int64)) * beta).sum((-1, -2))

    def fire(self, mask):
        if self.scu is not None:
            self.scu.masked_fill_(mask[..., None, None], 0)
            self.mr.masked_fill_(mask[..., None, None], 0)

    def reset(self):
        self.stats.finish_batch(self.peak)
        self.scu = self.mr = self.peak = None


def quantize_int5(weight):
    """Per-output-channel symmetric PTQ; -16 is supported, range uses -15..15."""
    scale = weight.detach().flatten(1).abs().amax(1).clamp_min(1e-12) / 15
    q = (weight.detach() / scale[:, None, None, None]).round().clamp(-16, 15).to(torch.int32)
    return q, scale


class ClusterConv2d(nn.Module):
    """Actual bit-plane PMAC, SCU carry and temporal MR accumulation.

    Only accepts binary spikes multiplied by a known VoltageScaler constant.
    The continuous stem and final classifier must remain digital boundaries.
    Bias is added digitally EACH step. The reconstructed cumulative MAC is
    differenced to feed the digital IF accumulator without double integration.
    Its hard-reset spike clears all MR/SCU lanes of this output neuron.
    """
    def __init__(self, conv, input_scale, mr_bits=8, overflow='wide_reference'):
        super().__init__()
        if conv.groups != 1 or conv.dilation != (1, 1) or conv.padding_mode != 'zeros':
            raise ValueError('only ordinary zero-padded Conv2d supported')
        if not math.isfinite(input_scale) or input_scale <= 0:
            raise ValueError('input scale must be positive')
        self.input_scale = input_scale
        self.kernel_size, self.stride, self.padding = conv.kernel_size, conv.stride, conv.padding
        self.in_channels, self.out_channels = conv.in_channels, conv.out_channels
        q, scale = quantize_int5(conv.weight)
        self.register_buffer('qweight', q)
        self.register_buffer('weight_scale', scale)
        self.register_buffer('bias', torch.zeros(conv.out_channels) if conv.bias is None else conv.bias.detach().clone())
        k = q[0].numel()
        self.folds = math.ceil(k / 576)
        self.fan_in = k
        # [F,M,36,O*5] for batched macro-wise matrix multiplication.
        flat = F.pad(q.flatten(1), (0, self.folds * 576 - k))
        planes = ((flat[..., None] & 31) >> torch.arange(5)) & 1
        packed = planes.reshape(conv.out_channels, self.folds, 16, 36, 5)
        self.register_buffer('bit_weights', packed.permute(1, 2, 3, 0, 4).reshape(self.folds, 16, 36, -1).float().contiguous())
        self.bank = ClusterAccumulator(mr_bits, overflow)
        self.previous_total = None
        self.calls = 0
        self.observer = None  # Optional read-only diagnostic; disabled by default.

    def forward(self, x):
        # Validate the mapping instead of silently thresholding continuous inputs.
        raw = x / self.input_scale
        binary = raw.round()
        if not torch.allclose(raw, binary, atol=1e-5, rtol=0) or binary.min() < 0 or binary.max() > 1:
            raise ValueError('Cluster input is not scaled binary spikes')
        patches = F.unfold(binary, self.kernel_size, padding=self.padding, stride=self.stride)
        b, _, p = patches.shape
        patches = F.pad(patches, (0, 0, 0, self.folds * 576 - self.fan_in))
        patches = patches.reshape(b, self.folds, 16, 36, p)
        for fold in range(self.folds):
            inputs = patches[:, fold].permute(1, 0, 3, 2).reshape(16, b * p, 36)
            # float32 represents all 0..36 integer popcounts exactly.
            counts = torch.bmm(inputs, self.bit_weights[fold])
            counts = counts.reshape(16, b, p, self.out_channels, 5).permute(1, 3, 2, 0, 4).contiguous().to(torch.int32)
            self.bank.add_counts(counts)
            if self.observer is not None:
                self.observer.on_fold(self, counts, fold, inputs)
        total = self.bank.reduce()
        increment = total if self.previous_total is None else total - self.previous_total
        self.previous_total = total
        h = (x.shape[2] + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        w = (x.shape[3] + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        self.calls += 1
        return (increment.float() * (self.weight_scale * self.input_scale)[None, :, None]
                + self.bias[None, :, None]).reshape(b, self.out_channels, h, w)

    def fire(self, spike):
        mask = spike.flatten(2).bool()
        self.bank.fire(mask)
        if self.previous_total is not None:
            self.previous_total.masked_fill_(mask, 0)

    def reset(self):
        self.bank.reset()
        self.previous_total = None
