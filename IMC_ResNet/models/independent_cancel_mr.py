"""Experimental independent plane MR bank with exact local cancellation."""
import torch
from .cluster_backend import BETA, MRStats
from .mr_pair_cancel import add_carries_with_positive_forward, clear_one_b1_pair, clear_one_b2_pair


class IndependentCancelMR:
    def __init__(self):
        self.scu = self.mr = self.shadow = self.peak = None
        self.stats = MRStats(5)
        self.raw_max = self.max_integer_error = 0
        self.plane_max = torch.zeros(5, dtype=torch.int32)

    def add_counts(self, counts):
        if self.scu is None:
            self.scu = torch.zeros_like(counts)
            self.mr = torch.zeros_like(counts)
            self.shadow = torch.zeros_like(counts, dtype=torch.int64)
            self.peak = torch.zeros_like(counts)
        z = self.scu + counts
        carry, self.scu = z // 16, z % 16
        self.shadow += carry
        self.raw_max = max(self.raw_max, int((self.mr + carry).max()))
        self.mr, _, _ = add_carries_with_positive_forward(self.mr, carry)
        self.mr, _ = clear_one_b1_pair(self.mr)
        self.mr, _ = clear_one_b2_pair(self.mr)
        beta = self.shadow.new_tensor(BETA)
        error = int(((self.mr.long() - self.shadow) * beta).sum(-1).abs().max())
        self.max_integer_error = max(self.max_integer_error, error)
        assert error == 0
        self.stats.observe(self.mr)
        self.peak = torch.maximum(self.peak, self.mr)
        self.plane_max = torch.maximum(self.plane_max, self.mr.reshape(-1,5).amax(0))

    def reduce(self):
        beta = self.shadow.new_tensor(BETA)
        return ((self.scu.long() + 16*self.mr.long()) * beta).sum((-1,-2))

    def fire(self, mask):
        if self.scu is not None:
            for value in (self.scu, self.mr, self.shadow):
                value.masked_fill_(mask[...,None,None], 0)

    def reset(self):
        self.stats.finish_batch(self.peak)
        self.scu = self.mr = self.shadow = self.peak = None

    def report(self):
        return {**self.stats.report(), 'raw_pre_cancel_max': self.raw_max,
                'max_local_integer_error': self.max_integer_error,
                'plane_max': self.plane_max.tolist()}
