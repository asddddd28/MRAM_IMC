"""Exact per-Macro signed coarse accumulator; independent experimental bank."""
import torch
from .cluster_backend import BETA


def signed_bits(lo, hi):
    """Smallest ordinary two's-complement register covering [lo, hi]."""
    return max(1, int(hi).bit_length()+1 if hi >= 0 else 1,
               (int(-lo)-1).bit_length()+1 if lo < 0 else 1)


def guaranteed_bound(steps, folds=1, rows=36):
    maximum = rows * steps * folds // 16
    lo, hi = -16*maximum, 15*maximum
    return dict(steps=steps, folds=folds, rows=rows, max_unsigned_plane=maximum,
                coarse_min=lo, coarse_max=hi, shared_signed_bits=signed_bits(lo,hi),
                unsigned_plane_bits=max(1,maximum.bit_length()))


class SharedMacroMR:
    def __init__(self):
        self.scu = self.coarse = self.shadow = None
        self.minimum = self.maximum = 0
        self.min_map = self.max_map = None
        self.fold_updates = 0
        self.max_integer_error = 0

    def add_counts(self, counts):
        if self.scu is None:
            self.scu = torch.zeros_like(counts)
            self.coarse = torch.zeros_like(counts[...,0],dtype=torch.int64)
            # Diagnostic shadow ONLY: not part of proposed hardware storage.
            self.shadow = torch.zeros_like(counts,dtype=torch.int64)
        z = self.scu + counts
        carry, self.scu = z // 16, z % 16
        beta = torch.tensor(BETA,dtype=torch.int64)
        self.coarse += (carry.long()*beta).sum(-1)
        self.shadow += carry
        error = int((self.coarse-(self.shadow*beta).sum(-1)).abs().max())
        self.max_integer_error=max(self.max_integer_error,error)
        assert error == 0
        self.minimum=min(self.minimum,int(self.coarse.min()))
        self.maximum=max(self.maximum,int(self.coarse.max()))
        low=self.coarse.amin((0,2))
        high=self.coarse.amax((0,2))
        self.min_map=low if self.min_map is None else torch.minimum(self.min_map,low)
        self.max_map=high if self.max_map is None else torch.maximum(self.max_map,high)
        self.fold_updates+=1

    def reduce(self):
        beta=torch.tensor(BETA,dtype=torch.int64)
        return ((self.scu.long()*beta).sum(-1)+16*self.coarse).sum(-1)

    def fire(self, mask):
        if self.scu is not None:
            self.scu.masked_fill_(mask[...,None,None],0)
            self.coarse.masked_fill_(mask[...,None],0)
            self.shadow.masked_fill_(mask[...,None,None],0)

    def reset(self):
        self.scu=self.coarse=self.shadow=None

    def report(self):
        return dict(minimum=self.minimum, maximum=self.maximum,
                    required_signed_bits=signed_bits(self.minimum,self.maximum),
                    fold_updates=self.fold_updates,max_local_integer_error=self.max_integer_error,
                    min_by_channel_macro=self.min_map.tolist(),max_by_channel_macro=self.max_map.tolist())
