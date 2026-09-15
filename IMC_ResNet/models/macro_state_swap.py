"""Exact pairwise relocation of all five MR/SCU states within one neuron."""
import torch


def swap_macro_pairs(mr, scu, rate, *, active_macros=16, adaptive=True, gap=4):
    """Pair high/low recent rates; swap only if high-rate state is fuller.

    rate has shape [...,16], states [...,16,5]. Rates belong to physical
    input mappings and MUST NOT move with state. No future inputs are used.
    Fixed mode pairs index i with i+active_macros/2.
    """
    if active_macros not in (8,16):
        raise ValueError('supported groups: 8 or 16')
    n=active_macros
    shape=mr.shape[:-2]
    mapping=torch.arange(16,device=mr.device).expand(*shape,16).clone()
    if adaptive:
        order=rate[...,:n].argsort(dim=-1,stable=True)
        low=order[...,:n//2]
        high=order[...,n//2:].flip(-1)
    else:
        low=torch.arange(n//2,device=mr.device).expand(*shape,n//2)
        high=low+n//2
    pressure=(mr.float()+scu.float()/16).amax(-1)
    lp=pressure.gather(-1,low);hp=pressure.gather(-1,high)
    if adaptive:
        selected=(rate.gather(-1,high)>rate.gather(-1,low)) & (hp>lp+gap)
    else:
        left=mr.gather(-2,low[...,None].expand(*low.shape,5))
        right=mr.gather(-2,high[...,None].expand(*high.shape,5))
        sl=scu.gather(-2,low[...,None].expand(*low.shape,5))
        sr=scu.gather(-2,high[...,None].expand(*high.shape,5))
        selected=(left!=right).any(-1)|(sl!=sr).any(-1)
    mapping.scatter_(-1,low,torch.where(selected,high,low))
    mapping.scatter_(-1,high,torch.where(selected,low,high))
    idx=mapping[...,None].expand_as(mr)
    return mr.gather(-2,idx),scu.gather(-2,idx),int(selected.sum())

