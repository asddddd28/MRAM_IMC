"""Architecture experiment: exact equal-weight cancellation within each Macro.

MR indices are weight planes, so MR4[0] has weight -16, and
MR3[1]/MR2[2]/MR1[3]/MR0[4] each have weight +16 in coarse units.
No deployed Cluster backend is replaced by this module.
"""
import torch


def clear_one_b2_pair(mr):
    """Post-b1 stage: MR4[2] against ONE of MR3[3], MR2[4].

    Priority MR3 then MR2; no MR1/MR0 connections. Each pair removes
    four MR4 counter units while preserving signed coarse potential.
    """
    out = mr.clone()
    available = (out[..., 4] & 4) != 0
    pairs = torch.zeros_like(out[..., 4])
    for b in (3, 2):
        bit = 1 << (6 - b)
        hit = available & ((out[..., b] & bit) != 0)
        out[..., b] -= hit.to(out.dtype) * bit
        out[..., 4] -= hit.to(out.dtype) * 4
        pairs += hit.to(out.dtype)
        available &= ~hit
    return out, pairs


def clear_one_b1_pair(mr, include_mr0=False):
    """Clear MR4[1] against ONE of MR3[2], MR2[3], MR1[4].

    Post-update stage, priority MR3 then MR2 then MR1. No MR0[5]
    connection and no new forwarding or high-bit carry interception.
    Return fresh state and number of pairs (each equals two MR4 units).
    include_mr0 adds MR0[5] as the last-priority partner, still one pair.
    """
    out = mr.clone()
    available = (out[..., 4] & 2) != 0
    pairs = torch.zeros_like(out[..., 4])
    for b in ((3, 2, 1, 0) if include_mr0 else (3, 2, 1)):
        bit = 1 << (5 - b)
        hit = available & ((out[..., b] & bit) != 0)
        out[..., b] -= hit.to(out.dtype) * bit
        out[..., 4] -= hit.to(out.dtype) * 2
        pairs += hit.to(out.dtype)
        available &= ~hit
    return out, pairs


def clear_one_b0_pair(mr):
    """One priority-selected positive bit per set MR4[0], never clear all hits.

    Input is a nonnegative integer tensor [...,5]. Return fresh state and
    a per-Macro 0/1 count of cancelled coarse negative units.
    Priority MR3, MR2, MR1, MR0 is a design choice, not an optimum claim.
    """
    out = mr.clone()
    available = (out[..., 4] & 1).bool()
    cancelled = torch.zeros_like(out[..., 4])
    for b in (3, 2, 1, 0):
        bit = 1 << (4 - b)
        hit = available & ((out[..., b] & bit) != 0)
        out[..., b] -= hit.to(out.dtype) * bit
        out[..., 4] -= hit.to(out.dtype)
        cancelled += hit.to(out.dtype)
        available &= ~hit
    return out, cancelled


def add_carries_with_b0_cancel(mr, carry, trace=None):
    """Intercept each pending negative carry before allowing binary ripple.

    First form all positive candidate counters, then consume available
    aligned positive bits against pending MR4 carry units. Commit remaining
    carries and service one existing MR4[0] pair. Since carry can be 0..3
    for SCU radix 16 with 36 inputs, a Boolean overflow flag is insufficient.
    """
    out = mr + carry
    out[..., 4] = mr[..., 4]
    pending = carry[..., 4].clone()
    if trace is not None:
        trace['incoming_negative_units'] = pending.clone()
        trace['before_b0_interception'] = out.clone()
    cancelled = torch.zeros_like(pending)
    for b in (3, 2, 1, 0):
        bit = 1 << (4 - b)
        hit = (pending > 0) & ((out[..., b] & bit) != 0)
        out[..., b] -= hit.to(out.dtype) * bit
        pending -= hit.to(out.dtype)
        cancelled += hit.to(out.dtype)
    if trace is not None:
        trace['intercepted_negative_units'] = cancelled.clone()
    out[..., 4] += pending
    if trace is not None:
        trace['before_existing_b0'] = out.clone()
    out, existing = clear_one_b0_pair(out)
    return out, cancelled + existing


def add_carries_with_positive_forward(mr, carry, cancel_negative=True, trace=None, toward_lsb=False):
    """Route new +16 bit events from lower to higher weight-plane indices.

    Unit-carry microsteps capture every 0->1 transition, including carry=3.
    Source order MR0..MR3, destination priority MR3 down to source+1.
    A received bit cannot itself re-forward; only a local carry creates an
    event. Occupancy is updated serially, so destinations cannot collide.
    The experiment assumes SCU radix 16 and at most 36 inputs (carry<=3).
    toward_lsb reverses the route: sources MR3..MR0, destinations MR0
    upwards to source-1. Equal-weight bit positions remain unchanged.
    """
    out = mr.clone()
    moved = torch.zeros_like(out[..., 4])
    for b in (range(3, -1, -1) if toward_lsb else range(4)):
        source_bit = 1 << (4 - b)
        for unit in range(3):
            was_zero = (out[..., b] & source_bit) == 0
            active = carry[..., b] > unit
            out[..., b] += active.to(out.dtype)
            pending = active & was_zero & ((out[..., b] & source_bit) != 0)
            for target in (range(b) if toward_lsb else range(3, b, -1)):
                target_bit = 1 << (4 - target)
                hit = pending & ((out[..., target] & target_bit) == 0)
                out[..., b] -= hit.to(out.dtype) * source_bit
                out[..., target] += hit.to(out.dtype) * target_bit
                moved += hit.to(out.dtype)
                pending &= ~hit
    negative_carry = torch.zeros_like(carry)
    negative_carry[..., 4] = carry[..., 4]
    if cancel_negative:
        out, cancelled = add_carries_with_b0_cancel(out, negative_carry, trace=trace)
    else:
        out += negative_carry
        cancelled = torch.zeros_like(moved)
    return out, cancelled, moved
