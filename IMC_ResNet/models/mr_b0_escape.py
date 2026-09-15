"""Count actual b0 carry-out events, never repeated unmatched waiting states."""
import torch


def has_partner(state):
    hit = torch.zeros_like(state[..., 4], dtype=torch.bool)
    for b in range(4):
        hit |= (state[..., b] & (1 << (4-b))) != 0
    return hit


class B0Escape:
    def __init__(self):
        self.seen_partner = None
        self.rollovers = self.strict_rollovers = self.rollover_updates = 0

    def observe(self, previous, trace, after_b0):
        bit = (previous[..., 4] & 1).bool()
        if self.seen_partner is None:
            self.seen_partner = torch.zeros_like(bit)
        # Preserve whether this surviving b0=1 episode has ever seen a
        # partner at the pre-interception check, even if incoming carry
        # consumed that partner instead of clearing the resident b0.
        self.seen_partner &= bit
        self.seen_partner |= bit & has_partner(trace['before_b0_interception'])
        pending = trace['incoming_negative_units'] - trace['intercepted_negative_units']
        assert not ((pending > 0) & has_partner(trace['before_existing_b0'])).any()
        updates = torch.zeros_like(bit)
        for unit in range(3):
            active = pending > unit
            rollover = active & bit
            self.rollovers += int(rollover.sum())
            self.strict_rollovers += int((rollover & ~self.seen_partner).sum())
            updates |= rollover
            bit ^= active
            # Each admitted unit changes b0: any new residence starts
            # after all positive partners have been exhausted.
            self.seen_partner &= ~active
        self.rollover_updates += int(updates.sum())
        self.seen_partner &= (after_b0[..., 4] & 1).bool()

    def reset(self, fired):
        if self.seen_partner is not None:
            self.seen_partner.masked_fill_(fired[..., 0], False)
