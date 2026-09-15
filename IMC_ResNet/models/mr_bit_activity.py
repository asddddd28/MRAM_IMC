"""Read-only counters: register-boundary activity and conditional pair hits."""
import torch


class BitActivity:
    def __init__(self):
        self.observations = 0
        self.rising = torch.zeros((5, 5), dtype=torch.int64)
        self.falling = self.rising.clone()
        self.reset_falling = self.rising.clone()
        self.ones = self.rising.clone()

    def observe(self, previous, candidate, fired):
        self.observations += previous[..., 0].numel()
        for k in range(5):
            old = ((previous >> k) & 1).bool()
            new = ((candidate >> k) & 1).bool()
            self.rising[:, k] += ((~old) & new).reshape(-1, 5).sum(0)
            self.falling[:, k] += (old & (~new)).reshape(-1, 5).sum(0)
            self.reset_falling[:, k] += (new & fired).reshape(-1, 5).sum(0)
            self.ones[:, k] += new.reshape(-1, 5).sum(0)

    def report(self):
        out = {'denominator_per_plane_bit': self.observations,
               'definition': 'rows MR0..MR4, columns b0..b4; update from previous post-reset state to current pre-reset candidate; reset separate; no internal glitches/microstep toggles'}
        for name in ('rising', 'falling', 'reset_falling', 'ones'):
            value = getattr(self, name)
            out[name + '_counts'] = value.tolist()
            out[name + '_rates'] = (value.double()/self.observations).tolist()
        out['update_toggle_rates'] = ((self.rising+self.falling).double()/self.observations).tolist()
        return out


class PairOpportunity:
    def __init__(self, k):
        self.k = k
        self.partners = [(b, k+4-b) for b in (3,2,1,0) if k+4-b <= 4]
        self.observations = self.requested = self.matched = 0
        self.partner_matches = [0]*len(self.partners)

    def observe(self, state):
        request = (state[..., 4] & (1 << self.k)) != 0
        any_hit = torch.zeros_like(request)
        self.observations += request.numel()
        self.requested += int(request.sum())
        for i, (b, k) in enumerate(self.partners):
            hit = request & ((state[..., b] & (1 << k)) != 0)
            self.partner_matches[i] += int(hit.sum())
            any_hit |= hit
        self.matched += int(any_hit.sum())

    def report(self):
        return dict(negative_bit=self.k, implemented=self.k<=2,
                    stage='after negative carry interception, before existing b0 clear' if self.k==0 else 'before b1 clear' if self.k==1 else 'before b2 clear' if self.k==2 else 'after existing b0/b1/b2; hypothetical only',
                    observations=self.observations, requested=self.requested, matched=self.matched,
                    request_rate=self.requested/self.observations if self.observations else None,
                    conditional_match_rate=self.matched/self.requested if self.requested else None,
                    partners=[dict(plane=b,bit=k,matches=n,conditional_match_rate=n/self.requested if self.requested else None)
                              for (b,k),n in zip(self.partners,self.partner_matches)])
