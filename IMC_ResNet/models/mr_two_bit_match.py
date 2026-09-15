"""Read-only full subtraction opportunities, without changing trajectories."""
import torch


class TwoBitMatch:
    def __init__(self, planes):
        self.planes = list(planes)
        self.total = self.requests = self.ge = self.gt = 0
        self.by_x = torch.zeros((4, 3), dtype=torch.int64)
        self.per_plane = {b: [0, 0] for b in planes}

    def observe(self, state):
        x = state[..., 4] & 3
        request = x != 0
        ge = torch.zeros_like(request)
        gt = torch.zeros_like(request)
        self.total += x.numel()
        self.requests += int(request.sum())
        for b in self.planes:
            y = (state[..., b] >> (4-b)) & 3
            hit_ge = request & (y >= x)
            hit_gt = request & (y > x)
            self.per_plane[b][0] += int(hit_ge.sum())
            self.per_plane[b][1] += int(hit_gt.sum())
            ge |= hit_ge
            gt |= hit_gt
        self.ge += int(ge.sum())
        self.gt += int(gt.sum())
        for n in (1,2,3):
            mask = x == n
            self.by_x[n] += torch.tensor([int(mask.sum()),int((mask & ge).sum()),int((mask & gt).sum())])

    def report(self):
        return dict(planes=self.planes, observations=self.total, requests=self.requests,
                    full_cancel_ge=self.ge, strict_larger_gt=self.gt,
                    full_cancel_rate=self.ge/self.requests if self.requests else None,
                    strict_larger_rate=self.gt/self.requests if self.requests else None,
                    by_negative_value={str(n):dict(requests=int(self.by_x[n,0]), ge=int(self.by_x[n,1]),gt=int(self.by_x[n,2])) for n in (1,2,3)},
                    per_plane_ge_gt=self.per_plane)
