"""Read-only diagnostics of simultaneous MR states, not merged peak maps.

Positive bit planes are NOT positive weights: low bits of a negative int5
weight contribute to the positive terms too. Signed contributions always use
[1,2,4,8,-16]. Snapshots are taken per fold, before firing clears the bank.
"""
import numpy as np
import torch
from .cluster_backend import bit_histogram


BETA = np.array([1, 2, 4, 8, -16], dtype=np.int64)


def contribution_metrics(mr, scu=None):
    """Return exact per-macro terms and split cancellation into local/cross-macro.

    min(P,N) counts cancelled magnitude ONCE, not twice. The cancellation ratio
    uses 2*min(P,N)/(P+N). These are algebraic terms, not separate positive weights.
    """
    count = 16 * np.asarray(mr, dtype=np.int64)
    if scu is not None:
        count = count + np.asarray(scu, dtype=np.int64)
    positive = (count[..., :4] * BETA[:4]).sum(-1)
    negative = count[..., 4] * 16
    p, n = positive.sum(-1), negative.sum(-1)
    local = np.minimum(positive, negative).sum(-1)
    cancelled = np.minimum(p, n)
    return {'positive_by_macro': positive, 'negative_by_macro': negative,
            'net_by_macro': positive - negative, 'positive': p, 'negative': n,
            'net': p - n, 'local_cancelled': local,
            'cross_macro_cancelled': cancelled - local,
            'cancellation_ratio': 2 * cancelled / np.maximum(p + n, 1),
            'local_share_of_cancelled': local / np.maximum(cancelled, 1)}


def serial_metrics(mr, scu=None):
    return {key: value.tolist() for key, value in contribution_metrics(mr, scu).items()}


def new_summary():
    return {'neuron_fold_observations': 0, 'high_lane_observations': 0,
            'both_sides_high': 0, 'same_macro_both_high': 0,
            'different_macros_both_high': 0, 'age_sum': 0, 'age_max': 0,
            'age_min': 10**9, 'cancellation_ratio_sum': 0.,
            'local_cancelled_sum': 0, 'cross_macro_cancelled_sum': 0,
            'charged_voltage_sum': 0., 'negative_charged_voltage_count': 0,
            'fired_same_step': 0}


class MRObserver:
    """Record high-width conditions and bounded full 16x5 snapshots per layer."""
    def __init__(self, layer, sink, top_k=8, thresholds=(32, 48, 64)):
        if top_k < 1 or not thresholds or any(t < 1 for t in thresholds):
            raise ValueError("top_k and diagnostic thresholds must be positive")
        self.layer, self.sink = layer, sink
        self.top_k, self.thresholds = top_k, thresholds
        self.summaries = {str(t): new_summary() for t in thresholds}
        self.top = {}
        self.events_7bit = []
        self.peak_bits_by_macro = [{} for _ in range(16)]
        self.peak_bits_by_plane = [{} for _ in range(5)]
        self.step = 0
        self.pending = []
        self.trace_targets = set()
        self.trace = []
        self.batch_ids = []
        self.last_fire = None
        self.trace_cumulative = {}
        self.all_age_sum = self.all_neuron_steps = 0
        self.qpacked = None
        self.kcounts = None

    def begin_batch(self, sample_ids):
        self.batch_ids = list(sample_ids)
        self.last_fire = None
        self.pending = []
        self.trace_cumulative = {}

    def _weights(self, conv):
        if self.qpacked is None:
            q = conv.qweight.detach().cpu().numpy().reshape(conv.out_channels, -1)
            q = np.pad(q, ((0, 0), (0, conv.folds * 576 - q.shape[1])))
            self.qpacked = q.reshape(conv.out_channels, conv.folds, 16, 36)
            bits = ((self.qpacked[..., None] & 31) >> np.arange(5)) & 1
            self.kcounts = bits.sum(axis=(1, 3))

    def on_fold(self, conv, counts, fold, inputs):
        if conv.bank.overflow != "wide_reference":
            raise ValueError("MRObserver requires wide_reference: snapshots must precede any truncation")
        mr, scu = conv.bank.mr.numpy(), conv.bank.scu.numpy()
        b, o, positions = mr.shape[:3]
        if self.last_fire is None:
            self.last_fire = np.zeros((b, o, positions), dtype=np.int32)
        if fold == 0:
            self.all_age_sum += int((self.step - self.last_fire).sum())
            self.all_neuron_steps += self.last_fire.size
        self._weights(conv)
        selected = np.empty((0, 3), dtype=int)
        maximum = np.empty(0, dtype=int)
        snapshots = []
        # A historical max below the diagnostic cutoff safely skips this scan.
        if len(conv.bank.stats.hist) - 1 >= min(self.thresholds):
            peak = mr.max(axis=(-1, -2))
            selected = np.argwhere(peak >= min(self.thresholds))
            if len(selected):
                bi, ci, pi = selected.T
                values = mr[bi, ci, pi]
                maximum = peak[bi, ci, pi]
                ages = self.step - self.last_fire[bi, ci, pi]
                metrics = contribution_metrics(values)
                for limit in self.thresholds:
                    mask = maximum >= limit
                    if not mask.any():
                        continue
                    high = values[mask] >= limit
                    ph, nh = high[..., :4].any(-1), high[..., 4]
                    same = (ph & nh).sum(-1)
                    cross = ph.sum(-1) * nh.sum(-1) - same
                    stat = self.summaries[str(limit)]
                    stat['neuron_fold_observations'] += int(mask.sum())
                    stat['high_lane_observations'] += int(high.sum())
                    stat['both_sides_high'] += int((ph.any(-1) & nh.any(-1)).sum())
                    stat['same_macro_both_high'] += int((same > 0).sum())
                    stat['different_macros_both_high'] += int((cross > 0).sum())
                    stat['age_sum'] += int(ages[mask].sum())
                    stat['age_max'] = max(stat['age_max'], int(ages[mask].max()))
                    stat['age_min'] = min(stat['age_min'], int(ages[mask].min()))
                    stat['cancellation_ratio_sum'] += float(metrics['cancellation_ratio'][mask].sum())
                    stat['local_cancelled_sum'] += int(metrics['local_cancelled'][mask].sum())
                    stat['cross_macro_cancelled_sum'] += int(metrics['cross_macro_cancelled'][mask].sum())
                for idx, (bb, cc, pp) in enumerate(selected):
                    score = int(maximum[idx])
                    key = (self.batch_ids[bb], int(cc), int(pp))
                    old = self.top.get(key)
                    eligible = ((old is not None and score > old['max_mr']) or
                                (old is None and (len(self.top) < self.top_k or score > min(v['max_mr'] for v in self.top.values()))))
                    if score < 64 and not eligible:
                        continue
                    snap = self._snapshot(conv, mr, scu, counts, int(bb), int(cc), int(pp), fold)
                    snapshots.append(snap)
                    if score >= 64:
                        self.events_7bit.append(snap)
                    if eligible:
                        if old is None and len(self.top) >= self.top_k:
                            del self.top[min(self.top, key=lambda k: self.top[k]['max_mr'])]
                        self.top[key] = snap
        # Targeted second-pass traces retain every time step, including resets.
        for bb, sample in enumerate(self.batch_ids):
            for wanted_sample, cc, pp in self.trace_targets:
                if sample != wanted_sample:
                    continue
                snap = self._snapshot(conv, mr, scu, counts, bb, cc, pp, fold)
                input_patch = inputs[:, bb * positions + pp, :].detach().numpy().astype(np.int64)
                weight = self.qpacked[cc, fold].astype(np.int64)
                pos = int((input_patch * np.maximum(weight, 0)).sum())
                neg = int((input_patch * np.maximum(-weight, 0)).sum())
                key = (sample, cc, pp)
                cumulative = self.trace_cumulative.setdefault(key, [0, 0])
                cumulative[0] += pos
                cumulative[1] += neg
                snap['true_weight_positive_step_fold'] = pos
                snap['true_weight_negative_step_fold'] = neg
                snap['true_weight_positive_since_reset'] = cumulative[0]
                snap['true_weight_negative_since_reset'] = cumulative[1]
                snap['input_spikes_by_macro_this_fold'] = input_patch.sum(-1).tolist()
                snap['integer_decomposition_error'] = int(snap['with_scu']['net'] - (cumulative[0] - cumulative[1]))
                self.trace.append(snap)
                snapshots.append(snap)
        self.pending.append((selected, maximum, snapshots))

    def _snapshot(self, conv, mr, scu, counts, b, channel, pos, fold):
        values = mr[b, channel, pos]
        macro, bit = np.unravel_index(values.argmax(), values.shape)
        age = self.step - int(self.last_fire[b, channel, pos])
        total = 16 * int(values[macro, bit]) + int(scu[b, channel, pos, macro, bit])
        k = int(self.kcounts[channel, macro, bit])
        return {'layer': self.layer, 'sample': self.batch_ids[b], 'batch_index': b,
                'channel': channel, 'position': pos, 'step': self.step, 'fold': fold,
                'age_since_reset': age, 'last_fire_step': int(self.last_fire[b, channel, pos]),
                'max_mr': int(values.max()), 'max_macro': int(macro), 'max_bit': int(bit),
                'hot_lane_weight_ones_across_folds': k,
                'hot_lane_mean_pmac_per_step_since_reset': total / max(age, 1),
                'hot_lane_active_fraction_since_reset': total / max(age*k, 1),
                'mr': values.tolist(), 'scu': scu[b, channel, pos].tolist(),
                'pmac': counts[b, channel, pos].tolist(),
                'mr_only': serial_metrics(values), 'with_scu': serial_metrics(values, scu[b, channel, pos]),
                'weight_scale': float(conv.weight_scale[channel]), 'bias_per_step': float(conv.bias[channel]),
                'bias_since_reset': float(conv.bias[channel]) * age}

    def before_if(self, module, args):
        current = args[0].detach().flatten(2).numpy()
        previous = module.v
        previous = previous.detach().flatten(2).numpy() if isinstance(previous, torch.Tensor) else np.full(current.shape, previous)
        for selected, maximum, snapshots in self.pending:
            if len(selected):
                charged = previous[tuple(selected.T)] + current[tuple(selected.T)]
                for limit in self.thresholds:
                    mask = maximum >= limit
                    stat = self.summaries[str(limit)]
                    stat['charged_voltage_sum'] += float(charged[mask].sum(dtype=np.float64))
                    stat['negative_charged_voltage_count'] += int((charged[mask] < 0).sum())
            for snap in snapshots:
                idx = (snap['batch_index'], snap['channel'], snap['position'])
                snap['v_before'] = float(previous[idx])
                snap['if_input_this_step'] = float(current[idx])
                snap['v_charged'] = float(previous[idx] + current[idx])
                snap['if_threshold'] = float(module.v_threshold)

    def after_if(self, module, args, output):
        spike = output.detach().flatten(2).numpy().astype(bool)
        for selected, maximum, snapshots in self.pending:
            if len(selected):
                fired = spike[tuple(selected.T)]
                for limit in self.thresholds:
                    self.summaries[str(limit)]['fired_same_step'] += int(fired[maximum >= limit].sum())
            for snap in snapshots:
                idx = (snap['batch_index'], snap['channel'], snap['position'])
                snap['fired'] = bool(spike[idx])
                if snap['fired']:
                    self.trace_cumulative.pop((snap['sample'], snap['channel'], snap['position']), None)
        if self.last_fire is not None:
            self.last_fire[spike] = self.step
        self.pending = []

    def finish_batch(self, conv):
        if conv.bank.peak is None:
            return
        peak = conv.bank.peak.numpy()
        for macro in range(16):
            self._merge(self.peak_bits_by_macro[macro], bit_histogram(np.bincount(peak[..., macro, :].reshape(-1))))
        for bit in range(5):
            self._merge(self.peak_bits_by_plane[bit], bit_histogram(np.bincount(peak[..., bit].reshape(-1))))

    @staticmethod
    def _merge(target, source):
        for key, value in source.items():
            target[key] = target.get(key, 0) + value

    def report(self):
        # Reporting must not mutate live counters (reports may be read mid-run).
        summaries = {key: value.copy() for key, value in self.summaries.items()}
        for stat in summaries.values():
            n = stat['neuron_fold_observations']
            stat['mean_age'] = stat['age_sum'] / max(n, 1)
            stat['mean_cancellation_ratio'] = stat['cancellation_ratio_sum'] / max(n, 1)
            stat['mean_charged_voltage'] = stat['charged_voltage_sum'] / max(n, 1)
            if not n:
                stat['age_min'] = None
        return {'threshold_conditions': summaries,
                'all_neuron_step_mean_age': self.all_age_sum / max(self.all_neuron_steps, 1),
                'peak_bits_by_macro': self.peak_bits_by_macro,
                'peak_bits_by_plane': self.peak_bits_by_plane,
                'top_cases': sorted(self.top.values(), key=lambda x: -x['max_mr']),
                'events_7bit': self.events_7bit, 'trace': self.trace}
