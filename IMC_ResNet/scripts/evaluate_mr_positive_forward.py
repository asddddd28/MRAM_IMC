"""Read-only trajectory replay of proposed Macro-local bit-pair cancellation."""
import json
import sys
from pathlib import Path
import torch
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from spikingjelly.activation_based import functional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from IMC_ResNet.models.finetuned_snn import load_finetuned, fuse_explicit_hidden, make_explicit_quantized_reference
from IMC_ResNet.models.cluster_backend import MRStats, quantize_int5
from IMC_ResNet.models.mr_pair_cancel import clear_one_b0_pair, add_carries_with_b0_cancel, add_carries_with_positive_forward, clear_one_b1_pair, clear_one_b2_pair


def main(high_bit=False, b2=False, lsb=False, lsb_b1=False, checkpoint=None, output=None, rows=36, best_only=False, indices=None):
    lsb = lsb or lsb_b1
    b2 = b2 or lsb
    high_bit = high_bit or b2
    torch.set_num_threads(4)
    snn = ROOT/'SNN_ResNet10'
    indices = indices if indices is not None else [0, 1, 2, 3, 4, 5, 64, 115]
    assert rows in (18, 36)
    data = datasets.FashionMNIST(snn/'data', train=False, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((.2860,), (.3530,))]))
    loader = DataLoader(Subset(data, indices), 2)
    model, meta = load_finetuned(checkpoint or snn/'checkpoints/hard_reset_finetuned.pt', snn/'checkpoints/calibrated.json')
    model = make_explicit_quantized_reference(fuse_explicit_hidden(model))
    conv = model.layer1.conv1
    q, _ = quantize_int5(conv.weight)
    assert q.flatten(1).shape == (32, 288)
    flat = F.pad(q.flatten(1), (0, 16*rows-288))
    bits = (((flat[..., None] & 31) >> torch.arange(5)) & 1).float()
    packed = bits.reshape(32, 16, rows, 5).permute(1, 2, 0, 3).reshape(16, rows, 160)
    variants = ['baseline', 'carry_intercept_b0', 'positive_forward_only', 'positive_forward_cancel']
    if high_bit:
        variants = ['baseline', 'carry_intercept_b0', 'carry_intercept_b0_b1', 'positive_forward_cancel', 'positive_forward_cancel_b1']
    if b2:
        variants = ['baseline', 'carry_intercept_b0_b1', 'carry_intercept_b0_b1_b2', 'positive_forward_cancel_b1', 'positive_forward_cancel_b1_b2']
    if lsb:
        variants = ['baseline', 'carry_intercept_b0_b1_b2', 'positive_forward_cancel_b1_b2', 'positive_forward_lsb_cancel_b1_b2']
    if lsb_b1:
        variants = ['baseline', 'positive_forward_cancel_b1_b2', 'positive_forward_lsb_cancel_b1_b2', 'positive_forward_lsb_mr0b5_cancel_b1_b2']
    if best_only:
        variants = ['baseline', 'positive_forward_cancel_b1_b2']
    mr0b5_pairs = {n: 0 for n in variants}
    b2_pairs = {n: 0 for n in variants}
    high_pairs = {n: 0 for n in variants}
    stats = {n: MRStats(5) for n in variants}
    before_max = {n: 0 for n in variants}
    cancelled = {n: 0 for n in variants}
    forwarded = {n: 0 for n in variants}
    plane_max = {n: torch.zeros(5, dtype=torch.int32) for n in variants}
    beta = torch.tensor([1, 2, 4, 8, -16], dtype=torch.int64)
    max_error = {n: 0 for n in variants}
    out = ROOT/'IMC_ResNet/checkpoints/mr_positive_forward_counterfactual.json'
    if high_bit:
        out = out.with_name('mr_b1_cancel_counterfactual.json')
    if b2:
        out = out.with_name('mr_b2_cancel_counterfactual.json')
    if lsb:
        out = out.with_name('mr_lsb_forward_counterfactual.json')
    if lsb_b1:
        out = out.with_name('mr_lsb_mr0b5_counterfactual.json')
    if output is not None:
        out = Path(output)
    report = {'status': 'running', 'checkpoint': meta, 'indices': indices, 'steps': 64,
              'layer': 'layer1.conv1', 'mapping': f'16 macros, {rows} assigned rows each, zero padding to 36 hardware rows',
              'positive_partner_priority': [3, 2, 1, 0],
              'b2_extension': 'after b1: MR4[2] with one of MR3[3], MR2[4], priority 3,2; no other connections' if b2 else None,
              'b1_extension': 'post existing update: MR4[1] paired with one of MR3[2], MR2[3], MR1[4], priority 3,2,1; no MR0[5]' if high_bit else None,
              'mr0b5_extension': 'MR0[5] appended to b1 priority MR3,MR2,MR1,MR0; only for mr0b5 variant' if lsb_b1 else None,
              'lsb_forwarding': 'source MR3..MR0, destination MR0 first then higher below source; same +16 connections and same b0/b1/b2 cancel' if lsb else None,
              'forwarding': 'source MR0..MR3; only new 0->1 local carry events; choose highest empty higher-index plane; three unit carry microsteps; then negative cancellation',
              'scope': 'all spatial positions; preserve each Macro signed state and fixed digital IF reset masks; no hardware/full-network remap validation',
              'statistics': 'candidate after cancellation and BEFORE firing reset; raw pre-cancellation maximum separately recorded'}
    out.write_text(json.dumps(report, indent=2), encoding='utf8')
    with torch.no_grad():
        for batch, (x, _) in enumerate(loader):
            functional.reset_net(model)
            banks = {n: None for n in variants}
            peaks = {n: None for n in variants}
            scu = None
            for t in range(64):
                spike = model.stem(x)
                patches = F.pad(F.unfold(spike, 3, padding=1), (0, 0, 0, 16*rows-288))
                b, _, p = patches.shape
                inp = patches.reshape(b, 16, rows, p).permute(1, 0, 3, 2).reshape(16, b*p, rows)
                counts = torch.bmm(inp, packed).reshape(16, b, p, 32, 5).permute(1, 3, 2, 0, 4).contiguous().to(torch.int32)
                if scu is None:
                    scu = torch.zeros_like(counts)
                    banks = {n: torch.zeros_like(counts) for n in variants}
                    peaks = {n: torch.zeros_like(counts) for n in variants}
                z = scu + counts
                carry, scu = z // 16, z % 16
                for name in variants:
                    raw = banks[name] + carry
                    before_max[name] = max(before_max[name], int(raw.max()))
                    if name.startswith('positive_forward'):
                        candidate, removed, moved = add_carries_with_positive_forward(banks[name], carry, cancel_negative='_cancel' in name, toward_lsb='_lsb_' in name)
                        forwarded[name] += int(moved.sum())
                    elif name == 'post_add_b0':
                        candidate, removed = clear_one_b0_pair(raw)
                    elif name.startswith('carry_intercept_b0'):
                        candidate, removed = add_carries_with_b0_cancel(banks[name], carry)
                    else:
                        candidate, removed = raw, torch.zeros(1, dtype=torch.int32)
                    if '_b1' in name:
                        extended = '_mr0b5_' in name
                        if extended:
                            available = ((candidate[...,4] & 2) != 0) & ((candidate[...,0] & 32) != 0)
                            for plane in (3,2,1):
                                available &= (candidate[...,plane] & (1 << (5-plane))) == 0
                            mr0b5_pairs[name] += int(available.sum())
                        candidate, pairs = clear_one_b1_pair(candidate, include_mr0=extended)
                        high_pairs[name] += int(pairs.sum())
                        removed = removed + 2 * pairs
                    if name.endswith('_b2'):
                        candidate, pairs = clear_one_b2_pair(candidate)
                        b2_pairs[name] += int(pairs.sum())
                        removed = removed + 4 * pairs
                    cancelled[name] += int(removed.sum())
                    banks[name] = candidate
                    stats[name].observe(candidate)
                    peaks[name] = torch.maximum(peaks[name], candidate)
                    plane_max[name] = torch.maximum(plane_max[name], candidate.reshape(-1, 5).amax(0))
                baseline_value = (banks['baseline'].long() * beta).sum(-1)
                for name in variants[1:]:
                    error = int(((banks[name].long() * beta).sum(-1) - baseline_value).abs().max())
                    max_error[name] = max(max_error[name], error)
                    assert error == 0, (batch, t, name, error)
                fired = model.layer1.lif1(conv(spike)).flatten(2).bool()[..., None, None]
                scu.masked_fill_(fired, 0)
                for name in variants:
                    banks[name].masked_fill_(fired, 0)
            for name in variants:
                stats[name].finish_batch(peaks[name])
            print(f'samples={min((batch+1)*2,len(indices))}/{len(indices)}', flush=True)
    report.update(status='complete', variants={name: {**stats[name].report(),
                  'mr0b5_cancelled_pairs': mr0b5_pairs[name], 'b2_cancelled_pairs': b2_pairs[name], 'b1_cancelled_pairs': high_pairs[name], 'forwarded_positive_units': forwarded[name], 'raw_pre_cancel_max_mr': before_max[name], 'cancelled_mr4_units': cancelled[name],
                  'max_by_weight_plane_0_to_4': plane_max[name].tolist(),
                  'max_local_coarse_integer_error': max_error[name]} for name in variants})
    out.write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps({name: {k: v[k] for k in ['max_mr', 'mr_ge48', 'mr_ge64', 'max_by_weight_plane_0_to_4', 'raw_pre_cancel_max_mr', 'cancelled_mr4_units', 'max_local_coarse_integer_error', 'forwarded_positive_units', 'b1_cancelled_pairs', 'b2_cancelled_pairs', 'mr0b5_cancelled_pairs']}
                      for name, v in report['variants'].items()}, indent=2))


if __name__ == '__main__':
    main()
