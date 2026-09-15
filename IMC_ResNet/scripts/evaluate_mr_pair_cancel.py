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
from IMC_ResNet.models.mr_pair_cancel import clear_one_b0_pair, add_carries_with_b0_cancel


def main():
    torch.set_num_threads(4)
    snn = ROOT/'SNN_ResNet10'
    indices = [0, 1, 2, 3, 4, 5, 64, 115]
    data = datasets.FashionMNIST(snn/'data', train=False, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((.2860,), (.3530,))]))
    loader = DataLoader(Subset(data, indices), 2)
    model, meta = load_finetuned(snn/'checkpoints/hard_reset_finetuned.pt', snn/'checkpoints/calibrated.json')
    model = make_explicit_quantized_reference(fuse_explicit_hidden(model))
    conv = model.layer1.conv1
    q, _ = quantize_int5(conv.weight)
    assert q.flatten(1).shape == (32, 288)
    flat = F.pad(q.flatten(1), (0, 576-288))
    bits = (((flat[..., None] & 31) >> torch.arange(5)) & 1).float()
    packed = bits.reshape(32, 16, 36, 5).permute(1, 2, 0, 3).reshape(16, 36, 160)
    variants = ['baseline', 'post_add_b0', 'carry_intercept_b0']
    stats = {n: MRStats(6) for n in variants}
    before_max = {n: 0 for n in variants}
    cancelled = {n: 0 for n in variants}
    plane_max = {n: torch.zeros(5, dtype=torch.int32) for n in variants}
    beta = torch.tensor([1, 2, 4, 8, -16], dtype=torch.int64)
    max_error = {n: 0 for n in variants}
    out = ROOT/'IMC_ResNet/checkpoints/mr_pair_cancel_counterfactual.json'
    report = {'status': 'running', 'checkpoint': meta, 'indices': indices, 'steps': 64,
              'layer': 'layer1.conv1', 'mapping': 'original 8x36 + 8 padded macros',
              'positive_partner_priority': [3, 2, 1, 0],
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
                patches = F.pad(F.unfold(spike, 3, padding=1), (0, 0, 0, 288))
                b, _, p = patches.shape
                inp = patches.reshape(b, 16, 36, p).permute(1, 0, 3, 2).reshape(16, b*p, 36)
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
                    if name == 'post_add_b0':
                        candidate, removed = clear_one_b0_pair(raw)
                    elif name == 'carry_intercept_b0':
                        candidate, removed = add_carries_with_b0_cancel(banks[name], carry)
                    else:
                        candidate, removed = raw, torch.zeros(1, dtype=torch.int32)
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
            print(f'samples={(batch+1)*2}/8', flush=True)
    report.update(status='complete', variants={name: {**stats[name].report(),
                  'raw_pre_cancel_max_mr': before_max[name], 'cancelled_mr4_units': cancelled[name],
                  'max_by_weight_plane_0_to_4': plane_max[name].tolist(),
                  'max_local_coarse_integer_error': max_error[name]} for name in variants})
    out.write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps({name: {k: v[k] for k in ['max_mr', 'mr_ge48', 'mr_ge64', 'max_by_weight_plane_0_to_4', 'raw_pre_cancel_max_mr', 'cancelled_mr4_units', 'max_local_coarse_integer_error']}
                      for name, v in report['variants'].items()}, indent=2))


if __name__ == '__main__':
    main()
