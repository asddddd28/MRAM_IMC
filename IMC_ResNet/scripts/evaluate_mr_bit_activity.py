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
    from IMC_ResNet.models.mr_bit_activity import BitActivity, PairOpportunity
    activity = BitActivity()
    opportunities = [PairOpportunity(k) for k in range(5)]
    beta = torch.tensor([1, 2, 4, 8, -16], dtype=torch.int64)
    incoming = intercepted = forwarded = b0_pairs = b1_pairs = b2_pairs = 0
    peak = torch.zeros(5, dtype=torch.int32)
    with torch.no_grad():
        for batch, (x, _) in enumerate(loader):
            functional.reset_net(model)
            bank = scu = reference = None
            for t in range(64):
                spike = model.stem(x)
                patches = F.pad(F.unfold(spike, 3, padding=1), (0, 0, 0, 288))
                b, _, p = patches.shape
                inp = patches.reshape(b, 16, 36, p).permute(1, 0, 3, 2).reshape(16, b*p, 36)
                counts = torch.bmm(inp, packed).reshape(16, b, p, 32, 5).permute(1, 3, 2, 0, 4).contiguous().to(torch.int32)
                # Exclude the eight structurally padded Macros from denominators.
                assert counts[..., 8:, :].count_nonzero() == 0
                counts = counts[..., :8, :].contiguous()
                if scu is None:
                    scu = torch.zeros_like(counts)
                    bank = torch.zeros_like(counts)
                    reference = torch.zeros_like(counts)
                z = scu + counts
                carry, scu = z // 16, z % 16
                reference += carry
                trace = {}
                candidate, removed, moved = add_carries_with_positive_forward(bank, carry, trace=trace)
                incoming += int(trace['incoming_negative_units'].sum())
                intercepted += int(trace['intercepted_negative_units'].sum())
                b0_pairs += int((removed - trace['intercepted_negative_units']).sum())
                forwarded += int(moved.sum())
                opportunities[0].observe(trace['before_existing_b0'])
                opportunities[1].observe(candidate)
                candidate, pairs = clear_one_b1_pair(candidate)
                b1_pairs += int(pairs.sum())
                opportunities[2].observe(candidate)
                candidate, pairs = clear_one_b2_pair(candidate)
                b2_pairs += int(pairs.sum())
                opportunities[3].observe(candidate)
                opportunities[4].observe(candidate)
                assert torch.equal((candidate.long()*beta).sum(-1), (reference.long()*beta).sum(-1))
                peak = torch.maximum(peak, candidate.reshape(-1,5).amax(0))
                fired = model.layer1.lif1(conv(spike)).flatten(2).bool()[..., None, None]
                activity.observe(bank, candidate, fired)
                bank = candidate.masked_fill(fired, 0)
                scu.masked_fill_(fired, 0)
                reference.masked_fill_(fired, 0)
            print(f'samples={(batch+1)*2}/8', flush=True)
    report = dict(status='complete', indices=indices, steps=64, layer='layer1.conv1',
        scope='original weights/mapping; current forwarding+b0/b1/b2; eight active Macros only; all spatial positions including silent neurons',
        activity=activity.report(), cancellation_opportunities=[o.report() for o in opportunities],
        incoming_negative_units=incoming, intercepted_negative_units=intercepted,
        carry_interception_rate=intercepted/incoming if incoming else None,
        existing_b0_pairs=b0_pairs, b1_pairs=b1_pairs, b2_pairs=b2_pairs,
        forwarded_positive_units=forwarded, max_by_weight_plane_0_to_4=peak.tolist(),
        max_local_coarse_integer_error=0)
    assert opportunities[0].matched == b0_pairs
    assert opportunities[1].matched == b1_pairs
    assert opportunities[2].matched == b2_pairs
    out = ROOT/'IMC_ResNet/checkpoints/mr_bit_activity.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k != 'activity'}, indent=2))


if __name__ == '__main__':
    main()
