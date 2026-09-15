"""Counterfactual only: split each of layer1.conv1's 8 macros into two.

Preserves weights, inputs, IF decisions and signed integer sum. Does not
modify deployed mapping or claim hardware input routing supports the layout.
"""
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
from IMC_ResNet.models.cluster_backend import ClusterAccumulator, quantize_int5


def main():
    torch.set_num_threads(4)
    snn = ROOT/'SNN_ResNet10'
    ds = datasets.FashionMNIST(snn/'data', train=False, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((.2860,), (.3530,))]))
    indices = [0, 1, 2, 3, 4, 5, 64, 115]
    loader = DataLoader(Subset(ds, indices), 2)
    results = {'scope': 'layer1.conv1 only; same digital IF reset masks; all spatial positions; T64',
               'indices': indices, 'hardware_routing_validated': False, 'models': {}}
    for name, path in [('original', snn/'checkpoints/hard_reset_finetuned.pt'),
                       ('mr_qat', snn/'checkpoints/mr_qat_t64_balanced/alpha_10.pt')]:
        model, meta = load_finetuned(path, snn/'checkpoints/calibrated.json')
        model = make_explicit_quantized_reference(fuse_explicit_hidden(model))
        conv = model.layer1.conv1
        q, _ = quantize_int5(conv.weight)
        assert q.flatten(1).shape == (32, 288)
        bits = (((q.flatten(1)[..., None] & 31) >> torch.arange(5)) & 1).float()
        packed = {}
        for layout, rows in [('original_8x36', 36), ('split_16x18', 18)]:
            planes = F.pad(bits, (0, 0, 0, 16 * rows - 288))
            packed[layout] = planes.reshape(32, 16, rows, 5).permute(1, 2, 0, 3).reshape(16, rows, 160)
        banks = {key: ClusterAccumulator(6) for key in packed}
        max_error = 0
        with torch.no_grad():
            for batch, (x, _) in enumerate(loader):
                functional.reset_net(model)
                for bank in banks.values():
                    bank.reset()
                for t in range(64):
                    spikes = model.stem(x)
                    patches = F.unfold(spikes, 3, padding=1)
                    b, _, p = patches.shape
                    for layout, rows in [('original_8x36', 36), ('split_16x18', 18)]:
                        padded = F.pad(patches, (0, 0, 0, 16 * rows - 288))
                        inp = padded.reshape(b, 16, rows, p).permute(1, 0, 3, 2).reshape(16, b*p, rows)
                        counts = torch.bmm(inp, packed[layout]).reshape(16, b, p, 32, 5).permute(1, 3, 2, 0, 4).contiguous().to(torch.int32)
                        banks[layout].add_counts(counts)
                    error = int((banks['original_8x36'].reduce() - banks['split_16x18'].reduce()).abs().max())
                    max_error = max(max_error, error)
                    assert error == 0
                    fired = model.layer1.lif1(conv(spikes)).flatten(2).bool()
                    for bank in banks.values():
                        bank.fire(fired)
                print(f'{name} samples={(batch+1)*2}/8', flush=True)
        for bank in banks.values():
            bank.reset()
        results['models'][name] = {'metadata': meta, 'max_signed_integer_error': max_error,
                                  **{key: bank.stats.report() for key, bank in banks.items()}}
        out = ROOT/'IMC_ResNet/checkpoints/mr_remap_counterfactual.json'
        out.write_text(json.dumps(results, indent=2), encoding='utf8')
    print(json.dumps({name: {key: value['max_mr'] for key, value in row.items() if key in packed}
                      for name, row in results['models'].items()}, indent=2))


if __name__ == '__main__':
    main()
