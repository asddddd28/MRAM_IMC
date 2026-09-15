"""Bounded T=64 MR-aware QAT with matched zero-penalty controls.

Freeze BN statistics and train only layer1.conv1, the observed overflow layer.
All hidden weights are fake-quantized AFTER BN folding. Full temporal BPTT;
spatially sampled banks reduce CPU memory, not sequence length.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from spikingjelly.activation_based import neuron

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from SNN_ResNet10.models.snn_resnet10 import reset
from SNN_ResNet10.models.mr_qat import QATConv2d, MRProxy, quantized_weight
from IMC_ResNet.models.finetuned_snn import load_finetuned, fuse_explicit_hidden, explicit_targets


def evaluate(model, loader):
    model.eval()
    correct = {16: 0, 64: 0}
    with torch.no_grad():
        for x, y in loader:
            reset(model)
            total = 0
            for t in range(1, 65):
                total = total + model(x.unsqueeze(0))[0]
                if t in correct:
                    correct[t] += int((total.argmax(1) == y).sum())
    reset(model)
    return {str(t): n / len(loader.dataset) for t, n in correct.items()}


def export(model, args, alpha, history, path):
    reset(model)
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for name, conv in model.named_modules():
        if isinstance(conv, QATConv2d):
            q, scale = quantized_weight(conv.weight)
            state[name + '.weight'] = (q * scale[:, None, None, None]).detach()
    torch.save({'model': state, 'latent_model': model.state_dict(),
                'hard_reset': True, 'hidden_bn_fused': True,
                'leakage': args.leakage, 'steps': args.steps,
                'detach_reset_gradient': getattr(args, 'detach_reset', True),
                'quantization': 'fused-hidden-int5-materialized',
                'if_thresholds': [m.v_threshold for m in model.modules() if isinstance(m, neuron.IFNode)],
                'mr_qat': {'layer': 'layer1.conv1', 'target': args.target,
                           'positions': args.positions, 'alpha': alpha,
                           'shared_bits': getattr(args, 'shared_bits', None), 'shared_margin': 8 if getattr(args, 'shared_bits', None) else None,
                           'beta': args.beta if alpha else 0, 'gamma': args.gamma if alpha else 0,
                           'balance_mr_gradient': getattr(args, 'balance_mr_gradient', False),
                           'surrogate': 'hard two-complement bits + Gaussian categorical STE'},
                'history': history}, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--init-checkpoint', type=Path, default=ROOT/'checkpoints/leakage_qat_from_hard/leakage_0.990_qat.pt')
    p.add_argument('--thresholds', type=Path, default=ROOT/'checkpoints/calibrated.json')
    p.add_argument('--data', type=Path, default=ROOT/'data')
    p.add_argument('--output-dir', type=Path, default=ROOT/'checkpoints/mr_qat_t64')
    p.add_argument('--alphas', nargs='+', type=float, default=[0., 10.])
    p.add_argument('--beta', type=float, default=1.)
    p.add_argument('--gamma', type=float, default=.1)
    p.add_argument('--shared-bits', type=int, choices=range(6,17), help='Use signed shared-MR overflow loss instead of unsigned plane loss')
    p.add_argument('--target', type=float, default=56)
    p.add_argument('--leakage', type=float, default=.99)
    p.add_argument('--positions', type=int, default=64)
    p.add_argument('--steps', type=int, default=64)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--train-size', type=int, default=128)
    p.add_argument('--test-size', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--detach-reset', action=argparse.BooleanOptionalAction, default=True,
                   help='Detach IF reset in backward only to avoid long-window gradient explosion')
    p.add_argument('--balance-mr-gradient', action='store_true',
                   help='Match MR and CE gradient norms, detached multiplier capped at 1e5')
    args = p.parse_args()
    if min(args.steps, args.epochs, args.train_size, args.test_size, args.batch_size, args.threads) < 1:
        p.error('sizes must be positive')
    if not 0 < args.leakage <= 1 or args.target <= 0 or args.positions < 0 or min(args.alphas + [args.beta, args.gamma]) < 0:
        p.error('invalid leakage/loss/sampling configuration')
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.2860,), (.3530,))])
    train = datasets.FashionMNIST(args.data, train=True, download=False, transform=tf)
    test = datasets.FashionMNIST(args.data, train=False, download=False, transform=tf)
    test_loader = DataLoader(Subset(test, range(min(args.test_size, len(test)))), 32)
    source, metadata = load_finetuned(args.init_checkpoint, args.thresholds)
    base = fuse_explicit_hidden(source)
    for node in base.modules():
        if isinstance(node, neuron.IFNode):
            node.leakage = args.leakage
            node.detach_reset = args.detach_reset
    for name, _, _ in explicit_targets(base):
        conv = base.get_submodule(name)
        qat = QATConv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                        conv.padding, bias=conv.bias is not None)
        qat.load_state_dict(conv.state_dict())
        base.set_submodule(name, qat)
    base.eval()
    baseline = evaluate(base, test_loader)
    export(base, args, 0, [], args.output_dir/'baseline.pt')
    report = {'status': 'running', 'source': metadata,
              'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'baseline_accuracy': baseline, 'candidates': [],
              'scope': 'layer1.conv1 only; sampled spatial MR banks; T=64; exploratory test-prefix evaluation'}
    def save():
        path = args.output_dir/'summary.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2), encoding='utf8')
        tmp.replace(path)
    save()
    print('baseline ' + json.dumps(baseline), flush=True)
    for alpha in args.alphas:
        torch.manual_seed(20260913)
        model = copy.deepcopy(base).eval()
        for name, param in model.named_parameters():
            param.requires_grad_(name == 'layer1.conv1.weight')
        proxy = MRProxy(model.layer1.conv1, args.target, args.positions, shared_bits=args.shared_bits)
        handles = [model.layer1.conv1.register_forward_hook(proxy.observe),
                   model.layer1.lif1.register_forward_hook(proxy.fire)]
        opt = torch.optim.AdamW([model.layer1.conv1.weight], lr=args.lr, weight_decay=0)
        train_loader = DataLoader(Subset(train, range(min(args.train_size, len(train)))), args.batch_size,
                                  shuffle=True, generator=torch.Generator().manual_seed(20260913))
        history = []
        start = time.perf_counter()
        for epoch in range(args.epochs):
            for batch, (x, y) in enumerate(train_loader):
                reset(model)
                proxy.reset()
                opt.zero_grad(set_to_none=True)
                logits = model(x.unsqueeze(0).expand(args.steps, -1, -1, -1, -1)).mean(0)
                cls = F.cross_entropy(logits, y)
                terms = proxy.losses()
                mr_loss = alpha * terms['overflow']
                if alpha:
                    mr_loss = mr_loss + args.beta * terms['peak'] + args.gamma * terms['quiet']
                balance = 1.
                ce_norm = mr_norm = None
                if alpha and args.balance_mr_gradient:
                    ce_grad = torch.autograd.grad(cls, model.layer1.conv1.weight, retain_graph=True)[0]
                    mr_grad = torch.autograd.grad(mr_loss, model.layer1.conv1.weight, retain_graph=True)[0]
                    ce_norm, mr_norm = float(ce_grad.norm()), float(mr_grad.norm())
                    balance = min(1e5, ce_norm / max(mr_norm, 1e-12))
                loss = cls + balance * mr_loss
                loss.backward()
                grad = float(model.layer1.conv1.weight.grad.norm())
                if not torch.isfinite(loss) or not torch.isfinite(model.layer1.conv1.weight.grad).all():
                    raise FloatingPointError('non-finite loss/gradient')
                torch.nn.utils.clip_grad_norm_([model.layer1.conv1.weight], 1.)
                opt.step()
                row = {'epoch': epoch + 1, 'batch': batch + 1, 'classification_loss': float(cls.detach()),
                       **{k: float(v.detach()) for k, v in terms.items()}, 'gradient_norm': grad,
                       'mr_gradient_multiplier': balance, 'ce_gradient_norm': ce_norm,
                       'mr_gradient_norm_before_balance': mr_norm, **proxy.report()}
                history.append(row)
                if batch % 8 == 0 or batch + 1 == len(train_loader):
                    print(f'alpha={alpha} epoch={epoch+1} batch={batch+1}/{len(train_loader)} '
                          f'CE={row["classification_loss"]:.4f} MR={row["max_mr"]} '
                          f'overflow={row["overflow"]:.6f} grad={grad:.4g} elapsed={time.perf_counter()-start:.1f}s', flush=True)
        for handle in handles:
            handle.remove()
        proxy.reset()
        accuracy = evaluate(model, test_loader)
        checkpoint = args.output_dir/f'alpha_{alpha:g}.pt'
        export(model, args, alpha, history, checkpoint)
        report['candidates'].append({'alpha': alpha, 'accuracy': accuracy, 'history': history,
                                     'checkpoint': str(checkpoint.resolve()), 'seconds': time.perf_counter()-start})
        save()
        print(f'alpha={alpha} accuracy={accuracy}', flush=True)
    report['status'] = 'complete'
    save()


if __name__ == '__main__':
    main()
