"""Calibrate ANN activation scales, then evaluate the transferred SNN.

This is a diagnostic calibration pass: ANN ReLU percentiles become IF
thresholds. It also records per-layer spike rates so threshold choices can be
reviewed instead of treated as a black box.
"""
import argparse, json, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

snn_mod = load_module("snn_model", ROOT / "models" / "snn_resnet10.py")
ann_mod = load_module("ann_model", ROOT.parent / "ANN_ResNet10" / "models" / "resnet10.py")
SNNResNet10, convert_ann_weights, reset = snn_mod.SNNResNet10, snn_mod.convert_ann_weights, snn_mod.reset
ResNet10 = ann_mod.ResNet10
from spikingjelly.activation_based import neuron


def load_state(path):
    c = torch.load(path, map_location="cpu", weights_only=False)
    return c["model"] if "model" in c else c


def calibrate_ann(model, loader, batches, percentile):
    stats, handles = {}, []
    def hook(name):
        def fn(_, __, out):
            v = out.detach().flatten()
            stats.setdefault(name, []).append(v[torch.randperm(v.numel())[:min(20000, v.numel())]].cpu())
        return fn
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.ReLU): handles.append(m.register_forward_hook(hook(name)))
    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            model(x)
            if i + 1 >= batches: break
    for h in handles: h.remove()
    return {n: float(torch.quantile(torch.cat(v), percentile)) for n, v in stats.items()}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--ann-checkpoint',type=Path,required=True); p.add_argument('--data',type=Path,default=ROOT/'data'); p.add_argument('--batch-size',type=int,default=128); p.add_argument('--steps',type=int,default=16); p.add_argument('--calibration-batches',type=int,default=10); p.add_argument('--percentile',type=float,default=.999); p.add_argument('--test-size',type=int,default=2000); p.add_argument('--output',type=Path,default=ROOT/'checkpoints/calibrated.json'); a=p.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); tf=transforms.ToTensor()
    train=datasets.FashionMNIST(a.data,train=True,download=True,transform=tf); test=datasets.FashionMNIST(a.data,train=False,download=True,transform=tf)
    cal=DataLoader(Subset(train,range(min(a.calibration_batches*a.batch_size,len(train)))),a.batch_size,shuffle=False,num_workers=0)
    test=Subset(test,range(min(a.test_size,len(test)))); loader=DataLoader(test,a.batch_size,shuffle=False,num_workers=0)
    ann=ResNet10().to(device); ann.load_state_dict(load_state(a.ann_checkpoint)); scales=calibrate_ann(ann,cal,a.calibration_batches,a.percentile)
    # ReLU order in the ANN maps to stem IF, then block output IFs. The first
    # block IF is also calibrated from the first block ReLU as a practical approximation.
    ordered=list(scales.values()); snn=SNNResNet10(v_threshold=1.0).to(device); info=convert_ann_weights(snn,a.ann_checkpoint)
    if_nodes=[m for m in snn.modules() if isinstance(m,neuron.IFNode)]
    thresholds=[]
    for i,m in enumerate(if_nodes):
        t=max(1e-3, ordered[min(i,len(ordered)-1)]) if ordered else 1.0; m.v_threshold=t; thresholds.append(t)
    counts={f'if_{i}':0.0 for i in range(len(if_nodes))}; totals={k:0 for k in counts}; handles=[]
    for i,m in enumerate(if_nodes):
        def spike_hook(_, __, out, i=i):
            counts[f'if_{i}'] += float(out.detach().sum())
            totals[f'if_{i}'] += out.numel()
        handles.append(m.register_forward_hook(spike_hook))
    correct=total=0
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device),y.to(device); reset(snn); out=snn(x.unsqueeze(0).repeat(a.steps,1,1,1,1)); correct+=(out.mean(0).argmax(1)==y).sum().item(); total+=y.numel()
    for h in handles:h.remove()
    result={'accuracy':correct/total,'samples':total,'steps':a.steps,'calibration_batches':a.calibration_batches,'percentile':a.percentile,'ann_relu_thresholds':scales,'snn_if_thresholds':thresholds,'firing_rate':{k:counts[k]/totals[k] if totals[k] else 0 for k in counts},'weight_transfer':{'copied_tensors':len(info['copied']),'source_tensors':info['source_tensors']},'device':str(device)}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2),encoding='utf8'); print(json.dumps(result,indent=2))

if __name__=='__main__': main()




