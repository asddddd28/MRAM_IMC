"""Fine-tune the explicit SNN ResNet-10 with hard-reset IF neurons.

This is the trainable digital SNN baseline for the later Cluster model:
weights start from the ANN checkpoint, IF neurons use v_reset=0, and the loss
is applied to time-averaged logits through SpikingJelly surrogate gradients.
"""
import argparse, json, sys, time
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'models'))
sys.path.insert(0, str(ROOT.parent / 'ANN_ResNet10' / 'models'))
from snn_resnet10 import SNNResNet10, convert_ann_weights, reset
from resnet10 import ResNet10
from spikingjelly.activation_based import neuron


def load_state(path):
    obj = torch.load(path, map_location='cpu', weights_only=False)
    return obj['model'] if isinstance(obj, dict) and 'model' in obj else obj


def set_hard_reset(model):
    for module in model.modules():
        if isinstance(module, neuron.IFNode):
            module.v_reset = 0.0


def evaluate(model, loader, steps, device):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device); reset(model)
            logits = model(x.unsqueeze(0).repeat(steps, 1, 1, 1, 1)).mean(0)
            correct += int((logits.argmax(1) == y).sum()); total += y.numel()
    return correct / total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ann-checkpoint', type=Path, default=ROOT.parent/'ANN_ResNet10/checkpoints/resnet10_fashionmnist_5ep.pt')
    p.add_argument('--data', type=Path, default=ROOT/'data')
    p.add_argument('--output', type=Path, default=ROOT/'checkpoints/hard_reset_finetuned.pt')
    p.add_argument('--epochs', type=int, default=2); p.add_argument('--steps', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=64); p.add_argument('--train-size', type=int, default=10000)
    p.add_argument('--test-size', type=int, default=2000); p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--device', default='cpu'); a = p.parse_args(); device = torch.device(a.device)
    torch.manual_seed(20260912); torch.set_num_threads(min(8, torch.get_num_threads()))
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.2860,), (.3530,))])
    train = datasets.FashionMNIST(a.data, train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(a.data, train=False, download=True, transform=tf)
    train_loader = DataLoader(Subset(train, range(min(a.train_size, len(train)))), a.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(Subset(test, range(min(a.test_size, len(test)))), a.batch_size, shuffle=False, num_workers=0)
    model = SNNResNet10().to(device); info = convert_ann_weights(model, a.ann_checkpoint); set_hard_reset(model)
    # Start with calibrated activation scale when available; otherwise 1.0.
    cal_file = ROOT/'checkpoints/calibrated.json'
    if cal_file.exists():
        cal = json.loads(cal_file.read_text(encoding='utf8'))
        thresholds = cal.get('snn_if_thresholds', [])
        for i, node in enumerate(m for m in model.modules() if isinstance(m, neuron.IFNode)):
            if i < len(thresholds): node.v_threshold = max(float(thresholds[i]), 1e-3)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(); history=[]; start=time.perf_counter()
    before = evaluate(model, test_loader, a.steps, device)
    for epoch in range(a.epochs):
        model.train(); running=correct=total=0
        for batch, (x, y) in enumerate(train_loader, 1):
            x,y=x.to(device),y.to(device); reset(model); opt.zero_grad(set_to_none=True)
            logits = model(x.unsqueeze(0).repeat(a.steps,1,1,1,1)).mean(0)
            loss=criterion(logits,y); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            running += float(loss); correct += int((logits.detach().argmax(1)==y).sum()); total += y.numel()
            if batch % 25 == 0: print(f'epoch {epoch+1}/{a.epochs} batch {batch}/{len(train_loader)} loss={running/batch:.4f} train_acc={correct/total:.4f}', flush=True)
        val=evaluate(model,test_loader,a.steps,device); history.append({'epoch':epoch+1,'train_loss':running/len(train_loader),'train_accuracy':correct/total,'test_accuracy':val}); print(f'epoch {epoch+1}: test_acc={val:.4f}', flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True); torch.save({'model':model.state_dict(),'steps':a.steps,'hard_reset':True,'history':history},a.output)
    result={'before_accuracy':before,'after_accuracy':history[-1]['test_accuracy'],'epochs':a.epochs,'steps':a.steps,'train_size':len(train_loader.dataset),'test_size':len(test_loader.dataset),'lr':a.lr,'hard_reset':True,'weight_transfer':{'copied_tensors':len(info['copied']),'source_tensors':info['source_tensors']},'history':history,'seconds':time.perf_counter()-start,'checkpoint':str(a.output.resolve())}
    a.output.with_suffix('.json').write_text(json.dumps(result,indent=2),encoding='utf8'); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
