"""Leakage + int5 fake-quantization fine-tuning for the explicit hard-reset SNN.

Leakage is applied before each IF charge, while reset remains hard (v_reset=0).
Convolution weights use per-output-channel int5 fake quantization with STE,
matching the later Cluster quantizer. This script is intentionally bounded and
saves one checkpoint/JSON per leakage candidate.
"""
import argparse, copy, json, sys, time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from spikingjelly.activation_based import neuron, surrogate, functional

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'models')); sys.path.insert(0,str(ROOT.parent/'ANN_ResNet10'/'models'))
from snn_resnet10 import SNNResNet10, convert_ann_weights, reset

class LeakageIFNode(neuron.IFNode):
    """IF node with leaky subthreshold state and unchanged hard reset."""
    def __init__(self, *args, leakage=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 < leakage <= 1: raise ValueError('leakage must be in (0,1]')
        self.leakage=float(leakage)
    def neuronal_charge(self, x):
        self.v = self.v * self.leakage + x
    def single_step_forward(self, x):
        return neuron.BaseNode.single_step_forward(self, x)

class FakeQuantConv2d(nn.Conv2d):
    """Conv2d whose forward weight is STE per-output-channel symmetric int5."""
    def _fake_weight(self):
        scale=self.weight.detach().flatten(1).abs().amax(1).clamp_min(1e-8)/15
        q=(self.weight/scale[:,None,None,None]).round().clamp(-15,15)
        return self.weight+(q*scale[:,None,None,None]-self.weight).detach()
    def forward(self,x):
        return F.conv2d(x,self._fake_weight(),self.bias,self.stride,self.padding,self.dilation,self.groups)

def replace_modules(module, leakage):
    for name, child in list(module.named_children()):
        if isinstance(child, neuron.IFNode):
            node=LeakageIFNode(v_threshold=child.v_threshold, v_reset=0., surrogate_function=surrogate.ATan(), leakage=leakage)
            setattr(module,name,node)
        elif isinstance(child, nn.Conv2d):
            repl=FakeQuantConv2d(child.in_channels,child.out_channels,child.kernel_size,child.stride,child.padding,child.dilation,child.groups,child.bias is not None)
            repl.load_state_dict(child.state_dict()); setattr(module,name,repl)
        else: replace_modules(child,leakage)

def thresholds(model, path):
    vals=json.loads(path.read_text())['snn_if_thresholds']
    for i,m in enumerate(x for x in model.modules() if isinstance(x,LeakageIFNode)):
        m.v_threshold=max(float(vals[i]),1e-3)

def eval_model(model, loader, steps):
    model.eval(); correct=total=0
    with torch.no_grad():
        for x,y in loader:
            reset(model); out=model(x.unsqueeze(0).repeat(steps,1,1,1,1)).mean(0)
            correct+=int((out.argmax(1)==y).sum()); total+=len(y)
    return correct/total

def train_candidate(args, leakage):
    torch.manual_seed(20260912); torch.set_num_threads(args.threads)
    tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.2860,),(.3530,))])
    train=datasets.FashionMNIST(args.data,train=True,download=False,transform=tf); test=datasets.FashionMNIST(args.data,train=False,download=False,transform=tf)
    train_loader=DataLoader(Subset(train,range(min(args.train_size,len(train)))),args.batch_size,shuffle=True)
    test_loader=DataLoader(Subset(test,range(min(args.test_size,len(test)))),args.batch_size,shuffle=False)
    model=SNNResNet10()
    if args.init_checkpoint is not None:
        state=torch.load(args.init_checkpoint,map_location='cpu')
        if isinstance(state,dict) and 'model' in state: state=state['model']
        model.load_state_dict(state)
    else:
        convert_ann_weights(model,args.ann_checkpoint)
    replace_modules(model,leakage); thresholds(model,args.thresholds)
    before=eval_model(model,test_loader,args.steps); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4); lossfn=nn.CrossEntropyLoss(); history=[]; start=time.perf_counter()
    for epoch in range(args.epochs):
        model.train(); total=correct=0; running=0.
        for x,y in train_loader:
            reset(model); opt.zero_grad(set_to_none=True); logits=model(x.unsqueeze(0).repeat(args.steps,1,1,1,1)).mean(0); loss=lossfn(logits,y); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); running+=float(loss);correct+=int((logits.detach().argmax(1)==y).sum());total+=len(y)
        val=eval_model(model,test_loader,args.steps); history.append({'epoch':epoch+1,'train_loss':running/len(train_loader),'train_accuracy':correct/total,'test_accuracy':val}); print(f'leakage={leakage} epoch={epoch+1} val={val:.4f}',flush=True)
    out=args.output_dir/f'leakage_{leakage:.3f}_qat.pt'; torch.save({'model':model.state_dict(),'steps':args.steps,'hard_reset':True,'leakage':leakage,'quantization':'per-output-channel symmetric int5 STE','history':history},out)
    result={'leakage':leakage,'before_accuracy':before,'after_accuracy':history[-1]['test_accuracy'],'history':history,'checkpoint':str(out.resolve()),'seconds':time.perf_counter()-start,'quantization':'int5 STE'}
    out.with_suffix('.json').write_text(json.dumps(result,indent=2),encoding='utf8'); return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--leakages',type=float,nargs='+',default=[1.,.99,.97,.95]); p.add_argument('--epochs',type=int,default=1);p.add_argument('--steps',type=int,default=16);p.add_argument('--train-size',type=int,default=5000);p.add_argument('--test-size',type=int,default=2000);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--threads',type=int,default=4);p.add_argument('--lr',type=float,default=2e-4);p.add_argument('--data',type=Path,default=ROOT/'data');p.add_argument('--ann-checkpoint',type=Path,default=ROOT.parent/'ANN_ResNet10/checkpoints/resnet10_fashionmnist_5ep.pt');p.add_argument('--init-checkpoint',type=Path,default=None);p.add_argument('--thresholds',type=Path,default=ROOT/'checkpoints/calibrated.json');p.add_argument('--output-dir',type=Path,default=ROOT/'checkpoints/leakage_qat');a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True); results=[train_candidate(a,l) for l in a.leakages];(a.output_dir/'summary.json').write_text(json.dumps(results,indent=2),encoding='utf8');print(json.dumps(results,indent=2))
if __name__=='__main__':main()

