"""Evaluate SpikingJelly's official ``ann2snn.Converter`` on ResNet-10."""
import argparse, json, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from spikingjelly.activation_based import ann2snn, functional, neuron

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'ANN_ResNet10'))
from models import ResNet10

def main():
 p=argparse.ArgumentParser(); p.add_argument('--ann-checkpoint',type=Path,required=True); p.add_argument('--data',type=Path,default=ROOT/'data'); p.add_argument('--steps',type=int,default=8); p.add_argument('--calibration-batches',type=int,default=10); p.add_argument('--test-size',type=int,default=2000); p.add_argument('--mode',default='99.9%'); p.add_argument('--output',type=Path,default=ROOT/'checkpoints/official_ann2snn.json'); a=p.parse_args()
 tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.2860,),(.3530,))]); train=datasets.FashionMNIST(a.data,train=True,download=True,transform=tf); test=datasets.FashionMNIST(a.data,train=False,download=True,transform=tf)
 cal=DataLoader(Subset(train,range(min(a.calibration_batches*128,len(train)))),128,shuffle=False); loader=DataLoader(Subset(test,range(min(a.test_size,len(test)))),128,shuffle=False)
 ann=ResNet10(); ck=torch.load(a.ann_checkpoint,map_location='cpu',weights_only=False); ann.load_state_dict(ck['model'] if 'model' in ck else ck); ann.eval()
 converter=ann2snn.Converter(cal,device='cpu',mode=a.mode,fuse_flag=True); snn=converter(ann); snn.eval()
 nodes=[m for m in snn.modules() if isinstance(m,neuron.IFNode)]; count=[0.0]*len(nodes); totalv=[0]*len(nodes); hs=[]
 def make(i):
  def h(_,__,out): count[i]+=float(out.detach().sum()); totalv[i]+=out.numel()
  return h
 for i,m in enumerate(nodes): hs.append(m.register_forward_hook(make(i)))
 correct=total=0
 with torch.no_grad():
  for x,y in loader:
   functional.reset_net(snn); outs=[]
   for _ in range(a.steps): outs.append(snn(x))
   pred=torch.stack(outs).mean(0).argmax(1); correct+=(pred==y).sum().item(); total+=len(y)
 for h in hs:h.remove()
 result={'accuracy':correct/total,'samples':total,'steps':a.steps,'mode':a.mode,'converter':'spikingjelly.activation_based.ann2snn.Converter','fuse_conv_bn':True,'if_nodes':len(nodes),'firing_rate':{f'if_{i}':count[i]/totalv[i] if totalv[i] else 0 for i in range(len(nodes))},'device':'cpu'}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2),encoding='utf8'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
