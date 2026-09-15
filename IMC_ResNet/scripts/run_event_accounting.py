import argparse,json,sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader,Subset
from torchvision import datasets,transforms
from spikingjelly.activation_based import ann2snn,functional,neuron
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT.parent/'ANN_ResNet10'/'models')); sys.path.insert(0,str(ROOT/'models'))
from resnet10 import ResNet10
from cluster_runtime import ClusterStats
def main():
 p=argparse.ArgumentParser(); p.add_argument('--ann-checkpoint',type=Path,required=True); p.add_argument('--data',type=Path,default=ROOT/'data'); p.add_argument('--steps',type=int,default=64); p.add_argument('--calibration-batches',type=int,default=10); p.add_argument('--test-size',type=int,default=2000); p.add_argument('--output',type=Path,default=ROOT/'checkpoints/cluster_t64.json'); a=p.parse_args()
 tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.2860,),(.3530,))]); train=datasets.FashionMNIST(a.data,train=True,download=True,transform=tf); test=datasets.FashionMNIST(a.data,train=False,download=True,transform=tf)
 cal=DataLoader(Subset(train,range(min(a.calibration_batches*128,len(train)))),128,shuffle=False); loader=DataLoader(Subset(test,range(min(a.test_size,len(test)))),128,shuffle=False)
 ann=ResNet10(); ck=torch.load(a.ann_checkpoint,map_location='cpu',weights_only=False); ann.load_state_dict(ck['model'] if 'model' in ck else ck); ann.eval(); snn=ann2snn.Converter(cal,device='cpu',mode='99.9%',fuse_flag=True)(ann); snn.eval()
 nodes=[m for m in snn.modules() if isinstance(m,neuron.IFNode)]; counts=[0]*len(nodes); neurons=[0]*len(nodes); hooks=[]
 def make(i):
  def h(_,__,out): counts[i]+=int(out.detach().sum()); neurons[i]+=out.numel()
  return h
 for i,n in enumerate(nodes): hooks.append(n.register_forward_hook(make(i)))
 correct=total=0
 with torch.no_grad():
  for x,y in loader:
   functional.reset_net(snn); outs=[snn(x) for _ in range(a.steps)]; pred=torch.stack(outs).mean(0).argmax(1); correct+=int((pred==y).sum()); total+=len(y)
 for h in hooks:h.remove()
 s=ClusterStats(steps=a.steps,samples=total)
 for i in range(len(nodes)): s.record_layer('if_'+str(i),counts[i],neurons[i])
 result={'accuracy':correct/total,'samples':total,'steps':a.steps,'cluster_model':'simplified event/TDP accounting','tdp_bits':5,'cluster_stats':s.as_dict(),'firing_rate':{'if_'+str(i):counts[i]/neurons[i] for i in range(len(nodes))}}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()

