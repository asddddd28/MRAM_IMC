"""Read-only per-Macro input and weight-gated reception across all layers."""
import argparse,json,math,sys
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,Subset
from torchvision import datasets,transforms
from spikingjelly.activation_based import functional
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from IMC_ResNet.models.finetuned_snn import load_finetuned,fuse_explicit_hidden,make_explicit_quantized_reference,explicit_targets
from IMC_ResNet.models.cluster_backend import quantize_int5


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'IMC_ResNet/checkpoints/macro_input_activity.json');a=p.parse_args()
 torch.set_num_threads(4);snn=ROOT/'SNN_ResNet10'
 model,_=load_finetuned(snn/'checkpoints/hard_reset_finetuned.pt',snn/'checkpoints/calibrated.json');model=make_explicit_quantized_reference(fuse_explicit_hidden(model))
 indices=[0,1,2,3,4,5,64,115]+list(range(128,136))
 ds=datasets.FashionMNIST(snn/'data',train=False,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.2860,),(.3530,))]));loader=DataLoader(Subset(ds,indices),2)
 state={};handles=[]
 def install(name,c):
  k=c.weight.flatten(1).shape[1];folds=math.ceil(k/576);q,_=quantize_int5(c.weight)
  bits=(((F.pad(q.flatten(1),(0,folds*576-k))[...,None]&31)>>torch.arange(5))&1).float()
  coeff=bits.reshape(c.out_channels,folds,16,36,5).mean(0)
  valid=F.pad(torch.ones(k),(0,folds*576-k)).reshape(folds,16,36).sum(-1)
  row=dict(input=torch.zeros(folds,16,dtype=torch.float64),gated=torch.zeros(16,5,dtype=torch.float64),den=0,valid=valid,windows=[])
  row['local_windows']=[]
  state[name]=row;window_node=None;window=torch.zeros(folds,16,dtype=torch.float64);windowden=0;step=0
  def hook(m,inputs,output):
   nonlocal window,windowden,step,window_node
   patches=F.unfold(inputs[0],c.kernel_size,padding=c.padding,stride=c.stride);b,_,pos=patches.shape
   local=F.pad(patches,(0,0,0,folds*576-k)).reshape(b,folds,16,36,pos).sum((1,3)).permute(0,2,1)
   window_node=local if window_node is None else window_node+local
   summed=F.pad(patches.sum((0,2)),(0,folds*576-k)).reshape(folds,16,36)
   rates=summed.sum(-1).double();row['input']+=rates;row['den']+=b*pos
   row['gated']+=(summed[...,None]*coeff).sum((0,2)).double()
   window+=rates;windowden+=b*pos;step+=1
   if step%8==0:
    active=valid.sum(0)>0
    local_rates=window_node[...,active]/valid.sum(0)[active].clamp_min(1)
    avg=local_rates.mean(-1);nonzero=avg>0
    chosen=local_rates[nonzero];mean=avg[nonzero]
    cv=chosen.std(-1,unbiased=False)/mean;ratio=chosen.amax(-1)/mean
    row['local_windows'].append(dict(nonzero_neuron_windows=int(nonzero.sum()),all_neuron_windows=nonzero.numel(),cv_sum=float(cv.sum()),max_to_mean_sum=float(ratio.sum()),max_to_mean_gt2=int((ratio>2).sum()),any_inactive_macro=int((chosen==0).any(-1).sum())))
    window_node=None
    row['windows'].append((window.sum(0)/windowden).tolist());window.zero_();windowden=0
  handles.append(c.register_forward_hook(hook))
 for name,_,_ in explicit_targets(model):install(name,model.get_submodule(name))
 with torch.no_grad():
  for i,(x,_) in enumerate(loader):
   functional.reset_net(model)
   for t in range(64):model(x.unsqueeze(0))
   print(f'samples={(i+1)*2}/{len(indices)}',flush=True)
 for h in handles:h.remove()
 layers={}
 for n,r in state.items():
  inp=r['input']/r['den'];macro=inp.sum(0);valid=r['valid'].sum(0)
  normalized=macro/valid.clamp_min(1);active=valid>0;v=normalized[active];g=r['gated']/r['den']
  # Ratios exclude empty padding macros; normalized metric accounts for partial folds.
  local={k:sum(w[k] for w in r['local_windows']) for k in r['local_windows'][0]}
  local['mean_cv']=local['cv_sum']/max(1,local['nonzero_neuron_windows'])
  local['mean_max_to_mean']=local['max_to_mean_sum']/max(1,local['nonzero_neuron_windows'])
  local['fraction_max_to_mean_gt2']=local['max_to_mean_gt2']/max(1,local['nonzero_neuron_windows'])
  layers[n]=dict(local_8step_input_imbalance=local,assigned_rows_across_folds=valid.tolist(),input_per_step_macro=macro.tolist(),input_per_assigned_row=normalized.tolist(),input_per_step_fold_macro=inp.tolist(),
   gated_popcount_per_step_macro_plane=g.tolist(),active_macro_input_rate_max_min=float(v.max()/v.min()) if float(v.min())>0 else None,
   active_macro_input_rate_cv=float(v.std(unbiased=False)/v.mean()),window_input_per_step_macro=r['windows'])
 a.output.write_text(json.dumps(dict(status='complete',indices=indices,steps=64,scope='all 11 convolutions; original weights and mapping; activity only; no state swaps',layers=layers),indent=2),encoding='utf8')
 print(json.dumps({n:dict(ratio=r['active_macro_input_rate_max_min'],cv=r['active_macro_input_rate_cv']) for n,r in layers.items()}),flush=True)

if __name__=='__main__':main()

