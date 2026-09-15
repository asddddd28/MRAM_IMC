"""First-layer input imbalance and exact Macro-state swap counterfactuals."""
import argparse,json,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,Subset
from torchvision import datasets,transforms
from spikingjelly.activation_based import functional
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from IMC_ResNet.models.finetuned_snn import load_finetuned,fuse_explicit_hidden,make_explicit_quantized_reference
from IMC_ResNet.models.cluster_backend import quantize_int5,MRStats
from IMC_ResNet.models.mr_pair_cancel import add_carries_with_positive_forward,clear_one_b1_pair,clear_one_b2_pair
from IMC_ResNet.models.macro_state_swap import swap_macro_pairs
from IMC_ResNet.models.macro_state_ram import MacroStateRAM


def main():
 p=argparse.ArgumentParser();p.add_argument('--rows',type=int,choices=[18,36],default=36)
 p.add_argument('--indices',type=int,nargs='+',default=[0,1,2,3,4,5,64,115])
 p.add_argument('--ram-restore',action='store_true')
 p.add_argument('--cancel-mode',choices=['full','none'],default='full')
 p.add_argument('--period',type=int,default=8);p.add_argument('--gap',type=float,default=4)
 p.add_argument('--variants',nargs='+',default=['baseline','fixed_active','adaptive_active','fixed_all','adaptive_all'])
 p.add_argument('--output',type=Path,default=ROOT/'IMC_ResNet/checkpoints/macro_state_swap.json')
 a=p.parse_args();torch.set_num_threads(4)
 assert 'baseline' in a.variants and a.period>0
 assert not a.ram_restore or all(n in ['baseline','fixed_active','fixed_all'] for n in a.variants)
 snn=ROOT/'SNN_ResNet10';model,_=load_finetuned(snn/'checkpoints/hard_reset_finetuned.pt',snn/'checkpoints/calibrated.json')
 model=make_explicit_quantized_reference(fuse_explicit_hidden(model));conv=model.layer1.conv1
 q,_=quantize_int5(conv.weight);flat=F.pad(q.flatten(1),(0,16*a.rows-288))
 bits=(((flat[...,None]&31)>>torch.arange(5))&1).float()
 packed=bits.reshape(32,16,a.rows,5).permute(1,2,0,3).reshape(16,a.rows,160)
 ds=datasets.FashionMNIST(snn/'data',train=False,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.2860,),(.3530,))]))
 loader=DataLoader(Subset(ds,a.indices),2);beta=torch.tensor([1,2,4,8,-16])
 stats={n:MRStats(5) for n in a.variants};swaps={n:0 for n in a.variants};errors={n:0 for n in a.variants}
 rawpeak={n:0 for n in a.variants};planepeak={n:torch.zeros(5,dtype=torch.int32) for n in a.variants}
 macropeak={n:torch.zeros(16,dtype=torch.int32) for n in a.variants}
 plane_counts=torch.zeros(16,5,dtype=torch.float64);input_counts=torch.zeros(16,dtype=torch.float64)
 input_den=0;count_den=0;basemr_sum=torch.zeros(16,dtype=torch.float64);basemr_den=0
 local_min=torch.zeros(16,dtype=torch.int64);local_max=local_min.clone();window_rates=[]
 ram_counts={n:dict(macro_state_records_read=0,macro_state_records_written=0,remapped_read_records=0) for n in a.variants}
 snapshots=[];start=time.perf_counter();raw_mr_max=0
 def save(status):
  result=dict(status=status,transport_note=('swap payload fields describe ownership changes included in normal restore, not extra traffic; access counts assume all 16 records read/written each step' if a.ram_restore else 'swap payload excludes address and handshake overhead'),ram_restore=a.ram_restore,cancel_mode=a.cancel_mode,ram_access=ram_counts,indices=a.indices,rows=a.rows,steps=64,period=a.period,gap=a.gap,completed_samples=sum(v['samples'] for v in snapshots),
   scope='layer1.conv1 all spatial; full MR+SCU pair swap before next fold; weights and physical inputs fixed; wide arithmetic; global neuron integer equality each step',
   rate_source='previous period sum of actual five-plane popcounts; physical input history does not move',
   input_spikes_per_step_macro=(input_counts/max(input_den,1)).tolist(),
   input_rate_per_assigned_row=(input_counts/max(input_den*a.rows,1)).tolist(),
   popcount_per_plane_step_macro=(plane_counts/max(count_den,1)).tolist(),
   baseline_mean_mr_per_macro=(basemr_sum/max(basemr_den,1)).tolist(),
   baseline_local_potential_min=local_min.tolist(),baseline_local_potential_max=local_max.tolist(),
   window_input_rates=window_rates,raw_no_cancel_max=raw_mr_max,batches=snapshots,
   variants={n:{**stats[n].report(), 'raw_pre_cancel_max':rawpeak[n],'plane_max':planepeak[n].tolist(),'macro_max':macropeak[n].tolist(),
     'swap_pairs':swaps[n],'payload_bits_at_5bit_mr':swaps[n]*2*5*(5+4),'payload_bits_at_6bit_mr':swaps[n]*2*5*(6+4),
     'max_global_integer_error':errors[n]} for n in a.variants})
  tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2),encoding='utf8');tmp.replace(a.output)
 with torch.no_grad():
  for batch,(x,_) in enumerate(loader):
   functional.reset_net(model);rams={};banks={};scus={};peaks={};recent=None;rawmr=rawscu=None;window_input=torch.zeros(16)
   for t in range(64):
    spike=model.stem(x);patches=F.pad(F.unfold(spike,3,padding=1),(0,0,0,16*a.rows-288));b,_,pos=patches.shape
    inp=patches.reshape(b,16,a.rows,pos).permute(1,0,3,2).reshape(16,b*pos,a.rows)
    counts=torch.bmm(inp,packed).reshape(16,b,pos,32,5).permute(1,3,2,0,4).contiguous().to(torch.int32)
    ins=inp.sum((1,2));input_counts+=ins.double();input_den+=b*pos;window_input+=ins
    plane_counts+=counts.sum((0,1,2)).double();count_den+=b*pos*32
    if not banks:
     for n in a.variants:banks[n]=torch.zeros_like(counts);scus[n]=torch.zeros_like(counts);peaks[n]=torch.zeros_like(counts)
     if a.ram_restore:
      rams={n:MacroStateRAM(counts) for n in a.variants}
     recent=torch.zeros_like(counts[...,0]);rawmr=torch.zeros_like(counts);rawscu=torch.zeros_like(counts)
    if a.ram_restore:
     for n in a.variants:
      active=8 if a.rows==36 and n.endswith('_active') else 16
      exchange=n!='baseline' and t>0 and t%a.period==0
      # Compare restore-address remap to the direct-swap reference exactly.
      expected_m,expected_s=banks[n],scus[n]
      if exchange:
       expected_m,expected_s,moved=swap_macro_pairs(expected_m,expected_s,recent,active_macros=active,adaptive=False)
       swaps[n]+=moved
      banks[n],scus[n]=rams[n].restore(exchange,active)
      assert torch.equal(banks[n],expected_m) and torch.equal(scus[n],expected_s)
    if not a.ram_restore and t and t%a.period==0:
     for n in a.variants:
      if n=='baseline':continue
      active=8 if a.rows==36 and n.endswith('_active') else 16
      banks[n],scus[n],moved=swap_macro_pairs(banks[n],scus[n],recent,active_macros=active,adaptive=n.startswith('adaptive'),gap=a.gap)
      swaps[n]+=moved
     recent.zero_()
    recent+=counts.sum(-1)
    z=rawscu+counts;rawmr+=z//16;rawscu=z%16;raw_mr_max=max(raw_mr_max,int(rawmr.max()))
    reference=((rawscu.long()+16*rawmr.long())*beta).sum((-1,-2))
    for n in a.variants:
     z=scus[n]+counts;carry,scus[n]=z//16,z%16
     rawpeak[n]=max(rawpeak[n],int((banks[n]+carry).max()))
     if a.cancel_mode=='full':
      bank,_,_=add_carries_with_positive_forward(banks[n],carry);bank,_=clear_one_b1_pair(bank);bank,_=clear_one_b2_pair(bank)
     else:bank=banks[n]+carry
     banks[n]=bank;stats[n].observe(bank);peaks[n]=torch.maximum(peaks[n],bank)
     planepeak[n]=torch.maximum(planepeak[n],bank.reshape(-1,5).amax(0));macropeak[n]=torch.maximum(macropeak[n],bank.amax((0,1,2,4)))
     local=((scus[n].long()+16*bank.long())*beta).sum(-1)
     error=int((local.sum(-1)-reference).abs().max());errors[n]=max(errors[n],error);assert error==0,(batch,t,n,error)
     if n=='baseline':
      basemr_sum+=bank.sum((0,1,2,4)).double();basemr_den+=b*pos*32*5
      local_min=torch.minimum(local_min,local.amin((0,1,2)));local_max=torch.maximum(local_max,local.amax((0,1,2)))
    fired=model.layer1.lif1(conv(spike)).flatten(2).bool()[...,None,None]
    rawmr.masked_fill_(fired,0);rawscu.masked_fill_(fired,0)
    for n in a.variants:banks[n].masked_fill_(fired,0);scus[n].masked_fill_(fired,0)
    if a.ram_restore:
     for n in a.variants:rams[n].writeback(banks[n],scus[n])
    if (t+1)%a.period==0:
     window_rates.append(dict(batch=batch,end_step=t+1,mean_input_per_step=(window_input/(b*pos*a.period)).tolist()));window_input.zero_()
   for n in a.variants:
    stats[n].finish_batch(peaks[n])
    if a.ram_restore:
     for key,value in rams[n].report().items():ram_counts[n][key]+=value
   snapshots.append(dict(samples=len(x),seconds=time.perf_counter()-start));save('running')
   print('BATCH '+json.dumps(dict(completed=sum(v['samples'] for v in snapshots),seconds=time.perf_counter()-start,maximum={n:stats[n].report()['max_mr'] for n in a.variants})),flush=True)
 save('complete')
 print('COMPLETE '+str(a.output),flush=True)

if __name__=='__main__':main()
