import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from IMC_ResNet.models.macro_state_swap import swap_macro_pairs
from IMC_ResNet.models.mr_pair_cancel import add_carries_with_positive_forward,clear_one_b1_pair,clear_one_b2_pair


def test_pair_swap_is_permutation_and_moves_full_state():
 mr=torch.arange(16,dtype=torch.int32).reshape(1,16,1).expand(1,16,5).clone()
 scu=(mr%16).clone();rate=torch.arange(16).reshape(1,16)
 m,s,n=swap_macro_pairs(mr,scu,rate,gap=0)
 assert n==8
 assert torch.equal(m,mr.flip(-2)) and torch.equal(s,scu.flip(-2))
 assert torch.equal(rate,torch.arange(16).reshape(1,16))


def test_swap_then_updates_preserve_neuron_sum_and_reset():
 torch.manual_seed(20260915);beta=torch.tensor([1,2,4,8,-16])
 ref_m=torch.zeros(2,3,16,5,dtype=torch.int32);ref_s=ref_m.clone();m=ref_m.clone();s=ref_m.clone()
 rate=torch.rand(2,3,16)
 for t in range(24):
  if t%4==0:m,s,_=swap_macro_pairs(m,s,rate,adaptive=(t%8==0),gap=0)
  c=torch.randint(0,37,m.shape,dtype=torch.int32)
  z=ref_s+c;ref_m+=z//16;ref_s=z%16
  z=s+c;carry,s=z//16,z%16
  m,_,_=add_carries_with_positive_forward(m,carry);m,_=clear_one_b1_pair(m);m,_=clear_one_b2_pair(m)
  value=lambda a,b:((16*a.long()+b.long())*beta).sum((-1,-2))
  assert torch.equal(value(m,s),value(ref_m,ref_s))
  mask=(torch.rand(2,3)<.1)[...,None,None]
  for state in [m,s,ref_m,ref_s]:state.masked_fill_(mask,0)


def test_active_only_and_no_gain_no_swap():
 m=torch.zeros(1,16,5,dtype=torch.int32);s=m.clone();m[:,8:]=99
 rate=torch.arange(16).reshape(1,16)
 out,_,n=swap_macro_pairs(m,s,rate,active_macros=8)
 assert n==0 and torch.equal(out,m)
