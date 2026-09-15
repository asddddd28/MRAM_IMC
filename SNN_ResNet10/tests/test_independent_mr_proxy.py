import sys
sys.path.insert(0,'MRAM_IMC')
import torch
from SNN_ResNet10.models.mr_qat import MRProxy
from IMC_ResNet.models.mr_pair_cancel import add_carries_with_positive_forward,clear_one_b1_pair,clear_one_b2_pair

def test_independent_proxy_temporal_exact_and_gradient():
 torch.manual_seed(4)
 conv=torch.nn.Conv2d(1,1,1)
 proxy=MRProxy(conv,target=29,independent_cancel=True)
 total=torch.zeros(2,1,3,16,5)
 bank=torch.zeros_like(total,dtype=torch.int32);scu=bank.clone()
 for step in range(30):
  counts=torch.randint(0,37,bank.shape,dtype=torch.int32)
  z=scu+counts;carry=z//16;scu=z%16
  bank,_,_=add_carries_with_positive_forward(bank,carry)
  bank,_=clear_one_b1_pair(bank);bank,_=clear_one_b2_pair(bank)
  total=total+counts
  continuous=(total/16).requires_grad_()
  observed=proxy.encode_independent(continuous)
  assert torch.equal(observed.detach(),bank.float())
  observed.sum().backward();assert torch.equal(continuous.grad,torch.ones_like(continuous))
  if step%5==0:
   # Exercise the actual hook reset on selected neuron positions.
   proxy.indices=torch.arange(3);proxy.total=total;spike=torch.tensor([[[[1.,0.,0.]]],[[[0.,1.,0.]]]])
   proxy.fire(None,None,spike)
   mask=spike.flatten(2).bool()[...,None,None]
   bank.masked_fill_(mask,0);scu.masked_fill_(mask,0);total=proxy.total
