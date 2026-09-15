import sys
sys.path.insert(0,'MRAM_IMC')
import torch
from IMC_ResNet.models.independent_cancel_mr import IndependentCancelMR
from IMC_ResNet.models.cluster_backend import ClusterAccumulator


def test_independent_bank_fold_reduce_reset():
 torch.manual_seed(9)
 bank=IndependentCancelMR();reference=ClusterAccumulator()
 for step in range(20):
  for fold in range(4):
   counts=torch.randint(0,37,(2,3,4,16,5),dtype=torch.int32)
   bank.add_counts(counts);reference.add_counts(counts)
   assert torch.equal(bank.reduce(),reference.reduce())
  fired=torch.rand(2,3,4)<.1
  bank.fire(fired);reference.fire(fired)
  assert torch.equal(bank.reduce(),reference.reduce())
 bank.reset();reference.reset()
 assert bank.report()['max_local_integer_error']==0
 assert bank.report()['fold_observations']==80
