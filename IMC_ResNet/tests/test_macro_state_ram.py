import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from IMC_ResNet.models.macro_state_ram import MacroStateRAM
from IMC_ResNet.models.macro_state_swap import swap_macro_pairs


def test_restore_remap_matches_direct_exchange_and_access_count():
 torch.manual_seed(15)
 template=torch.zeros(2,3,16,5,dtype=torch.int32)
 base=MacroStateRAM(template);ram=MacroStateRAM(template)
 m=template.clone();s=template.clone()
 for t in range(40):
  exchange=t>0 and t%8==0
  expected_m,expected_s=m,s
  if exchange:expected_m,expected_s,_=swap_macro_pairs(m,s,torch.zeros_like(m[...,0]),adaptive=False)
  got_m,got_s=ram.restore(exchange)
  assert torch.equal(got_m,expected_m) and torch.equal(got_s,expected_s)
  base.restore(False)
  m=got_m+torch.randint(0,4,template.shape,dtype=torch.int32);s=torch.randint(0,16,template.shape,dtype=torch.int32)
  m[0,0]=0;s[0,0]=0
  ram.writeback(m,s);base.writeback(m,s)
 for key in ['macro_state_records_read','macro_state_records_written']:
  assert ram.report()[key]==base.report()[key]==40*2*3*16
 assert ram.report()['remapped_read_records']==4*2*3*16


def test_remap_active_eight_keeps_other_slots():
 template=torch.arange(16,dtype=torch.int32).reshape(1,16,1).expand(1,16,5)
 ram=MacroStateRAM(template);ram.writeback(template,template)
 m,s=ram.restore(True,8)
 assert torch.equal(m[:,:4],template[:,4:8])
 assert torch.equal(m[:,4:8],template[:,:4])
 assert torch.equal(m[:,8:],template[:,8:]) and torch.equal(m,s)
