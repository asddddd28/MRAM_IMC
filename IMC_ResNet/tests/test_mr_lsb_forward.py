import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from IMC_ResNet.models.mr_pair_cancel import (
    add_carries_with_positive_forward, clear_one_b1_pair, clear_one_b2_pair,
)


def test_lsb_priority_and_occupied_destination():
    for old, expected in [
        ([0,0,0,1,0], [16,0,0,0,0]),
        ([16,0,0,1,0], [16,8,0,0,0]),
        ([16,8,4,1,0], [16,8,4,2,0]),
    ]:
        state = torch.tensor(old)
        out, _, _ = add_carries_with_positive_forward(state, torch.tensor([0,0,0,1,0]), False, toward_lsb=True)
        assert out.tolist() == expected and state.tolist() == old


def test_lsb_received_positive_bit_can_cancel_negative_carry():
    state = torch.tensor([0,0,0,1,0])
    out, removed, moved = add_carries_with_positive_forward(state,torch.tensor([0,0,0,1,1]),toward_lsb=True)
    assert out.count_nonzero()==0 and removed==1 and moved==1


def test_lsb_temporal_conservation_with_multi_carries_and_resets():
    torch.manual_seed(917)
    reference=torch.zeros((3,16,5),dtype=torch.int32)
    bank=reference.clone()
    beta=torch.tensor([1,2,4,8,-16],dtype=torch.int64)
    for t in range(192):
        carry=torch.randint(0,4,bank.shape,dtype=torch.int32)
        reference+=carry
        bank,_,_=add_carries_with_positive_forward(bank,carry,toward_lsb=True)
        bank,_=clear_one_b1_pair(bank)
        bank,_=clear_one_b2_pair(bank)
        assert torch.equal((bank.long()*beta).sum(-1),(reference.long()*beta).sum(-1))
        assert (bank>=0).all()
        if t%13==0:
            bank[0].zero_()
            reference[0].zero_()


def test_mr0b5_extension_and_existing_partner_priority():
    for old, expected in [([32,0,0,0,2],[0,0,0,0,0]),
                          ([32,0,0,4,3],[32,0,0,0,1]),
                          ([32,0,0,0,4],[32,0,0,0,4])]:
        state=torch.tensor(old)
        out,pairs=clear_one_b1_pair(state,include_mr0=True)
        assert out.tolist()==expected and state.tolist()==old
        assert int(pairs)==int(old!=expected)


def test_lsb_extended_b1_preserves_temporal_state():
    torch.manual_seed(918)
    reference=torch.zeros((3,16,5),dtype=torch.int32)
    bank=reference.clone()
    beta=torch.tensor([1,2,4,8,-16],dtype=torch.int64)
    for t in range(192):
        carry=torch.randint(0,4,bank.shape,dtype=torch.int32)
        reference+=carry
        bank,_,_=add_carries_with_positive_forward(bank,carry,toward_lsb=True)
        bank,_=clear_one_b1_pair(bank,include_mr0=True)
        bank,_=clear_one_b2_pair(bank)
        assert torch.equal((bank.long()*beta).sum(-1),(reference.long()*beta).sum(-1))
        assert (bank>=0).all()
        if t%13==0:
            bank[0].zero_()
            reference[0].zero_()
