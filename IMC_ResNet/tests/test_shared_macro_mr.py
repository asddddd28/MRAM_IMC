import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from IMC_ResNet.models.shared_macro_mr import SharedMacroMR, signed_bits, guaranteed_bound


def test_signed_width_boundaries():
    for lo,hi,bits in [(0,0,1),(-1,0,1),(-128,127,8),(-129,127,9),(-128,128,9),(-2304,2160,13)]:
        assert signed_bits(lo,hi)==bits
    assert guaranteed_bound(64)['shared_signed_bits']==13


def test_shared_bank_matches_full_integer_potential_with_resets():
    torch.manual_seed(919)
    bank=SharedMacroMR()
    total=torch.zeros((2,3,4,16,5),dtype=torch.int64)
    beta=torch.tensor([1,2,4,8,-16])
    for t in range(100):
        counts=torch.randint(0,37,total.shape,dtype=torch.int32)
        total+=counts
        bank.add_counts(counts)
        assert torch.equal(bank.reduce(),(total*beta).sum((-1,-2)))
        mask=torch.rand(2,3,4)<.1
        bank.fire(mask)
        total.masked_fill_(mask[...,None,None],0)
    bank.reset()
    assert bank.coarse is None


def test_generic_extreme_bounds_are_attainable_without_reset():
    for plane_pattern,expected in [([0,0,0,0,36],-2304),([36,36,36,36,0],2160)]:
        bank=SharedMacroMR()
        counts=torch.tensor(plane_pattern,dtype=torch.int32).expand(1,1,1,16,5)
        for _ in range(64): bank.add_counts(counts)
        assert (bank.coarse==expected).all()
