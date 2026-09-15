import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from IMC_ResNet.models.mr_pair_cancel import (
    clear_one_b1_pair, add_carries_with_b0_cancel,
    add_carries_with_positive_forward,
)


def value(state):
    return (state.long() * torch.tensor([1, 2, 4, 8, -16])).sum(-1)


def test_each_b1_pair_preserves_value_and_input():
    for b in (3, 2, 1):
        state = torch.zeros(5, dtype=torch.int32)
        state[4] = 2
        state[b] = 1 << (5 - b)
        before = state.clone()
        out, pairs = clear_one_b1_pair(state)
        assert out.count_nonzero() == 0 and pairs == 1
        assert torch.equal(state, before)


def test_b1_arbitration_and_excluded_mr0():
    state = torch.tensor([32, 16, 8, 4, 3])
    out, pairs = clear_one_b1_pair(state)
    assert out.tolist() == [32, 16, 8, 0, 1] and pairs == 1
    for state in (torch.tensor([32, 0, 0, 0, 2]), torch.tensor([0, 16, 8, 4, 4])):
        out, pairs = clear_one_b1_pair(state)
        assert torch.equal(out, state) and pairs == 0


def test_extension_preserves_every_macro_through_updates_and_resets():
    torch.manual_seed(914)
    reference = torch.zeros((3, 16, 5), dtype=torch.int32)
    banks = [reference.clone(), reference.clone()]
    for t in range(192):
        carry = torch.randint(0, 4, reference.shape, dtype=torch.int32)
        reference += carry
        banks[0], _ = add_carries_with_b0_cancel(banks[0], carry)
        banks[1], _, _ = add_carries_with_positive_forward(banks[1], carry)
        for i in range(2):
            banks[i], _ = clear_one_b1_pair(banks[i])
            assert torch.equal(value(banks[i]), value(reference))
            assert (banks[i] >= 0).all()
        if t % 13 == 0:
            for state in [reference] + banks:
                state[0].zero_()
