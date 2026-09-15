import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from IMC_ResNet.models.mr_pair_cancel import (
    clear_one_b1_pair, clear_one_b2_pair, add_carries_with_b0_cancel,
    add_carries_with_positive_forward,
)


def value(state):
    return (state.long() * torch.tensor([1, 2, 4, 8, -16])).sum(-1)


def test_each_b2_connection_and_one_hot_priority():
    for state, expected in [
        ([0, 0, 0, 8, 4], [0, 0, 0, 0, 0]),
        ([0, 0, 16, 0, 4], [0, 0, 0, 0, 0]),
        ([0, 0, 16, 8, 7], [0, 0, 16, 0, 3]),
    ]:
        original = torch.tensor(state)
        out, pairs = clear_one_b2_pair(original)
        assert out.tolist() == expected and pairs == 1
        assert original.tolist() == state and value(out) == value(original)


def test_b2_excludes_other_planes_and_requires_negative_bit():
    for state in ([64, 32, 0, 0, 4], [0, 0, 16, 8, 8]):
        original = torch.tensor(state)
        out, pairs = clear_one_b2_pair(original)
        assert torch.equal(out, original) and pairs == 0


def test_b2_temporal_conservation_with_and_without_forwarding():
    torch.manual_seed(915)
    reference = torch.zeros((3, 16, 5), dtype=torch.int32)
    banks = [reference.clone(), reference.clone()]
    for t in range(192):
        carry = torch.randint(0, 4, reference.shape, dtype=torch.int32)
        reference += carry
        banks[0], _ = add_carries_with_b0_cancel(banks[0], carry)
        banks[1], _, _ = add_carries_with_positive_forward(banks[1], carry)
        for i in range(2):
            banks[i], _ = clear_one_b1_pair(banks[i])
            banks[i], _ = clear_one_b2_pair(banks[i])
            assert torch.equal(value(banks[i]), value(reference))
            assert (banks[i] >= 0).all()
        if t % 13 == 0:
            for state in [reference] + banks:
                state[0].zero_()
