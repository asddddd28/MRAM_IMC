import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from IMC_ResNet.models.mr_pair_cancel import clear_one_b0_pair, add_carries_with_b0_cancel, add_carries_with_positive_forward

BETA = torch.tensor([1, 2, 4, 8, -16], dtype=torch.int64)


def test_positive_forward_priority_and_blocked_destination():
    carry = torch.tensor([1, 0, 0, 0, 0])
    for state, expected, moved in [
        ([15, 0, 0, 0, 0], [0, 0, 0, 2, 0], 1),
        ([15, 0, 0, 2, 0], [0, 0, 4, 2, 0], 1),
        ([15, 8, 4, 2, 0], [16, 8, 4, 2, 0], 0),
        ([16, 0, 0, 0, 0], [17, 0, 0, 0, 0], 0),
    ]:
        result, _, n = add_carries_with_positive_forward(torch.tensor(state), carry, False)
        assert result.tolist() == expected and n == moved


def test_simultaneous_sources_do_not_overwrite_received_bits():
    state = torch.tensor([15, 7, 3, 0, 0])
    carry = torch.tensor([1, 1, 1, 0, 0])
    result, _, n = add_carries_with_positive_forward(state, carry, False)
    assert result.tolist() == [0, 0, 8, 2, 0] and n == 2
    assert value(result) == value(state + carry)


def test_positive_forward_temporal_carries_and_resets():
    torch.manual_seed(713)
    original = torch.zeros((3, 16, 5), dtype=torch.int32)
    states = [original.clone(), original.clone()]
    for t in range(192):
        carry = torch.randint(0, 4, original.shape, dtype=torch.int32)
        original += carry
        for index in range(2):
            states[index], _, _ = add_carries_with_positive_forward(states[index], carry, bool(index))
            assert torch.equal(value(states[index]), value(original))
            assert (states[index] >= 0).all()
        if t % 13 == 0:
            for state in [original] + states:
                state[0].zero_()


def value(mr):
    return (mr.to(torch.int64) * BETA).sum(-1)


def test_each_proposed_connection_has_equal_and_opposite_weight():
    for b in range(4):
        state = torch.zeros(5, dtype=torch.int32)
        state[4] = 1
        state[b] = 1 << (4-b)
        result, n = clear_one_b0_pair(state)
        assert result.count_nonzero() == 0 and n == 1


def test_multiple_hits_clear_only_one_partner():
    state = torch.tensor([16, 8, 4, 2, 1], dtype=torch.int32)
    result, n = clear_one_b0_pair(state)
    assert result.tolist() == [16, 8, 4, 0, 0]
    assert n == 1 and value(result) == value(state)
    assert state.tolist() == [16, 8, 4, 2, 1]


def test_no_hit_and_even_negative_counter_remain_unchanged():
    for state in [torch.tensor([0, 0, 0, 0, 65]), torch.tensor([16, 8, 4, 2, 64])]:
        result, n = clear_one_b0_pair(state)
        assert torch.equal(result, state) and n == 0


def test_carry_interception_prevents_missed_binary_ripple():
    state = torch.tensor([0, 0, 0, 2, 1], dtype=torch.int32)
    carry = torch.tensor([0, 0, 0, 0, 1], dtype=torch.int32)
    post, _ = clear_one_b0_pair(state + carry)
    intercepted, n = add_carries_with_b0_cancel(state, carry)
    assert post.tolist() == [0, 0, 0, 2, 2]
    assert intercepted.tolist() == [0, 0, 0, 0, 1]
    assert value(post) == value(intercepted) and n == 1


def test_random_temporal_sequences_preserve_every_macro_and_reset():
    torch.manual_seed(91)
    shape = (3, 16, 5)
    original = torch.zeros(shape, dtype=torch.int32)
    post = original.clone()
    intercepted = original.clone()
    scu = original.clone()
    for t in range(192):
        counts = torch.randint(0, 37, shape, dtype=torch.int32)
        z = scu + counts
        carry, scu = z // 16, z % 16
        original += carry
        post, _ = clear_one_b0_pair(post + carry)
        intercepted, _ = add_carries_with_b0_cancel(intercepted, carry)
        assert torch.equal(value(original), value(post))
        assert torch.equal(value(original), value(intercepted))
        assert (post >= 0).all() and (intercepted >= 0).all()
        if t % 13 == 0:
            # A spike clears every Macro belonging to that output neuron.
            for state in (original, post, intercepted, scu):
                state[0].zero_()
