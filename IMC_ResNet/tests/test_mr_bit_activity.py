import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from IMC_ResNet.models.mr_bit_activity import BitActivity, PairOpportunity
from IMC_ResNet.models.mr_pair_cancel import add_carries_with_positive_forward


def test_activity_separates_update_and_reset():
    stats = BitActivity()
    stats.observe(torch.tensor([[1, 0, 0, 0, 0]]),
                  torch.tensor([[2, 0, 0, 0, 0]]), torch.tensor([[True]]))
    r = stats.report()
    assert r['denominator_per_plane_bit'] == 1
    assert r['rising_counts'][0] == [0, 1, 0, 0, 0]
    assert r['falling_counts'][0] == [1, 0, 0, 0, 0]
    assert r['reset_falling_counts'][0] == [0, 1, 0, 0, 0]


def test_match_probability_uses_union_and_request_denominator():
    stats = PairOpportunity(2)
    stats.observe(torch.tensor([[0,0,16,8,4], [0,0,0,0,4], [0,0,16,8,0]]))
    assert stats.requested == 2 and stats.matched == 1
    assert stats.report()['conditional_match_rate'] == .5
    assert stats.partner_matches == [1,1]
    high = PairOpportunity(4)
    high.observe(torch.tensor([[0,0,0,16,16]]))
    assert high.requested == 1 and high.matched == 0 and high.partners == []


def test_trace_is_read_only_and_counts_actual_b0_pairs():
    torch.manual_seed(916)
    state = torch.randint(0,32,(100,5))
    carry = torch.randint(0,4,(100,5))
    original = add_carries_with_positive_forward(state,carry)
    trace = {}
    traced = add_carries_with_positive_forward(state,carry,trace=trace)
    assert all(torch.equal(a,b) for a,b in zip(original,traced))
    stat = PairOpportunity(0)
    stat.observe(trace['before_existing_b0'])
    assert stat.matched == int((traced[1]-trace['intercepted_negative_units']).sum())
