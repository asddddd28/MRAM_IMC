import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from IMC_ResNet.models.mr_b0_escape import B0Escape
from IMC_ResNet.models.mr_pair_cancel import add_carries_with_b0_cancel


def step(stats, state, carry):
    state = torch.tensor([state])
    trace = {}
    result, _ = add_carries_with_b0_cancel(state, torch.tensor([carry]), trace)
    stats.observe(state, trace, result)
    return result


def test_waiting_is_not_escape_and_multiple_carries_are_counted():
    s = B0Escape()
    step(s, [0,0,0,0,1], [0,0,0,0,0])
    assert s.rollovers == 0
    step(s, [0,0,0,0,1], [0,0,0,0,3])
    assert s.rollovers == 2 and s.strict_rollovers == 2
    assert s.rollover_updates == 1


def test_interception_is_not_carry_out():
    s = B0Escape()
    step(s, [0,0,0,2,1], [0,0,0,0,1])
    assert s.rollovers == 0
    step(s, [0,0,0,0,1], [0,0,0,0,1])
    assert s.rollovers == 1 and s.strict_rollovers == 0


def test_reset_clears_residence_history():
    s = B0Escape()
    step(s, [0,0,0,2,1], [0,0,0,0,1])
    s.reset(torch.tensor([[True]]))
    step(s, [0,0,0,0,0], [0,0,0,0,2])
    assert s.rollovers == 1 and s.strict_rollovers == 1
