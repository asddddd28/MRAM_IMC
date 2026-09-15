import sys
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from SNN_ResNet10.scripts.sweep_shared_mr_qat import task_loss,rate_loss


def test_multi_window_loss_preserves_both_time_horizons_and_teacher_identity():
    torch.manual_seed(920)
    z=torch.randn(64,2,10,requires_grad=True)
    labels=torch.tensor([1,2])
    expected=(F.cross_entropy(z[:16].mean(0),labels)+F.cross_entropy(z.mean(0),labels))/2
    torch.testing.assert_close(task_loss(z,labels,z.detach(),1.),expected)
    task_loss(z,labels,z.detach(),1.).backward()
    assert torch.isfinite(z.grad).all()
    assert z.grad[:16].abs().sum()>0 and z.grad[16:].abs().sum()>0


def test_window_budget_detects_burst_even_at_low_sequence_average():
    teacher=[torch.ones(2)*.5 for _ in range(64)]
    steady=[torch.ones(2)*.25 for _ in range(64)]
    burst=[torch.ones(2,requires_grad=True) if t<16 else torch.zeros(2,requires_grad=True) for t in range(64)]
    assert rate_loss(steady,teacher)==0
    loss=rate_loss(burst,teacher)
    assert loss>0
    loss.backward()
    assert burst[0].grad.abs().sum()>0
