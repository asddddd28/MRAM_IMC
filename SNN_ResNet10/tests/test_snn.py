import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from SNN_ResNet10.models import SNNResNet10

def test_snn_shape():
 m=SNNResNet10(); y=m(torch.rand(4,2,1,28,28)); assert y.shape==(4,2,10)


def test_single_step_rollout_matches_temporal_forward():
    """Evaluation may collect T-prefix logits without resetting between steps."""
    from SNN_ResNet10.models.snn_resnet10 import reset

    torch.manual_seed(42)
    model = SNNResNet10(base_channels=2).eval()
    images = torch.randn(2, 1, 28, 28)
    with torch.inference_mode():
        reset(model)
        sequence = model(images.unsqueeze(0).repeat(4, 1, 1, 1, 1))
        reset(model)
        streamed = torch.stack([model(images.unsqueeze(0))[0] for _ in range(4)])
        torch.testing.assert_close(streamed, sequence, rtol=0, atol=0)
        reset(model)
        repeated = model(images.unsqueeze(0).repeat(4, 1, 1, 1, 1))
        torch.testing.assert_close(repeated, sequence, rtol=0, atol=0)
    reset(model)

