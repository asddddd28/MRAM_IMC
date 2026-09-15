import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from models import ResNet10

def test_resnet10_shape_and_parameter_count():
    model=ResNet10(); x=torch.randn(2,1,28,28); y=model(x)
    assert y.shape==(2,10)
    assert sum(p.numel() for p in model.parameters()) > 100_000
