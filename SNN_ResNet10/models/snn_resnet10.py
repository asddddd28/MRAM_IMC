"""SpikingJelly implementation of the small ANN FashionMNIST ResNet-10.

The topology intentionally mirrors ``ANN_ResNet10/models/resnet10.py``:
there is one residual block in each stage and the same Conv/BN/Linear tensor
shapes.  IF neurons keep state across calls to ``forward``; callers must call
``reset(model)`` between independent samples/batches.
"""
import torch
from torch import nn
from spikingjelly.activation_based import functional, neuron, surrogate


class SpikingBasicBlock(nn.Module):
    """ANN BasicBlock with integrate-and-fire activations.

    The second convolution is followed by residual addition, matching the ANN
    ordering ``conv2 -> add -> ReLU``.  Therefore only the first activation and
    the block-output activation have IF nodes.
    """

    def __init__(self, in_channels, out_channels, stride=1, v_threshold=1.0):
        super().__init__()
        lif = lambda: neuron.IFNode(
            v_threshold=v_threshold, surrogate_function=surrogate.ATan()
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.lif1 = lif()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )
        self.lif_out = lif()

    def forward(self, x):
        identity = self.downsample(x)
        residual = self.lif1(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))
        return self.lif_out(residual + identity)


class SNNResNet10(nn.Module):
    """Spiking ResNet-10 accepting input shaped ``[T, B, 1, 28, 28]``."""

    def __init__(self, num_classes=10, base_channels=32, v_threshold=1.0):
        super().__init__()
        self.in_channels = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            neuron.IFNode(v_threshold=v_threshold, surrogate_function=surrogate.ATan()),
        )
        self.layer1 = self._make_layer(base_channels, 1, v_threshold)
        self.layer2 = self._make_layer(base_channels * 2, 2, v_threshold)
        self.layer3 = self._make_layer(base_channels * 4, 2, v_threshold)
        self.layer4 = self._make_layer(base_channels * 8, 2, v_threshold)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_channels * 8, num_classes)

    def _make_layer(self, channels, stride, v_threshold):
        block = SpikingBasicBlock(self.in_channels, channels, stride, v_threshold)
        self.in_channels = channels
        return block

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(f"expected [T,B,C,H,W], got shape {tuple(x.shape)}")
        outputs = []
        for t in range(x.shape[0]):
            xt = self.stem(x[t])
            xt = self.layer1(xt)
            xt = self.layer2(xt)
            xt = self.layer3(xt)
            xt = self.layer4(xt)
            outputs.append(self.fc(torch.flatten(self.pool(xt), 1)))
        return torch.stack(outputs, 0)


def reset(model):
    """Reset membrane states before evaluating a new independent sequence."""
    functional.reset_net(model)


def convert_ann_weights(snn_model, ann_checkpoint):
    """Copy every ANN tensor with an identical key and shape into the SNN.

    IF neurons have no ANN counterpart.  The returned report makes transfer
    coverage explicit instead of silently ignoring incompatible tensors.
    """
    checkpoint = torch.load(ann_checkpoint, map_location="cpu", weights_only=False)
    source = checkpoint["model"] if "model" in checkpoint else checkpoint
    target = snn_model.state_dict()
    copied, missing = [], []
    for key, value in target.items():
        if key in source and source[key].shape == value.shape:
            target[key] = source[key]
            copied.append(key)
        elif not ("lif" in key or key.endswith(".v")):
            missing.append(key)
    snn_model.load_state_dict(target)
    return {"copied": copied, "missing": missing, "source_tensors": len(source)}
