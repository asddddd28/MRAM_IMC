"""A small ResNet-10 for 28x28 FashionMNIST images."""
import torch
from torch import nn

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels),
        ) if stride != 1 or in_channels != out_channels else nn.Identity()
    def forward(self, x):
        identity = self.downsample(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + identity)

class ResNet10(nn.Module):
    """ResNet-10: one BasicBlock in each of four residual stages."""
    def __init__(self, num_classes=10, base_channels=32):
        super().__init__()
        self.in_channels = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(base_channels, 1)
        self.layer2 = self._make_layer(base_channels * 2, 2)
        self.layer3 = self._make_layer(base_channels * 4, 2)
        self.layer4 = self._make_layer(base_channels * 8, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_channels * 8, num_classes)
        self._init_weights()
    def _make_layer(self, channels, stride):
        block = BasicBlock(self.in_channels, channels, stride)
        self.in_channels = channels
        return block
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d): nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d): nn.init.constant_(module.weight, 1); nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear): nn.init.normal_(module.weight, 0, 0.01); nn.init.constant_(module.bias, 0)
    def forward(self, x):
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.pool(x), 1))

def resnet10(**kwargs): return ResNet10(**kwargs)
