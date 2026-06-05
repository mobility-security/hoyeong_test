import torch
import torch.nn as nn


class SeparableConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel,
            stride=stride, padding=padding, groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class BlockA(nn.Module):
    """SepConv(3→256, k=3, stride=3) → ReLU → BN → MaxPool(2,2). No skip."""

    def __init__(self):
        super().__init__()
        self.conv = SeparableConv2d(3, 256, kernel=3, stride=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn   = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.pool(self.bn(self.relu(self.conv(x))))


class BlockB(nn.Module):
    """SepConv(256→256, k=3, stride=1) → ReLU → BN + skip(F(x)+x). ×5"""

    def __init__(self):
        super().__init__()
        self.conv = SeparableConv2d(256, 256, kernel=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn   = nn.BatchNorm2d(256)

    def forward(self, x):
        return self.bn(self.relu(self.conv(x))) + x


class BlockC(nn.Module):
    """SepConv(256→512, k=3, stride=3) → GlobalAvgPool → Flatten."""

    def __init__(self):
        super().__init__()
        self.conv = SeparableConv2d(256, 512, kernel=3, stride=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn   = nn.BatchNorm2d(512)
        self.gap  = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.gap(self.bn(self.relu(self.conv(x))))
        return x.flatten(1)


class DCNN(nn.Module):
    """
    TOW-IDS DCNN: Block A(×1) → Block B(×5) → Block C(×1) → FC head.
    num_classes=2 for Phase 1 (binary), 6 for Phase 2 (multiclass).
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.block_a = BlockA()
        self.block_b = nn.Sequential(*[BlockB() for _ in range(5)])
        self.block_c = BlockC()
        self.head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 64),  nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.block_a(x)
        x = self.block_b(x)
        x = self.block_c(x)
        return self.head(x)
