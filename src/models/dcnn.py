import torch.nn as nn


class SeparableConv2d(nn.Module):
    """Depthwise + Pointwise 분리 합성곱 — 파라미터 수를 줄이면서 표현력 유지."""

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
    """특징 추출 블록: SepConv(3→256, k=3, stride=3) → ReLU → BN → MaxPool(2,2). 스킵 연결 없음."""

    def __init__(self):
        super().__init__()
        self.conv = SeparableConv2d(3, 256, kernel=3, stride=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn   = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.pool(self.bn(self.relu(self.conv(x))))


class BlockB(nn.Module):
    """잔차 블록: SepConv(256→256, k=3, stride=1) → ReLU → BN + 스킵 연결(F(x)+x). ×5 반복."""

    def __init__(self):
        super().__init__()
        self.conv = SeparableConv2d(256, 256, kernel=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn   = nn.BatchNorm2d(256)

    def forward(self, x):
        return self.bn(self.relu(self.conv(x))) + x


class BlockC(nn.Module):
    """채널 확장 블록: SepConv(256→512, k=3, stride=3) → 전역 평균 풀링 → Flatten."""

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
    TOW-IDS DCNN 백본: Block A(×1) → Block B(×5) → Block C(×1) → FC 헤드.
    num_classes=2: Phase 1 이진 분류 / num_classes=6: Phase 2 공격 유형 분류.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        if num_classes < 2:
            raise ValueError(f'num_classes must be at least 2, got {num_classes}')
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f'dropout must be in [0, 1), got {dropout}')
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
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f'input must have shape (N,3,H,W), got {tuple(x.shape)}')
        x = self.block_a(x)
        x = self.block_b(x)
        x = self.block_c(x)
        return self.head(x)
