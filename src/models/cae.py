"""
정상(Normal) 트래픽 복원 전용 Convolutional Autoencoder.
정상 샘플만으로 학습하며, 공격 입력은 복원 오차(MSE)가 높게 나타남.
input_shape를 완전히 파라미터화해 H×W 하드코딩 없이 어떤 해상도도 지원.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CAE(nn.Module):
    def __init__(self, input_shape: tuple = (3, 32, 32),
                 latent_dim: int = 128,
                 noise_std: float = 0.05):
        super().__init__()
        C, H, W = input_shape
        self.noise_std = noise_std

        # 인코더: stride-2 합성곱 3단계 → 공간 해상도 ÷8
        self.encoder = nn.Sequential(
            nn.Conv2d(C,  32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )

        # 임의의 H, W에 대해 실제 인코더 출력 크기를 프로브로 확인 (수식 가정 없음)
        with torch.no_grad():
            probe = self.encoder(torch.zeros(1, C, H, W))
        _, C_enc, H_enc, W_enc = probe.shape
        self._shape_enc = (C_enc, H_enc, W_enc)
        flat_dim = C_enc * H_enc * W_enc

        # 병목(Bottleneck): flat_dim → latent_dim → flat_dim
        self.enc_fc = nn.Linear(flat_dim, latent_dim)
        self.dec_fc = nn.Linear(latent_dim, flat_dim)

        # 디코더: 인코더의 역순 전치 합성곱
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,  32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32,   C, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]

        # 학습 중 노이즈 추가: 항등 함수로 과적합하는 것을 방지 (Denoising CAE)
        if training and self.noise_std > 0:
            x_in = torch.clamp(x + torch.randn_like(x) * self.noise_std, 0.0, 1.0)
        else:
            x_in = x

        # 인코딩 → 잠재 벡터
        z = self.enc_fc(self.encoder(x_in).flatten(1))

        # 디코딩
        C_enc, H_enc, W_enc = self._shape_enc
        xhat = self.decoder(self.dec_fc(z).view(-1, C_enc, H_enc, W_enc))

        # 출력 해상도가 입력과 다를 경우 보간으로 맞춤
        if xhat.shape[2:] != (H, W):
            xhat = F.interpolate(xhat, size=(H, W), mode='bilinear', align_corners=False)

        return xhat

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
