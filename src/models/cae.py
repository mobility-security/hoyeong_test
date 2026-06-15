"""
정상(Normal) 트래픽 복원 전용 Convolutional Autoencoder.
정상 샘플만으로 학습하며, 공격 입력은 복원 오차(MSE)가 높게 나타남.

내부적으로 cae_input_size×cae_input_size(기본 32×32)로 다운샘플하여 연산하고,
use_detail_channels=True 시 detail-energy 채널(3ch)을 concat해 6ch 입력으로
이상 신호를 강화한다.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CAE(nn.Module):
    def __init__(self, input_shape: tuple = (3, 64, 64),
                 latent_dim: int = 32,
                 noise_std: float = 0.10,
                 cae_input_size: int = 32,
                 use_detail_channels: bool = True):
        super().__init__()
        if len(input_shape) != 3 or any(int(size) <= 0 for size in input_shape):
            raise ValueError(f'input_shape must be positive (C,H,W), got {input_shape}')
        if latent_dim < 1:
            raise ValueError(f'latent_dim must be positive, got {latent_dim}')
        if noise_std < 0:
            raise ValueError(f'noise_std must be non-negative, got {noise_std}')
        if cae_input_size < 1:
            raise ValueError(f'cae_input_size must be positive, got {cae_input_size}')

        C, H, W = map(int, input_shape)
        self.input_shape = (C, H, W)
        self.noise_std = noise_std
        self.cae_input_size = cae_input_size
        self.use_detail_channels = use_detail_channels

        in_ch = 6 if use_detail_channels else 3

        # 4-block encoder: stride-2 × 4 → spatial / 16
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 32,  3, stride=2, padding=1),
            nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.Conv2d(32,  64,  3, stride=2, padding=1),
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.Conv2d(64,  128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )

        # probe on cae_input_size × cae_input_size to determine flat_dim
        self.encoder.eval()
        with torch.no_grad():
            probe = self.encoder(torch.zeros(1, in_ch, cae_input_size, cae_input_size))
        self.encoder.train()
        _, C_enc, H_enc, W_enc = probe.shape
        self._shape_enc = (C_enc, H_enc, W_enc)
        flat_dim = C_enc * H_enc * W_enc

        self.enc_fc = nn.Linear(flat_dim, latent_dim)
        self.dec_fc = nn.Linear(latent_dim, flat_dim)

        # 4-block decoder: ConvTranspose2d stride-2 × 4
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64,  3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,  32,  3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32,   3,  3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def _prepare_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Downsample to cae_input_size and optionally append detail-energy channels."""
        x_ll = F.interpolate(x, size=(self.cae_input_size, self.cae_input_size),
                             mode='bilinear', align_corners=False)
        if self.use_detail_channels:
            detail = torch.abs(x_ll - F.avg_pool2d(x_ll, 3, stride=1, padding=1))
            return torch.cat([x_ll, detail], dim=1), x_ll
        return x_ll, x_ll

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct x; output has the same (N, C, H, W) shape as input."""
        if x.ndim != 4 or tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(
                f'input must have shape (N,{self.input_shape[0]},'
                f'{self.input_shape[1]},{self.input_shape[2]}), got {tuple(x.shape)}')
        H, W = x.shape[2], x.shape[3]
        x_prepared, _ = self._prepare_input(x)
        C_enc, H_enc, W_enc = self._shape_enc
        z = self.enc_fc(self.encoder(x_prepared).flatten(1))
        xhat_small = self.decoder(self.dec_fc(z).view(-1, C_enc, H_enc, W_enc))
        if xhat_small.shape[2:] != (H, W):
            xhat_small = F.interpolate(xhat_small, size=(H, W),
                                       mode='bilinear', align_corners=False)
        return xhat_small

    def mse_loss(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        """Per-sample reconstruction MSE at cae_input_size resolution. Returns (N,)."""
        if x.ndim != 4 or tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(
                f'input must have shape (N,{self.input_shape[0]},'
                f'{self.input_shape[1]},{self.input_shape[2]}), got {tuple(x.shape)}')
        x_prepared, x_ll = self._prepare_input(x)
        if training and self.noise_std > 0:
            x_in = torch.clamp(
                x_prepared + torch.randn_like(x_prepared) * self.noise_std, 0.0, 1.0)
        else:
            x_in = x_prepared
        C_enc, H_enc, W_enc = self._shape_enc
        z = self.enc_fc(self.encoder(x_in).flatten(1))
        xhat_small = self.decoder(self.dec_fc(z).view(-1, C_enc, H_enc, W_enc))
        if xhat_small.shape[2:] != x_ll.shape[2:]:
            xhat_small = F.interpolate(
                xhat_small, size=x_ll.shape[2:], mode='bilinear', align_corners=False)
        return ((xhat_small - x_ll) ** 2).mean(dim=(1, 2, 3))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
