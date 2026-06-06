"""
Convolutional Autoencoder for Normal traffic reconstruction.
Trained exclusively on Normal samples; attack inputs yield high MSE.
input_shape is fully parameterized — no hardcoded H×W.
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

        # Encoder: three stride-2 conv blocks → spatial dims ÷8
        self.encoder = nn.Sequential(
            nn.Conv2d(C,  32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )

        # Probe actual encoder output shape (works for any H, W, no formula assumed)
        with torch.no_grad():
            probe = self.encoder(torch.zeros(1, C, H, W))
        _, C_enc, H_enc, W_enc = probe.shape
        self._shape_enc = (C_enc, H_enc, W_enc)
        flat_dim = C_enc * H_enc * W_enc

        # Bottleneck (compression ratio: flat_dim → 128)
        self.enc_fc = nn.Linear(flat_dim, latent_dim)
        self.dec_fc = nn.Linear(latent_dim, flat_dim)

        # Decoder: mirror of encoder
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

        # Denoising prevents identity-mapping overfitting during training
        if training and self.noise_std > 0:
            x_in = torch.clamp(x + torch.randn_like(x) * self.noise_std, 0.0, 1.0)
        else:
            x_in = x

        # Encode → bottleneck
        z = self.enc_fc(self.encoder(x_in).flatten(1))

        # Decode
        C_enc, H_enc, W_enc = self._shape_enc
        xhat = self.decoder(self.dec_fc(z).view(-1, C_enc, H_enc, W_enc))

        # Guarantee output spatial size == input spatial size
        if xhat.shape[2:] != (H, W):
            xhat = F.interpolate(xhat, size=(H, W), mode='bilinear', align_corners=False)

        return xhat

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
