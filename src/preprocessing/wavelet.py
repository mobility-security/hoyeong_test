"""3-wavelet 변환 (Team A / 김민세) — TOW-IDS 핵심 전처리.

각 이미지를 3개 wavelet(coif1, db3, rbio1.3)로 2D-DWT 한 뒤 LL(approximation)
sub-band 만 채널로 쌓아 3채널 텐서를 만든다.

주의 (R4):
  - single-level dwt2 는 공간 차원을 ~절반으로 줄인다 (64x64 → 32x32).
  - rbio1.3 은 biorthogonal 이라 LL 크기가 다른 wavelet 과 1px 어긋날 수 있다
    → 공통 min(H,W) 로 crop 후 stack.
  - DWT 계수는 [0,1] 범위를 벗어날 수 있다 → 채널별 min-max 로 [0,1] 재정규화
    (dataset.npz 계약 충족).
출력: (3, H, W) float32, channel-first, [0,1].
"""
from __future__ import annotations

import numpy as np
import pywt

DEFAULT_WAVELETS = ("coif1", "db3", "rbio1.3")


def wavelet_3ch(img2d: np.ndarray, names=DEFAULT_WAVELETS, level: int = 1,
                mode: str = "periodization", channel_norm: str = "minmax01") -> np.ndarray:
    """단일 2D 이미지 → (3, H, W) 3채널 wavelet 텐서."""
    img2d = np.asarray(img2d)
    if img2d.ndim != 2:
        raise ValueError(f"img2d must be 2-D, got {img2d.shape}")
    if not np.isfinite(img2d).all():
        raise ValueError('img2d contains nan or inf')
    if len(names) != 3:
        raise ValueError(f'exactly 3 wavelets are required, got {len(names)}')
    if level < 1:
        raise ValueError(f'level must be positive, got {level}')
    if channel_norm != 'minmax01':
        raise ValueError(f'unsupported channel_norm: {channel_norm}')

    lls = []
    for w in names:
        coeffs = pywt.wavedec2(img2d.astype(np.float32), w, level=level, mode=mode)
        lls.append(np.asarray(coeffs[0], dtype=np.float32))  # coeffs[0] = LL

    h = min(a.shape[0] for a in lls)
    w_ = min(a.shape[1] for a in lls)
    lls = [a[:h, :w_] for a in lls]
    out = np.stack(lls, axis=0)  # (3, H, W) channel-first

    if channel_norm == "minmax01":
        for c in range(out.shape[0]):
            ch = out[c]
            lo, hi = float(ch.min()), float(ch.max())
            out[c] = (ch - lo) / (hi - lo) if hi > lo else np.zeros_like(ch)
    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    return np.ascontiguousarray(out)


def wavelet_batch(images: np.ndarray, **kw) -> np.ndarray:
    """이미지 배치 (M, H, W) → (M, 3, H', W')."""
    images = np.asarray(images)
    if images.ndim != 3:
        raise ValueError(f'images must have shape (M,H,W), got {images.shape}')
    if len(images) == 0:
        raise ValueError('images batch must not be empty')
    return np.stack([wavelet_3ch(im, **kw) for im in images], axis=0).astype(np.float32)


def output_hw(img_size: int, level: int = 1, mode: str = "periodization") -> tuple[int, int]:
    """주어진 입력 정사각 크기에서 wavelet 후 (H, W) 계산 (계약·stub 일치용)."""
    probe = wavelet_3ch(np.zeros((img_size, img_size), np.float32), level=level, mode=mode)
    return probe.shape[1], probe.shape[2]
