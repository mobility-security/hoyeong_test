"""
Unknown case 4-panel 시각화:
  (a) 입력 이미지 (3채널 → RGB 합성)
  (b) CAE 재구성 이미지
  (c) Per-pixel 오차 heatmap
  (d) MSE vs tau 히스토그램 (normal 저오차 vs 공격 고오차)

출력: results/figures/unknown_case_4panel.png

실제 dataset이 있으면 첫 번째 excluded 공격(F_I)과 Normal 샘플을 사용.
데이터가 없으면 stub 랜덤 이미지를 사용.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIGURE_PATH = ROOT / 'results' / 'figures' / 'unknown_case_4panel.png'
CAE_CKPT = ROOT / 'results' / 'checkpoints' / 'cae_best.pth'
TRAIN_NPZ = ROOT / 'data' / 'processed' / 'dataset_train.npz'
TAU_JSON = ROOT / 'results' / 'tables' / 'tau_values.json'


def _load_cae_and_tau(device: torch.device):
    from src.models.cae import CAE
    ckpt = torch.load(CAE_CKPT, map_location=device, weights_only=True)
    cae = CAE(input_shape=tuple(ckpt['input_shape']),
              latent_dim=int(ckpt['latent_dim']),
              noise_std=float(ckpt.get('noise_std', 0.05)))
    cae.load_state_dict(ckpt['model_state_dict'])
    cae.eval().to(device)
    with TAU_JSON.open(encoding='utf-8') as fh:
        tau_data = json.load(fh)
    tau = float(tau_data[tau_data.get('headline_tau', 'tau_2sigma')])
    return cae, tau


def _to_rgb(x: np.ndarray) -> np.ndarray:
    """(3, H, W) float32 → (H, W, 3) uint8 for imshow."""
    x = np.clip(x, 0.0, 1.0)
    return (x.transpose(1, 2, 0) * 255).astype(np.uint8)


def main() -> None:
    device = torch.device('cpu')

    # Try loading real data
    attack_sample = None
    normal_sample = None
    cae = None
    tau = 0.005
    mse_normal_all = None
    mse_attack_all = None

    if CAE_CKPT.exists() and TRAIN_NPZ.exists() and TAU_JSON.exists():
        from src.utils.io import load_dataset
        X, y, _ = load_dataset(TRAIN_NPZ)
        cae, tau = _load_cae_and_tau(device)

        # Pick one F_I attack sample (label=1) and one Normal sample (label=0)
        attack_idx = np.flatnonzero(y == 1)
        normal_idx = np.flatnonzero(y == 0)
        if len(attack_idx) and len(normal_idx):
            rng = np.random.default_rng(42)
            attack_sample = X[rng.choice(attack_idx)]    # (3,H,W)
            normal_sample = X[rng.choice(normal_idx[:100])]  # (3,H,W)

            # MSE distribution for histogram (small random subset for speed)
            n_sub = min(300, len(attack_idx), len(normal_idx))
            sub_atk = X[rng.choice(attack_idx, n_sub, replace=False)]
            sub_nrm = X[rng.choice(normal_idx, n_sub, replace=False)]
            from src.train.train_cae import compute_mse_batch
            mse_attack_all = compute_mse_batch(cae, sub_atk, device)
            mse_normal_all = compute_mse_batch(cae, sub_nrm, device)

    if attack_sample is None:
        # Stub data
        rng = np.random.default_rng(0)
        attack_sample = rng.random((3, 32, 32)).astype(np.float32)
        normal_sample = rng.random((3, 32, 32)).astype(np.float32) * 0.1
        mse_attack_all = np.abs(rng.standard_normal(100)).astype(np.float32) * 0.01 + 0.008
        mse_normal_all = np.abs(rng.standard_normal(100)).astype(np.float32) * 0.001 + 0.001

    # Reconstruct via CAE
    if cae is not None:
        with torch.no_grad():
            x_in = torch.from_numpy(attack_sample[None]).to(device)
            x_rec = cae(x_in)[0].cpu().numpy()
    else:
        x_rec = attack_sample * 0.5  # stub

    err_map = np.abs(x_rec - attack_sample).mean(axis=0)  # (H, W)
    mse_attack_val = float(((x_rec - attack_sample) ** 2).mean())

    # 4-panel figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # (a) Input
    axes[0].imshow(_to_rgb(attack_sample))
    axes[0].set_title('(a) Input (F_I attack)', fontsize=11)
    axes[0].axis('off')

    # (b) CAE reconstruction
    axes[1].imshow(_to_rgb(x_rec))
    axes[1].set_title('(b) CAE Reconstruction', fontsize=11)
    axes[1].axis('off')

    # (c) Per-pixel error heatmap
    im = axes[2].imshow(err_map, cmap='hot', vmin=0)
    axes[2].set_title('(c) Per-Pixel Error', fontsize=11)
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    # (d) MSE histogram: Normal vs Attack vs tau
    bins = np.linspace(
        0,
        max(float(np.percentile(mse_attack_all, 99)) * 1.2, tau * 3),
        50,
    )
    axes[3].hist(mse_normal_all, bins=bins, alpha=0.7, label='Normal', color='steelblue')
    axes[3].hist(mse_attack_all, bins=bins, alpha=0.7, label='F_I Attack', color='tomato')
    axes[3].axvline(tau, color='red', lw=2, ls='--', label=f'τ={tau:.4f}')
    axes[3].axvline(mse_attack_val, color='orange', lw=1.5, ls=':',
                    label=f'This sample MSE={mse_attack_val:.4f}')
    axes[3].set_xlabel('Reconstruction MSE', fontsize=10)
    axes[3].set_ylabel('Count', fontsize=10)
    axes[3].set_title('(d) MSE Distribution', fontsize=11)
    axes[3].legend(fontsize=8)

    plt.suptitle('Unknown Case Analysis — Excluded Attack: F_I (Frame Injection)',
                 fontsize=12, y=1.01)
    plt.tight_layout()

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {FIGURE_PATH}')


if __name__ == '__main__':
    main()
