"""
S2 6-class + S3 7-class confusion matrix 확정본 PNG 생성.

S2: 6-class DCNN (기존 s2_seed_*.pth 앙상블 또는 단일 best seed)
S3: 두-단계 파이프라인 (CAE gate + S2, Unknown class 포함)

출력:
  results/figures/cm_s2_6class.png  — S2 6-class CM
  results/figures/cm_s3_7class.png  — S3 7-class CM (Unknown 포함)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.cae import CAE
from src.models.dcnn import DCNN
from src.train.common import select_device
from src.train.train_cae import compute_mse_batch
from src.utils.io import LABEL_NAMES, NUM_CLASSES, load_dataset

SEEDS = [0, 1, 2, 3, 4]
CONF_THR = 0.5
CLASS_NAMES_S2 = LABEL_NAMES                     # 6 classes
CLASS_NAMES_S3 = LABEL_NAMES + ['Unknown']        # 7 classes


def _load_s2(seed: int, device: torch.device) -> DCNN | None:
    path = ROOT / 'results' / 'checkpoints' / f's2_seed_{seed}_best.pth'
    if not path.exists():
        return None
    ck = torch.load(path, map_location=device, weights_only=True)
    model = DCNN(num_classes=NUM_CLASSES, dropout=float(ck.get('dropout', 0.5)))
    model.load_state_dict(ck['model_state_dict'])
    return model.eval().to(device)


def _load_cae(device: torch.device) -> tuple[CAE, float] | tuple[None, None]:
    ckpt_path = ROOT / 'results' / 'checkpoints' / 'cae_best.pth'
    tau_path = ROOT / 'results' / 'tables' / 'tau_values.json'
    if not ckpt_path.exists() or not tau_path.exists():
        return None, None
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    cae = CAE(input_shape=tuple(ck['input_shape']), latent_dim=int(ck['latent_dim']))
    cae.load_state_dict(ck['model_state_dict'])
    cae.eval().to(device)
    with tau_path.open(encoding='utf-8') as fh:
        td = json.load(fh)
    tau = float(td[td.get('headline_tau', 'tau_2sigma')])
    return cae, tau


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                       labels: list) -> np.ndarray:
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y_true, y_pred, labels=labels)


def _save_cm(cm: np.ndarray, class_names: list[str],
             title: str, out_path: Path) -> None:
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, mat, fmt, tag in [
        (axes[0], cm_norm, '.3f', 'Normalized'),
        (axes[1], cm, 'd', 'Raw counts'),
    ]:
        sns.heatmap(mat, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.5)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('True', fontsize=10)
        ax.set_title(f'{title} — {tag}', fontsize=11)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def main() -> None:
    device = select_device()
    test_npz = ROOT / 'data' / 'processed' / 'dataset_test.npz'
    if not test_npz.exists():
        print(f'[ERROR] {test_npz} not found.')
        sys.exit(1)

    X_test, y_test, _ = load_dataset(test_npz)
    print(f'Test: {X_test.shape}')

    # --- S2: ensemble predictions across available seeds ---
    all_logits = []
    for seed in SEEDS:
        model = _load_s2(seed, device)
        if model is None:
            continue
        with torch.no_grad():
            logits = model(torch.from_numpy(X_test).to(device))
        all_logits.append(logits.cpu())

    if all_logits:
        mean_logits = torch.stack(all_logits).mean(0)
        y_pred_s2 = mean_logits.argmax(dim=1).numpy().astype(np.int64)
        cm_s2 = _confusion_matrix(y_test, y_pred_s2, labels=list(range(NUM_CLASSES)))
        _save_cm(cm_s2, CLASS_NAMES_S2, 'S2 (6-class DCNN)',
                 ROOT / 'results' / 'figures' / 'cm_s2_6class.png')
    else:
        print('[WARN] No S2 checkpoints found — skipping S2 CM.')

    # --- S3: two-stage with CAE gate ---
    cae, tau = _load_cae(device)
    if cae is None:
        print('[WARN] CAE checkpoint not found — skipping S3 CM.')
        return
    if not all_logits:
        print('[WARN] No S2 checkpoints for S3 CM — skipping.')
        return

    # Use ensemble S2 for S3 as well
    mse = compute_mse_batch(cae, X_test, device)
    anomalous = mse > tau
    n = len(X_test)
    y_pred_s3 = np.zeros(n, dtype=np.int64)
    UNKNOWN = NUM_CLASSES  # 6

    routed = np.flatnonzero(anomalous)
    if len(routed):
        routed_logits = mean_logits[routed]
        probs = F.softmax(routed_logits, dim=1).numpy()
        max_probs = probs.max(axis=1)
        y_pred_s3[routed] = probs.argmax(axis=1)
        y_pred_s3[routed[max_probs < CONF_THR]] = UNKNOWN

    cm_s3 = _confusion_matrix(y_test, y_pred_s3, labels=list(range(NUM_CLASSES + 1)))
    # Trim zero Unknown rows/cols to keep matrix readable if no Unknowns
    _save_cm(cm_s3, CLASS_NAMES_S3, 'S3 (Two-stage: CAE+S2, 7-class)',
             ROOT / 'results' / 'figures' / 'cm_s3_7class.png')


if __name__ == '__main__':
    main()
