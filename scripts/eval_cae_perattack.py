"""C1 / m11: CAE 단독 frozen-test 평가 — 공격 클래스별 분리 성능 보고.

현재 comparison_table.py 의 CAE 행은 전부 "—" 이며, ROC-AUC/TPR 수치는
val 셋 기준(P_I 포함 2클래스)으로 계산된 것임(C1 이슈).
이 스크립트는 frozen test 전체(6클래스)를 사용해 공격별 분리 성능을 정직하게 보고한다.

사용:
    python scripts/eval_cae_perattack.py
    python scripts/eval_cae_perattack.py --allow-stale-provenance   # 재학습 전 진단용
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.cae import CAE
from src.train.common import (
    load_manifest, select_device, validate_artifact_provenance,
    validate_checkpoint_provenance,
)
from src.train.train_cae import compute_mse_batch, compute_tau
from src.utils.config import load_experiment_config
from src.utils.io import LABEL_NAMES, load_dataset

try:
    from sklearn.metrics import roc_auc_score, roc_curve
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def _per_attack_metrics(
    mse: np.ndarray,
    y: np.ndarray,
    tau: float,
    label_names: list[str],
) -> list[dict]:
    """공격별 탐지율(TPR), Normal FPR, ROC-AUC 계산."""
    rows = []
    normal_mask = y == 0
    fpr_at_tau = float((mse[normal_mask] > tau).mean()) if normal_mask.any() else float('nan')

    for cls_id, cls_name in enumerate(label_names):
        mask = y == cls_id
        if not mask.any():
            rows.append({'class': cls_name, 'n': 0,
                         'tpr_at_tau': float('nan'),
                         'fpr_normal': float('nan'),
                         'roc_auc': float('nan')})
            continue
        tpr = float((mse[mask] > tau).mean()) if cls_id != 0 else float('nan')
        if _HAS_SKLEARN and cls_id != 0:
            binary = np.zeros(len(y), dtype=np.int64)
            binary[mask] = 1
            eval_mask = normal_mask | mask
            try:
                auc = float(roc_auc_score(binary[eval_mask], mse[eval_mask]))
            except ValueError:
                auc = float('nan')
        else:
            auc = float('nan')
        rows.append({
            'class': cls_name,
            'n': int(mask.sum()),
            'tpr_at_tau': tpr,
            'fpr_normal': fpr_at_tau if cls_id != 0 else float(
                (mse[normal_mask] > tau).mean()),
            'roc_auc': auc,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--allow-stale-provenance', action='store_true',
                        help='skip manifest SHA check (use before retraining for diagnostics)')
    args = parser.parse_args()

    cfg_exp = load_experiment_config()
    output_dir = Path(args.out_dir or str(cfg_exp.output_dir))
    device = select_device()

    X_train, y_train, _ = load_dataset(cfg_exp.train_npz_path)
    X_test, y_test, _ = load_dataset(cfg_exp.test_npz_path)

    manifest = load_manifest(
        cfg_exp.manifest_path, len(X_train), len(X_test),
        train_npz_path=cfg_exp.train_npz_path,
        test_npz_path=cfg_exp.test_npz_path)

    ckpt_path = output_dir / 'checkpoints' / 'cae_best.pth'
    if not ckpt_path.exists():
        raise FileNotFoundError(f'CAE checkpoint not found: {ckpt_path}')

    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    if not args.allow_stale_provenance:
        validate_checkpoint_provenance(ck, manifest, 'configs/cae.yaml')
    else:
        print('[WARN] --allow-stale-provenance: skipping manifest SHA check '
              '(retrain to resolve)')

    for key in ('noise_std', 'use_detail_channels', 'cae_input_size'):
        if key not in ck:
            raise ValueError(
                f'CAE checkpoint missing key "{key}"; '
                'retrain with train_cae.py to generate a complete checkpoint')

    cae = CAE(
        input_shape=tuple(ck['input_shape']),
        latent_dim=int(ck['latent_dim']),
        noise_std=float(ck['noise_std']),
        cae_input_size=int(ck['cae_input_size']),
        use_detail_channels=bool(ck['use_detail_channels']),
    ).to(device)
    cae.load_state_dict(ck['model_state_dict'])
    cae.eval()

    # M4: derive tau from current val-Normal (not cached JSON)
    val_idx = np.asarray(manifest['val_idx'], dtype=np.int64)
    y_val = y_train[val_idx]
    X_val = X_train[val_idx]
    normal_val_mask = y_val == 0
    if not normal_val_mask.any():
        raise ValueError('val split has no Normal samples')

    mse_normal_val = compute_mse_batch(cae, X_val[normal_val_mask], device)
    taus = compute_tau(mse_normal_val)
    tau = float(taus['tau_2sigma'])
    print(f'tau_2sigma={tau:.6f}  (from {normal_val_mask.sum()} val-Normal samples)')

    # Evaluate on frozen test set
    print(f'\nEvaluating CAE on frozen test (N={len(X_test)}) ...')
    mse_test = compute_mse_batch(cae, X_test, device)

    rows = _per_attack_metrics(mse_test, y_test, tau, LABEL_NAMES)

    # Overall attack TPR (all non-Normal)
    attack_mask = y_test != 0
    overall_tpr = float((mse_test[attack_mask] > tau).mean()) if attack_mask.any() else float('nan')
    normal_fpr = float((mse_test[y_test == 0] > tau).mean())

    # ROC-AUC over all classes
    if _HAS_SKLEARN:
        try:
            overall_auc = float(roc_auc_score((y_test != 0).astype(int), mse_test))
        except ValueError:
            overall_auc = float('nan')
    else:
        overall_auc = float('nan')

    print('\n=== CAE frozen-test per-attack breakdown (C1 / m11) ===')
    header = f"{'class':<10} {'n':>6}  {'TPR@tau':>8}  {'FPR_Normal':>10}  {'ROC-AUC':>8}"
    print(header)
    print('-' * len(header))
    for row in rows:
        cls = row['class']
        n = row['n']
        tpr = f"{row['tpr_at_tau']:.4f}" if not np.isnan(row['tpr_at_tau']) else '   —   '
        fpr = f"{row['fpr_normal']:.4f}" if not np.isnan(row['fpr_normal']) else '   —   '
        auc = f"{row['roc_auc']:.4f}" if not np.isnan(row['roc_auc']) else '   —   '
        print(f'{cls:<10} {n:>6}  {tpr:>8}  {fpr:>10}  {auc:>8}')
    print('-' * len(header))
    print(f'{"ALL_ATTACK":<10} {int(attack_mask.sum()):>6}  '
          f'{overall_tpr:>8.4f}  {normal_fpr:>10.4f}  {overall_auc:>8.4f}')

    # Save results
    tables_dir = output_dir / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    result = {
        'tau_2sigma': tau,
        'tau_source': 'val-Normal (fold-recalibrated)',
        'n_val_normal': int(normal_val_mask.sum()),
        'overall_attack_tpr': overall_tpr,
        'normal_fpr': normal_fpr,
        'overall_roc_auc': overall_auc,
        'per_class': rows,
        'manifest_sha256': manifest['sha256'],
        'note': (
            'C1/m11: CAE evaluated on frozen test with all 6 classes. '
            'tau derived from val-Normal (M4 fold-recalibrated).'
        ),
    }
    out_path = tables_dir / 'cae_perattack_test.json'
    out_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'\n[OK] → {out_path}')


if __name__ == '__main__':
    main()
