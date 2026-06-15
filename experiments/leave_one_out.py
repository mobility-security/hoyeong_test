"""
Phase 4: Leave-One-Attack-Out (LOAO) 제로데이 탐지 평가 프로토콜.

공격 유형 하나를 학습에서 제외하고, 해당 공격을 Unknown으로 탐지할 수 있는지 평가.

사용법:
  python -m experiments.leave_one_out           # 전체 실행: 5 fold × 5 seed
  python -m experiments.leave_one_out --smoke   # fold 1개, seed 1개, 1 epoch
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.cae import CAE
from src.models.dcnn import DCNN
from src.pipeline.two_stage import _effective_anomalous
from src.train.common import (
    EarlyStopping, checkpoint_provenance, config_sha256, load_manifest,
    make_supervised_loader,
    result_bundle_provenance, select_device, validate_artifact_provenance,
    validate_checkpoint_provenance,
)
from src.train.train_cae import compute_mse_batch
from src.utils.config import load_experiment_config, resolve_conf_threshold
from src.utils.focal_loss import FocalLoss
from src.utils.io import LABEL_MAP, LABEL_NAMES, load_dataset
from src.utils.seed import set_seed
from src.utils.split import leak_safe_trainval_split

ATTACK_CLASSES: List[str] = ['F_I', 'P_I', 'M_F', 'C_D', 'C_R']
SEEDS: List[int] = [0, 1, 2, 3, 4]
UNKNOWN_LABEL = 6   # pipeline Unknown sentinel
# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def _remap_labels_exclude(y: np.ndarray, excl_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Exclude class `excl_id` and return (keep_mask, remapped_y).

    Remapping rule: classes 0..excl_id-1 stay the same; classes excl_id+1..5
    are shifted down by one, yielding 5 contiguous labels (0..4).
    """
    keep = y != excl_id
    y_out = y[keep].copy()
    y_out[y_out > excl_id] -= 1
    return keep, y_out


# ---------------------------------------------------------------------------
# CAE loading
# ---------------------------------------------------------------------------

def load_cae(
    ckpt_path: str | Path,
    device: torch.device,
    manifest: dict,
) -> CAE:
    """Load CAE checkpoint. Tau is computed per-fold from val-Normal MSE."""
    ckpt_path = Path(ckpt_path)
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    validate_checkpoint_provenance(ck, manifest, 'configs/cae.yaml')
    for key in ('model_state_dict', 'input_shape', 'latent_dim'):
        if key not in ck:
            raise ValueError(f'CAE checkpoint missing key: {key}')
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
    )
    cae.load_state_dict(ck['model_state_dict'])
    cae.eval().to(device)
    return cae


# ---------------------------------------------------------------------------
# Stage-2 training (5-class, per fold)
# ---------------------------------------------------------------------------

def _infer_5class(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (y_true, y_pred, y_prob_all_classes)."""
    model.eval()
    trues, preds, probs = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            p = F.softmax(logits, dim=1).cpu().numpy()
            probs.append(p)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            trues.append(yb.numpy())
    return (
        np.concatenate(trues).astype(np.int64),
        np.concatenate(preds).astype(np.int64),
        np.concatenate(probs).astype(np.float32),
    )


def train_s2_fold(
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg_train,
    device: torch.device,
    num_classes: int = 5,
    max_epochs: int | None = None,
    checkpoint_path: str | Path | None = None,
    provenance: dict | None = None,
    excluded_attack: str | None = None,
) -> nn.Module:
    """Train 5-class Stage-2 DCNN for one fold/seed."""
    set_seed(seed)

    epochs = max_epochs if max_epochs is not None else int(cfg_train.epochs)
    lr = float(cfg_train.lr)
    bs = int(cfg_train.batch_size)
    patience = int(cfg_train.patience)

    classes = np.arange(num_classes)
    missing = np.setdiff1d(classes, np.unique(y_train))
    if len(missing):
        raise ValueError(f'train split missing classes: {missing.tolist()}')

    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = torch.FloatTensor(weights).to(device)

    pin = device.type == 'cuda'
    train_loader = make_supervised_loader(X_train, y_train, bs, shuffle=True, pin_memory=pin)
    val_loader = make_supervised_loader(X_val, y_val, bs, pin_memory=pin)

    model = DCNN(num_classes=num_classes, dropout=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = FocalLoss(gamma=3.0, weight=class_weights)
    stopper = EarlyStopping(patience=patience, mode='max')
    val_labels = sorted(np.unique(y_val).tolist())

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=pin)
            yb = yb.to(device, non_blocking=pin)
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(xb), yb).backward()
            optimizer.step()

        _, y_vp, _ = _infer_5class(model, val_loader, device)
        val_macro_f1 = float(f1_score(y_val, y_vp, labels=val_labels,
                                      average='macro', zero_division=0))
        if stopper.step(val_macro_f1, model):
            break

    stopper.restore_best(model)
    model.eval()
    if checkpoint_path is not None:
        if provenance is None:
            raise ValueError('provenance is required when saving a LOAO checkpoint')
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'num_classes': num_classes,
            'dropout': 0.5,
            'seed': seed,
            'excluded_attack': excluded_attack,
            'val_macro_f1': float(stopper.best),
            **provenance,
        }, checkpoint_path)
    return model


# ---------------------------------------------------------------------------
# Per-fold evaluation
# ---------------------------------------------------------------------------

def _predict_loao(
    cae: CAE | None,
    s2_model: nn.Module,
    X: np.ndarray,
    tau: float,
    conf_thr: float,
    device: torch.device,
    batch_size: int = 64,
    use_cae: bool = True,
    mse: np.ndarray | None = None,
    routing_mode: str = 'strict_cascade',
) -> tuple[np.ndarray, np.ndarray]:
    """Run LOAO pipeline on X.

    Returns:
        mse:    (N,) CAE reconstruction error
        y_pred: (N,) final label — 0..num_classes-1 or UNKNOWN_LABEL(6)
    """
    if use_cae:
        if cae is None:
            raise ValueError('cae is required when use_cae=True')
        if mse is None:
            mse = compute_mse_batch(cae, X, device, batch_size)
        mse = np.asarray(mse, dtype=np.float32)
        if mse.shape != (len(X),):
            raise ValueError(f'mse must have shape ({len(X)},), got {mse.shape}')
        cae_anomalous = mse > tau
    else:
        mse = np.full(len(X), np.nan, dtype=np.float32)
        cae_anomalous = np.ones(len(X), dtype=bool)

    n = len(X)
    if n == 0:
        return mse, np.empty(0, dtype=np.int64)
    strict_gate = use_cae and routing_mode == 'strict_cascade'
    candidate_mask = cae_anomalous if strict_gate else np.ones(n, dtype=bool)
    candidate_idx = np.flatnonzero(candidate_mask)
    preds = np.zeros(n, dtype=np.int64)
    max_probs = np.full(n, np.nan, dtype=np.float32)
    probability_parts = []
    with torch.no_grad():
        for start in range(0, len(candidate_idx), batch_size):
            batch_idx = candidate_idx[start:start + batch_size]
            logits = s2_model(torch.from_numpy(X[batch_idx]).to(device))
            probability_parts.append(F.softmax(logits, dim=1).cpu().numpy())
    if probability_parts:
        probs = np.concatenate(probability_parts)
        preds[candidate_idx] = probs.argmax(axis=1).astype(np.int64)
        max_probs[candidate_idx] = probs.max(axis=1)
    routed_mask = _effective_anomalous(
        cae_anomalous, preds,
        's2_recovery' if not use_cae else routing_mode)
    y_pred = np.zeros(n, dtype=np.int64)
    y_pred[routed_mask] = preds[routed_mask]
    y_pred[routed_mask & (max_probs < conf_thr)] = UNKNOWN_LABEL

    return mse, y_pred


def _calibrate_known_confidence(
    model: nn.Module,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    target_reject_rate: float,
    max_threshold: float = 0.95,
) -> tuple[float, dict]:
    """Calibrate confidence using every known validation sample."""
    loader = make_supervised_loader(X_val, y_val, batch_size=128)
    y_true, y_pred, probs = _infer_5class(model, loader, device)
    if not 0.0 <= target_reject_rate < 1.0:
        raise ValueError('target_reject_rate must be in [0, 1)')
    if not 0.0 <= max_threshold < 1.0:
        raise ValueError('max_threshold must be in [0, 1)')
    present = sorted(np.unique(y_true).tolist())
    expected = list(range(probs.shape[1]))
    if present != expected:
        raise ValueError(
            f'known validation must cover every fold class: present={present}, expected={expected}')
    max_conf = probs.max(axis=1)
    candidates = np.unique(np.concatenate(([0.0, max_threshold], max_conf[max_conf <= max_threshold])))
    valid = [float(value) for value in candidates
             if float((max_conf < value).mean()) <= target_reject_rate]
    threshold = max(valid) if valid else 0.0
    rejected = max_conf < threshold
    per_class_reject = {
        str(label): float(rejected[y_true == label].mean())
        for label in expected
    }
    diagnostics = {
        'actual_known_reject_rate': float(rejected.mean()),
        'validation_coverage': float((~rejected).mean()),
        'validation_accuracy': float(accuracy_score(y_true, y_pred)),
        'validation_macro_f1': float(f1_score(
            y_true, y_pred, labels=expected, average='macro', zero_division=0)),
        'per_class_reject_rate': per_class_reject,
        'n_validation': int(len(y_true)),
    }
    return threshold, diagnostics


def run_one_fold(
    excluded_attack: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cae: CAE | None,
    tau: float,
    cfg_train,
    device: torch.device,
    conf_thr: float,
    seeds: List[int] = SEEDS,
    max_epochs: int | None = None,
    use_cae: bool = True,
    conf_thr_mode: str = 'fixed',
    target_known_reject_rate: float = 0.05,
    out_dir: str | Path = 'results',
    manifest: dict | None = None,
    mse_test: np.ndarray | None = None,
    routing_mode: str = 'strict_cascade',
    max_conf_thr: float = 0.95,
) -> List[dict]:
    """Train Stage-2 per seed (excluded class removed), evaluate on test."""
    excl_id = LABEL_MAP[excluded_attack]

    # Build 5-class train and val sets (no excluded class)
    train_keep, y_train_5 = _remap_labels_exclude(y_train, excl_id)
    X_train_5 = X_train[train_keep]

    val_keep, y_val_5 = _remap_labels_exclude(y_val, excl_id)
    X_val_5 = X_val[val_keep]

    # Test slices: excluded attack + Normal (for FPR)
    excl_test_mask = y_test == excl_id
    normal_test_mask = y_test == 0

    # Combined eval slice
    eval_mask = excl_test_mask | normal_test_mask
    X_eval = X_test[eval_mask]
    # y_eval: 0=Normal, keep original excl_id so compute_loao_metrics can distinguish
    y_eval = y_test[eval_mask]
    known_test_mask = ~excl_test_mask
    _, y_known_5 = _remap_labels_exclude(y_test, excl_id)
    provenance = checkpoint_provenance(
        manifest or {}, 'configs/model.yaml', 'configs/train.yaml')

    if use_cae and mse_test is None:
        if cae is None:
            raise ValueError('cae is required when use_cae=True')
        mse_test = compute_mse_batch(cae, X_test, device)

    mse_eval = mse_test[eval_mask] if mse_test is not None else None
    mse_excl = mse_test[excl_test_mask] if mse_test is not None else None
    mse_normal = mse_test[normal_test_mask] if mse_test is not None else None
    mse_known = mse_test[known_test_mask] if mse_test is not None else None
    if use_cae and len(np.unique((y_eval != 0).astype(np.int64))) == 2:
        mse_auc = float(roc_auc_score((y_eval != 0).astype(np.int64), mse_eval))
    else:
        mse_auc = float('nan')

    seed_results = []
    for seed in seeds:
        print(f'    [fold={excluded_attack}, seed={seed}] training Stage-2 (5-class)...')
        checkpoint_path = (
            Path(out_dir) / 'checkpoints' / 'loao'
            / f'exclude_{excluded_attack}_seed_{seed}.pth')
        s2 = train_s2_fold(
            seed, X_train_5, y_train_5, X_val_5, y_val_5,
            cfg_train, device, num_classes=5, max_epochs=max_epochs,
            checkpoint_path=checkpoint_path, provenance=provenance,
            excluded_attack=excluded_attack,
        )
        seed_conf_thr = conf_thr
        calibration = {
            'actual_known_reject_rate': None,
            'validation_coverage': None,
            'validation_accuracy': None,
            'validation_macro_f1': None,
            'per_class_reject_rate': {},
            'n_validation': int(len(y_val_5)),
        }
        if conf_thr_mode == 'validation':
            try:
                seed_conf_thr, calibration = _calibrate_known_confidence(
                    s2, X_val_5, y_val_5, device, target_known_reject_rate,
                    max_threshold=max_conf_thr)
            except ValueError as exc:
                print(f'    [WARN] confidence calibration fallback: {exc}')
        elif conf_thr_mode == 'fixed':
            validation_loader = make_supervised_loader(
                X_val_5, y_val_5, batch_size=128)
            y_cal_true, y_cal_pred, cal_probs = _infer_5class(
                s2, validation_loader, device)
            max_conf = cal_probs.max(axis=1)
            rejected = max_conf < seed_conf_thr
            calibration = {
                'actual_known_reject_rate': float(rejected.mean()),
                'validation_coverage': float((~rejected).mean()),
                'validation_accuracy': float(
                    accuracy_score(y_cal_true, y_cal_pred)),
                'validation_macro_f1': float(f1_score(
                    y_cal_true, y_cal_pred, labels=list(range(5)),
                    average='macro', zero_division=0)),
                'per_class_reject_rate': {
                    str(label): float(rejected[y_cal_true == label].mean())
                    for label in range(5)
                },
                'n_validation': int(len(y_cal_true)),
            }
        else:
            raise ValueError(f'unsupported loao_conf_thr_mode: {conf_thr_mode}')
        threshold_path = (
            Path(out_dir) / 'tables' / 'loao_thresholds'
            / f'exclude_{excluded_attack}_seed_{seed}.json')
        threshold_path.parent.mkdir(parents=True, exist_ok=True)
        threshold_path.write_text(json.dumps({
            'excluded_attack': excluded_attack,
            'seed': seed,
            'conf_thr': seed_conf_thr,
            'mode': conf_thr_mode,
            'target_known_reject_rate': target_known_reject_rate,
            'max_conf_thr': max_conf_thr,
            'routing_mode': routing_mode,
            **calibration,
            'manifest_sha256': (manifest or {}).get('sha256'),
            'train_sha256': (manifest or {}).get('train_sha256'),
            'test_sha256': (manifest or {}).get('test_sha256'),
            'config_sha256': config_sha256(
                'configs/model.yaml', 'configs/train.yaml',
                'configs/cae.yaml', 'configs/experiment.yaml'),
        }, indent=2), encoding='utf-8')

        _, y_pred_eval = _predict_loao(
            cae, s2, X_eval, tau, seed_conf_thr, device,
            use_cae=use_cae, mse=mse_eval, routing_mode=routing_mode)

        # Also compute on ALL test attacks of the excluded class (not just those in eval)
        _, y_pred_excl = _predict_loao(
            cae, s2, X_test[excl_test_mask], tau, seed_conf_thr, device,
            use_cae=use_cae, mse=mse_excl, routing_mode=routing_mode)
        _, y_pred_known = _predict_loao(
            cae, s2, X_test[known_test_mask], tau, seed_conf_thr, device,
            use_cae=use_cae, mse=mse_known, routing_mode=routing_mode)

        cae_recall_only = (
            float((mse_excl > tau).mean())
            if use_cae and len(mse_excl) else float('nan'))
        unknown_rate_only = float((y_pred_excl == UNKNOWN_LABEL).mean()) if len(y_pred_excl) else float('nan')
        if (use_cae and routing_mode == 'strict_cascade'
                and np.isfinite(unknown_rate_only)
                and unknown_rate_only > cae_recall_only + 1e-12):
            raise ValueError('strict LOAO Unknown rate cannot exceed CAE anomaly recall')
        cae_normal_fpr = (
            float((mse_normal > tau).mean())
            if use_cae and len(mse_normal) else float('nan'))
        normal_pred = y_pred_eval[y_eval == 0]
        normal_fpr = float((normal_pred != 0).mean()) if len(normal_pred) else float('nan')
        known_accuracy = float(accuracy_score(y_known_5, y_pred_known))
        known_macro_f1 = float(f1_score(
            y_known_5, y_pred_known, labels=list(range(5)),
            average='macro', zero_division=0))

        result = {
            'excluded_attack': excluded_attack,
            'seed': seed,
            'cae_anomaly_recall': cae_recall_only,
            'unknown_rate': unknown_rate_only,
            'normal_fpr': normal_fpr,
            'pipeline_normal_fpr': normal_fpr,
            'cae_normal_fpr': cae_normal_fpr,
            'mse_roc_auc': mse_auc,
            'known_accuracy': known_accuracy,
            'known_macro_f1': known_macro_f1,
            'conf_thr': seed_conf_thr,
            'known_reject_rate': calibration['actual_known_reject_rate'],
            'routing_mode': routing_mode,
            'checkpoint_path': str(checkpoint_path),
            'threshold_path': str(threshold_path),
            'manifest_sha256': (manifest or {}).get('sha256'),
            'n_excl_test': int(excl_test_mask.sum()),
            'n_normal_test': int(normal_test_mask.sum()),
        }
        seed_results.append(result)
        print(f'    → cae_recall={cae_recall_only:.4f}  '
              f'unknown_rate={unknown_rate_only:.4f}  '
              f'normal_fpr={normal_fpr:.4f}')
    return seed_results


def run_loao(
    excluded_attack: str,
    config: dict,
    cae_model,
    split_manifest: dict,
) -> dict:
    """
    지정된 공격 클래스를 제외하고 LOAO 평가 수행.
    CAE는 정상 데이터로 한 번만 학습하고 모든 fold에서 공유.

    반환값:
        {
          'cae_anomaly_recall': float,
          'unknown_rate': float,
          'normal_fpr': float,
          'auc_roc': float,
          'seed_results': List[dict],
        }
    """
    required = {
        'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test',
        'tau', 'cfg_train', 'device', 'conf_thr',
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f'run_loao config missing keys: {sorted(missing)}')
    rows = run_one_fold(
        excluded_attack=excluded_attack, cae=cae_model,
        manifest=split_manifest, **{key: config[key] for key in required},
        seeds=config.get('seeds', SEEDS),
        max_epochs=config.get('max_epochs'),
        use_cae=bool(config.get('use_cae', True)),
        conf_thr_mode=str(config.get('conf_thr_mode', 'fixed')),
        target_known_reject_rate=float(config.get('target_known_reject_rate', 0.05)),
        out_dir=config.get('out_dir', 'results'),
        mse_test=config.get('mse_test'),
        routing_mode=str(config.get('routing_mode', 'strict_cascade')),
        max_conf_thr=float(config.get('max_conf_thr', 0.95)),
    )
    return {'seed_results': rows, **{
        key: float(pd.DataFrame(rows)[key].mean())
        for key in ('cae_anomaly_recall', 'unknown_rate', 'normal_fpr', 'mse_roc_auc')
    }}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true',
                        help='1 fold (F_I), 1 seed (0), 1 epoch')
    parser.add_argument('--cae-ckpt', default=None)
    parser.add_argument('--out-dir', default=None)
    args = parser.parse_args()

    cfg_train = OmegaConf.load('configs/train.yaml').train
    cfg_exp = load_experiment_config()

    device = select_device()
    print(f'Device: {device}')

    X_train, y_train, _ = load_dataset(cfg_exp.train_npz_path)
    X_test, y_test, _ = load_dataset(cfg_exp.test_npz_path)

    manifest = load_manifest(
        cfg_exp.manifest_path, len(X_train), len(X_test),
        train_npz_path=cfg_exp.train_npz_path,
        test_npz_path=cfg_exp.test_npz_path)
    conf_thr_mode = str(cfg_exp.get('loao_conf_thr_mode', 'fixed'))
    routing_mode = str(cfg_exp.get('routing_mode', 'strict_cascade'))
    conf_thr = resolve_conf_threshold(cfg_exp, manifest)
    use_cae = bool(cfg_exp.get('use_cae', True))

    train_idx = np.asarray(manifest['train_idx'], dtype=np.int64)
    val_idx = np.asarray(manifest['val_idx'], dtype=np.int64)
    X_tr = X_train[train_idx]
    y_tr = y_train[train_idx]
    X_vl = X_train[val_idx]
    y_vl = y_train[val_idx]

    print(f'Train: {X_tr.shape}  Val: {X_vl.shape}  Test: {X_test.shape}')

    # Load CAE once — reuse across ALL folds
    out_dir = args.out_dir or str(cfg_exp.output_dir)
    cae_ckpt = args.cae_ckpt or str(Path(out_dir) / 'checkpoints' / 'cae_best.pth')
    cae = None
    tau = float('nan')
    if use_cae:
        print(f'Loading CAE from {cae_ckpt}')
        cae = load_cae(cae_ckpt, device, manifest)
        # M4: fold-specific tau — derive from current val-Normal MSE, not cached JSON
        normal_val_mask = y_vl == 0
        if not normal_val_mask.any():
            raise ValueError('val split contains no Normal samples; cannot calibrate tau')
        mse_normal_val = compute_mse_batch(cae, X_vl[normal_val_mask], device)
        mu = float(mse_normal_val.mean())
        sigma = float(mse_normal_val.std(ddof=1)) if len(mse_normal_val) > 1 else 0.0
        tau = mu + 2 * sigma
        print(f'tau recalibrated from val-Normal (N={normal_val_mask.sum()}): '
              f'mu={mu:.6f}  sigma={sigma:.6f}  tau={tau:.6f}')
        mse_test = compute_mse_batch(cae, X_test, device)
    else:
        print('CAE disabled by experiment.use_cae; running confidence-only open-set mode')
        mse_test = None
    print(f'tau={tau:.6f}')
    print(f'conf_thr={conf_thr:.3f}')

    folds = ATTACK_CLASSES[:1] if args.smoke else ATTACK_CLASSES
    seeds = SEEDS[:1] if args.smoke else SEEDS
    max_epochs = 1 if args.smoke else None

    all_rows: List[dict] = []
    for excluded in folds:
        print(f'\n=== Fold: exclude={excluded} ===')
        rows = run_one_fold(
            excluded_attack=excluded,
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_vl,
            y_val=y_vl,
            X_test=X_test,
            y_test=y_test,
            cae=cae,
            tau=tau,
            cfg_train=cfg_train,
            device=device,
            conf_thr=conf_thr,
            seeds=seeds,
            max_epochs=max_epochs,
            use_cae=use_cae,
            conf_thr_mode=conf_thr_mode,
            target_known_reject_rate=float(
                cfg_exp.get('target_known_reject_rate', 0.05)),
            out_dir=out_dir,
            manifest=manifest,
            mse_test=mse_test,
            routing_mode=routing_mode,
            max_conf_thr=float(cfg_exp.get('loao_max_conf_thr', 0.95)),
        )
        all_rows.extend(rows)

    # --- Save per-fold CSV ---
    df = pd.DataFrame(all_rows)
    out_tables = os.path.join(out_dir, 'tables')
    os.makedirs(out_tables, exist_ok=True)

    per_fold_path = os.path.join(out_tables, 'loao_per_fold.csv')
    df.to_csv(per_fold_path, index=False)
    print(f'\nSaved: {per_fold_path}')

    # --- Summary: 5-fold mean±std ---
    metric_cols = [
        'unknown_rate', 'normal_fpr', 'cae_normal_fpr',
        'known_accuracy', 'known_macro_f1', 'known_reject_rate',
    ]
    agg = df.groupby('excluded_attack')[metric_cols].agg(['mean', 'std']).reset_index()
    agg.columns = ['excluded_attack'] + [f'{m}_{s}' for m, s in agg.columns[1:]]
    agg = agg.fillna(0.0)

    # Grand mean±std across folds
    fold_means = df.groupby('excluded_attack')[metric_cols].mean()
    grand = {col: fold_means[col].mean() for col in metric_cols}
    grand_std = {
        col: float(fold_means[col].std()) if len(fold_means) > 1 else 0.0
        for col in metric_cols
    }
    summary_rows = []
    for _, row in agg.iterrows():
        summary_rows.append({
            'excluded_attack': row['excluded_attack'],
            'cae_recall': df.loc[
                df['excluded_attack'] == row['excluded_attack'],
                'cae_anomaly_recall'].iloc[0],
            'mse_roc_auc': df.loc[
                df['excluded_attack'] == row['excluded_attack'],
                'mse_roc_auc'].iloc[0],
            'unknown_rate_mean': row['unknown_rate_mean'],
            'unknown_rate_std': row['unknown_rate_std'],
            'unknown_rate_ci95': 1.96 * row['unknown_rate_std'] / np.sqrt(len(seeds)),
            'normal_fpr_mean': row['normal_fpr_mean'],
            'normal_fpr_std': row['normal_fpr_std'],
            'normal_fpr_ci95': 1.96 * row['normal_fpr_std'] / np.sqrt(len(seeds)),
            'cae_normal_fpr_mean': row['cae_normal_fpr_mean'],
            'cae_normal_fpr_std': row['cae_normal_fpr_std'],
            'cae_normal_fpr_ci95': 1.96 * row['cae_normal_fpr_std'] / np.sqrt(len(seeds)),
            'known_reject_rate_mean': row['known_reject_rate_mean'],
            'known_reject_rate_std': row['known_reject_rate_std'],
            'known_reject_rate_ci95': 1.96 * row['known_reject_rate_std'] / np.sqrt(len(seeds)),
            'known_accuracy_mean': row['known_accuracy_mean'],
            'known_accuracy_std': row['known_accuracy_std'],
            'known_accuracy_ci95': 1.96 * row['known_accuracy_std'] / np.sqrt(len(seeds)),
            'known_macro_f1_mean': row['known_macro_f1_mean'],
            'known_macro_f1_std': row['known_macro_f1_std'],
            'known_macro_f1_ci95': 1.96 * row['known_macro_f1_std'] / np.sqrt(len(seeds)),
        })
    summary_rows.append({
        'excluded_attack': 'grand_mean',
        'cae_recall': df.groupby('excluded_attack')['cae_anomaly_recall'].first().mean(),
        'mse_roc_auc': df.groupby('excluded_attack')['mse_roc_auc'].first().mean(),
        'unknown_rate_mean': grand['unknown_rate'],
        'unknown_rate_std': grand_std['unknown_rate'],
        'unknown_rate_ci95': 1.96 * grand_std['unknown_rate'] / np.sqrt(max(len(fold_means), 1)),
        'normal_fpr_mean': grand['normal_fpr'],
        'normal_fpr_std': grand_std['normal_fpr'],
        'normal_fpr_ci95': 1.96 * grand_std['normal_fpr'] / np.sqrt(max(len(fold_means), 1)),
        'cae_normal_fpr_mean': grand['cae_normal_fpr'],
        'cae_normal_fpr_std': grand_std['cae_normal_fpr'],
        'cae_normal_fpr_ci95': 1.96 * grand_std['cae_normal_fpr'] / np.sqrt(max(len(fold_means), 1)),
        'known_reject_rate_mean': grand['known_reject_rate'],
        'known_reject_rate_std': grand_std['known_reject_rate'],
        'known_reject_rate_ci95': 1.96 * grand_std['known_reject_rate'] / np.sqrt(max(len(fold_means), 1)),
        'known_accuracy_mean': grand['known_accuracy'],
        'known_accuracy_std': grand_std['known_accuracy'],
        'known_accuracy_ci95': 1.96 * grand_std['known_accuracy'] / np.sqrt(max(len(fold_means), 1)),
        'known_macro_f1_mean': grand['known_macro_f1'],
        'known_macro_f1_std': grand_std['known_macro_f1'],
        'known_macro_f1_ci95': 1.96 * grand_std['known_macro_f1'] / np.sqrt(max(len(fold_means), 1)),
    })
    df_summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_tables, 'loao_summary.csv')
    df_summary.to_csv(summary_path, index=False)

    # M7: per-seed raw metrics (Recall 포함) CSV — 통계 보고 투명성
    from src.utils.io import LABEL_NAMES as _LABEL_NAMES, LABEL_MAP as _LABEL_MAP
    raw_rows = []
    for row in all_rows:
        excl = row['excluded_attack']
        excl_id = _LABEL_MAP.get(excl, -1)
        raw_row = {
            'excluded_attack': excl,
            'seed': row['seed'],
            'unknown_rate': row['unknown_rate'],
            'cae_anomaly_recall': row.get('cae_anomaly_recall', float('nan')),
            'normal_fpr': row.get('normal_fpr', float('nan')),
            'known_accuracy': row.get('known_accuracy', float('nan')),
            'known_macro_f1': row.get('known_macro_f1', float('nan')),
            'conf_thr': row.get('conf_thr', float('nan')),
        }
        raw_rows.append(raw_row)
    df_raw = pd.DataFrame(raw_rows)
    raw_path = os.path.join(out_tables, 'loao_per_seed_raw.csv')
    df_raw.to_csv(raw_path, index=False)
    print(f'Saved: {raw_path}')

    provenance_path = os.path.join(out_tables, 'loao.provenance.json')
    Path(provenance_path).write_text(json.dumps(result_bundle_provenance(
        manifest,
        {'per_fold': per_fold_path, 'summary': summary_path},
        'configs/model.yaml', 'configs/train.yaml', 'configs/cae.yaml',
        'configs/experiment.yaml',
        routing_mode=routing_mode,
        use_cae=use_cae,
        folds=folds,
        seeds=seeds,
    ), indent=2), encoding='utf-8')
    print(f'Saved: {summary_path}')
    print(f'Saved: {provenance_path}')
    print(df_summary.to_string(index=False))


if __name__ == '__main__':
    main()
