"""
3종 비교표: S1(이진) / S2(6-class) / S3(2단계 파이프라인) 를 동일 frozen test에서 평가.

출력:
  results/tables/comparison_table.csv
  results/tables/comparison_table.md

사용법:
  python scripts/comparison_table.py           # 전체 (5 seed)
  python scripts/comparison_table.py --smoke   # 1 epoch, seed 0만
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.cae import CAE
from src.models.dcnn import DCNN
from src.train.common import EarlyStopping, load_manifest, make_supervised_loader, select_device
from src.train.train_cae import compute_mse_batch
from src.train.train_s1 import binarize, train_one_seed as train_s1_seed
from src.train.train_s2 import train_one_seed as train_s2_seed
from src.utils.focal_loss import FocalLoss
from src.utils.io import LABEL_NAMES, NUM_CLASSES, load_dataset
from src.utils.seed import set_seed

SEEDS = [0, 1, 2, 3, 4]
OUT_DIR = 'results'


# ---------------------------------------------------------------------------
# S3 helpers (two-stage)
# ---------------------------------------------------------------------------

def _load_cae(device: torch.device) -> tuple[CAE, float]:
    import json
    ckpt = torch.load('results/checkpoints/cae_best.pth',
                      map_location=device, weights_only=True)
    cae = CAE(
        input_shape=tuple(ckpt['input_shape']),
        latent_dim=int(ckpt['latent_dim']),
        noise_std=float(ckpt.get('noise_std', 0.05)),
    )
    cae.load_state_dict(ckpt['model_state_dict'])
    cae.eval().to(device)
    with open('results/tables/tau_values.json', encoding='utf-8') as fh:
        tau_data = json.load(fh)
    tau = float(tau_data[tau_data.get('headline_tau', 'tau_2sigma')])
    return cae, tau


def _s3_predict(
    cae: CAE,
    s2_model: nn.Module,
    X: np.ndarray,
    tau: float,
    conf_thr: float,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, float]:
    """Returns (y_pred_7class, latency_ms_per_sample).

    y_pred values: 0..5 = known classes, 6 = Unknown.
    """
    mse = compute_mse_batch(cae, X, device, batch_size)
    anomalous = mse > tau
    n = len(X)
    y_pred = np.zeros(n, dtype=np.int64)
    UNKNOWN = NUM_CLASSES  # 6

    t0 = time.perf_counter()
    routed = np.flatnonzero(anomalous)
    if len(routed):
        with torch.no_grad():
            logits = s2_model(torch.from_numpy(X[routed]).to(device))
            probs = F.softmax(logits, dim=1).cpu().numpy()
        max_probs = probs.max(axis=1)
        y_pred[routed] = probs.argmax(axis=1)
        y_pred[routed[max_probs < conf_thr]] = UNKNOWN
    latency_ms = (time.perf_counter() - t0) * 1000.0 / max(n, 1)
    return y_pred, latency_ms


# ---------------------------------------------------------------------------
# Metric aggregation helpers
# ---------------------------------------------------------------------------

def _normal_fpr_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    normal_mask = y_true == 0
    if not normal_mask.any():
        return float('nan')
    return float((y_pred[normal_mask] != 0).mean())


def _normal_fpr_cae(y_true: np.ndarray, mse: np.ndarray, tau: float) -> float:
    normal_mask = y_true == 0
    if not normal_mask.any():
        return float('nan')
    return float((mse[normal_mask] > tau).mean())


def _aggregate(rows: list[dict], num_cols: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    n = len(df)
    mean_row = {'seed': 'mean', **{c: df[c].mean() for c in num_cols}}
    std_row = {'seed': 'std', **{c: df[c].std() for c in num_cols}}
    ci_row = {'seed': '95%CI',
              **{c: 1.96 * df[c].std() / (n ** 0.5) for c in num_cols}}
    return pd.concat([df, pd.DataFrame([mean_row, std_row, ci_row])], ignore_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true',
                        help='1 epoch, seed 0 only')
    parser.add_argument('--out-dir', default=OUT_DIR)
    args = parser.parse_args()

    cfg_train = OmegaConf.load('configs/train.yaml').train
    cfg_exp = OmegaConf.load('configs/experiment.yaml').experiment

    device = select_device()
    print(f'Device: {device}')

    X_tr, y_tr, _ = load_dataset(cfg_exp.train_npz_path)
    X_test, y_test, _ = load_dataset(cfg_exp.test_npz_path)
    y_bin_tr = binarize(y_tr)
    y_bin_test = binarize(y_test)

    manifest = load_manifest(cfg_exp.manifest_path, len(X_tr), len(X_test))

    seeds = [0] if args.smoke else SEEDS
    max_epochs = 1 if args.smoke else None

    out_tables = os.path.join(args.out_dir, 'tables')
    os.makedirs(out_tables, exist_ok=True)
    conf_thr = float(cfg_exp.conf_thr)

    # ----- S1: binary DCNN -----
    print('\n=== S1: Binary DCNN ===')
    s1_rows = []
    for seed in seeds:
        print(f'  seed={seed}')
        t0 = time.perf_counter()
        m = train_s1_seed(
            seed, X_tr, y_bin_tr, X_test, y_bin_test,
            manifest, cfg_train, device, max_epochs,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0 / max(len(X_test), 1)
        # S1 doesn't produce per-class labels, so macro-F1 is binary
        s1_rows.append({
            'seed': seed,
            'accuracy': m['accuracy'],
            'macro_f1': m['f1_binary'],
            'normal_fpr': m['fpr'],
            'zeroday_rate': 0.0,
            'latency_ms': latency_ms,
        })
    df_s1 = _aggregate(s1_rows, ['accuracy', 'macro_f1', 'normal_fpr', 'zeroday_rate', 'latency_ms'])

    # ----- S2: 6-class DCNN -----
    print('\n=== S2: 6-class DCNN ===')
    s2_rows = []
    for seed in seeds:
        print(f'  seed={seed}')
        # Try to load existing checkpoint
        ckpt_path = f'results/checkpoints/s2_seed_{seed}_best.pth'
        if Path(ckpt_path).exists() and max_epochs is None:
            ck = torch.load(ckpt_path, map_location=device, weights_only=True)
            s2_model = DCNN(num_classes=NUM_CLASSES,
                            dropout=float(ck.get('dropout', 0.5))).to(device)
            s2_model.load_state_dict(ck['model_state_dict'])
            s2_model.eval()
            t0 = time.perf_counter()
            with torch.no_grad():
                logits = s2_model(torch.from_numpy(X_test).to(device))
                y_pred = logits.argmax(dim=1).cpu().numpy().astype(np.int64)
            latency_ms = (time.perf_counter() - t0) * 1000.0 / max(len(X_test), 1)
        else:
            m = train_s2_seed(
                seed, X_tr, y_tr, X_test, y_test,
                manifest, cfg_train, device,
                os.path.join(args.out_dir),
                max_epochs,
            )
            ckpt_path = m['checkpoint_path']
            ck = torch.load(ckpt_path, map_location=device, weights_only=True)
            s2_model = DCNN(num_classes=NUM_CLASSES).to(device)
            s2_model.load_state_dict(ck['model_state_dict'])
            s2_model.eval()
            t0 = time.perf_counter()
            with torch.no_grad():
                logits = s2_model(torch.from_numpy(X_test).to(device))
                y_pred = logits.argmax(dim=1).cpu().numpy().astype(np.int64)
            latency_ms = (time.perf_counter() - t0) * 1000.0 / max(len(X_test), 1)

        acc = float(accuracy_score(y_test, y_pred))
        macro_f1 = float(f1_score(y_test, y_pred, labels=list(range(NUM_CLASSES)),
                                   average='macro', zero_division=0))
        normal_fpr = _normal_fpr_from_preds(y_test, y_pred)
        s2_rows.append({
            'seed': seed,
            'accuracy': acc,
            'macro_f1': macro_f1,
            'normal_fpr': normal_fpr,
            'zeroday_rate': 0.0,
            'latency_ms': latency_ms,
        })
    df_s2 = _aggregate(s2_rows, ['accuracy', 'macro_f1', 'normal_fpr', 'zeroday_rate', 'latency_ms'])

    # ----- S3: Two-stage (CAE + S2) -----
    print('\n=== S3: Two-stage (CAE + S2) ===')
    if not Path('results/checkpoints/cae_best.pth').exists():
        print('  cae_best.pth not found — skipping S3')
        df_s3 = None
    else:
        cae, tau = _load_cae(device)
        s3_rows = []
        for seed in seeds:
            print(f'  seed={seed}')
            ckpt_path = f'results/checkpoints/s2_seed_{seed}_best.pth'
            if Path(ckpt_path).exists() and max_epochs is None:
                ck = torch.load(ckpt_path, map_location=device, weights_only=True)
                s2_model = DCNN(num_classes=NUM_CLASSES,
                                dropout=float(ck.get('dropout', 0.5))).to(device)
                s2_model.load_state_dict(ck['model_state_dict'])
                s2_model.eval()
            else:
                ck = torch.load(f'results/checkpoints/s2_seed_0_best.pth' if Path('results/checkpoints/s2_seed_0_best.pth').exists() else ckpt_path,
                                map_location=device, weights_only=True)
                s2_model = DCNN(num_classes=NUM_CLASSES).to(device)
                s2_model.load_state_dict(ck['model_state_dict'])
                s2_model.eval()

            y_pred_s3, latency_ms = _s3_predict(
                cae, s2_model, X_test, tau, conf_thr, device)

            # For S3, map Unknown(6) to "anomaly" for accuracy/F1 computation
            # Use known-class accuracy only (ignore Unknown)
            known_mask = y_pred_s3 < NUM_CLASSES
            acc = float(accuracy_score(y_test[known_mask], y_pred_s3[known_mask])) if known_mask.any() else float('nan')
            macro_f1 = float(f1_score(
                y_test[known_mask], y_pred_s3[known_mask],
                labels=list(range(NUM_CLASSES)), average='macro', zero_division=0,
            )) if known_mask.any() else float('nan')
            normal_fpr = _normal_fpr_from_preds(y_test, y_pred_s3)
            # Zero-day: use LOAO summary if available, else CAE FPR on attacks
            loao_path = os.path.join(out_tables, 'loao_summary.csv')
            if Path(loao_path).exists():
                loao_df = pd.read_csv(loao_path)
                grand = loao_df[loao_df['excluded_attack'] == 'grand_mean']
                zeroday = float(grand['unknown_rate_mean'].iloc[0]) if len(grand) else float('nan')
            else:
                zeroday = float((y_pred_s3[y_test != 0] == NUM_CLASSES).mean()) if (y_test != 0).any() else float('nan')

            s3_rows.append({
                'seed': seed,
                'accuracy': acc,
                'macro_f1': macro_f1,
                'normal_fpr': normal_fpr,
                'zeroday_rate': zeroday,
                'latency_ms': latency_ms,
            })
        df_s3 = _aggregate(s3_rows, ['accuracy', 'macro_f1', 'normal_fpr', 'zeroday_rate', 'latency_ms'])

    # ----- Save comparison table -----
    num_cols = ['accuracy', 'macro_f1', 'normal_fpr', 'zeroday_rate', 'latency_ms']

    def _mean_std_row(df: pd.DataFrame | None, stage: str) -> dict:
        if df is None:
            return {'stage': stage, **{c: 'N/A' for c in num_cols}}
        mean_row = df[df['seed'] == 'mean'].iloc[0]
        std_row = df[df['seed'] == 'std'].iloc[0]
        ci_row = df[df['seed'] == '95%CI'].iloc[0]
        row = {'stage': stage}
        for c in num_cols:
            m, s, ci = float(mean_row[c]) if mean_row[c] != 'N/A' else float('nan'), \
                       float(std_row[c]) if std_row[c] != 'N/A' else float('nan'), \
                       float(ci_row[c]) if ci_row[c] != 'N/A' else float('nan')
            row[f'{c}_mean'] = m
            row[f'{c}_std'] = s
            row[f'{c}_ci95'] = ci
        return row

    comparison = pd.DataFrame([
        _mean_std_row(df_s1, 'S1_binary'),
        _mean_std_row(df_s2, 'S2_6class'),
        _mean_std_row(df_s3, 'S3_twostage'),
    ])

    csv_path = os.path.join(out_tables, 'comparison_table.csv')
    comparison.to_csv(csv_path, index=False)
    print(f'\nSaved: {csv_path}')

    # Markdown
    md_cols = ['stage'] + [f'{c}_mean' for c in num_cols]
    md_df = comparison[md_cols].rename(columns={
        f'{c}_mean': c for c in num_cols
    })
    md_lines = ['# Model Comparison (frozen test set)', '',
                md_df.to_markdown(index=False, floatfmt='.4f'), '']
    md_path = os.path.join(out_tables, 'comparison_table.md')
    Path(md_path).write_text('\n'.join(md_lines), encoding='utf-8')
    print(f'Saved: {md_path}')
    print(md_df.to_string(index=False))


if __name__ == '__main__':
    main()
