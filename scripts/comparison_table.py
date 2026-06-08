"""Build a consistent S1/S2/S3 comparison on the full frozen test set."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.cae import CAE
from src.models.dcnn import DCNN
from src.pipeline.two_stage import TwoStagePipeline
from src.train.common import load_manifest, select_device
from src.train.train_s1 import binarize, train_one_seed as train_s1_seed
from src.train.train_s2 import train_one_seed as train_s2_seed
from src.utils.benchmark import benchmark_predict
from src.utils.config import load_experiment_config, resolve_conf_threshold
from src.utils.io import NUM_CLASSES, load_dataset


def _predict_dcnn(
    model: DCNN,
    X: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[start:start + batch_size]).to(device)
            predictions.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions).astype(np.int64)


def _predict_two_stage(
    pipeline: TwoStagePipeline,
    X: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    predictions = []
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[start:start + batch_size]).to(device)
        predictions.extend(pipeline.predict(xb)['class_id'])
    return np.asarray(predictions, dtype=np.int64)


def _normal_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    normal = y_true == 0
    return float((y_pred[normal] != 0).mean()) if normal.any() else float('nan')


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: list[int],
) -> dict[str, float]:
    known = np.isin(y_pred, labels)
    coverage = float(known.mean())
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(
            y_true, y_pred, labels=labels, average='macro', zero_division=0)),
        'coverage': coverage,
        'unknown_rate': float((~known).mean()),
        'selective_accuracy': float('nan'),
        'selective_macro_f1': float('nan'),
    }
    if known.any():
        metrics['selective_accuracy'] = float(accuracy_score(y_true[known], y_pred[known]))
        metrics['selective_macro_f1'] = float(f1_score(
            y_true[known], y_pred[known], labels=labels,
            average='macro', zero_division=0))
    return metrics


def _aggregate(rows: list[dict], numeric_columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    n = len(df)
    std = {column: float(df[column].std(ddof=1)) if n > 1 else 0.0
           for column in numeric_columns}
    mean_row = {'seed': 'mean', **{column: df[column].mean() for column in numeric_columns}}
    std_row = {'seed': 'std', **std}
    ci_row = {'seed': '95%CI',
              **{column: 1.96 * std[column] / np.sqrt(n) for column in numeric_columns}}
    return pd.concat([df, pd.DataFrame([mean_row, std_row, ci_row])], ignore_index=True)


def _load_cae(output_dir: Path, device: torch.device) -> tuple[CAE, float]:
    import json
    checkpoint = torch.load(
        output_dir / 'checkpoints' / 'cae_best.pth',
        map_location=device, weights_only=True)
    cae = CAE(
        input_shape=tuple(checkpoint['input_shape']),
        latent_dim=int(checkpoint['latent_dim']),
        noise_std=float(checkpoint.get('noise_std', 0.05)),
    ).to(device)
    cae.load_state_dict(checkpoint['model_state_dict'])
    cae.eval()
    tau_data = json.loads(
        (output_dir / 'tables' / 'tau_values.json').read_text(encoding='utf-8'))
    tau = float(tau_data[tau_data.get('headline_tau', 'tau_2sigma')])
    return cae, tau


def _load_dcnn(path: Path, num_classes: int, device: torch.device) -> DCNN:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = DCNN(
        num_classes=num_classes,
        dropout=float(checkpoint.get('dropout', 0.5)),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true', help='use seed 0 only')
    parser.add_argument('--out-dir', default=None)
    args = parser.parse_args()

    cfg_train = OmegaConf.load('configs/train.yaml').train
    cfg_exp = load_experiment_config()
    output_dir = Path(args.out_dir or str(cfg_exp.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device()

    X_train, y_train, _ = load_dataset(cfg_exp.train_npz_path)
    X_test, y_test, _ = load_dataset(cfg_exp.test_npz_path)
    manifest = load_manifest(cfg_exp.manifest_path, len(X_train), len(X_test))
    conf_thr = resolve_conf_threshold(cfg_exp, manifest)

    seeds = [0] if args.smoke else list(cfg_train.seeds)
    max_epochs = 1 if args.smoke else None
    batch_size = int(cfg_exp.get('benchmark_batch_size', 64))
    warmup = int(cfg_exp.get('benchmark_warmup', 2))
    repeats = int(cfg_exp.get('benchmark_repeats', 5))
    numeric_columns = [
        'accuracy', 'macro_f1', 'attack_f1', 'normal_fpr',
        'coverage', 'selective_accuracy', 'selective_macro_f1',
        'unknown_rate', 'loao_unknown_rate',
        'latency_ms_mean', 'latency_ms_std', 'latency_ms_p50',
        'latency_ms_p95', 'samples_per_sec',
    ]

    loao_unknown_rate = float('nan')
    loao_path = output_dir / 'tables' / 'loao_summary.csv'
    if loao_path.exists():
        loao = pd.read_csv(loao_path)
        grand = loao[loao['excluded_attack'] == 'grand_mean']
        if len(grand):
            loao_unknown_rate = float(grand['unknown_rate_mean'].iloc[0])

    results = []
    y_binary_test = binarize(y_test)
    y_binary_train = binarize(y_train)

    for stage in ('S1_binary', 'S2_6class', 'S3_twostage'):
        rows = []
        for seed in seeds:
            if stage == 'S1_binary':
                checkpoint_path = output_dir / 'checkpoints' / f's1_seed_{seed}_best.pth'
                if not checkpoint_path.exists():
                    train_s1_seed(
                        seed, X_train, y_binary_train, X_test, y_binary_test,
                        manifest, cfg_train, device, max_epochs,
                        out_dir=str(output_dir),
                    )
                model = _load_dcnn(checkpoint_path, 2, device)
                predict = lambda: _predict_dcnn(model, X_test, device, batch_size)
                y_pred, timing = benchmark_predict(
                    predict, n_samples=len(X_test), device=device,
                    warmup=warmup, repeats=repeats)
                metrics = _classification_metrics(
                    y_binary_test, y_pred, labels=[0, 1])
                metrics['attack_f1'] = float(f1_score(
                    y_binary_test, y_pred, average='binary', zero_division=0))
                metrics['normal_fpr'] = _normal_fpr(y_binary_test, y_pred)
                metrics['loao_unknown_rate'] = 0.0
                task = 'binary'
            elif stage == 'S2_6class':
                checkpoint_path = output_dir / 'checkpoints' / f's2_seed_{seed}_best.pth'
                if not checkpoint_path.exists():
                    train_s2_seed(
                        seed, X_train, y_train, X_test, y_test, manifest,
                        cfg_train, device, str(output_dir), max_epochs)
                model = _load_dcnn(checkpoint_path, NUM_CLASSES, device)
                predict = lambda: _predict_dcnn(model, X_test, device, batch_size)
                y_pred, timing = benchmark_predict(
                    predict, n_samples=len(X_test), device=device,
                    warmup=warmup, repeats=repeats)
                metrics = _classification_metrics(
                    y_test, y_pred, labels=list(range(NUM_CLASSES)))
                metrics['attack_f1'] = float('nan')
                metrics['normal_fpr'] = _normal_fpr(y_test, y_pred)
                metrics['loao_unknown_rate'] = 0.0
                task = '6class'
            else:
                cae, tau = _load_cae(output_dir, device)
                model = _load_dcnn(
                    output_dir / 'checkpoints' / f's2_seed_{seed}_best.pth',
                    NUM_CLASSES, device)
                pipeline = TwoStagePipeline(cae, model, tau, conf_thr, use_cae=True)
                predict = lambda: _predict_two_stage(
                    pipeline, X_test, device, batch_size)
                y_pred, timing = benchmark_predict(
                    predict, n_samples=len(X_test), device=device,
                    warmup=warmup, repeats=repeats)
                metrics = _classification_metrics(
                    y_test, y_pred, labels=list(range(NUM_CLASSES)))
                metrics['attack_f1'] = float('nan')
                metrics['normal_fpr'] = _normal_fpr(y_test, y_pred)
                metrics['loao_unknown_rate'] = loao_unknown_rate
                task = '6class_with_unknown'

            rows.append({'seed': seed, 'task': task, **metrics, **timing})

        aggregated = _aggregate(rows, numeric_columns)
        mean = aggregated[aggregated['seed'] == 'mean'].iloc[0].to_dict()
        mean['stage'] = stage
        mean['task'] = rows[0]['task']
        results.append(mean)

    comparison = pd.DataFrame(results)
    ordered = ['stage', 'task'] + numeric_columns
    comparison = comparison[ordered]
    tables_dir = output_dir / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / 'comparison_table.csv'
    md_path = tables_dir / 'comparison_table.md'
    comparison.to_csv(csv_path, index=False)
    md_path.write_text(
        '# Model Comparison (full frozen test set)\n\n'
        + comparison.to_markdown(index=False, floatfmt='.4f') + '\n',
        encoding='utf-8')
    print(comparison.to_string(index=False))
    print(f'Saved: {csv_path}')
    print(f'Saved: {md_path}')


if __name__ == '__main__':
    main()
