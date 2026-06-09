"""Build a consistent S1/S2/S3 comparison on the full frozen test set."""
from __future__ import annotations

import argparse
import json
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
from src.train.common import (
    load_manifest, select_device, validate_artifact_provenance,
    validate_checkpoint_provenance,
)
from src.train.train_s1 import binarize
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
        'weighted_f1': float(f1_score(
            y_true, y_pred, labels=labels, average='weighted', zero_division=0)),
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
    for label in labels:
        mask = y_true == label
        metrics[f'recall_class_{label}'] = (
            float((y_pred[mask] == label).mean()) if mask.any() else float('nan'))
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


def _to_markdown(frame: pd.DataFrame) -> str:
    """Serialize a DataFrame without pandas' optional tabulate dependency."""
    columns = [str(column) for column in frame.columns]

    def format_value(value) -> str:
        if pd.isna(value):
            return ''
        if isinstance(value, (float, np.floating)):
            return f'{float(value):.4f}'
        return str(value).replace('|', '\\|')

    lines = [
        '| ' + ' | '.join(columns) + ' |',
        '| ' + ' | '.join(['---'] * len(columns)) + ' |',
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join(format_value(value) for value in row) + ' |')
    return '\n'.join(lines)


def _load_cae(
    output_dir: Path,
    device: torch.device,
    manifest: dict,
) -> tuple[CAE, float]:
    import json
    checkpoint = torch.load(
        output_dir / 'checkpoints' / 'cae_best.pth',
        map_location=device, weights_only=True)
    validate_checkpoint_provenance(checkpoint, manifest, 'configs/cae.yaml')
    cae = CAE(
        input_shape=tuple(checkpoint['input_shape']),
        latent_dim=int(checkpoint['latent_dim']),
        noise_std=float(checkpoint.get('noise_std', 0.05)),
    ).to(device)
    cae.load_state_dict(checkpoint['model_state_dict'])
    cae.eval()
    tau_data = json.loads(
        (output_dir / 'tables' / 'tau_values.json').read_text(encoding='utf-8'))
    validate_artifact_provenance(tau_data, manifest, 'tau artifact')
    tau = float(tau_data[tau_data.get('headline_tau', 'tau_2sigma')])
    return cae, tau


def _load_dcnn(
    path: Path,
    num_classes: int,
    device: torch.device,
    manifest: dict,
) -> DCNN:
    if not path.exists():
        raise FileNotFoundError(
            f'required checkpoint is missing: {path}; run the training stage first')
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    validate_checkpoint_provenance(
        checkpoint, manifest, 'configs/model.yaml', 'configs/train.yaml')
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
    manifest = load_manifest(
        cfg_exp.manifest_path, len(X_train), len(X_test),
        train_npz_path=cfg_exp.train_npz_path,
        test_npz_path=cfg_exp.test_npz_path)
    conf_thr = resolve_conf_threshold(cfg_exp, manifest)

    seeds = [0] if args.smoke else list(cfg_train.seeds)
    max_epochs = 1 if args.smoke else None
    batch_size = int(cfg_exp.get('benchmark_batch_size', 64))
    warmup = int(cfg_exp.get('benchmark_warmup', 2))
    repeats = int(cfg_exp.get('benchmark_repeats', 5))
    numeric_columns = [
        'accuracy', 'macro_f1', 'weighted_f1', 'attack_f1', 'normal_fpr',
        'coverage', 'selective_accuracy', 'selective_macro_f1',
        'unknown_rate', 'loao_unknown_rate',
        'latency_ms_mean', 'latency_ms_std', 'latency_ms_p50',
        'latency_ms_p95', 'samples_per_sec',
    ] + [f'recall_class_{class_id}' for class_id in range(NUM_CLASSES)]

    loao_unknown_rate = float('nan')
    loao_path = output_dir / 'tables' / 'loao_summary.csv'
    if loao_path.exists():
        loao = pd.read_csv(loao_path)
        grand = loao[loao['excluded_attack'] == 'grand_mean']
        if len(grand):
            loao_unknown_rate = float(grand['unknown_rate_mean'].iloc[0])

    results = []
    detailed_results = []
    y_binary_test = binarize(y_test)
    y_binary_train = binarize(y_train)

    stages = ['S1_binary', 'S2_6class']
    if bool(cfg_exp.get('use_cae', True)):
        stages.append('S3_twostage')
    for stage in stages:
        rows = []
        for seed in seeds:
            if stage == 'S1_binary':
                checkpoint_path = output_dir / 'checkpoints' / f's1_seed_{seed}_best.pth'
                model = _load_dcnn(checkpoint_path, 2, device, manifest)
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
                model = _load_dcnn(checkpoint_path, NUM_CLASSES, device, manifest)
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
                cae, tau = _load_cae(output_dir, device, manifest)
                model = _load_dcnn(
                    output_dir / 'checkpoints' / f's2_seed_{seed}_best.pth',
                    NUM_CLASSES, device, manifest)
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
            for class_id in range(NUM_CLASSES):
                rows[-1].setdefault(f'recall_class_{class_id}', float('nan'))

        aggregated = _aggregate(rows, numeric_columns)
        aggregated.insert(0, 'stage', stage)
        aggregated['task'] = rows[0]['task']
        aggregated = aggregated[['stage', 'task'] + [
            column for column in aggregated.columns
            if column not in {'stage', 'task'}]]
        detailed_results.append(aggregated)
        mean = aggregated[aggregated['seed'] == 'mean'].iloc[0].to_dict()
        std = aggregated[aggregated['seed'] == 'std'].iloc[0]
        ci = aggregated[aggregated['seed'] == '95%CI'].iloc[0]
        summary = {
            'schema_version': 2,
            'stage': stage,
            'task': rows[0]['task'],
            'supports_unknown': stage == 'S3_twostage',
            'uses_cae': stage == 'S3_twostage',
        }
        for column in numeric_columns:
            summary[column] = mean[column]
            summary[f'{column}_std'] = std[column]
            summary[f'{column}_ci95'] = ci[column]
        results.append(summary)

    comparison = pd.DataFrame(results)
    tables_dir = output_dir / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / 'comparison_table.csv'
    md_path = tables_dir / 'comparison_table.md'
    detailed_path = tables_dir / 'comparison_table_detailed.csv'
    tex_path = tables_dir / 'comparison_table.tex'
    comparison.to_csv(csv_path, index=False)
    pd.concat(detailed_results, ignore_index=True).to_csv(detailed_path, index=False)
    md_path.write_text(
        '# Model Comparison (full frozen test set)\n\n'
        + _to_markdown(comparison) + '\n',
        encoding='utf-8')
    tex_path.write_text(comparison.to_latex(index=False, float_format='%.4f'), encoding='utf-8')
    (tables_dir / 'comparison_table.provenance.json').write_text(json.dumps({
        'schema_version': 2,
        'manifest_sha256': manifest['sha256'],
        'train_sha256': manifest['train_sha256'],
        'test_sha256': manifest['test_sha256'],
        'use_cae': bool(cfg_exp.get('use_cae', True)),
        'seeds': seeds,
    }, indent=2), encoding='utf-8')
    print(comparison.to_string(index=False))
    print(f'Saved: {csv_path}')
    print(f'Saved: {md_path}')
    print(f'Saved: {detailed_path}')
    print(f'Saved: {tex_path}')


if __name__ == '__main__':
    main()
