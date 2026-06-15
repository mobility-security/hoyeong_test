"""Generate batch-safe, single-seed S2/S3 confusion matrices."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.comparison_table import _load_cae, _load_dcnn, _predict_dcnn, _predict_two_stage
from src.pipeline.two_stage import TwoStagePipeline
from src.train.common import load_manifest, select_device
from src.utils.config import load_experiment_config, resolve_conf_threshold
from src.utils.io import LABEL_NAMES, NUM_CLASSES, load_dataset


def _save_cm(cm: np.ndarray, class_names: list[str], title: str, out_path: Path) -> None:
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, matrix, fmt, tag in (
        (axes[0], cm_norm, '.3f', 'Normalized'),
        (axes[1], cm, 'd', 'Raw counts'),
    ):
        sns.heatmap(matrix, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'{title} - {tag}')
        ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    cfg = load_experiment_config()
    output_dir = Path(str(cfg.output_dir))
    device = select_device()
    batch_size = int(cfg.get('benchmark_batch_size', 64))

    X_train, _, _ = load_dataset(cfg.train_npz_path)
    X_test, y_test, _ = load_dataset(cfg.test_npz_path)
    manifest = load_manifest(
        cfg.manifest_path, len(X_train), len(X_test),
        train_npz_path=cfg.train_npz_path, test_npz_path=cfg.test_npz_path)
    s2 = _load_dcnn(
        output_dir / 'checkpoints' / f's2_seed_{args.seed}_best.pth',
        NUM_CLASSES, device, manifest)

    y_pred_s2 = _predict_dcnn(s2, X_test, device, batch_size)
    cm_s2 = confusion_matrix(y_test, y_pred_s2, labels=list(range(NUM_CLASSES)))
    _save_cm(cm_s2, LABEL_NAMES, f'S2 seed={args.seed}',
             output_dir / 'figures' / 'cm_s2_6class.png')

    if not bool(cfg.get('use_cae', True)):
        print('CAE disabled; S3 confusion matrix was not generated.')
        return
    cae, tau = _load_cae(output_dir, device, manifest)
    conf_thr = resolve_conf_threshold(cfg, manifest)
    pipeline = TwoStagePipeline(
        cae, s2, tau, conf_thr, use_cae=True,
        routing_mode=str(cfg.get('routing_mode', 'strict_cascade')))
    y_pred_s3 = _predict_two_stage(pipeline, X_test, device, batch_size)
    cm_s3 = confusion_matrix(
        y_test, y_pred_s3, labels=list(range(NUM_CLASSES + 1)))
    _save_cm(cm_s3, LABEL_NAMES + ['Unknown'], f'S3 seed={args.seed}',
             output_dir / 'figures' / 'cm_s3_7class.png')


if __name__ == '__main__':
    main()
