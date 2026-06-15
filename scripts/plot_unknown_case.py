"""Visualize a real held-out LOAO sample that the pipeline predicted Unknown."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.leave_one_out import UNKNOWN_LABEL, _predict_loao, load_cae
from src.models.dcnn import DCNN
from src.train.common import (
    load_manifest, select_device, validate_checkpoint_provenance,
    validate_result_bundle,
)
from src.utils.config import load_experiment_config
from src.utils.io import LABEL_MAP, LABEL_NAMES, load_dataset


def _to_rgb(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).transpose(1, 2, 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack', choices=list(LABEL_MAP)[1:], default='F_I')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--allow-missing', action='store_true',
                        help='exit successfully when no actual Unknown case exists')
    args = parser.parse_args()

    cfg = load_experiment_config()
    if not bool(cfg.get('use_cae', True)):
        raise ValueError('Unknown CAE reconstruction figure requires experiment.use_cae=true')

    from src.train.train_cae import compute_mse_batch, compute_tau as _compute_tau

    output_dir = Path(str(cfg.output_dir))
    X_train, y_train, _ = load_dataset(cfg.train_npz_path)
    X_test, y_test, _ = load_dataset(cfg.test_npz_path)
    manifest = load_manifest(
        cfg.manifest_path, len(X_train), len(X_test),
        train_npz_path=cfg.train_npz_path, test_npz_path=cfg.test_npz_path)
    device = select_device()

    cae = load_cae(output_dir / 'checkpoints' / 'cae_best.pth', device, manifest)
    _val_idx = np.asarray(manifest['val_idx'], dtype=np.int64)
    _normal_val_mask = y_train[_val_idx] == 0
    _mse_nv = compute_mse_batch(cae, X_train[_val_idx][_normal_val_mask], device)
    tau = float(_compute_tau(_mse_nv)['tau_2sigma'])
    s2_path = (output_dir / 'checkpoints' / 'loao'
               / f'exclude_{args.attack}_seed_{args.seed}.pth')
    checkpoint = torch.load(s2_path, map_location=device, weights_only=True)
    validate_checkpoint_provenance(
        checkpoint, manifest, 'configs/model.yaml', 'configs/train.yaml')
    if checkpoint.get('excluded_attack') != args.attack:
        raise ValueError('LOAO checkpoint excluded class does not match the requested attack')
    s2 = DCNN(
        num_classes=int(checkpoint['num_classes']),
        dropout=float(checkpoint.get('dropout', 0.5)),
    ).to(device)
    s2.load_state_dict(checkpoint['model_state_dict'])
    s2.eval()

    rows = pd.read_csv(output_dir / 'tables' / 'loao_per_fold.csv')
    provenance_path = output_dir / 'tables' / 'loao.provenance.json'
    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    validate_result_bundle(
        provenance, manifest,
        {
            'per_fold': output_dir / 'tables' / 'loao_per_fold.csv',
            'summary': output_dir / 'tables' / 'loao_summary.csv',
        },
        'configs/model.yaml', 'configs/train.yaml', 'configs/cae.yaml',
        'configs/experiment.yaml')
    selected_row = rows[
        (rows['excluded_attack'] == args.attack) & (rows['seed'] == args.seed)]
    if selected_row.empty:
        raise ValueError('LOAO result row is missing for the requested fold/seed')
    conf_thr = float(selected_row.iloc[0]['conf_thr'])

    attack_id = LABEL_MAP[args.attack]
    attack_indices = np.flatnonzero(y_test == attack_id)
    if len(attack_indices) == 0:
        raise ValueError(f'test dataset has no {args.attack} samples')
    X_attack = X_test[attack_indices]
    mse, predictions = _predict_loao(
        cae, s2, X_attack, tau, conf_thr, device, use_cae=True)
    unknown_local = np.flatnonzero(predictions == UNKNOWN_LABEL)
    if len(unknown_local) == 0:
        if args.allow_missing:
            print(f'[WARN] fold {args.attack}/seed {args.seed} has no actual Unknown sample')
            return
        raise ValueError(
            f'fold {args.attack}/seed {args.seed} produced no actual Unknown sample')

    local_index = int(unknown_local[np.argmax(mse[unknown_local])])
    dataset_index = int(attack_indices[local_index])
    sample = X_attack[local_index]
    with torch.no_grad():
        tensor = torch.from_numpy(sample[None]).to(device)
        reconstruction = cae(tensor)[0].cpu().numpy()
        probabilities = F.softmax(s2(tensor), dim=1)[0].cpu().numpy()

    squared_error = ((reconstruction - sample) ** 2).mean(axis=0)
    max_probability = float(probabilities.max())
    known_names = [name for name in LABEL_NAMES if name != args.attack]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    axes[0].imshow(_to_rgb(sample))
    axes[0].set_title(f'(a) Held-out {args.attack} input')
    axes[0].axis('off')

    axes[1].imshow(_to_rgb(reconstruction))
    axes[1].set_title('(b) CAE reconstruction')
    axes[1].axis('off')

    image = axes[2].imshow(squared_error, cmap='hot', vmin=0)
    axes[2].set_title(f'(c) Squared error\nMSE={mse[local_index]:.5f}, tau={tau:.5f}')
    axes[2].axis('off')
    fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].bar(known_names, probabilities, color='#4C78A8')
    axes[3].axhline(conf_thr, color='red', linestyle='--',
                    label=f'Unknown threshold={conf_thr:.3f}')
    axes[3].set_ylim(0, 1)
    axes[3].tick_params(axis='x', rotation=45)
    axes[3].set_ylabel('Softmax confidence')
    axes[3].set_title(f'(d) S2 confidence\nmax={max_probability:.3f} -> Unknown')
    axes[3].legend(fontsize=8)

    fig.suptitle(
        f'Actual LOAO Unknown case: exclude={args.attack}, seed={args.seed}, '
        f'test_index={dataset_index}')
    fig.tight_layout()
    figure_path = output_dir / 'figures' / 'unknown_case_4panel.png'
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    figure_path.with_suffix('.json').write_text(json.dumps({
        'excluded_attack': args.attack,
        'seed': args.seed,
        'test_index': dataset_index,
        'prediction': 'Unknown',
        'mse': float(mse[local_index]),
        'tau': tau,
        'max_probability': max_probability,
        'conf_thr': conf_thr,
        'manifest_sha256': manifest['sha256'],
        'checkpoint_path': str(s2_path),
    }, indent=2), encoding='utf-8')
    print(f'Saved: {figure_path}')


if __name__ == '__main__':
    main()
