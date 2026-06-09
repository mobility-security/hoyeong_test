"""Validate active datasets, manifest, checkpoints, and threshold provenance."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train.common import (
    load_manifest, validate_artifact_provenance, validate_checkpoint_provenance,
)
from src.utils.config import load_experiment_config
from src.utils.io import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true', help='validate seed 0 only')
    args = parser.parse_args()
    cfg = load_experiment_config()
    output_dir = Path(str(cfg.output_dir))
    X_train, _, _ = load_dataset(cfg.train_npz_path)
    X_test, _, _ = load_dataset(cfg.test_npz_path)
    manifest = load_manifest(
        cfg.manifest_path, len(X_train), len(X_test),
        train_npz_path=cfg.train_npz_path, test_npz_path=cfg.test_npz_path)

    configured_seeds = list(OmegaConf.load('configs/train.yaml').train.seeds)
    seeds = [0] if args.smoke else configured_seeds
    for prefix in ('s1', 's2'):
        for seed in seeds:
            path = output_dir / 'checkpoints' / f'{prefix}_seed_{seed}_best.pth'
            if not path.exists():
                raise FileNotFoundError(f'required checkpoint is missing: {path}')
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            validate_checkpoint_provenance(
                checkpoint, manifest, 'configs/model.yaml', 'configs/train.yaml')

    if bool(cfg.get('use_cae', True)):
        cae_path = output_dir / 'checkpoints' / 'cae_best.pth'
        tau_path = output_dir / 'tables' / 'tau_values.json'
        if not cae_path.exists() or not tau_path.exists():
            raise FileNotFoundError('CAE checkpoint or tau artifact is missing')
        validate_checkpoint_provenance(
            torch.load(cae_path, map_location='cpu', weights_only=True),
            manifest, 'configs/cae.yaml')
        validate_artifact_provenance(
            json.loads(tau_path.read_text(encoding='utf-8')), manifest, 'tau artifact')
    print('[OK] active datasets, manifest, and checkpoints are compatible')


if __name__ == '__main__':
    main()
