"""Shared training helpers for S1 and S2."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.split import (
    validate_manifest_indices,
    validate_manifest_provenance,
    validate_trainval_indices,
)
from src.utils.io import sha256_file


class EarlyStopping:
    """Metric-based early stopping with CPU-resident best weights."""

    def __init__(self, patience: int = 5, mode: str = 'min'):
        if patience < 1:
            raise ValueError(f'patience must be positive, got {patience}')
        if mode not in ('min', 'max'):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        self.patience = patience
        self.mode = mode
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.best_weights = None

    def _improved(self, score: float) -> bool:
        return score < self.best if self.mode == 'min' else score > self.best

    def step(self, score: float, model: nn.Module) -> bool:
        """Return True when training should stop."""
        if not np.isfinite(score):
            raise ValueError(f'early-stopping score must be finite, got {score}')
        if self._improved(score):
            self.best = float(score)
            self.counter = 0
            self.best_weights = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


def make_supervised_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
    pin_memory: bool = False,
) -> DataLoader:
    if batch_size < 1:
        raise ValueError(f'batch_size must be positive, got {batch_size}')
    if len(X) != len(y):
        raise ValueError(f'len(X)={len(X)} != len(y)={len(y)}')
    if len(X) == 0:
        raise ValueError('cannot create a DataLoader from an empty split')
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=pin_memory,
    )


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_manifest(
    path: str | Path,
    n_train: int,
    n_test: int | None = None,
    *,
    train_npz_path: str | Path | None = None,
    test_npz_path: str | Path | None = None,
) -> dict:
    path = Path(path)
    try:
        with path.open(encoding='utf-8') as file:
            manifest = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'failed to load split manifest {path}: {exc}') from exc
    if not isinstance(manifest, dict):
        raise ValueError(f'split manifest must be a JSON object, got {type(manifest)}')
    if n_test is None:
        validate_trainval_indices(manifest, n_train)
    else:
        validate_manifest_indices(manifest, n_train, n_test)
    train_path = train_npz_path or manifest.get('train_source')
    test_path = test_npz_path or manifest.get('test_source')
    if train_path is None or test_path is None:
        raise ValueError('dataset paths are required for manifest provenance validation')
    validate_manifest_provenance(manifest, train_path, test_path)
    return manifest


def config_sha256(*paths: str | Path) -> str:
    """Hash ordered configuration files into one reproducibility identifier."""
    digest = hashlib.sha256()
    for path in paths:
        path = Path(path)
        digest.update(str(path).encode('utf-8'))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def checkpoint_provenance(manifest: dict, *config_paths: str | Path) -> dict:
    """Build the provenance payload embedded into every model checkpoint."""
    required = {'sha256', 'train_sha256', 'test_sha256'}
    missing = required - set(manifest)
    if missing:
        raise ValueError(f'manifest missing checkpoint provenance: {sorted(missing)}')
    return {
        'manifest_sha256': manifest['sha256'],
        'train_sha256': manifest['train_sha256'],
        'test_sha256': manifest['test_sha256'],
        'config_sha256': config_sha256(*config_paths),
    }


def validate_checkpoint_provenance(
    checkpoint: dict,
    manifest: dict,
    *config_paths: str | Path,
) -> None:
    """Reject checkpoints trained against another dataset or split manifest."""
    for artifact_key, manifest_key in (
        ('manifest_sha256', 'sha256'),
        ('train_sha256', 'train_sha256'),
        ('test_sha256', 'test_sha256'),
    ):
        if (not checkpoint.get(artifact_key)
                or checkpoint[artifact_key] != manifest.get(manifest_key)):
            raise ValueError(
                f'checkpoint {artifact_key} does not match the active manifest')
    if config_paths:
        expected_config_sha = config_sha256(*config_paths)
        if checkpoint.get('config_sha256') != expected_config_sha:
            raise ValueError('checkpoint config_sha256 does not match active configs')


def validate_artifact_provenance(
    artifact: dict,
    manifest: dict,
    artifact_name: str,
) -> None:
    """Validate non-checkpoint JSON/CSV sidecar provenance fields."""
    for artifact_key, manifest_key in (
        ('manifest_sha256', 'sha256'),
        ('train_sha256', 'train_sha256'),
        ('test_sha256', 'test_sha256'),
    ):
        if (not artifact.get(artifact_key)
                or artifact[artifact_key] != manifest.get(manifest_key)):
            raise ValueError(
                f'{artifact_name} {artifact_key} does not match the active manifest')


def result_bundle_provenance(
    manifest: dict,
    artifacts: dict[str, str | Path],
    *config_paths: str | Path,
    **extra,
) -> dict:
    """Build provenance for result tables/figures derived from checkpoints."""
    payload = {
        'schema_version': 2,
        'manifest_sha256': manifest['sha256'],
        'train_sha256': manifest['train_sha256'],
        'test_sha256': manifest['test_sha256'],
        'config_sha256': config_sha256(*config_paths),
        'artifacts': {
            name: {'path': str(path), 'sha256': sha256_file(path)}
            for name, path in artifacts.items()
        },
    }
    payload.update(extra)
    return payload


def validate_result_bundle(
    provenance: dict,
    manifest: dict,
    artifacts: dict[str, str | Path],
    *config_paths: str | Path,
) -> None:
    """Reject stale or modified result files before downstream reporting."""
    validate_artifact_provenance(provenance, manifest, 'result bundle')
    if provenance.get('config_sha256') != config_sha256(*config_paths):
        raise ValueError('result bundle config_sha256 does not match active configs')
    recorded = provenance.get('artifacts')
    if not isinstance(recorded, dict):
        raise ValueError('result bundle is missing artifact hashes')
    for name, path in artifacts.items():
        entry = recorded.get(name)
        if not isinstance(entry, dict) or entry.get('sha256') != sha256_file(path):
            raise ValueError(f'result artifact does not match provenance: {name}')
