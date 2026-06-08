"""Shared training helpers for S1 and S2."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.split import validate_manifest_indices, validate_trainval_indices


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


def load_manifest(path: str | Path, n_train: int, n_test: int | None = None) -> dict:
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
    return manifest
