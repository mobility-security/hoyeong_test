"""Consistent end-to-end inference benchmarking helpers."""
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import torch


def synchronize_device(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps' and hasattr(torch, 'mps'):
        torch.mps.synchronize()


def benchmark_predict(
    predict_fn: Callable[[], np.ndarray],
    *,
    n_samples: int,
    device: torch.device,
    warmup: int = 2,
    repeats: int = 5,
) -> tuple[np.ndarray, dict[str, float]]:
    """Benchmark one full-dataset prediction callable using synchronized timing."""
    if n_samples < 1:
        raise ValueError('n_samples must be positive')
    if warmup < 0 or repeats < 1:
        raise ValueError('warmup must be non-negative and repeats must be positive')
    for _ in range(warmup):
        predict_fn()
        synchronize_device(device)

    durations = []
    predictions = None
    for _ in range(repeats):
        synchronize_device(device)
        started = time.perf_counter()
        current = np.asarray(predict_fn(), dtype=np.int64)
        synchronize_device(device)
        durations.append(time.perf_counter() - started)
        if current.shape != (n_samples,):
            raise ValueError(
                f'predict_fn must return shape ({n_samples},), got {current.shape}')
        if predictions is not None and not np.array_equal(predictions, current):
            raise ValueError('predict_fn returned inconsistent predictions across repeats')
        predictions = current

    per_sample_ms = np.asarray(durations, dtype=np.float64) * 1000.0 / n_samples
    mean_duration = float(np.mean(durations))
    return predictions, {
        'latency_ms_mean': float(per_sample_ms.mean()),
        'latency_ms_std': float(per_sample_ms.std(ddof=0)),
        'latency_ms_p50': float(np.percentile(per_sample_ms, 50)),
        'latency_ms_p95': float(np.percentile(per_sample_ms, 95)),
        'samples_per_sec': float(n_samples / mean_duration),
    }
