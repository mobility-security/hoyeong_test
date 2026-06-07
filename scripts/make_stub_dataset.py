"""Generate a contract-compliant stub dataset for smoke tests."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.io import build_meta, load_dataset, save_dataset  # noqa: E402

OUT_PATH = Path('data/processed/dataset_v0.npz')


def main() -> None:
    if OUT_PATH.exists():
        print(f'Already exists: {OUT_PATH}, skipping.')
        return

    n_samples, channels, height, width = 200, 3, 32, 32
    rng = np.random.default_rng(42)
    X = rng.random((n_samples, channels, height, width)).astype(np.float32)
    y = rng.integers(0, 6, size=n_samples, dtype=np.int64)
    pcap_id = np.repeat(np.arange(20), n_samples // 20).astype(np.int64)
    meta = build_meta(
        height,
        width,
        created_by='make_stub_dataset.py',
        note='stub',
    )

    output = save_dataset(OUT_PATH, X, y, meta, pcap_id=pcap_id)
    with output.open('rb') as file:
        sha256 = hashlib.sha256(file.read()).hexdigest()
    print(f'Stub saved: X={X.shape}, y={y.shape}')
    print(f'sha256: {sha256}')

    dataset_md = (
        f'## dataset_v0.npz\n\nN={n_samples}, '
        f'shape=(N,3,{height},{width}), labels 0-5 (stub)\n\nsha256: {sha256}\n')
    Path('DATASET.md').write_text(dataset_md, encoding='utf-8')
    print('DATASET.md written.')

    X_loaded, y_loaded, _ = load_dataset(output)
    if X_loaded.shape != (n_samples, channels, height, width):
        raise RuntimeError(f'shape mismatch: {X_loaded.shape}')
    if y_loaded.shape != (n_samples,):
        raise RuntimeError(f'y shape mismatch: {y_loaded.shape}')
    print('validate_schema: PASS')


if __name__ == '__main__':
    main()
