"""Generate deterministic contract-compliant datasets for smoke tests."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.io import build_meta, load_dataset, save_dataset  # noqa: E402


def build_stub_dataset(
    path: str | Path,
    *,
    seed: int,
    n_groups: int,
    samples_per_group: int = 10,
) -> Path:
    """Create class-covered capture groups with lightweight learnable signal."""
    if n_groups < 6:
        raise ValueError('n_groups must be at least 6')
    if samples_per_group < 1:
        raise ValueError('samples_per_group must be positive')
    path = Path(path)
    rng = np.random.default_rng(seed)
    group_labels = np.arange(n_groups, dtype=np.int64) % 6
    y = np.repeat(group_labels, samples_per_group)
    pcap_id = np.repeat(np.arange(n_groups, dtype=np.int64), samples_per_group)
    X = rng.random((len(y), 3, 32, 32), dtype=np.float32) * 0.35
    X[:, 0] += y[:, None, None].astype(np.float32) * 0.1
    X = np.clip(X, 0.0, 1.0).astype(np.float32)
    starts_one = np.arange(samples_per_group, dtype=np.int64) * 64
    packet_start = np.tile(starts_one, n_groups)
    packet_end = packet_start + 64
    meta = build_meta(
        32, 32, seed=seed, stub=True,
        group_semantics='capture',
        packet_range_semantics='zero_based_half_open',
        created_by='make_stub_dataset.py',
        note='STUB dataset - synthetic class-conditional signal, NOT real traffic',
    )
    return save_dataset(
        path, X, y, meta, pcap_id=pcap_id,
        packet_start=packet_start, packet_end=packet_end)


def _print_hash(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f'Stub saved: {path}  sha256={digest}')
    return digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='data/processed')
    parser.add_argument('--smoke-layout', action='store_true',
                        help='write distinct dataset_train/test.npz files')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke_layout:
        outputs = [
            (out_dir / 'dataset_train.npz', 42, 24),
            (out_dir / 'dataset_test.npz', 314, 12),
        ]
    else:
        outputs = [(out_dir / 'dataset_v0.npz', 42, 20)]

    hashes = []
    for path, seed, n_groups in outputs:
        if path.exists() and not args.force:
            print(f'Already exists: {path}, skipping.')
        else:
            build_stub_dataset(path, seed=seed, n_groups=n_groups)
        hashes.append((path, _print_hash(path)))
        X, y, _ = load_dataset(path)
        print(f'  validate_schema: PASS  X={X.shape} y={y.shape}')

    if not args.smoke_layout:
        path, digest = hashes[0]
        X, y, _ = load_dataset(path)
        Path('DATASET.md').write_text(
            f'## dataset_v0.npz\n\nN={len(y)}, shape={X.shape}, labels 0-5 (stub)\n\n'
            f'sha256: {digest}\n',
            encoding='utf-8',
        )


if __name__ == '__main__':
    main()
