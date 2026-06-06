"""
Leakage-safe train/val/test split for wavelet image datasets.

Wavelet images from the same PCAP share packets (sliding-window), so random
split causes temporal leakage.  When pcap_id is available we split by PCAP
block; otherwise we fall back to positional (index-order) split and apply a
scaled guard gap to reduce leakage.

Usage (as script):
  python -m src.utils.split
  python -m src.utils.split --npz data/processed/dataset_v0.npz
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.io import load_dataset

CLASS_NAMES = {0: 'Normal', 1: 'F_I', 2: 'P_I', 3: 'M_F', 4: 'C_D', 5: 'C_R'}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_split_manifest(
    npz_path: str,
    out_path: str = 'data/processed/split_manifest.json',
    ratio: tuple = (0.70, 0.15, 0.15),
    guard_gap: int = 64,
    seed: int = 42,
) -> dict:
    """
    Build a frozen train/val/test split and save it to out_path.
    Once generated, this file must NEVER be modified — all S1/S2/S3 models
    share the same test_idx for a fair comparison.

    Returns the manifest dict.
    """
    assert abs(sum(ratio) - 1.0) < 1e-6, f'ratio must sum to 1, got {sum(ratio)}'

    np.random.seed(seed)
    X, y, meta = load_dataset(npz_path)
    N = len(X)
    print(f'[split] N={N}, classes={sorted(set(y.tolist()))}, ratio={ratio}, guard_gap={guard_gap}')

    if 'pcap_id' in meta and meta.get('pcap_id') is not None:
        print('[split] Using pcap_id-based temporal split.')
        train_idx, val_idx, test_idx = _split_by_pcap(y, meta, ratio, guard_gap)
        strict_ratio_check = True
    else:
        print('[split] pcap_id not found → index-order fallback split.')
        train_idx, val_idx, test_idx = _split_by_index(N, ratio, guard_gap)
        strict_ratio_check = False

    # Overlap asserts (hard failure)
    assert len(set(train_idx) & set(test_idx)) == 0, \
        f'train/test overlap: {len(set(train_idx) & set(test_idx))} samples'
    assert len(set(train_idx) & set(val_idx)) == 0, \
        f'train/val overlap: {len(set(train_idx) & set(val_idx))} samples'
    assert len(set(val_idx) & set(test_idx)) == 0, \
        f'val/test overlap: {len(set(val_idx) & set(test_idx))} samples'
    print(f'[OK] No overlap — train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}')

    # Class ratio check
    _check_class_ratio(y, train_idx, strict=strict_ratio_check)

    # Normal-only indices for CAE (Phase 3)
    normal_train_idx = [i for i in train_idx if y[i] == 0]
    normal_val_idx   = [i for i in val_idx   if y[i] == 0]
    print(f'[OK] Normal samples — train={len(normal_train_idx)}, val={len(normal_val_idx)}')

    # Label counts per split
    def _counts(idx):
        if not idx:
            return {}
        vals, cnts = np.unique(y[list(idx)], return_counts=True)
        return {CLASS_NAMES.get(int(v), str(v)): int(c) for v, c in zip(vals, cnts)}

    manifest = {
        'train_idx':        [int(i) for i in train_idx],
        'val_idx':          [int(i) for i in val_idx],
        'test_idx':         [int(i) for i in test_idx],
        'normal_train_idx': [int(i) for i in normal_train_idx],
        'normal_val_idx':   [int(i) for i in normal_val_idx],
        'label_counts':     {'train': _counts(train_idx),
                             'val':   _counts(val_idx),
                             'test':  _counts(test_idx)},
        'seed':             seed,
        'sha256':           '',
        'created_at':       datetime.now(timezone.utc).isoformat(),
    }

    # Save manifest
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    body = json.dumps(manifest, indent=2)
    sha  = hashlib.sha256(body.encode()).hexdigest()
    manifest['sha256'] = sha
    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'[OK] split_manifest.json → {out_path}  sha256={sha[:16]}...')

    # Save normal_only_idx.npy (all normal samples in train+val — Phase 3 hand-off)
    normal_only = np.concatenate([
        np.array(normal_train_idx, dtype=np.int64),
        np.array(normal_val_idx,   dtype=np.int64),
    ]) if (normal_train_idx or normal_val_idx) else np.array([], dtype=np.int64)
    npy_dir  = os.path.dirname(os.path.abspath(out_path))
    npy_path = os.path.join(npy_dir, 'normal_only_idx.npy')
    np.save(npy_path, normal_only)
    print(f'[OK] normal_only_idx.npy → {npy_path}  ({len(normal_only)} normal samples total)')

    return manifest


# ---------------------------------------------------------------------------
# Split strategies
# ---------------------------------------------------------------------------

def _split_by_pcap(y, meta, ratio, guard_gap):
    """Temporal split by pcap_id. Applies guard_gap at PCAP block boundaries."""
    pcap_ids   = np.array(meta['pcap_id'])
    unique_pcaps = sorted(set(pcap_ids.tolist()))
    n = len(unique_pcaps)

    n_tr  = int(n * ratio[0])
    n_val = int(n * ratio[1])

    train_pcaps = set(unique_pcaps[:n_tr])
    val_pcaps   = set(unique_pcaps[n_tr : n_tr + n_val])
    test_pcaps  = set(unique_pcaps[n_tr + n_val :])

    def pcap_indices(pcap_set):
        return sorted(i for i, p in enumerate(pcap_ids) if p in pcap_set)

    train_raw = pcap_indices(train_pcaps)
    val_raw   = pcap_indices(val_pcaps)
    test_raw  = pcap_indices(test_pcaps)

    # Guard gap: remove guard_gap samples at each boundary end
    def _trim(idx_list, drop_head, drop_tail):
        if len(idx_list) <= drop_head + drop_tail:
            return idx_list
        end = len(idx_list) - drop_tail if drop_tail else len(idx_list)
        return idx_list[drop_head:end]

    train_idx = _trim(train_raw, 0,         guard_gap)
    val_idx   = _trim(val_raw,   guard_gap, guard_gap)
    test_idx  = _trim(test_raw,  guard_gap, 0)
    return train_idx, val_idx, test_idx


def _split_by_index(N, ratio, guard_gap):
    """
    Fallback: positional split when pcap_id is unavailable.
    Scales guard_gap down if the dataset is too small to fit all three regions.
    """
    n_train = int(N * ratio[0])
    n_val   = int(N * ratio[1])
    n_test  = N - n_train - n_val

    # Scale guard_gap to ensure no empty split
    eff_gap = min(guard_gap, n_val // 4, n_test // 4)
    if eff_gap != guard_gap:
        print(f'[WARN] guard_gap {guard_gap} → {eff_gap} (N={N} too small for full gap)')

    b1 = n_train
    b2 = n_train + n_val

    train_idx = list(range(0,            b1 - eff_gap))
    val_idx   = list(range(b1 + eff_gap, b2 - eff_gap))
    test_idx  = list(range(b2 + eff_gap, N))

    # Emergency fallback: no guard gap if any split became empty
    if not train_idx or not val_idx or not test_idx:
        print('[WARN] Empty split after guard_gap — falling back to zero-gap positional split.')
        train_idx = list(range(0,      n_train))
        val_idx   = list(range(n_train, n_train + n_val))
        test_idx  = list(range(n_train + n_val, N))

    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _check_class_ratio(y, train_idx, strict: bool = True, tol: float = 0.01):
    """Compare class proportions in train vs overall dataset."""
    overall = {int(c): (y == c).sum() / len(y) for c in np.unique(y)}
    train_y = y[list(train_idx)] if train_idx else np.array([], dtype=y.dtype)

    violations = []
    for c, p_all in overall.items():
        p_tr = ((train_y == c).sum() / len(train_y)) if len(train_y) > 0 else 0.0
        if abs(p_tr - p_all) > tol:
            violations.append(
                f'class {CLASS_NAMES.get(c, c)}: overall={p_all:.3f}, train={p_tr:.3f}'
            )
    if violations:
        msg = 'Class ratio deviation >±1%: ' + ' | '.join(violations)
        if strict:
            raise AssertionError(msg)
        print(f'[WARN] {msg}  (fallback mode — assert skipped)')
    else:
        print('[OK] Class ratio within ±1%')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz',  default='data/processed/dataset_v0.npz')
    parser.add_argument('--out',  default='data/processed/split_manifest.json')
    parser.add_argument('--guard-gap', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    manifest = make_split_manifest(
        npz_path=args.npz,
        out_path=args.out,
        guard_gap=args.guard_gap,
        seed=args.seed,
    )
    print(f'\nSplit sizes: train={len(manifest["train_idx"])}, '
          f'val={len(manifest["val_idx"])}, test={len(manifest["test_idx"])}')
    print(f'Normal (CAE): train={len(manifest["normal_train_idx"])}, '
          f'val={len(manifest["normal_val_idx"])}')
