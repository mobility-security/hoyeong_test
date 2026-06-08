"""
누수 안전 train/val 분할과 frozen test manifest 생성.

- dataset_train.npz: pcap_id가 있으면 capture group 단위로 train/val 분할
- dataset_test.npz: 전체를 frozen test로 사용하며 학습/보정에는 사용하지 않음
- pcap_id가 없는 legacy 데이터는 명시적 옵션에서만 unsafe fallback 허용
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.io import LABEL_NAMES, load_dataset


# ---------------------------------------------------------------------------
# Group-safe split helpers
# ---------------------------------------------------------------------------

def _validate_group_inputs(groups: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    groups = np.asarray(groups)
    y = np.asarray(y)
    if groups.ndim != 1 or y.ndim != 1:
        raise ValueError(f'groups and y must be 1-D, got {groups.shape} and {y.shape}')
    if len(groups) != len(y):
        raise ValueError(f'len(groups)={len(groups)} != len(y)={len(y)}')
    if len(y) == 0:
        raise ValueError('cannot split an empty dataset')
    if not np.issubdtype(groups.dtype, np.integer):
        raise ValueError('groups must contain integer identifiers')
    if not np.issubdtype(y.dtype, np.integer) or np.any(y < 0):
        raise ValueError('y must contain non-negative integer labels')
    return groups, y


def make_contiguous_groups(n: int, block_size: int) -> np.ndarray:
    """group 정보가 없을 때 연속 block 단위 group id를 생성."""
    if n < 0:
        raise ValueError(f'n must be non-negative, got {n}')
    if block_size <= 0:
        raise ValueError(f'block_size must be positive, got {block_size}')
    return (np.arange(n) // block_size).astype(np.int64)


def _group_dominant_label(groups: np.ndarray, y: np.ndarray) -> dict[int, int]:
    """각 group의 dominant label 계산. 전체 배열을 group마다 재스캔하지 않는다."""
    order = np.argsort(groups, kind='stable')
    groups_sorted = groups[order]
    y_sorted = y[order]
    unique_groups, starts = np.unique(groups_sorted, return_index=True)
    ends = np.r_[starts[1:], len(groups_sorted)]

    dominant = {}
    for group, start, end in zip(unique_groups, starts, ends):
        dominant[int(group)] = int(np.bincount(y_sorted[start:end]).argmax())
    return dominant


def _indices_for_groups(groups: np.ndarray, selected: list[int]) -> np.ndarray:
    if not selected:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero(np.isin(groups, np.asarray(selected, dtype=groups.dtype)))


def assert_no_group_leak(groups: np.ndarray, *splits: np.ndarray) -> None:
    """split 간 group 교집합이 있으면 ValueError."""
    groups = np.asarray(groups)
    group_sets = [set(np.unique(groups[np.asarray(split, dtype=np.int64)]).tolist())
                  for split in splits]
    for i in range(len(group_sets)):
        for j in range(i + 1, len(group_sets)):
            overlap = group_sets[i] & group_sets[j]
            if overlap:
                raise ValueError(
                    f'group leak between split {i} and {j}: {sorted(overlap)[:5]}')


def leak_safe_split(
    groups: np.ndarray,
    y: np.ndarray,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """group 단위 stratified train/val/test split."""
    groups, y = _validate_group_inputs(groups, y)
    if len(ratios) != 3 or any(ratio < 0 for ratio in ratios):
        raise ValueError(f'ratios must contain three non-negative values, got {ratios}')
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f'ratios must sum to 1, got {ratios}')

    rng = np.random.default_rng(seed)
    dominant = _group_dominant_label(groups, y)
    by_class: dict[int, list[int]] = defaultdict(list)
    for group, label in dominant.items():
        by_class[label].append(group)

    train_groups, val_groups, test_groups = [], [], []
    for class_groups in by_class.values():
        class_groups = list(class_groups)
        rng.shuffle(class_groups)
        n_groups = len(class_groups)
        if n_groups == 1:
            n_train, n_val = 1, 0
        elif n_groups == 2:
            n_train, n_val = 1, 0
        else:
            n_test = max(1, round(n_groups * ratios[2]))
            n_val = max(1, round(n_groups * ratios[1]))
            n_train = n_groups - n_val - n_test
            while n_train < 1:
                if n_val > 1:
                    n_val -= 1
                elif n_test > 1:
                    n_test -= 1
                else:
                    break
                n_train = n_groups - n_val - n_test
        train_groups.extend(class_groups[:n_train])
        val_groups.extend(class_groups[n_train:n_train + n_val])
        test_groups.extend(class_groups[n_train + n_val:])

    train_idx = _indices_for_groups(groups, train_groups)
    val_idx = _indices_for_groups(groups, val_groups)
    test_idx = _indices_for_groups(groups, test_groups)
    assert_no_group_leak(groups, train_idx, val_idx, test_idx)

    empty_splits = [name for name, split in (('val', val_idx), ('test', test_idx))
                    if len(split) == 0]
    if empty_splits:
        per_class = {
            int(label): sum(group_label == label for group_label in dominant.values())
            for label in sorted(set(dominant.values()))
        }
        raise ValueError(
            f'leak_safe_split: {empty_splits} split is empty; group counts={per_class}')
    return train_idx, val_idx, test_idx


def leak_safe_trainval_split(
    groups: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.15,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """group 단위 stratified train/val split."""
    groups, y = _validate_group_inputs(groups, y)
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f'val_ratio must be in (0, 1), got {val_ratio}')

    rng = np.random.default_rng(seed)
    dominant = _group_dominant_label(groups, y)
    by_class: dict[int, list[int]] = defaultdict(list)
    for group, label in dominant.items():
        by_class[label].append(group)

    train_groups, val_groups = [], []
    for class_groups in by_class.values():
        class_groups = list(class_groups)
        rng.shuffle(class_groups)
        n_groups = len(class_groups)
        n_val = round(n_groups * val_ratio)
        n_val = min(max(n_val, 1), n_groups - 1) if n_groups >= 2 else 0
        val_groups.extend(class_groups[:n_val])
        train_groups.extend(class_groups[n_val:])

    train_idx = _indices_for_groups(groups, train_groups)
    val_idx = _indices_for_groups(groups, val_groups)
    assert_no_group_leak(groups, train_idx, val_idx)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError('train/val split is empty; at least two capture groups are required')
    return train_idx, val_idx


def normal_only_indices(y: np.ndarray) -> np.ndarray:
    """CAE 학습용 Normal(0) 샘플 인덱스."""
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f'y must be 1-D, got {y.shape}')
    return np.flatnonzero(y == 0)


# ---------------------------------------------------------------------------
# Manifest validation and creation
# ---------------------------------------------------------------------------

def validate_trainval_indices(manifest: dict, n_train: int) -> None:
    """manifest train/val 인덱스의 범위, 중복, 교집합을 검증."""
    required = {'train_idx', 'val_idx'}
    missing = required - set(manifest)
    if missing:
        raise ValueError(f'manifest missing keys: {sorted(missing)}')

    splits = {}
    for name in ('train', 'val'):
        idx = np.asarray(manifest[f'{name}_idx'])
        if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer):
            raise ValueError(f'{name}_idx must be a 1-D integer array')
        idx = idx.astype(np.int64, copy=False)
        if len(idx) == 0:
            raise ValueError(f'{name} split is empty')
        if len(np.unique(idx)) != len(idx):
            raise ValueError(f'{name}_idx contains duplicate indices')
        if idx.min() < 0 or idx.max() >= n_train:
            raise ValueError(f'{name}_idx contains indices outside dataset size {n_train}')
        splits[name] = idx

    overlap = np.intersect1d(splits['train'], splits['val'])
    if len(overlap):
        raise ValueError(f'train/val overlap: {overlap[:5].tolist()}')


def validate_manifest_indices(manifest: dict, n_train: int, n_test: int) -> None:
    """train/val과 frozen test 전체 사용을 검증."""
    validate_trainval_indices(manifest, n_train)
    if 'test_idx' not in manifest:
        raise ValueError('manifest missing keys: [\'test_idx\']')
    test_idx = np.asarray(manifest['test_idx'])
    if test_idx.ndim != 1 or not np.issubdtype(test_idx.dtype, np.integer):
        raise ValueError('test_idx must be a 1-D integer array')
    test_idx = test_idx.astype(np.int64, copy=False)
    if len(test_idx) == 0:
        raise ValueError('test split is empty')
    if len(np.unique(test_idx)) != len(test_idx):
        raise ValueError('test_idx contains duplicate indices')
    if test_idx.min() < 0 or test_idx.max() >= n_test:
        raise ValueError(f'test_idx contains indices outside dataset size {n_test}')
    if not np.array_equal(np.sort(test_idx), np.arange(n_test)):
        raise ValueError('test_idx must contain the entire frozen test dataset')


def _label_counts(indices: list[int], y: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(y[np.asarray(indices, dtype=np.int64)], return_counts=True)
    return {
        LABEL_NAMES[int(value)] if int(value) < len(LABEL_NAMES) else str(value): int(count)
        for value, count in zip(values, counts)
    }


def make_split_manifest(
    train_npz_path: str,
    test_npz_path: str,
    out_path: str = 'data/processed/split_manifest.json',
    val_ratio: float = 0.10,
    guard_gap: int = 64,
    seed: int = 42,
    allow_unsafe_fallback: bool = False,
) -> dict:
    """train/val 및 frozen test manifest를 생성."""
    if Path(train_npz_path).resolve() == Path(test_npz_path).resolve():
        raise ValueError('train and test datasets must be different files')

    X_train, y_train, _, pcap_id = load_dataset(train_npz_path, with_pcap_id=True)
    X_test, y_test, _ = load_dataset(test_npz_path)
    n_train = len(X_train)
    n_test = len(X_test)
    print(f'[split] dataset_train N={n_train}  dataset_test N={n_test}')
    print(f'[split] val_ratio={val_ratio}  guard_gap={guard_gap}  seed={seed}')

    if pcap_id is not None:
        train_idx_arr, val_idx_arr = leak_safe_trainval_split(
            pcap_id, y_train, val_ratio=val_ratio, seed=seed)
        split_strategy = 'pcap_group_stratified'
        assert_no_group_leak(pcap_id, train_idx_arr, val_idx_arr)
    else:
        if not allow_unsafe_fallback:
            raise ValueError(
                'pcap_id is required for leak-safe train/val splitting; '
                'set allow_unsafe_fallback=True only for legacy smoke data')
        print('[WARN] pcap_id missing; using unsafe sample-level stratified fallback')
        all_idx = np.arange(n_train)
        try:
            train_idx_arr, val_idx_arr = train_test_split(
                all_idx, test_size=val_ratio, stratify=y_train, random_state=seed)
        except ValueError as exc:
            raise ValueError(f'failed to create stratified train/val split: {exc}') from exc
        split_strategy = 'sample_stratified_fallback'

    train_idx = train_idx_arr.astype(np.int64).tolist()
    val_idx = val_idx_arr.astype(np.int64).tolist()
    test_idx = list(range(n_test))
    normal_train_idx = [index for index in train_idx if y_train[index] == 0]
    normal_val_idx = [index for index in val_idx if y_train[index] == 0]

    manifest = {
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'normal_train_idx': normal_train_idx,
        'normal_val_idx': normal_val_idx,
        'label_counts': {
            'train': _label_counts(train_idx, y_train),
            'val': _label_counts(val_idx, y_train),
            'test': _label_counts(test_idx, y_test),
        },
        'train_source': train_npz_path,
        'test_source': test_npz_path,
        'split_strategy': split_strategy,
        'val_ratio': val_ratio,
        'guard_gap': guard_gap,
        'seed': seed,
        'created_at': datetime.now(timezone.utc).isoformat(),
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(',', ':'))
    manifest['sha256'] = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    validate_manifest_indices(manifest, n_train, n_test)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as file:
        json.dump(manifest, file, indent=2)
    print(f'[OK] split_manifest.json -> {out_path}  sha256={manifest["sha256"][:16]}...')

    normal_only_path = out_path.with_name('normal_only_idx.npy')
    np.save(normal_only_path, np.asarray(normal_train_idx, dtype=np.int64))
    print(f'[OK] normal_only_idx.npy -> {normal_only_path} '
          f'({len(normal_train_idx)} train-only samples)')
    return manifest


# ---------------------------------------------------------------------------
# Guard gap 적용
# ---------------------------------------------------------------------------

def _apply_guard_gap(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    guard_gap: int,
    N: int,
) -> tuple:
    """
    train/val 경계 근방에서 guard_gap 개 샘플을 제거.
    stratified split 후 인덱스 정렬 기준으로 경계를 판단.
    guard_gap이 val 크기의 1/4을 초과하면 자동 축소.

    NOTE: make_split_manifest()에서 호출하지 않음.
    이유 ①: train/test가 dataset_train.npz / dataset_test.npz 파일 수준에서
             이미 분리되어 있어 파일 경계가 1차 누수 방지막이다.
    이유 ②: build_tow_dataset.py가 stride=window_size(=64)로 윈도우를 생성하므로
             인접 윈도우 간 패킷 공유가 0이다.
    이유 ③: sklearn stratified split으로 val 인덱스가 0..N-1 전체에 분산되므로
             val_min≈0, val_max≈N-1이 되어 이 함수를 호출하면 train 샘플
             대부분이 제거되는 오작동이 발생한다.
    contiguous(시간 순서) split으로 전환할 경우에만 활성화할 것.
    """
    train_sorted = np.sort(train_idx)
    val_sorted   = np.sort(val_idx)

    if len(val_sorted) == 0:
        return list(train_sorted), []

    val_min = int(val_sorted[0])
    val_max = int(val_sorted[-1])

    eff_gap = min(guard_gap, len(val_sorted) // 4)
    if eff_gap != guard_gap:
        print(f'[WARN] guard_gap {guard_gap} → {eff_gap} (val이 너무 작음)')

    # val 경계 ±eff_gap 범위의 train 샘플 제거
    train_filtered = [
        int(i) for i in train_sorted
        if not (val_min - eff_gap <= i <= val_max + eff_gap)
    ]
    val_filtered = list(map(int, val_sorted))

    return train_filtered, val_filtered


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-npz', default='data/processed/dataset_train.npz')
    parser.add_argument('--test-npz', default='data/processed/dataset_test.npz')
    parser.add_argument('--out', default='data/processed/split_manifest.json')
    parser.add_argument('--val-ratio', type=float, default=0.10)
    parser.add_argument('--guard-gap', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--allow-unsafe-fallback', action='store_true')
    args = parser.parse_args()

    result = make_split_manifest(
        train_npz_path=args.train_npz,
        test_npz_path=args.test_npz,
        out_path=args.out,
        val_ratio=args.val_ratio,
        guard_gap=args.guard_gap,
        seed=args.seed,
        allow_unsafe_fallback=args.allow_unsafe_fallback,
    )
    print(f'\nSplit sizes: train={len(result["train_idx"])}, '
          f'val={len(result["val_idx"])}, test={len(result["test_idx"])}')
    print(f'Normal (CAE): train={len(result["normal_train_idx"])}, '
          f'val={len(result["normal_val_idx"])}')
