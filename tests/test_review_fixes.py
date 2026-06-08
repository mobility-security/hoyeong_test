import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.comparison_table import _classification_metrics
from scripts.make_stub_dataset import build_stub_dataset
from src.utils.benchmark import benchmark_predict
from src.utils.config import load_experiment_config, resolve_conf_threshold
from src.utils.io import load_dataset
from src.utils.split import make_split_manifest


def test_stub_train_and_test_are_distinct_and_capture_grouped(tmp_path):
    train = build_stub_dataset(tmp_path / 'dataset_train.npz', seed=1, n_groups=12)
    test = build_stub_dataset(tmp_path / 'dataset_test.npz', seed=2, n_groups=12)
    X_train, _, meta_train, groups = load_dataset(train, with_pcap_id=True)
    X_test, _, meta_test = load_dataset(test)
    assert meta_train['stub'] and meta_test['stub']
    assert meta_train['group_semantics'] == 'capture'
    assert len(np.unique(groups)) == 12
    assert not np.array_equal(X_train, X_test)


def test_runtime_config_isolates_smoke_paths(monkeypatch):
    monkeypatch.setenv('TOW_IDS_TRAIN_NPZ', 'data/smoke/dataset_train.npz')
    monkeypatch.setenv('TOW_IDS_TEST_NPZ', 'data/smoke/dataset_test.npz')
    monkeypatch.setenv('TOW_IDS_MANIFEST', 'data/smoke/split_manifest.json')
    monkeypatch.setenv('TOW_IDS_OUTPUT_DIR', 'results/smoke')
    monkeypatch.setenv(
        'TOW_IDS_CONF_THR_ARTIFACT', 'results/smoke/tables/conf_threshold.json')
    cfg = load_experiment_config()
    assert str(cfg.train_npz_path).startswith('data/smoke/')
    assert str(cfg.test_npz_path).startswith('data/smoke/')
    assert str(cfg.manifest_path).startswith('data/smoke/')
    assert str(cfg.output_dir) == 'results/smoke'
    assert str(cfg.conf_thr_artifact).startswith('results/smoke/')


def test_single_capture_manifest_uses_temporal_guard(tmp_path):
    n_per_class = 6
    y = np.repeat(np.arange(6), n_per_class).astype(np.int64)
    X = np.random.default_rng(0).random((len(y), 3, 4, 4), dtype=np.float32)
    starts = np.arange(len(y), dtype=np.int64) * 64
    ends = starts + 64
    from src.utils.io import build_meta, save_dataset
    train_path = save_dataset(
        tmp_path / 'train.npz', X, y,
        build_meta(4, 4, group_semantics='capture'),
        pcap_id=np.zeros(len(y), dtype=np.int64),
        packet_start=starts, packet_end=ends)
    test_path = build_stub_dataset(tmp_path / 'test.npz', seed=4, n_groups=6)
    manifest = make_split_manifest(
        str(train_path), str(test_path), str(tmp_path / 'manifest.json'),
        val_ratio=0.2, guard_gap_packets=64)
    assert manifest['split_strategy'] == 'temporal_segment_stratified'
    assert manifest['guard_removed_samples'] > 0
    assert set(y[manifest['train_idx']]) == set(range(6))
    assert set(y[manifest['val_idx']]) == set(range(6))


def test_s3_full_metrics_count_unknown_as_error():
    y_true = np.array([0, 1, 2, 2], dtype=np.int64)
    y_pred = np.array([0, 1, 6, 2], dtype=np.int64)
    metrics = _classification_metrics(y_true, y_pred, labels=[0, 1, 2])
    assert metrics['accuracy'] == pytest.approx(0.75)
    assert metrics['coverage'] == pytest.approx(0.75)
    assert metrics['unknown_rate'] == pytest.approx(0.25)
    assert metrics['selective_accuracy'] == pytest.approx(1.0)


def test_benchmark_uses_consistent_prediction_repeats():
    calls = 0

    def predict():
        nonlocal calls
        calls += 1
        return np.array([0, 1, 1], dtype=np.int64)

    predictions, timing = benchmark_predict(
        predict, n_samples=3, device=torch.device('cpu'), warmup=1, repeats=2)
    assert calls == 3
    assert predictions.tolist() == [0, 1, 1]
    assert timing['latency_ms_mean'] >= 0
    assert timing['samples_per_sec'] > 0


def test_threshold_artifact_must_match_manifest(tmp_path):
    artifact = tmp_path / 'threshold.json'
    artifact.write_text(json.dumps({
        'conf_thr': 0.7,
        'target_fpr': 0.05,
        'manifest_sha256': 'expected',
    }), encoding='utf-8')
    cfg = {
        'conf_thr_source': 'artifact',
        'conf_thr_artifact': str(artifact),
        'output_dir': str(tmp_path),
    }
    assert resolve_conf_threshold(cfg, {'sha256': 'expected'}) == 0.7
    with pytest.raises(ValueError, match='does not match'):
        resolve_conf_threshold(cfg, {'sha256': 'different'})
