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
from src.train.common import (
    config_sha256, load_manifest, result_bundle_provenance,
    validate_checkpoint_provenance, validate_result_bundle,
)
from src.utils.split import compute_manifest_sha256, validate_manifest_provenance


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
    assert manifest['split_strategy'] == 'class_temporal_tail'
    assert manifest['guard_removed_samples'] > 0
    train_idx = np.asarray(manifest['train_idx'])
    val_idx = np.asarray(manifest['val_idx'])
    assert set(y[train_idx]) == set(range(6))
    assert set(y[val_idx]) == set(range(6))
    for train_index in train_idx:
        assert not np.any(
            (starts[val_idx] < ends[train_index] + 64)
            & (ends[val_idx] > starts[train_index] - 64))


def test_cae_supports_non_multiple_internal_size():
    from src.models.cae import CAE
    model = CAE(input_shape=(3, 64, 64), cae_input_size=30)
    X = torch.zeros(2, 3, 64, 64)
    assert model.mse_loss(X).shape == (2,)


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
        'train_sha256': 'train',
        'test_sha256': 'test',
    }), encoding='utf-8')
    cfg = {
        'conf_thr_source': 'artifact',
        'conf_thr_artifact': str(artifact),
        'output_dir': str(tmp_path),
    }
    manifest = {
        'sha256': 'expected', 'train_sha256': 'train', 'test_sha256': 'test'}
    assert resolve_conf_threshold(cfg, manifest) == 0.7
    with pytest.raises(ValueError, match='does not match'):
        resolve_conf_threshold(cfg, {**manifest, 'sha256': 'different'})


def test_checkpoint_provenance_maps_manifest_digest_key():
    manifest = {'sha256': 'm', 'train_sha256': 'tr', 'test_sha256': 'te'}
    checkpoint = {
        'manifest_sha256': 'm', 'train_sha256': 'tr', 'test_sha256': 'te'}
    validate_checkpoint_provenance(checkpoint, manifest)
    with pytest.raises(ValueError, match='manifest_sha256'):
        validate_checkpoint_provenance(
            {**checkpoint, 'manifest_sha256': 'different'}, manifest)


def test_manifest_and_dataset_tampering_are_rejected(tmp_path):
    train = build_stub_dataset(tmp_path / 'train.npz', seed=3, n_groups=12)
    test = build_stub_dataset(tmp_path / 'test.npz', seed=4, n_groups=12)
    manifest_path = tmp_path / 'manifest.json'
    manifest = make_split_manifest(str(train), str(test), str(manifest_path))
    X_train, _, _ = load_dataset(train)
    X_test, _, _ = load_dataset(test)
    load_manifest(
        manifest_path, len(X_train), len(X_test),
        train_npz_path=train, test_npz_path=test)

    tampered = {**manifest, 'seed': 999}
    manifest_path.write_text(json.dumps(tampered), encoding='utf-8')
    with pytest.raises(ValueError, match='SHA-256'):
        load_manifest(
            manifest_path, len(X_train), len(X_test),
            train_npz_path=train, test_npz_path=test)

    wrong_data_hash = {**manifest, 'train_sha256': '0' * 64}
    wrong_data_hash['sha256'] = compute_manifest_sha256(wrong_data_hash)
    with pytest.raises(ValueError, match='train dataset'):
        validate_manifest_provenance(wrong_data_hash, train, test)


def test_checkpoint_config_hash_is_enforced(tmp_path):
    config = tmp_path / 'config.yaml'
    config.write_text('value: 1\n', encoding='utf-8')
    manifest = {'sha256': 'm', 'train_sha256': 'tr', 'test_sha256': 'te'}
    checkpoint = {
        'manifest_sha256': 'm', 'train_sha256': 'tr', 'test_sha256': 'te',
        'config_sha256': config_sha256(config),
    }
    validate_checkpoint_provenance(checkpoint, manifest, config)
    config.write_text('value: 2\n', encoding='utf-8')
    with pytest.raises(ValueError, match='config_sha256'):
        validate_checkpoint_provenance(checkpoint, manifest, config)


def test_result_bundle_rejects_stale_table(tmp_path):
    config = tmp_path / 'config.yaml'
    table = tmp_path / 'table.csv'
    config.write_text('value: 1\n', encoding='utf-8')
    table.write_text('metric\n1\n', encoding='utf-8')
    manifest = {'sha256': 'm', 'train_sha256': 'tr', 'test_sha256': 'te'}
    provenance = result_bundle_provenance(
        manifest, {'table': table}, config, routing_mode='strict_cascade')
    validate_result_bundle(provenance, manifest, {'table': table}, config)
    table.write_text('metric\n2\n', encoding='utf-8')
    with pytest.raises(ValueError, match='does not match provenance'):
        validate_result_bundle(provenance, manifest, {'table': table}, config)
