"""dataset.npz 계약(io.py) 검증 테스트."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.io import (  # noqa: E402
    SchemaError, build_meta, load_dataset, save_dataset, validate_schema,
)


def _good(n=8, H=32, W=32):
    X = np.random.rand(n, 3, H, W).astype(np.float32)
    y = np.random.randint(0, 6, size=n).astype(np.int64)
    return X, y


def test_valid_passes():
    X, y = _good()
    validate_schema(X, y, build_meta(32, 32))


def test_roundtrip(tmp_path):
    X, y = _good()
    meta = build_meta(32, 32, seed=1)
    p = save_dataset(tmp_path / "d.npz", X, y, meta)
    X2, y2, meta2 = load_dataset(p)
    assert np.array_equal(X, X2) and np.array_equal(y, y2)
    assert meta2["wavelets"] == ["coif1", "db3", "rbio1.3"]
    assert meta2["seed"] == 1


def test_non_npz_suffix_uses_actual_output_path(tmp_path):
    X, y = _good()
    path = save_dataset(tmp_path / 'dataset.data', X, y, build_meta(32, 32))
    assert path.name == 'dataset.data.npz'
    assert path.exists()


def test_scalar_y_raises_schema_error():
    X, _ = _good(n=1)
    with pytest.raises(SchemaError, match='1-D'):
        validate_schema(X, np.array(0, dtype=np.int64))


def test_non_mapping_meta_raises_schema_error():
    X, y = _good()
    with pytest.raises(SchemaError, match='mapping'):
        validate_schema(X, y, meta=['not', 'a', 'mapping'])


def test_load_rejects_wrong_stored_dtype(tmp_path):
    X, y = _good()
    path = tmp_path / 'bad_dtype.npz'
    np.savez(path, X=X.astype(np.float64), y=y)
    with pytest.raises(SchemaError, match='float32'):
        load_dataset(path)


def test_save_bad_shape_raises_schema_error_without_meta(tmp_path):
    X = np.zeros((2, 3), dtype=np.float32)
    y = np.zeros(2, dtype=np.int64)
    with pytest.raises(SchemaError, match='X.shape'):
        save_dataset(tmp_path / 'bad.npz', X, y)


def test_pcap_id_roundtrip_and_sidecar(tmp_path):
    X, y = _good(n=8)
    pid = np.repeat(np.arange(4), 2).astype(np.int64)  # 4개 pcap
    p = save_dataset(tmp_path / "d.npz", X, y, build_meta(32, 32), pcap_id=pid)
    # with_pcap_id
    X2, y2, meta2, pid2 = load_dataset(p, with_pcap_id=True)
    assert np.array_equal(pid, pid2)
    # 하위호환: 3-tuple
    assert len(load_dataset(p)) == 3
    # meta.json sidecar 생성됨
    assert p.with_suffix(".meta.json").exists()


def test_save_rejects_fractional_labels_and_groups(tmp_path):
    X, _ = _good(n=2)
    with pytest.raises(SchemaError, match='integer dtype'):
        save_dataset(
            tmp_path / 'fractional_y.npz', X, np.array([0.9, 1.9]),
            build_meta(32, 32))
    with pytest.raises(SchemaError, match='integer dtype'):
        save_dataset(
            tmp_path / 'fractional_group.npz', X, np.array([0, 1], dtype=np.int64),
            build_meta(32, 32), pcap_id=np.array([1.2, 2.8]))


def test_packet_provenance_roundtrip(tmp_path):
    X, y = _good(n=3)
    starts = np.array([0, 64, 128], dtype=np.int64)
    ends = starts + 64
    p = save_dataset(
        tmp_path / 'ranges.npz', X, y, build_meta(32, 32),
        packet_start=starts, packet_end=ends)
    with np.load(p, allow_pickle=False) as data:
        assert np.array_equal(data['packet_start'], starts)
        assert np.array_equal(data['packet_end'], ends)


def test_pcap_id_length_mismatch_raises():
    X, y = _good(n=8)
    with pytest.raises(SchemaError, match="pcap_id"):
        validate_schema(X, y, None, np.arange(7, dtype=np.int64))


def test_numpy_typed_meta_serializes(tmp_path):
    X, y = _good()
    meta = build_meta(32, 32, n_samples=np.int64(8), tau=np.float32(0.5),
                      counts=np.array([1, 2, 3]))
    p = save_dataset(tmp_path / "m.npz", X, y, meta)
    _, _, meta2 = load_dataset(p)
    assert meta2["n_samples"] == 8 and meta2["counts"] == [1, 2, 3]


def test_meta_hw_mismatch_raises():
    X, y = _good(H=32, W=32)
    with pytest.raises(SchemaError, match="H/W"):
        validate_schema(X, y, build_meta(99, 99))


@pytest.mark.parametrize("mutate", [
    lambda X, y: (X.astype(np.float64), y),          # wrong dtype
    lambda X, y: (X[:, :2], y),                       # wrong channels
    lambda X, y: (X * 3.0, y),                        # out of [0,1]
    lambda X, y: (np.where(np.arange(X.size).reshape(X.shape) == 0,
                           np.inf, X).astype(np.float32), y),  # non-finite
    lambda X, y: (X, (y + 10).astype(np.int64)),      # label out of range
    lambda X, y: (X, y.astype(np.int32)),             # wrong y dtype
])
def test_invalid_raises(mutate):
    X, y = _good()
    Xb, yb = mutate(X, y)
    with pytest.raises(SchemaError):
        validate_schema(Xb, yb)
