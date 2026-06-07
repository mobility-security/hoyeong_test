"""누수 안전 split + 2단계 파이프라인 통합 스캐폴드 테스트."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.split import (  # noqa: E402
    assert_no_group_leak, leak_safe_split, leak_safe_trainval_split,
    normal_only_indices,
)
from src.pipeline.two_stage import (  # noqa: E402
    CallableScorer, IdentityScorer, TwoStageIDS, compute_tau,
)
from src.utils.io import UNKNOWN_LABEL, build_meta, save_dataset  # noqa: E402
from src.utils.split import make_split_manifest  # noqa: E402


def _toy(n_groups=24, per=60):
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(n_groups), per)
    y = np.repeat(rng.integers(0, 6, n_groups), per).astype(np.int64)  # group = 단일 클래스
    return groups, y


def test_split_no_leak_and_coverage():
    groups, y = _toy()
    tr, va, te = leak_safe_split(groups, y, seed=1)
    assert_no_group_leak(groups, tr, va, te)               # 위반 시 raise
    assert len(tr) + len(va) + len(te) == len(y)           # 전 샘플 커버
    assert len(set(tr) & set(te)) == 0
    assert len(tr) > 0 and len(va) > 0 and len(te) > 0     # 각 split 비어있지 않음


def test_normal_only_subset():
    _, y = _toy()
    idx = normal_only_indices(y)
    assert (y[idx] == 0).all()


def test_trainval_split_no_leak():
    groups, y = _toy()
    tr, va = leak_safe_trainval_split(groups, y, val_ratio=0.2, seed=3)
    assert_no_group_leak(groups, tr, va)
    assert len(tr) > 0 and len(va) > 0
    assert len(tr) + len(va) == len(y)
    assert len(set(tr) & set(va)) == 0


def test_manifest_uses_pcap_groups_and_train_only_normal_file(tmp_path):
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(12), 2).astype(np.int64)
    group_labels = np.tile(np.arange(6), 2)
    y_train = np.repeat(group_labels, 2).astype(np.int64)
    X_train = rng.random((len(y_train), 3, 4, 4), dtype=np.float32)
    y_test = np.arange(6, dtype=np.int64)
    X_test = rng.random((len(y_test), 3, 4, 4), dtype=np.float32)

    train_path = save_dataset(
        tmp_path / 'train.npz', X_train, y_train, build_meta(4, 4), pcap_id=groups)
    test_path = save_dataset(
        tmp_path / 'test.npz', X_test, y_test, build_meta(4, 4))
    manifest_path = tmp_path / 'split_manifest.json'
    manifest = make_split_manifest(
        str(train_path), str(test_path), str(manifest_path), val_ratio=0.25, seed=3)

    train_groups = set(groups[manifest['train_idx']].tolist())
    val_groups = set(groups[manifest['val_idx']].tolist())
    assert train_groups.isdisjoint(val_groups)
    assert manifest['test_idx'] == list(range(len(y_test)))
    normal_only = np.load(tmp_path / 'normal_only_idx.npy')
    assert np.array_equal(np.sort(normal_only), np.sort(manifest['normal_train_idx']))


def test_manifest_rejects_missing_pcap_id_by_default(tmp_path):
    rng = np.random.default_rng(2)
    y_train = np.tile(np.arange(6), 2).astype(np.int64)
    X_train = rng.random((len(y_train), 3, 4, 4), dtype=np.float32)
    y_test = np.arange(6, dtype=np.int64)
    X_test = rng.random((len(y_test), 3, 4, 4), dtype=np.float32)
    train_path = save_dataset(
        tmp_path / 'train.npz', X_train, y_train, build_meta(4, 4))
    test_path = save_dataset(
        tmp_path / 'test.npz', X_test, y_test, build_meta(4, 4))

    import pytest
    with pytest.raises(ValueError, match='pcap_id is required'):
        make_split_manifest(str(train_path), str(test_path), str(tmp_path / 'split.json'))


def test_compute_tau():
    s = np.array([0.1, 0.1, 0.1, 0.1])  # mean=0.1, std=0
    assert abs(compute_tau(s, k=2.0) - 0.1) < 1e-9
    s2 = np.array([0.0, 2.0])            # mean=1, std(ddof=1)=sqrt(2)
    assert compute_tau(s2, k=1.0) > 1.0


def test_compute_tau_single_sample_not_nan():
    assert compute_tau(np.array([0.3]), k=2.0) == 0.3   # σ 불가 → μ (nan 금지)
    assert compute_tau(np.array([]), k=2.0) == 0.0


def test_empty_split_raises_clear_error():
    import pytest
    # 한 클래스에 group 1개뿐 → val/test 생성 불가 → 명확한 ValueError
    groups = np.array([0, 0, 0, 1, 1, 1])      # 클래스0: group0,1 ; 클래스1: group?
    y = np.array([0, 0, 0, 0, 0, 0], np.int64)  # 전부 같은 클래스, group 2개 → val 빔
    with pytest.raises(ValueError, match="split"):
        leak_safe_split(groups, y, seed=0)


def test_stage2_1d_output_raises():
    import pytest
    X = np.random.rand(2, 3, 32, 32).astype(np.float32)
    scorer = CallableScorer(lambda x: np.array([9.0, 9.0]))   # 둘 다 이상
    bad_stage2 = lambda x: np.array([3, 4])                    # argmax(1-D) 실수
    p = TwoStageIDS(scorer, bad_stage2, tau=1.0)
    with pytest.raises(ValueError, match="softmax"):
        p.predict(X)


def test_pipeline_normal_when_identity_scorer():
    # IdentityScorer → 전부 score 0 <= tau → 전부 Normal
    X = np.random.rand(10, 3, 32, 32).astype(np.float32)
    stage2 = lambda x: np.tile([0.0, 0.2, 0.2, 0.2, 0.2, 0.2], (len(x), 1))
    p = TwoStageIDS(IdentityScorer(), stage2, tau=0.5, conf_thr=0.5)
    assert (p.predict(X) == 0).all()


def test_pipeline_routes_unknown_and_attack():
    X = np.random.rand(4, 3, 32, 32).astype(np.float32)
    scorer = CallableScorer(lambda x: np.array([0.0, 0.0, 9.0, 9.0]))  # 뒤 2개만 이상
    # 이상 샘플 중 하나는 자신만만(C_D), 하나는 저신뢰(→Unknown)
    probs = np.array([[0, 0, 0, 0, 0.95, 0.05],      # idx2: argmax=C_D(4), conf 0.95
                      [0.2, 0.2, 0.2, 0.15, 0.15, 0.1]])  # idx3: max 0.2 < 0.5 → Unknown
    stage2 = lambda x: probs
    p = TwoStageIDS(scorer, stage2, tau=1.0, conf_thr=0.5)
    out = p.predict(X)
    assert out[0] == 0 and out[1] == 0       # 정상
    assert out[2] == 4                        # CAN DoS
    assert out[3] == UNKNOWN_LABEL            # Unknown(6)


def test_pipeline_bypass_cae():
    X = np.random.rand(3, 3, 32, 32).astype(np.float32)
    stage2 = lambda x: np.tile([0.05, 0.9, 0.01, 0.01, 0.02, 0.01], (len(x), 1))
    p = TwoStageIDS(IdentityScorer(), stage2, tau=0.5, conf_thr=0.5, use_cae=False)
    assert (p.predict(X) == 1).all()          # use_cae=False → 전부 Stage-2(F_I=1)
