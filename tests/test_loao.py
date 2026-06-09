"""Tests for Phase 4 LOAO infrastructure and metrics."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.leave_one_out import (
    ATTACK_CLASSES,
    UNKNOWN_LABEL,
    _predict_loao,
    _remap_labels_exclude,
)
from src.utils.metrics import (
    compute_binary_metrics,
    compute_loao_metrics,
    compute_multiclass_metrics,
)


# ---------------------------------------------------------------------------
# _remap_labels_exclude
# ---------------------------------------------------------------------------

class TestRemapLabelsExclude:
    def test_excludes_class_and_remaps(self):
        y = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        mask, y_new = _remap_labels_exclude(y, excl_id=2)
        assert list(mask) == [True, True, False, True, True, True]
        assert list(y_new) == [0, 1, 2, 3, 4]

    def test_exclude_first_class(self):
        # y=[0,1,2,0,1], excl_id=1 → keep positions 0,2,3 (labels 0,2,0)
        y = np.array([0, 1, 2, 0, 1], dtype=np.int64)
        mask, y_new = _remap_labels_exclude(y, excl_id=1)
        assert mask.sum() == 3  # positions 1 and 4 (label=1) are excluded
        # 2 → 1 (shifted down), 0 stays 0
        assert list(y_new) == [0, 1, 0]

    def test_all_classes_contiguous(self):
        y = np.array([0, 1, 2, 3, 4, 5] * 10, dtype=np.int64)
        for excl_id in range(6):
            _, y_new = _remap_labels_exclude(y, excl_id)
            assert set(y_new.tolist()) == set(range(5)), f'excl_id={excl_id}'


# ---------------------------------------------------------------------------
# compute_binary_metrics
# ---------------------------------------------------------------------------

class TestComputeBinaryMetrics:
    def test_perfect_prediction(self):
        y_true = np.array([0, 0, 1, 1], dtype=np.int64)
        y_pred = np.array([0, 0, 1, 1], dtype=np.int64)
        y_prob = np.array([0.1, 0.1, 0.9, 0.9])
        m = compute_binary_metrics(y_true, y_pred, y_prob)
        assert m['accuracy'] == 1.0
        assert m['fpr'] == 0.0
        assert m['fnr'] == 0.0

    def test_all_wrong(self):
        y_true = np.array([0, 1], dtype=np.int64)
        y_pred = np.array([1, 0], dtype=np.int64)
        y_prob = np.array([0.9, 0.1])
        m = compute_binary_metrics(y_true, y_pred, y_prob)
        assert m['accuracy'] == 0.0
        assert m['fpr'] == 1.0
        assert m['fnr'] == 1.0


# ---------------------------------------------------------------------------
# compute_multiclass_metrics
# ---------------------------------------------------------------------------

class TestComputeMulticlassMetrics:
    def test_basic(self):
        y_true = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
        y_pred = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
        m = compute_multiclass_metrics(y_true, y_pred)
        assert m['accuracy'] == 1.0
        assert m['macro_f1'] == pytest.approx(1.0)

    def test_returns_confusion_matrix(self):
        y_true = np.array([0, 1, 2], dtype=np.int64)
        y_pred = np.array([0, 2, 1], dtype=np.int64)
        m = compute_multiclass_metrics(y_true, y_pred)
        assert len(m['confusion_matrix']) == 3
        assert len(m['per_class_recall']) == 3


# ---------------------------------------------------------------------------
# compute_loao_metrics
# ---------------------------------------------------------------------------

class TestComputeLoaoMetrics:
    def _make_inputs(self, n_excl=50, n_normal=100, cae_recall=0.8, fpr=0.05,
                     unknown_rate=0.6):
        rng = np.random.default_rng(0)
        tau = 0.01

        # Excluded attack samples: cae_recall fraction above tau
        mse_excl = np.where(
            rng.random(n_excl) < cae_recall,
            tau + rng.random(n_excl) * 0.01,  # above tau
            rng.random(n_excl) * tau * 0.5,   # below tau
        ).astype(np.float64)

        # Build y_pred for excluded samples (based on Unknown rate)
        # Among anomalous, unknown_rate fraction should be Unknown(6)
        anomalous_mask = mse_excl > tau
        y_pred_excl = np.zeros(n_excl, dtype=np.int64)
        anomalous_idx = np.flatnonzero(anomalous_mask)
        n_unknown = round(unknown_rate * len(anomalous_idx))
        y_pred_excl[anomalous_idx[:n_unknown]] = 6  # Unknown

        # Normal samples
        mse_normal = np.where(
            rng.random(n_normal) < fpr,
            tau + rng.random(n_normal) * 0.001,
            rng.random(n_normal) * tau * 0.5,
        ).astype(np.float64)
        y_pred_normal = np.zeros(n_normal, dtype=np.int64)

        # Combine: excl_id=1 for attack, 0 for normal
        y_true = np.concatenate([np.ones(n_excl, dtype=np.int64),
                                  np.zeros(n_normal, dtype=np.int64)])
        y_pred = np.concatenate([y_pred_excl, y_pred_normal])
        mse = np.concatenate([mse_excl, mse_normal])
        return y_true, y_pred, mse, tau

    def test_cae_recall_range(self):
        y_true, y_pred, mse, tau = self._make_inputs(cae_recall=0.8)
        m = compute_loao_metrics(y_true, y_pred, mse, tau)
        assert 0.0 <= m['cae_anomaly_recall'] <= 1.0

    def test_normal_fpr_range(self):
        y_true, y_pred, mse, tau = self._make_inputs(fpr=0.05)
        m = compute_loao_metrics(y_true, y_pred, mse, tau)
        assert 0.0 <= m['normal_fpr'] <= 1.0

    def test_unknown_rate_range(self):
        y_true, y_pred, mse, tau = self._make_inputs()
        m = compute_loao_metrics(y_true, y_pred, mse, tau)
        assert 0.0 <= m['unknown_rate'] <= 1.0

    def test_all_normal_no_attack(self):
        y_true = np.zeros(10, dtype=np.int64)
        y_pred = np.zeros(10, dtype=np.int64)
        mse = np.zeros(10, dtype=np.float64)
        m = compute_loao_metrics(y_true, y_pred, mse, 0.01)
        assert np.isnan(m['cae_anomaly_recall'])
        assert np.isnan(m['unknown_rate'])


# ---------------------------------------------------------------------------
# _predict_loao (integration test with stub CAE)
# ---------------------------------------------------------------------------

class StubCAE(torch.nn.Module):
    """Returns MSE=0 for all inputs (everything below threshold)."""
    def forward(self, x):
        return x  # perfect reconstruction → MSE ≈ 0

    def mse_loss(self, x, training=False):
        return torch.zeros(len(x), dtype=torch.float32)


class HighMseCAE(torch.nn.Module):
    """Returns constant high reconstruction error."""
    def forward(self, x):
        return torch.zeros_like(x)  # MSE = mean(x^2)

    def mse_loss(self, x, training=False):
        return (x ** 2).mean(dim=(1, 2, 3))


class FixedS2(torch.nn.Module):
    """Always predicts class 1 with high confidence."""
    def forward(self, x):
        logits = torch.zeros(len(x), 5)
        logits[:, 1] = 10.0  # high confidence class 1
        return logits


class LowConfS2(torch.nn.Module):
    """Uniform logits → all probabilities = 0.2 → max_prob < conf_thr → Unknown."""
    def forward(self, x):
        return torch.zeros(len(x), 5)  # uniform


class TestPredictLoao:
    def _make_X(self, n=8):
        return np.ones((n, 3, 4, 4), dtype=np.float32) * 0.5

    def test_s2_recovers_attack_below_cae_threshold(self):
        cae = StubCAE()
        # MSE ≈ 0 for perfect reconstruction → all below any reasonable tau
        X = self._make_X(8)
        # Since X is all 0.5 and reconstruction is X → MSE = 0
        mse, y_pred = _predict_loao(cae, FixedS2(), X, tau=0.1,
                                    conf_thr=0.5, device=torch.device('cpu'))
        assert np.all(mse == pytest.approx(0.0, abs=1e-6))
        assert np.all(y_pred == 1)

    def test_high_mse_cae_routes_to_s2(self):
        cae = HighMseCAE()
        X = self._make_X(8)  # all 0.5 → MSE = mean(0.5^2) = 0.25
        tau = 0.1
        mse, y_pred = _predict_loao(cae, FixedS2(), X, tau=tau,
                                    conf_thr=0.5, device=torch.device('cpu'))
        assert np.all(mse > tau)
        assert np.all(y_pred == 1)  # FixedS2 predicts class 1

    def test_low_conf_produces_unknown(self):
        cae = HighMseCAE()
        X = self._make_X(8)
        tau = 0.1
        mse, y_pred = _predict_loao(cae, LowConfS2(), X, tau=tau,
                                    conf_thr=0.5, device=torch.device('cpu'))
        assert np.all(mse > tau)
        assert np.all(y_pred == UNKNOWN_LABEL)
