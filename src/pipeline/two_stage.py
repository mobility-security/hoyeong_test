"""CAE anomaly gate와 6-class DCNN을 연결하는 2단계 IDS 파이프라인."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.io import LABEL_NAMES, NUM_CLASSES, UNKNOWN_LABEL

PIPELINE_LABEL_NAMES = LABEL_NAMES + ['Unknown']


# ---------------------------------------------------------------------------
# Shared decision helpers
# ---------------------------------------------------------------------------

def _validate_thresholds(tau: float, conf_thr: float) -> tuple[float, float]:
    tau = float(tau)
    conf_thr = float(conf_thr)
    if not np.isfinite(tau):
        raise ValueError(f'tau must be finite, got {tau}')
    if not 0.0 <= conf_thr <= 1.0:
        raise ValueError(f'conf_thr must be in [0, 1], got {conf_thr}')
    return tau, conf_thr


def _route_predictions(
    anomalous: np.ndarray,
    predicted_class: np.ndarray,
    max_probability: np.ndarray,
    conf_thr: float,
) -> np.ndarray:
    """공통 gate/Unknown 판정. 입력 배열은 전체 batch 길이를 공유한다."""
    anomalous = np.asarray(anomalous, dtype=bool)
    predicted_class = np.asarray(predicted_class, dtype=np.int64)
    max_probability = np.asarray(max_probability, dtype=np.float64)
    if not (anomalous.ndim == predicted_class.ndim == max_probability.ndim == 1):
        raise ValueError('routing inputs must be 1-D arrays')
    if not (len(anomalous) == len(predicted_class) == len(max_probability)):
        raise ValueError('routing inputs must have the same length')

    final = np.zeros(len(anomalous), dtype=np.int64)
    routed = np.flatnonzero(anomalous)
    if len(routed) == 0:
        return final
    if np.any((predicted_class[routed] < 0) | (predicted_class[routed] >= NUM_CLASSES)):
        raise ValueError('stage2 predicted a class outside the canonical label range')
    if not np.isfinite(max_probability[routed]).all():
        raise ValueError('stage2 produced non-finite probabilities')

    final[routed] = predicted_class[routed]
    final[routed[max_probability[routed] < conf_thr]] = UNKNOWN_LABEL
    return final


def _validate_numpy_probabilities(probs: np.ndarray, expected_rows: int) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    expected_shape = (expected_rows, NUM_CLASSES)
    if probs.shape != expected_shape:
        raise ValueError(
            f'stage2 softmax probabilities must have shape {expected_shape}, '
            f'got {probs.shape}')
    if not np.isfinite(probs).all():
        raise ValueError('stage2 probabilities contain nan or inf')
    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError('stage2 probabilities must be in [0, 1]')
    if expected_rows and not np.allclose(probs.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError('stage2 probabilities must sum to 1 for each sample')
    return probs


def compute_tau(val_normal_scores: np.ndarray, k: float = 2.0) -> float:
    """validation-normal score의 mean + k * sample standard deviation."""
    scores = np.asarray(val_normal_scores, dtype=np.float64).reshape(-1)
    if not np.isfinite(k) or k < 0:
        raise ValueError(f'k must be a non-negative finite value, got {k}')
    if not np.isfinite(scores).all():
        raise ValueError('val_normal_scores contain nan or inf')
    if len(scores) == 0:
        return 0.0
    if len(scores) == 1:
        return float(scores[0])
    return float(scores.mean() + k * scores.std(ddof=1))


# ---------------------------------------------------------------------------
# NumPy compatibility adapter
# ---------------------------------------------------------------------------

class IdentityScorer:
    """모든 입력을 정상 score=0으로 처리하는 smoke-test scorer."""

    def score(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X), dtype=np.float32)


class CallableScorer:
    """callable(X) -> scores를 scorer 인터페이스로 변환."""

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray]):
        if not callable(fn):
            raise TypeError('fn must be callable')
        self.fn = fn

    def score(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.fn(X), dtype=np.float32).reshape(-1)


class TwoStageIDS:
    """기존 NumPy scorer/probability API를 공통 판정 로직에 연결하는 어댑터."""

    def __init__(
        self,
        scorer,
        stage2_predict_proba: Callable[[np.ndarray], np.ndarray],
        tau: float,
        conf_thr: float = 0.5,
        use_cae: bool = True,
    ):
        if use_cae and scorer is None:
            raise ValueError('scorer is required when use_cae=True')
        if not callable(stage2_predict_proba):
            raise TypeError('stage2_predict_proba must be callable')
        self.tau, self.conf_thr = _validate_thresholds(tau, conf_thr)
        self.scorer = scorer
        self.stage2_predict_proba = stage2_predict_proba
        self.use_cae = bool(use_cae)

    def predict(self, X: np.ndarray, return_detail: bool = False):
        X = np.asarray(X)
        if X.ndim < 1:
            raise ValueError(f'X must include a batch dimension, got {X.shape}')
        n_samples = len(X)

        if self.use_cae:
            scores = np.asarray(self.scorer.score(X), dtype=np.float32).reshape(-1)
            if scores.shape != (n_samples,):
                raise ValueError(
                    f'scorer scores must have shape ({n_samples},), got {scores.shape}')
            if not np.isfinite(scores).all():
                raise ValueError('scorer produced nan or inf')
            anomalous = scores > self.tau
        else:
            scores = np.full(n_samples, np.nan, dtype=np.float32)
            anomalous = np.ones(n_samples, dtype=bool)

        predicted_class = np.zeros(n_samples, dtype=np.int64)
        max_probability = np.full(n_samples, np.nan, dtype=np.float64)
        routed = np.flatnonzero(anomalous)
        if len(routed):
            probs = _validate_numpy_probabilities(
                self.stage2_predict_proba(X[anomalous]), len(routed))
            predicted_class[routed] = probs.argmax(axis=1)
            max_probability[routed] = probs.max(axis=1)

        final = _route_predictions(
            anomalous, predicted_class, max_probability, self.conf_thr)
        if return_detail:
            return final, {
                'scores': scores,
                'anomalous': anomalous,
                'max_probability': max_probability,
                'tau': self.tau,
            }
        return final


# ---------------------------------------------------------------------------
# PyTorch pipeline
# ---------------------------------------------------------------------------

class TwoStagePipeline:
    def __init__(
        self,
        cae,
        s2_model,
        tau: float,
        conf_thr: float = 0.5,
        use_cae: bool = True,
    ):
        if s2_model is None:
            raise ValueError('s2_model is required')
        if use_cae and cae is None:
            raise ValueError('cae is required when use_cae=True')
        self.tau, self.conf_thr = _validate_thresholds(tau, conf_thr)
        self.cae = cae
        self.s2 = s2_model
        self.use_cae = bool(use_cae)
        self.class_names = PIPELINE_LABEL_NAMES

    @torch.no_grad()
    def _infer_components(
        self,
        x: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            shape = getattr(x, 'shape', None)
            raise ValueError(f'x must be a 4-D torch.Tensor, got {shape}')

        n_samples = len(x)
        if n_samples == 0:
            empty_float = np.empty(0, dtype=np.float32)
            empty_int = np.empty(0, dtype=np.int64)
            return empty_float, np.empty(0, dtype=bool), empty_int, empty_float

        self.s2.eval()
        if self.cae is not None:
            self.cae.eval()

        if self.use_cae:
            reconstructed = self.cae(x)
            if reconstructed.shape != x.shape:
                raise ValueError(
                    f'CAE output shape {tuple(reconstructed.shape)} != input {tuple(x.shape)}')
            mse_tensor = ((reconstructed - x) ** 2).mean(dim=(1, 2, 3))
            if not torch.isfinite(mse_tensor).all():
                raise ValueError('CAE produced non-finite reconstruction scores')
            anomalous_tensor = mse_tensor > self.tau
        else:
            mse_tensor = torch.full(
                (n_samples,), float('nan'), dtype=torch.float32, device=x.device)
            anomalous_tensor = torch.ones(n_samples, dtype=torch.bool, device=x.device)

        anomalous = anomalous_tensor.cpu().numpy()
        predicted_class = np.zeros(n_samples, dtype=np.int64)
        max_probability = np.full(n_samples, np.nan, dtype=np.float32)
        routed = np.flatnonzero(anomalous)
        if len(routed):
            logits = self.s2(x[anomalous_tensor])
            expected_shape = (len(routed), NUM_CLASSES)
            if tuple(logits.shape) != expected_shape:
                raise ValueError(
                    f's2 logits must have shape {expected_shape}, got {tuple(logits.shape)}')
            if not torch.isfinite(logits).all():
                raise ValueError('s2 model produced nan or inf logits')
            probabilities = F.softmax(logits, dim=1)
            max_prob_tensor, class_tensor = probabilities.max(dim=1)
            predicted_class[routed] = class_tensor.cpu().numpy()
            max_probability[routed] = max_prob_tensor.cpu().numpy()

        return (
            mse_tensor.cpu().numpy(),
            anomalous,
            predicted_class,
            max_probability,
        )

    def predict(self, x: torch.Tensor) -> dict:
        """배치 추론 결과를 label/stage/mse/max_prob 형태로 반환."""
        mse, anomalous, predicted_class, max_probability = self._infer_components(x)
        final = _route_predictions(
            anomalous, predicted_class, max_probability, self.conf_thr)
        stages = np.where(anomalous, 2, 1).astype(np.int64)
        return {
            'label': [PIPELINE_LABEL_NAMES[class_id] for class_id in final],
            'class_id': final.tolist(),
            'stage': stages.tolist(),
            'mse': mse.tolist(),
            'max_prob': [float(prob) if is_anomalous else None
                         for prob, is_anomalous in zip(max_probability, anomalous)],
        }

    def calibrate_conf_thr(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        device: torch.device,
        target_fpr: float = 0.05,
        artifact_path: str | Path | None = None,
        manifest_sha256: str | None = None,
    ) -> float:
        """validation normal FPR 제한을 만족하는 가장 높은 confidence threshold 선택."""
        X_val = np.asarray(X_val)
        y_val = np.asarray(y_val)
        if X_val.ndim != 4 or X_val.dtype != np.float32:
            raise ValueError(f'X_val must be float32 (N,C,H,W), got {X_val.shape}/{X_val.dtype}')
        if y_val.ndim != 1 or len(y_val) != len(X_val):
            raise ValueError(f'y_val must have shape ({len(X_val)},), got {y_val.shape}')
        if not 0.0 <= target_fpr <= 1.0:
            raise ValueError(f'target_fpr must be in [0, 1], got {target_fpr}')

        normal_mask = y_val == 0
        n_normal = int(normal_mask.sum())
        if n_normal == 0:
            return self.conf_thr

        X_tensor = torch.from_numpy(X_val).to(device)
        _, anomalous, predicted_class, max_probability = self._infer_components(X_tensor)
        candidates = np.concatenate(([0.0], np.arange(30, 95, 5) / 100.0))
        results = []
        for threshold in candidates:
            final = _route_predictions(
                anomalous, predicted_class, max_probability, float(threshold))
            false_positives = int(np.count_nonzero(normal_mask & (final != 0)))
            results.append((float(threshold), false_positives / n_normal))

        valid = [(threshold, fpr) for threshold, fpr in results
                 if fpr <= target_fpr]
        if valid:
            best_threshold, _ = max(valid, key=lambda item: item[0])
        else:
            best_threshold, _ = min(results, key=lambda item: (item[1], item[0]))
        self.conf_thr = best_threshold
        if artifact_path is not None:
            from src.utils.config import write_conf_threshold_artifact
            write_conf_threshold_artifact(
                artifact_path, best_threshold, target_fpr,
                manifest_sha256 or '')
        return best_threshold

    @classmethod
    def from_checkpoints(
        cls,
        cae_ckpt_path: str | Path | None,
        s2_model,
        tau_json_path: str | Path,
        conf_thr: float = 0.5,
        use_cae: bool = True,
        device: torch.device = torch.device('cpu'),
    ):
        """검증된 state-dict checkpoint와 tau JSON으로 파이프라인 구성."""
        from src.models.cae import CAE
        from src.models.dcnn import DCNN

        cae = None
        if use_cae:
            if cae_ckpt_path is None:
                raise ValueError('cae_ckpt_path is required when use_cae=True')
            cae_checkpoint = torch.load(
                cae_ckpt_path, map_location=device, weights_only=True)
            required = {'model_state_dict', 'input_shape', 'latent_dim'}
            missing = required - set(cae_checkpoint)
            if missing:
                raise ValueError(f'CAE checkpoint missing keys: {sorted(missing)}')
            cae = CAE(
                input_shape=tuple(cae_checkpoint['input_shape']),
                latent_dim=int(cae_checkpoint['latent_dim']),
                noise_std=float(cae_checkpoint.get('noise_std', 0.05)),
            )
            cae.load_state_dict(cae_checkpoint['model_state_dict'])
            cae.eval().to(device)

        if isinstance(s2_model, (str, Path)):
            s2_checkpoint = torch.load(
                s2_model, map_location=device, weights_only=True)
            if 'model_state_dict' not in s2_checkpoint:
                raise ValueError('S2 checkpoint missing model_state_dict')
            num_classes = int(s2_checkpoint.get('num_classes', NUM_CLASSES))
            if num_classes != NUM_CLASSES:
                raise ValueError(
                    f'S2 checkpoint num_classes must be {NUM_CLASSES}, got {num_classes}')
            s2_model = DCNN(
                num_classes=num_classes,
                dropout=float(s2_checkpoint.get('dropout', 0.5)),
            )
            s2_model.load_state_dict(s2_checkpoint['model_state_dict'])
        if s2_model is None:
            raise ValueError('s2_model is required')
        s2_model.eval().to(device)

        tau_json_path = Path(tau_json_path)
        try:
            with tau_json_path.open(encoding='utf-8') as file:
                tau_values = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f'failed to load tau JSON {tau_json_path}: {exc}') from exc
        headline = tau_values.get('headline_tau', 'tau_2sigma')
        if headline not in tau_values:
            raise ValueError(f'tau JSON missing headline value: {headline}')
        return cls(
            cae=cae,
            s2_model=s2_model,
            tau=tau_values[headline],
            conf_thr=conf_thr,
            use_cae=use_cae,
        )
