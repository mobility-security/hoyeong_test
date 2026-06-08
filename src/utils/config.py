"""Runtime experiment configuration and confidence-threshold helpers."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


_ENV_OVERRIDES = {
    'TOW_IDS_TRAIN_NPZ': 'train_npz_path',
    'TOW_IDS_TEST_NPZ': 'test_npz_path',
    'TOW_IDS_MANIFEST': 'manifest_path',
    'TOW_IDS_OUTPUT_DIR': 'output_dir',
    'TOW_IDS_CONF_THR_ARTIFACT': 'conf_thr_artifact',
}


def load_experiment_config(path: str | Path = 'configs/experiment.yaml'):
    """Load experiment config and apply explicit runtime path overrides."""
    cfg = OmegaConf.load(path).experiment
    for env_name, key in _ENV_OVERRIDES.items():
        value = os.environ.get(env_name)
        if value:
            cfg[key] = value
    return cfg


def resolve_conf_threshold(cfg, manifest: dict | None = None) -> float:
    """Resolve the shared confidence threshold from config or an artifact."""
    source = str(cfg.get('conf_thr_source', 'fixed'))
    if source == 'fixed':
        value = float(cfg.get('conf_thr', 0.5))
    elif source == 'artifact':
        artifact_value = cfg.get('conf_thr_artifact')
        if not artifact_value:
            artifact_value = str(
                Path(str(cfg.get('output_dir', 'results'))) / 'tables' / 'conf_threshold.json')
        artifact_path = Path(str(artifact_value))
        try:
            data = json.loads(artifact_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f'failed to load confidence-threshold artifact {artifact_path}: {exc}') from exc
        if 'conf_thr' not in data:
            raise ValueError(f'confidence-threshold artifact missing conf_thr: {artifact_path}')
        if manifest is not None:
            expected = manifest.get('sha256')
            actual = data.get('manifest_sha256')
            if not expected or actual != expected:
                raise ValueError(
                    'confidence-threshold artifact does not match the active split manifest')
        value = float(data['conf_thr'])
    else:
        raise ValueError(f'unsupported conf_thr_source: {source}')

    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f'conf_thr must be finite and in [0, 1], got {value}')
    return value


def write_conf_threshold_artifact(
    path: str | Path,
    conf_thr: float,
    target_fpr: float,
    manifest_sha256: str,
) -> Path:
    """Persist a calibrated threshold with the validation split identity."""
    conf_thr = float(conf_thr)
    target_fpr = float(target_fpr)
    if not 0.0 <= conf_thr <= 1.0:
        raise ValueError(f'conf_thr must be in [0, 1], got {conf_thr}')
    if not 0.0 <= target_fpr <= 1.0:
        raise ValueError(f'target_fpr must be in [0, 1], got {target_fpr}')
    if not manifest_sha256:
        raise ValueError('manifest_sha256 is required for a calibrated threshold')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        'conf_thr': conf_thr,
        'target_fpr': target_fpr,
        'manifest_sha256': manifest_sha256,
    }, indent=2), encoding='utf-8')
    return path
