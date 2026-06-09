import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import torch

from src.pipeline.two_stage import TwoStagePipeline


class FixedS2(torch.nn.Module):
    """첫 샘플은 Normal, 두 번째 샘플은 F_I로 약 0.8 confidence 예측."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        attack = x.mean(dim=(1, 2, 3)) > 0.5
        logits = torch.zeros((len(x), 6), device=x.device)
        logits[:, 0] = 3.0
        logits[attack, 0] = 0.0
        logits[attack, 1] = 3.0
        return logits


class TwoStageCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.X = np.stack([
            np.zeros((3, 4, 4), dtype=np.float32),
            np.ones((3, 4, 4), dtype=np.float32),
        ])
        self.y = np.zeros(2, dtype=np.int64)
        self.s2 = FixedS2()
        self.pipeline = TwoStagePipeline(
            cae=None,
            s2_model=self.s2,
            tau=0.0,
            conf_thr=0.5,
            use_cae=False,
        )

    def test_selects_highest_threshold_within_fpr_limit(self):
        threshold = self.pipeline.calibrate_conf_thr(
            self.X, self.y, torch.device('cpu'), target_fpr=0.5,
        )
        self.assertEqual(threshold, 0.8)

    def test_fallback_uses_minimum_fpr_threshold(self):
        threshold = self.pipeline.calibrate_conf_thr(
            self.X, self.y, torch.device('cpu'), target_fpr=0.1,
        )
        self.assertEqual(threshold, 0.0)

    def test_calibration_artifact_records_manifest(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'threshold.json'
            threshold = self.pipeline.calibrate_conf_thr(
                self.X, self.y, torch.device('cpu'), target_fpr=0.5,
                artifact_path=path, manifest={
                    'sha256': 'manifest-hash',
                    'train_sha256': 'train-hash',
                    'test_sha256': 'test-hash',
                },
            )
            self.assertEqual(threshold, 0.8)
            self.assertIn('manifest-hash', path.read_text(encoding='utf-8'))

    def test_cae_is_required_when_enabled(self):
        with self.assertRaises(ValueError):
            TwoStagePipeline(
                cae=None, s2_model=FixedS2(), tau=0.1, use_cae=True,
            )

    def test_predict_batches_stage2_once(self):
        X_tensor = torch.from_numpy(np.repeat(self.X, 4, axis=0))
        result = self.pipeline.predict(X_tensor)
        self.assertEqual(self.s2.calls, 1)
        self.assertEqual(len(result['class_id']), len(X_tensor))
        self.assertEqual(result['stage'], [2] * len(X_tensor))


if __name__ == '__main__':
    unittest.main()
