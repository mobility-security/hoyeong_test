"""
Two-Stage Gated Cascade Pipeline (S3).

Stage 1 — CAE gate:
  MSE(x) ≤ tau  →  Normal (stop)

Stage 2 — S2 classifier:
  max_softmax_prob ≥ conf_thr  →  one of F_I / P_I / M_F / C_D / C_R
  max_softmax_prob <  conf_thr →  Unknown

When use_cae=False (Day-9 FAIL gate), all inputs bypass Stage 1 and go
directly to Stage 2 (S2-only mode). No code changes needed — flip the flag.
"""
import json

import numpy as np
import torch
import torch.nn.functional as F


CLASS_NAMES = ['Normal', 'F_I', 'P_I', 'M_F', 'C_D', 'C_R', 'Unknown']
_S2_CLASSES  = CLASS_NAMES[:6]   # indices 0-5


class TwoStagePipeline:
    def __init__(self, cae, s2_model, tau: float,
                 conf_thr: float = 0.5, use_cae: bool = True):
        """
        cae      : trained CAE (can be None when use_cae=False)
        s2_model : trained 6-class DCNN
        tau      : MSE threshold (from tau_values.json, headline_tau)
        conf_thr : max-softmax threshold for Unknown detection
        use_cae  : False → S2-only mode (maps to configs/experiment.yaml use_cae)
        """
        self.cae      = cae
        self.s2       = s2_model
        self.tau      = tau
        self.conf_thr = conf_thr
        self.use_cae  = use_cae
        self.class_names = CLASS_NAMES

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict:
        """
        Batch inference.

        Returns:
          label    : List[str]   — one of 7 class names
          stage    : List[int]   — 1 (CAE Normal) or 2 (S2 branch)
          mse      : List[float] — per-image reconstruction error (nan if use_cae=False)
          max_prob : List[float|None] — S2 max-softmax (None for stage-1 samples)
        """
        self.s2.eval()
        if self.cae is not None:
            self.cae.eval()

        # Stage 1: CAE MSE
        if self.use_cae and self.cae is not None:
            xhat = self.cae(x)
            mse  = ((xhat - x) ** 2).mean(dim=(1, 2, 3))   # (B,)
        else:
            mse = torch.full((len(x),), float('nan'))

        labels, stages, max_probs = [], [], []

        for i in range(len(x)):
            mse_val = mse[i].item()

            # Stage 1 gate
            if self.use_cae and not np.isnan(mse_val) and mse_val <= self.tau:
                labels.append('Normal')
                stages.append(1)
                max_probs.append(None)
                continue

            # Stage 2: S2 classifier
            logits = self.s2(x[i].unsqueeze(0))
            probs  = F.softmax(logits, dim=1)
            mp, pc = probs.max(dim=1)
            mp_val = mp.item()
            max_probs.append(mp_val)

            if mp_val < self.conf_thr:
                labels.append('Unknown')
            else:
                labels.append(_S2_CLASSES[pc.item()])
            stages.append(2)

        return {
            'label':    labels,
            'stage':    stages,
            'mse':      mse.tolist(),
            'max_prob': max_probs,
        }

    def calibrate_conf_thr(self, X_val: np.ndarray, y_val: np.ndarray,
                           device: torch.device,
                           target_fpr: float = 0.05) -> float:
        """
        Find the smallest conf_thr such that Normal FPR ≤ target_fpr on val set.
        Searches conf_thr ∈ [0.30, 0.90) step 0.05.
        """
        n_normal = int((y_val == 0).sum())
        if n_normal == 0:
            print('[WARN] No normal samples in val — skipping calibration.')
            return self.conf_thr

        X_t = torch.from_numpy(X_val).to(device)
        best_thr = None

        for thr in np.arange(0.30, 0.95, 0.05):
            self.conf_thr = float(thr)
            preds = self.predict(X_t)
            n_fp  = sum(1 for i, lbl in enumerate(preds['label'])
                        if y_val[i] == 0 and lbl != 'Normal')
            fpr = n_fp / n_normal
            print(f'  calibrate: conf_thr={thr:.2f}  normal_fpr={fpr:.3f}')
            if fpr <= target_fpr:
                best_thr = float(thr)
                break

        if best_thr is None:
            print(f'[WARN] No conf_thr with FPR ≤ {target_fpr:.0%}. Using 0.90.')
            best_thr = 0.90

        self.conf_thr = best_thr
        print(f'[OK] Calibrated conf_thr = {best_thr:.2f}')
        return best_thr

    @classmethod
    def from_checkpoints(cls, cae_ckpt_path: str, s2_model,
                         tau_json_path: str,
                         conf_thr: float = 0.5,
                         use_cae: bool = True,
                         device: torch.device = torch.device('cpu')):
        """Convenience loader: build pipeline from saved artifacts."""
        from src.models.cae import CAE

        ckpt = torch.load(cae_ckpt_path, map_location=device)
        cae  = CAE(input_shape=tuple(ckpt['input_shape']),
                   latent_dim=ckpt['latent_dim'])
        cae.load_state_dict(ckpt['model_state_dict'])
        cae.eval().to(device)

        with open(tau_json_path) as f:
            taus = json.load(f)
        headline = taus.get('headline_tau', 'tau_2sigma')
        tau      = taus[headline]

        return cls(cae=cae, s2_model=s2_model, tau=tau,
                   conf_thr=conf_thr, use_cae=use_cae)
