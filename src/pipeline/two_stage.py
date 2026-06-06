"""
2단계 게이트 캐스케이드 파이프라인 (Stage 3).

Stage 1 — CAE 이상 탐지 게이트:
  MSE(x) ≤ tau  →  Normal (종료)

Stage 2 — S2 공격 분류기:
  max_softmax_prob ≥ conf_thr  →  F_I / P_I / M_F / C_D / C_R 중 하나
  max_softmax_prob <  conf_thr →  Unknown (제로데이 의심)

use_cae=False 시 Stage 1을 건너뛰고 S2 단독 모드로 동작.
configs/experiment.yaml의 use_cae 플래그만 바꾸면 됨 — 코드 수정 불필요.
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
        배치 추론.

        반환값:
          label    : List[str]        — 7개 클래스명 중 하나
          stage    : List[int]        — 1 (CAE Normal 판정) 또는 2 (S2 분류기 경유)
          mse      : List[float]      — 이미지별 복원 오차 (use_cae=False면 nan)
          max_prob : List[float|None] — S2 max-softmax 확률 (Stage 1 종료 샘플은 None)
        """
        self.s2.eval()
        if self.cae is not None:
            self.cae.eval()

        # Stage 1: CAE 복원 오차 계산
        if self.use_cae and self.cae is not None:
            xhat = self.cae(x)
            mse  = ((xhat - x) ** 2).mean(dim=(1, 2, 3))   # (B,)
        else:
            mse = torch.full((len(x),), float('nan'))

        labels, stages, max_probs = [], [], []

        for i in range(len(x)):
            mse_val = mse[i].item()

            # Stage 1 게이트: MSE ≤ τ 이면 Normal로 조기 종료
            if self.use_cae and not np.isnan(mse_val) and mse_val <= self.tau:
                labels.append('Normal')
                stages.append(1)
                max_probs.append(None)
                continue

            # Stage 2: S2 분류기 — max-softmax로 Unknown 판단
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
        val 세트에서 Normal FPR ≤ target_fpr을 만족하는 최소 conf_thr을 탐색.
        탐색 범위: conf_thr ∈ [0.30, 0.90), step=0.05.
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
        """저장된 체크포인트와 tau JSON으로부터 파이프라인을 바로 구성하는 편의 로더."""
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
