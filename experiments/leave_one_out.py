"""
Phase 4: Leave-One-Attack-Out (LOAO) 제로데이 탐지 평가 프로토콜.
공격 유형 하나를 학습에서 제외하고, 해당 공격을 Unknown으로 탐지할 수 있는지 평가.
"""
from typing import List


def run_loao(
    excluded_attack: str,   # 'F_I' | 'P_I' | 'M_F' | 'C_D' | 'C_R'
    config: dict,
    cae_model,              # 정상 데이터로 1회 학습 후 모든 fold에서 재사용
    split_manifest: dict,
) -> dict:
    """
    지정된 공격 클래스를 제외하고 LOAO 평가 수행.
    CAE는 정상 데이터로 한 번만 학습하고 모든 fold에서 공유.

    반환값:
        {
          'cae_anomaly_recall': float,   # P(MSE > tau | 제외된 공격 클래스)
          'unknown_rate': float,          # 엔드투엔드 Unknown 탐지 성공률
          'normal_fpr': float,
          'auc_roc': float,
          'seed_results': List[dict],
        }
    """
    raise NotImplementedError('run_loao는 Phase 4에서 구현 예정.')
