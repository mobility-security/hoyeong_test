"""Leave-One-Attack-Out (LOAO) zero-day detection protocol. Phase 0: signatures only."""
from typing import List


def run_loao(
    excluded_attack: str,   # 'F_I' | 'P_I' | 'M_F' | 'C_D' | 'C_R'
    config: dict,
    cae_model,              # trained once, reused across all folds
    split_manifest: dict,
) -> dict:
    """
    Run leave-one-attack-out evaluation excluding one attack class.
    CAE is trained once on normal-only data and reused for every fold.

    Returns:
        {
          'cae_anomaly_recall': float,   # P(MSE > tau | attack_k)
          'unknown_rate': float,          # end-to-end Unknown success rate
          'normal_fpr': float,
          'auc_roc': float,
          'seed_results': List[dict],
        }
    """
    raise NotImplementedError('run_loao is implemented in Phase 4.')
