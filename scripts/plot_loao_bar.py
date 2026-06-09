"""
LOAO 탐지율 bar chart: loao_per_fold.csv / loao_summary.csv 기반.

출력: results/figures/loao_bar_chart.png
  x=5 공격 유형, y=탐지율(CAE recall / Unknown rate), error bar=seed std
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ATTACK_ORDER = ['F_I', 'P_I', 'M_F', 'C_D', 'C_R']
ATTACK_LABELS = {
    'F_I': 'Frame\nInjection',
    'P_I': 'PTP\nSync',
    'M_F': 'MAC\nFlooding',
    'C_D': 'CAN\nDoS',
    'C_R': 'CAN\nReplay',
}


def main() -> None:
    from src.utils.config import load_experiment_config
    cfg_exp = load_experiment_config()
    output_dir = Path(str(cfg_exp.output_dir))
    summary_path = output_dir / 'tables' / 'loao_summary.csv'
    figure_path = output_dir / 'figures' / 'loao_bar_chart.png'
    if not summary_path.exists():
        print(f'[WARN] {summary_path} not found — generating from per-fold CSV.')
        per_fold_path = output_dir / 'tables' / 'loao_per_fold.csv'
        if not per_fold_path.exists():
            print(f'[ERROR] {per_fold_path} not found either. Run LOAO experiment first.')
            sys.exit(1)
        df_raw = pd.read_csv(per_fold_path)
        summary_rows = []
        for atk in ATTACK_ORDER:
            sub = df_raw[df_raw['excluded_attack'] == atk]
            if sub.empty:
                continue
            summary_rows.append({
                'excluded_attack': atk,
                'cae_recall': sub['cae_anomaly_recall'].iloc[0],
                'unknown_rate_mean': sub['unknown_rate'].mean(),
                'unknown_rate_std': sub['unknown_rate'].std(),
                'normal_fpr_mean': sub['normal_fpr'].mean(),
                'normal_fpr_std': sub['normal_fpr'].std(),
            })
        df = pd.DataFrame(summary_rows)
    else:
        full_summary = pd.read_csv(summary_path)
        grand_row = full_summary[full_summary['excluded_attack'] == 'grand_mean']
        df = full_summary[full_summary['excluded_attack'] != 'grand_mean'].copy()

    # Reorder to canonical attack order
    df['_order'] = df['excluded_attack'].map({a: i for i, a in enumerate(ATTACK_ORDER)})
    df = df.sort_values('_order').reset_index(drop=True)
    labels = [ATTACK_LABELS.get(a, a) for a in df['excluded_attack']]

    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    has_cae = 'cae_recall' in df and df['cae_recall'].notna().any()
    if has_cae:
        bars1 = ax.bar(
            x - width / 2, df['cae_recall'], width,
            label='CAE Anomaly Recall (one value per fold)',
            color='#2196F3', alpha=0.85,
        )
        unknown_x = x + width / 2
    else:
        bars1 = []
        unknown_x = x
    bars2 = ax.bar(
        unknown_x,
        df['unknown_rate_mean'],
        width,
        yerr=df['unknown_rate_std'],
        label='End-to-End Unknown Rate',
        color='#FF5722',
        alpha=0.85,
        capsize=5,
        error_kw={'linewidth': 1.5},
    )

    ax.set_xlabel('Excluded Attack Class', fontsize=12)
    ax.set_ylabel('Detection Rate', fontsize=12)
    per_fold_path = output_dir / 'tables' / 'loao_per_fold.csv'
    raw = pd.read_csv(per_fold_path) if per_fold_path.exists() else None
    n_folds = int(raw['excluded_attack'].nunique()) if raw is not None else len(df)
    n_seeds = int(raw.groupby('excluded_attack')['seed'].nunique().max()) if raw is not None else 0
    ax.set_title(f'LOAO Zero-Day Detection ({n_folds}-fold x {n_seeds}-seed)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    if 'grand_row' in locals() and not grand_row.empty:
        grand_unknown = float(grand_row['unknown_rate_mean'].iloc[0])
        ax.axhline(grand_unknown, color='#FF5722', linestyle=':', linewidth=1.5,
                   label=f'Grand Unknown mean={grand_unknown:.3f}')
        ax.legend(fontsize=9)

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f'{h:.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f'{h:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {figure_path}')


if __name__ == '__main__':
    main()
