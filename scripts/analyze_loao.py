"""M2: LOAO 결과 분산·이상치 진단 스크립트.

기존 loao_per_fold.csv 를 읽어 다음을 보고한다:
  - fold별 unknown_rate 평균 ± std, 이상치 (|z|>2 또는 절대값 임계)
  - CAE recall이 낮은 fold (C_D, C_R) 의 상세 분포
  - seed ensemble (다수결) vs 단순 평균의 비교
  - per-fold/seed 히트맵 (텍스트 표)

사용:
    python scripts/analyze_loao.py
    python scripts/analyze_loao.py --out-dir results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _zscore(series: pd.Series) -> pd.Series:
    mu, sigma = series.mean(), series.std(ddof=1)
    return (series - mu) / sigma if sigma > 0 else series * 0.0


def _text_heatmap(df: pd.DataFrame, value_col: str, row_col: str, col_col: str) -> str:
    pivot = df.pivot_table(index=row_col, columns=col_col, values=value_col)
    lines = [f'\n  {value_col} heatmap ({row_col} × {col_col}):']
    header = f"  {'':>12}" + ''.join(f'  seed={c}' for c in pivot.columns)
    lines.append(header)
    for row_label, row in pivot.iterrows():
        vals = ''.join(
            f'  {v:6.3f}' if not pd.isna(v) else '     —  '
            for v in row)
        lines.append(f'  {str(row_label):>12}{vals}')
    return '\n'.join(lines)


def _seed_ensemble(df: pd.DataFrame, excl: str, threshold: float = 0.5) -> float:
    """다수결 ensemble: seed별 unknown_rate >= 0.5 → 예측 Unknown."""
    sub = df[df['excluded_attack'] == excl]
    votes = (sub['unknown_rate'] >= threshold).mean()
    return float(votes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--outlier-z', type=float, default=2.0,
                        help='z-score threshold to flag outlier seeds')
    parser.add_argument('--low-recall-thr', type=float, default=0.10,
                        help='CAE recall below this is flagged as weak')
    args = parser.parse_args()

    from src.utils.config import load_experiment_config
    cfg_exp = load_experiment_config()
    out_dir = Path(args.out_dir or str(cfg_exp.output_dir))
    per_fold_path = out_dir / 'tables' / 'loao_per_fold.csv'
    summary_path = out_dir / 'tables' / 'loao_summary.csv'

    if not per_fold_path.exists():
        print(f'[ERROR] {per_fold_path} not found. Run experiments/leave_one_out.py first.')
        sys.exit(1)

    df = pd.read_csv(per_fold_path)
    summary = pd.read_csv(summary_path) if summary_path.exists() else None

    print('=' * 70)
    print('M2: LOAO 분산·이상치 진단')
    print('=' * 70)

    # --- 1. per-fold 요약 ---
    print('\n[1] fold별 unknown_rate 분포')
    fold_stats = df.groupby('excluded_attack')['unknown_rate'].agg(
        ['mean', 'std', 'min', 'max']).round(4)
    fold_stats.columns = ['mean', 'std', 'min', 'max']
    print(fold_stats.to_string())

    # --- 2. 이상치 탐지 ---
    print(f'\n[2] 이상치 (|z| > {args.outlier_z}) — per fold z-score')
    outliers = []
    for excl, sub in df.groupby('excluded_attack'):
        z = _zscore(sub['unknown_rate'])
        flagged = sub[z.abs() > args.outlier_z]
        for _, row in flagged.iterrows():
            outliers.append({
                'fold': excl,
                'seed': int(row['seed']),
                'unknown_rate': round(row['unknown_rate'], 4),
                'z_score': round(z[row.name], 3),
                'cae_recall': round(row.get('cae_anomaly_recall', float('nan')), 4),
            })
    if outliers:
        print(pd.DataFrame(outliers).to_string(index=False))
    else:
        print('  이상치 없음.')

    # --- 3. CAE recall 약점 분석 ---
    print(f'\n[3] CAE recall < {args.low_recall_thr} (weak detection)')
    weak = df[df['cae_anomaly_recall'] < args.low_recall_thr][
        ['excluded_attack', 'seed', 'cae_anomaly_recall', 'unknown_rate', 'normal_fpr']]
    if len(weak):
        print(weak.to_string(index=False))
        print(f'  → weak fold 수: {len(weak)} / {len(df)} '
              f'({100*len(weak)/len(df):.1f}%)')
    else:
        print('  없음.')

    # --- 4. seed ensemble vs 평균 비교 ---
    print('\n[4] seed ensemble (다수결, 임계=0.5) vs 단순 mean')
    header = f"  {'fold':<8}  {'mean_ur':>8}  {'ensemble_vote':>14}  {'min_ur':>7}  {'max_ur':>7}"
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for excl in sorted(df['excluded_attack'].unique()):
        sub = df[df['excluded_attack'] == excl]
        mean_ur = sub['unknown_rate'].mean()
        ens = _seed_ensemble(df, excl)
        min_ur = sub['unknown_rate'].min()
        max_ur = sub['unknown_rate'].max()
        print(f'  {excl:<8}  {mean_ur:>8.4f}  {ens:>14.2f}  {min_ur:>7.4f}  {max_ur:>7.4f}')

    # --- 5. 히트맵 ---
    print(_text_heatmap(df, 'unknown_rate', 'excluded_attack', 'seed'))
    print(_text_heatmap(df, 'cae_anomaly_recall', 'excluded_attack', 'seed'))

    # --- 6. 분산 완화 제안 ---
    high_var = fold_stats[fold_stats['std'] > 0.10]
    if len(high_var):
        print('\n[5] 분산 완화 권장 fold (std > 0.10):')
        for excl in high_var.index:
            print(f'  - {excl}: std={high_var.loc[excl, "std"]:.4f} '
                  '→ seed 수 늘리거나 OOD 스코어 앙상블 검토')
    else:
        print('\n[5] 모든 fold std ≤ 0.10 — 분산 양호.')

    # --- Save ---
    report = {
        'fold_stats': fold_stats.reset_index().to_dict(orient='records'),
        'outliers': outliers,
        'weak_cae_recall_count': len(weak),
        'weak_cae_recall_threshold': args.low_recall_thr,
    }
    out_path = out_dir / 'tables' / 'loao_variance_report.json'
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    print(f'\n[OK] → {out_path}')


if __name__ == '__main__':
    main()
