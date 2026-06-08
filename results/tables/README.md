# tables

학습 및 평가 지표의 원본과 요약을 기계가 읽을 수 있는 형식으로 보관합니다.

## 주요 결과

| 파일 | 내용 |
| --- | --- |
| `s1_baseline.csv` | Stage 1 이진 분류 baseline의 seed별 성능입니다. |
| `s2_summary*.csv` | Stage 2의 seed별 전체 성능입니다. |
| `s2_per_class*.csv` | Stage 2의 클래스별 precision, recall, F1입니다. |
| `tau_values.json` | CAE validation 오차로 계산한 threshold 후보입니다. |
| `tau_sensitivity.csv` | threshold 변화에 따른 탐지 성능입니다. |
| `loao_per_fold.csv` | 공격 제외 fold와 seed별 LOAO 결과입니다. |
| `loao_summary.csv` | 공격 클래스별 LOAO 집계 결과입니다. |
| `comparison_table.csv`, `comparison_table.md` | S1, S2, S3 비교표입니다. |

CSV와 JSON을 결과의 원본으로 사용하고 Markdown은 표시용 파생 파일로 취급합니다. 각 값은 생성 스크립트와 checkpoint에 종속되므로 수동 편집보다 재실행을 우선합니다.
