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
| `comparison_table_detailed.csv` | seed와 mean/std/95% CI 전체 행입니다. |
| `comparison_table.tex` | 보고서용 LaTeX 표입니다. |
| `comparison_table.provenance.json` | 데이터와 manifest 해시입니다. |

이 파일들은 생성물이며 현재 저장소에는 기준 수치를 커밋하지 않습니다. 결과를 인용할
때는 같은 디렉터리의 provenance와 checkpoint 해시를 함께 보관합니다.
