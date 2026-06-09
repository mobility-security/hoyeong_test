# results

학습과 평가에서 생성되는 결과 디렉터리입니다. 데이터 또는 manifest가 바뀌면 이전
결과는 유효하지 않으므로 표와 그림을 Git에 기준 결과로 보관하지 않습니다.

| 디렉터리 | 역할 |
| --- | --- |
| `checkpoints/` | CAE와 DCNN의 학습된 weight를 저장합니다. |
| `figures/` | confusion matrix, ROC, MSE 분포, LOAO 차트 등 보고서용 그림을 저장합니다. |
| `tables/` | seed/fold별 지표와 요약 결과를 CSV, JSON, Markdown으로 저장합니다. |

checkpoint와 threshold에는 manifest/data/config 해시가 포함되며, 불일치하면 로더가
실행을 중단합니다. 비교표에는 별도 `comparison_table.provenance.json`도 생성됩니다.

```bash
bash scripts/run.sh train
bash scripts/run.sh loao
bash scripts/run.sh visualize
```
