# results

학습과 평가에서 생성된 현재 기준 결과를 보관합니다.

| 디렉터리 | 역할 |
| --- | --- |
| `checkpoints/` | CAE와 DCNN의 학습된 weight를 저장합니다. |
| `figures/` | confusion matrix, ROC, MSE 분포, LOAO 차트 등 보고서용 그림을 저장합니다. |
| `tables/` | seed/fold별 지표와 요약 결과를 CSV, JSON, Markdown으로 저장합니다. |

결과 파일은 `scripts/run.sh`, `src/train/`, `experiments/`, 시각화 스크립트가 생성합니다. 수동으로 값을 수정하면 checkpoint 및 실행 설정과 불일치할 수 있으므로 가능한 한 생성 명령을 다시 실행합니다.

```bash
bash scripts/run.sh train
bash scripts/run.sh loao
bash scripts/run.sh visualize
```
