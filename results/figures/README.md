# figures

실험 결과를 보고서와 발표 자료에서 사용할 수 있도록 시각화한 PNG 파일을 보관합니다.

## 그림 종류

- `cm_s2_*.png`: Stage 2 분류 confusion matrix입니다.
- `cm_s3_7class.png`: Unknown을 포함한 2단계 파이프라인 confusion matrix입니다.
- `mse_histogram.png`, `roc_cae.png`: CAE 재구성 오차 분포와 ROC입니다.
- `loao_bar_chart.png`: 공격 클래스별 CAE recall과 Unknown 판정률입니다.
- `unknown_case_4panel.png`: 미지 공격 사례의 입력, 재구성, 오차맵, MSE 분포입니다.

그림은 `scripts/plot_*.py` 또는 `bash scripts/run.sh visualize`로 다시 생성합니다. 동일한 결과를 재현하려면 먼저 대응하는 `results/tables/`와 `results/checkpoints/`가 준비되어 있어야 합니다.
