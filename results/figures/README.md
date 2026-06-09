# figures

실험 결과를 보고서와 발표 자료에서 사용할 수 있도록 시각화한 PNG 파일을 보관합니다.

## 그림 종류

- `cm_s2_*.png`: Stage 2 분류 confusion matrix입니다.
- `cm_s3_7class.png`: Unknown을 포함한 2단계 파이프라인 confusion matrix입니다.
- `mse_histogram.png`, `roc_cae.png`: CAE 재구성 오차 분포와 ROC입니다.
- `loao_bar_chart.png`: 공격 클래스별 CAE recall과 Unknown 판정률입니다.
- `unknown_case_4panel.png`: 실제 held-out Unknown 사례의 입력, 재구성,
  squared-error map, S2 confidence입니다.

합성 fallback은 사용하지 않습니다. 실제 fold checkpoint와 Unknown 판정 샘플이 없으면
그림 생성은 실패합니다.
