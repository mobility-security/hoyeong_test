# src.pipeline

학습된 구성 요소를 결합해 최종 예측을 만드는 추론 파이프라인입니다.

`two_stage.py`는 두 가지 인터페이스를 제공합니다.

- `TwoStagePipeline`: PyTorch CAE와 DCNN을 배치 단위로 실행하는 실제 추론 구현입니다.
- `TwoStageIDS`: NumPy scorer와 callable 분류기를 받아 라우팅 규칙을 독립적으로 검증할 수 있는 경량 구현입니다.

CAE를 사용하더라도 S2는 전체 batch를 평가합니다. `CAE anomaly` 또는
`S2 prediction != Normal`인 샘플을 공격 후보로 간주하므로 CAE가 놓친 공격을 S2가
복구할 수 있습니다. 공격 후보의 최대 확률이 `conf_thr`보다 낮으면 Unknown(레이블 6),
두 모델이 모두 Normal이면 Normal을 반환합니다. `use_cae=False`이면 S2 confidence-only
모드로 동작합니다.
