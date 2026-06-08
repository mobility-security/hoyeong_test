# src.pipeline

학습된 구성 요소를 결합해 최종 예측을 만드는 추론 파이프라인입니다.

`two_stage.py`는 두 가지 인터페이스를 제공합니다.

- `TwoStagePipeline`: PyTorch CAE와 DCNN을 배치 단위로 실행하는 실제 추론 구현입니다.
- `TwoStageIDS`: NumPy scorer와 callable 분류기를 받아 라우팅 규칙을 독립적으로 검증할 수 있는 경량 구현입니다.

CAE를 사용하는 경우 재구성 MSE가 `tau` 이하인 샘플은 Normal로 종료합니다. `tau`를 넘은 샘플은 Stage 2로 전달하며 최대 class probability가 `conf_thr`보다 낮으면 Unknown(레이블 6)으로 반환합니다. `use_cae=False`이면 모든 샘플을 Stage 2로 직접 전달합니다.
