# src.pipeline

학습된 구성 요소를 결합해 최종 예측을 만드는 추론 파이프라인입니다.

`two_stage.py`는 두 가지 인터페이스를 제공합니다.

- `TwoStagePipeline`: PyTorch CAE와 DCNN을 배치 단위로 실행하는 실제 추론 구현입니다.
- `TwoStageIDS`: NumPy scorer와 callable 분류기를 받아 라우팅 규칙을 독립적으로 검증할 수 있는 경량 구현입니다.

기본 `routing_mode=strict_cascade`에서는 `MSE <= tau`인 샘플을 즉시
Normal로 반환하고 S2를 실행하지 않습니다. CAE anomaly 샘플만 S2에
전달하며 최대 확률이 `conf_thr`보다 낮으면 Unknown(레이블 6)으로
판정합니다.

`routing_mode=s2_recovery`는 CAE가 놓친 공격을 S2가 복구하는 ablation
전용입니다. headline LOAO에는 사용하지 않습니다. `use_cae=False`이면
S2 confidence-only 모드로 동작합니다.
