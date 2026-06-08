# src

전처리부터 2단계 추론까지 프로젝트의 재사용 가능한 Python 구현을 담은 패키지입니다.

| 패키지 | 역할 |
| --- | --- |
| `preprocessing/` | PCAP을 파싱하고 이미지 및 3-channel wavelet 텐서로 변환합니다. |
| `data/` | 기존 import 경로를 유지하기 위한 데이터 관련 호환 계층입니다. |
| `models/` | CAE와 DCNN 모델 구조를 정의합니다. |
| `train/` | S1, S2, CAE 학습 및 평가 루프를 구현합니다. |
| `pipeline/` | CAE gate와 Stage 2 분류기를 결합한 최종 추론을 구현합니다. |
| `utils/` | 데이터 계약, split, 지표, seed, loss 등 공통 기능을 제공합니다. |

의존 방향은 대체로 `utils` → `preprocessing/models` → `train/pipeline`입니다. 실행 진입점은 `python -m src.<module>` 형식을 우선하고, 전체 작업은 `scripts/`를 사용합니다.
