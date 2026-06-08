# src.models

PyTorch 기반 신경망 구조를 정의합니다. 학습 루프와 데이터 로드는 포함하지 않습니다.

| 파일 | 역할 |
| --- | --- |
| `cae.py` | 정상 트래픽 재구성 오차를 계산하는 Convolutional Autoencoder를 정의합니다. |
| `dcnn.py` | depthwise separable convolution block을 사용하는 TOW-IDS DCNN 분류기를 정의합니다. |

`CAE`는 입력 shape를 생성 시 고정하고 런타임 입력을 검증합니다. `DCNN`의 `num_classes`는 S1에서 2, 일반 S2에서 6, LOAO fold에서 5로 설정될 수 있습니다.
