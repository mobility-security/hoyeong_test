# checkpoints

학습된 PyTorch 모델 checkpoint를 저장합니다.

## 주요 이름

| 패턴 | 모델 |
| --- | --- |
| `cae_best.pth` | validation 정상 샘플의 재구성 오차를 기준으로 선택한 CAE입니다. |
| `s2_seed_<seed>_best.pth` | seed별 최적 Stage 2 DCNN입니다. |

`.pth` 파일은 용량 때문에 Git에서 제외됩니다. checkpoint를 읽을 때는 생성 당시의 모델 구조와 입력 크기 설정이 현재 `configs/`와 일치하는지 확인합니다.
