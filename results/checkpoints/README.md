# checkpoints

학습된 PyTorch 모델 checkpoint를 저장합니다.

## 주요 이름

| 패턴 | 모델 |
| --- | --- |
| `cae_best.pth` | validation 정상 샘플의 재구성 오차를 기준으로 선택한 CAE입니다. |
| `s2_seed_<seed>_best.pth` | seed별 최적 Stage 2 DCNN입니다. |
| `loao/exclude_<attack>_seed_<seed>.pth` | LOAO fold별 5-class DCNN입니다. |

`.pth` 파일은 용량 때문에 Git에서 제외됩니다. 각 checkpoint에는 manifest, train/test
dataset, config 해시가 저장되며 현재 manifest와 다르면 로딩이 거부됩니다.
