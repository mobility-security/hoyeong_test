# src.train

모델별 학습, validation, checkpoint 저장, 평가 결과 생성을 담당합니다.

| 파일 | 역할 |
| --- | --- |
| `common.py` | DataLoader 생성, device 선택, manifest 로드, early stopping을 제공합니다. |
| `train_s1.py` | Normal과 Attack을 구분하는 2-class DCNN baseline을 학습합니다. |
| `train_s2.py` | 6-class DCNN을 Focal Loss로 학습하고 seed별 성능을 집계합니다. |
| `train_cae.py` | 정상 샘플만으로 CAE를 학습하고 MSE 기반 `tau`와 관련 그림을 생성합니다. |

```bash
python -m src.train.train_s1
python -m src.train.train_s2
python -m src.train.train_cae
```

각 모듈은 빠른 연결 검증을 위한 `--smoke` 옵션을 지원합니다. 학습 전에 `data/processed/split_manifest.json`을 준비해야 하며 frozen test set은 최종 평가에만 사용합니다.
