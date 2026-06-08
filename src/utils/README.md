# src.utils

여러 단계에서 공유하는 데이터 및 학습 보조 기능을 제공합니다.

| 파일 | 역할 |
| --- | --- |
| `io.py` | NPZ schema, dtype, capture/packet provenance 검증과 데이터 저장 및 로드를 담당합니다. |
| `split.py` | 다중 캡처 group split 또는 단일 캡처 시간 구간 split과 manifest 생성을 담당합니다. |
| `config.py` | 실험 YAML과 runtime 경로 override, 공통 confidence threshold artifact를 처리합니다. |
| `benchmark.py` | device 동기화를 포함한 반복 추론 latency를 측정합니다. |
| `metrics.py` | binary, multiclass, LOAO 지표를 계산합니다. |
| `focal_loss.py` | 다중 클래스 Focal Loss를 구현합니다. |
| `seed.py` | Python, NumPy, PyTorch의 난수 seed를 고정합니다. |

split manifest는 다음 명령으로 생성합니다.

```bash
python -m src.utils.split \
  --train-npz data/processed/dataset_train.npz \
  --test-npz data/processed/dataset_test.npz \
  --guard-gap-packets 64
```

데이터 형식이나 label 정의를 바꿀 때는 `io.py`의 검증 규칙, 테스트, 기존 metadata의 호환성을 함께 갱신합니다.
