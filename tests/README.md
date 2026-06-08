# tests

전처리, 데이터 계약, split 누수 방지, 모델 유틸리티, 2단계 라우팅, LOAO 실험을 검증하는 pytest 테스트입니다. 대부분 합성 데이터를 사용하므로 원본 PCAP이나 전체 학습 checkpoint 없이 실행할 수 있습니다.

| 파일 | 검증 범위 |
| --- | --- |
| `test_io.py` | NPZ schema, metadata, 저장 및 로드 오류 처리 |
| `test_preprocessing.py` | PCAP round-trip, 이미징, wavelet shape와 값 범위 |
| `test_split_pipeline.py` | 그룹 누수 방지 split, manifest, 경량 2단계 라우팅 |
| `test_two_stage.py` | PyTorch 파이프라인 threshold 보정과 배치 추론 |
| `test_training_utils.py` | 모델 입력 검증, Focal Loss, early stopping, CAE threshold |
| `test_loao.py` | label remap, fold 학습 및 LOAO metric |

저장소 루트에서 실행합니다.

```bash
pytest -q
```

특정 영역만 확인하려면 파일 경로나 `-k` 표현식을 지정합니다.
