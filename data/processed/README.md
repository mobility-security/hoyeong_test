# processed

전처리와 데이터 분할 과정에서 생성된 모델 입력 및 메타데이터를 보관합니다.

## 주요 파일

| 패턴 | 내용 |
| --- | --- |
| `dataset_train.npz` | 학습 및 validation 분할의 원본 텐서입니다. |
| `dataset_test.npz` | 학습에 사용하지 않는 frozen test 텐서입니다. |
| `dataset_v0.npz` | 파이프라인 확인용 stub 데이터입니다. |
| `*.meta.json` | shape, wavelet, 정규화, label map, 생성 시점 등의 sidecar 메타데이터입니다. |
| `split_manifest.json` | train/validation/test 인덱스와 클래스 수를 기록합니다. |
| `normal_only_idx.npy` | CAE 학습에 사용할 정상 train 샘플 인덱스입니다. |

NPZ의 핵심 배열은 `(N, 3, H, W)` `float32` 형식의 `X`와 `(N,)` `int64` 형식의 `y`입니다. 학습 데이터에는 누수 방지를 위한 `pcap_id`와 각 window의 반열린 패킷 범위 `[packet_start, packet_end)`도 필요합니다. 여러 캡처는 `pcap_id` 그룹으로, 단일 캡처는 시간 구간과 guard gap으로 분할합니다. 정확한 검증 규칙은 `src/utils/io.py`에 있습니다.

`bash scripts/run.sh smoke`가 생성하는 임시 데이터는 `data/smoke/`에 격리되며 이 디렉터리의 운영 데이터와 manifest를 덮어쓰지 않습니다.

이 디렉터리의 파일은 직접 편집하지 말고 전처리 및 split 명령으로 다시 생성합니다.
