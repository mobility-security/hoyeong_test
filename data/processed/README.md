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

NPZ의 핵심 배열은 `(N, 3, H, W)` `float32` 형식의 `X`와 `(N,)` `int64`
형식의 `y`입니다. metadata schema v2는 wavelet 순서, frame content, padding,
window/stride, seed, Git SHA를 필수로 기록합니다. `pcap_id`가 있으면 각 window의
반열린 패킷 범위 `[packet_start, packet_end)`도 필수입니다.

manifest는 NPZ와 sidecar SHA-256을 저장합니다. 파일 내용이나 manifest JSON이 바뀌면
학습과 추론이 즉시 실패하므로 데이터 변경 후 반드시 manifest를 다시 생성해야 합니다.
단일 capture는 `class_temporal_tail` 전략으로 각 클래스의 시간상 마지막
샘플을 validation으로 사용하며 packet guard 위반 train window를 제거합니다.

`bash scripts/run.sh smoke`가 생성하는 임시 데이터는 `data/smoke/`에 격리되며 이 디렉터리의 운영 데이터와 manifest를 덮어쓰지 않습니다.

이 디렉터리의 파일은 직접 편집하지 말고 전처리 및 split 명령으로 다시 생성합니다.
