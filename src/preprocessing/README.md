# src.preprocessing

원본 패킷을 모델 입력 텐서로 변환하는 전처리 단계입니다.

## 처리 순서

1. `pcap_parser.py`가 PCAP/PCAPNG에서 패킷 바이트를 읽습니다.
2. `imaging.py`가 연속 패킷을 고정 크기 2차원 grayscale 이미지로 만듭니다.
3. `wavelet.py`가 `coif1`, `db3`, `rbio1.3`의 LL sub-band를 3개 채널로 쌓고 `[0, 1]`로 정규화합니다.
4. `build_dataset.py`가 위 단계를 조합해 데이터 계약을 만족하는 NPZ와 metadata를 저장합니다.

범용 manifest 기반 빌드는 다음과 같이 실행합니다.

```bash
python -m src.preprocessing.build_dataset \
  --manifest data/raw/manifest.csv \
  --out data/processed/dataset.npz
```

manifest는 `pcap_path`, `label_name` 열이 필요합니다. 고정된 TOW-IDS train/test 파일을 처리할 때는 `scripts/build_tow_dataset.py`를 사용합니다.
