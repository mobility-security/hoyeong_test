# data

원본 네트워크 캡처와 전처리된 학습 데이터를 보관합니다.

| 디렉터리 | 역할 |
| --- | --- |
| `raw/` | 외부에서 받은 PCAP 및 패킷 단위 레이블 CSV를 둡니다. |
| `processed/` | wavelet 텐서, 메타데이터, split manifest 등 모델 입력을 둡니다. |

데이터 파일은 크기가 크거나 재배포 제한이 있을 수 있어 대부분 Git에서 제외됩니다. 데이터 형식의 간단한 버전 정보는 저장소 루트의 `DATASET.md`, 상세 계약은 `src/utils/io.py`를 기준으로 확인합니다.

전처리는 다음 명령으로 실행합니다.

```bash
python scripts/build_tow_dataset.py
python -m src.utils.split
```
