# raw

TOW-IDS 원본 PCAP과 패킷 단위 레이블 CSV를 배치하는 입력 디렉터리입니다. 이 디렉터리의 데이터는 전처리 전 원본으로 취급하며 직접 수정하지 않습니다.

## 예상 파일

```text
Automotive_Ethernet_with_Attack_original_10_17_19_50_training.pcap
Automotive_Ethernet_with_Attack_original_10_17_20_04_test.pcap
y_train.csv
y_test.csv
```

CSV는 헤더가 없고 세 번째 열에 `Normal`, `F_I`, `P_I`, `M_F`, `C_D`, `C_R` 중 하나가 있어야 합니다. CSV 행과 PCAP 패킷 순서는 1:1로 대응해야 합니다.

원본 파일은 Git에 커밋하지 않습니다. 데이터 준비 절차는 저장소 루트 `README.md`의 "데이터 준비" 절을 참고합니다.
