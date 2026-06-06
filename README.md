# Automotive Ethernet IDS

차량용 이더넷 네트워크에서 발생하는 사이버 공격을 탐지·분류하는 딥러닝 기반 침입 탐지 시스템(IDS)입니다.  
PCAP 패킷 데이터를 웨이블릿 이미지로 변환한 뒤, **2단계 게이트 파이프라인**으로 공격 유무와 유형을 분류합니다.

---

## 탐지 대상 클래스

| 레이블 | 이름 | 설명 |
|--------|------|------|
| 0 | Normal | 정상 트래픽 |
| 1 | F\_I | Flooding/Injection |
| 2 | P\_I | Packet Injection |
| 3 | M\_F | MAC Flooding |
| 4 | C\_D | Content Disruption |
| 5 | C\_R | Content Replay |

---

## 파이프라인 구조

```
입력 이미지
    │
    ▼
[Stage 1: CAE 이상 탐지]
    MSE ≤ τ  ──────────────→  Normal (종료)
    MSE > τ
    │
    ▼
[Stage 2: DCNN 6-class 분류]
    max_prob ≥ thr  ────────→  F_I / P_I / M_F / C_D / C_R
    max_prob < thr  ────────→  Unknown (미탐 제로데이)
```

| 단계 | 모델 | 역할 |
|------|------|------|
| Phase 1 | DCNN (2-class) | 이진 분류 베이스라인 (Normal vs Attack) |
| Phase 2 | DCNN (6-class) | 공격 유형 분류, 클래스 불균형 보정 |
| Phase 3 | CAE + TwoStagePipeline | 비지도 이상 탐지 게이트 + 엔드투엔드 파이프라인 |
| Phase 4 | LOAO 평가 | Leave-One-Attack-Out 제로데이 평가 (미구현) |

---

## 프로젝트 구조

```
.
├── configs/
│   ├── cae.yaml          # CAE 하이퍼파라미터
│   ├── experiment.yaml   # 데이터 경로, use_cae 스위치 등
│   ├── model.yaml        # DCNN 구조 설정
│   └── train.yaml        # 학습률, 배치 크기, seeds 등
├── data/
│   ├── raw/              # 원본 PCAP + 레이블 CSV (git 제외 — 별도 공유)
│   └── processed/        # dataset_v0.npz, split_manifest.json (git 제외 — 별도 공유)
├── experiments/
│   └── leave_one_out.py  # Phase 4 LOAO 평가 (stub)
├── results/
│   ├── checkpoints/      # cae_best.pth (git 제외 — 별도 공유)
│   ├── figures/          # 혼동 행렬, MSE 히스토그램, ROC 곡선
│   └── tables/           # CSV 결과 테이블, tau_values.json
├── scripts/
│   └── make_stub_dataset.py  # 실제 데이터 없이 스모크 테스트용 stub 생성
├── src/
│   ├── models/
│   │   ├── dcnn.py       # TOW-IDS DCNN (SepConv 기반)
│   │   └── cae.py        # Convolutional Autoencoder
│   ├── pipeline/
│   │   └── two_stage.py  # TwoStagePipeline (CAE + S2 연결)
│   ├── train/
│   │   ├── train_s1.py   # Phase 1: 이진 분류 학습
│   │   ├── train_s2.py   # Phase 2: 6-class 분류 학습
│   │   └── train_cae.py  # Phase 3: CAE 학습 + tau 계산
│   └── utils/
│       ├── io.py          # 데이터 로드/저장
│       ├── metrics.py     # 평가 지표
│       ├── seed.py        # 재현성 seed 고정
│       └── split.py       # 시간적 누수 방지 train/val/test 분할
├── requirements.txt
└── spec_phase0_to_3.docx  # 전체 설계 스펙 문서
```

---

## 환경 세팅

Python **3.12** 권장.

```bash
# 1. 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. 의존성 설치
pip install -r requirements.txt
```

---

## 데이터 준비

PCAP 및 전처리 데이터는 용량 문제로 git에 포함되지 않습니다.  
팀원에게 별도로 받은 뒤 아래 경로에 배치하세요.

```
data/
├── raw/
│   ├── Automotive_Ethernet_with_Attack_original_10_17_19_50_training.pcap
│   ├── Automotive_Ethernet_with_Attack_original_10_17_20_04_test.pcap
│   ├── y_train.csv
│   └── y_test.csv
└── processed/
    ├── dataset_v0.npz        # 전처리된 웨이블릿 이미지 데이터
    └── split_manifest.json   # 고정된 train/val/test 인덱스
```

> 실제 데이터 없이 파이프라인만 테스트하려면 아래 **스모크 테스트** 참고.

---

## 실행 방법

> 모든 명령어는 **프로젝트 루트**(`MS/`)에서 실행합니다.

### 0. 데이터 분할 (최초 1회)

실제 `dataset_v0.npz`가 준비됐다면 split_manifest를 먼저 생성합니다.

```bash
python -m src.utils.split
```

### 1. Phase 1 — 이진 분류 학습 (Normal vs Attack)

```bash
python -m src.train.train_s1
```

결과: `results/tables/s1_baseline.csv`

### 2. Phase 2 — 6-class 공격 분류 학습

```bash
python -m src.train.train_s2
```

결과: `results/tables/s2_summary.csv`, `s2_per_class.csv`, `results/figures/cm_s2_*.png`

### 3. Phase 3 — CAE 학습 및 파이프라인 구성

```bash
python -m src.train.train_cae
```

결과: `results/checkpoints/cae_best.pth`, `results/tables/tau_values.json`, `results/figures/mse_histogram.png`, `roc_cae.png`

---

## 스모크 테스트 (실제 데이터 없을 때)

실제 데이터 없이 코드가 정상 동작하는지 빠르게 확인합니다.

```bash
# 1. stub 데이터 생성 (N=200, 랜덤 이미지)
python scripts/make_stub_dataset.py

# 2. split manifest 생성
python -m src.utils.split

# 3. 각 학습 스크립트 1-epoch 테스트
python -m src.train.train_s1 --smoke
python -m src.train.train_s2 --smoke
python -m src.train.train_cae --smoke
```

---

## 주요 설정 파일

### `configs/experiment.yaml`

```yaml
experiment:
  npz_path: data/processed/dataset_v0.npz
  manifest_path: data/processed/split_manifest.json
  use_cae: false   # true → Stage 1 CAE 게이트 활성화 / false → S2 단독 모드
  conf_thr: 0.5    # Stage 2 Unknown 판단 임계값
  output_dir: results/
```

### `configs/train.yaml`

```yaml
train:
  lr: 1.0e-3
  batch_size: 32
  epochs: 100
  patience: 5
  seeds: [0, 1, 2, 3, 4]   # 5개 seed 평균으로 결과 리포트
```

### `configs/cae.yaml`

```yaml
cae:
  latent_dim: 128
  lr: 1.0e-3
  batch_size: 64
  epochs: 150
  patience: 12
  noise_std: 0.05   # 노이즈 제거 학습 (denoising CAE)
```

---

## 재현성

모든 학습은 seed를 고정해 재현 가능합니다.

- S1 / S2: `configs/train.yaml`의 `seeds` 리스트 전체를 순회하고 평균±표준편차로 리포트
- CAE: `seed=42` 고정 (한 번만 학습 후 모든 fold에서 재사용)

---

## 주의사항

- `data/processed/split_manifest.json`은 **절대 수정하지 마세요.**  
  S1·S2·S3가 동일한 `test_idx`를 공유해야 공정한 비교가 됩니다.
- 데이터 분할은 시간적 누수(temporal leakage) 방지를 위해 랜덤 셔플 없이 인덱스 순서 기반으로 수행됩니다.
