# TOW-IDS Extension: 2-Stage Automotive Ethernet IDS

차량용 이더넷 네트워크에서 발생하는 사이버 공격을 탐지·분류하는 딥러닝 기반 2단계 침입 탐지 시스템입니다. [TOW-IDS (Han et al., 2023)](#references) 의 3-wavelet 이미징 파이프라인을 기반으로, CAE 이상 탐지 게이트와 제로데이 평가 프로토콜(LOAO)을 추가 구현했습니다.

> **현재 상태:** Phase 0~4 전체 구현 완료.  
> PCAP → wavelet 이미지 → S1/S2/S3 학습 → LOAO 평가 → 시각화까지 단일 스크립트(`run.sh`)로 실행 가능합니다.

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [전체 아키텍처 흐름](#2-전체-아키텍처-흐름)
3. [디렉터리 구조](#3-디렉터리-구조)
4. [환경 세팅](#4-환경-세팅)
5. [데이터 준비](#5-데이터-준비)
6. [전처리 실행](#6-전처리-실행)
7. [실험 실행 순서 (전체)](#7-실험-실행-순서-전체)
8. [현재 실험 결과 요약](#8-현재-실험-결과-요약)
9. [한계 및 향후 연구](#9-한계-및-향후-연구)
10. [주요 설정 파일](#10-주요-설정-파일)
11. [재현성](#11-재현성)
12. [References](#references)

---

## 1. 프로젝트 개요

**TOW-IDS Extension**은 차량용 이더넷(Automotive Ethernet) 환경에서의 6종 사이버 공격을 탐지하는 2단계 IDS입니다.

| 탐지 목표 | 설명 |
|-----------|------|
| 이상 탐지 (Stage 1) | CAE 재구성 오차 기반으로 "정상/비정상" 이진 판정 |
| 유형 분류 (Stage 2) | 6-class DCNN으로 공격 유형 특정 |
| 미지 공격 탐지 (S3) | 낮은 confidence 샘플을 Unknown으로 출력 (제로데이 후보) |

**탐지 대상 클래스**

| 레이블 | 이름 | 설명 |
|--------|------|------|
| 0 | Normal | 정상 트래픽 |
| 1 | F\_I | Frame Injection (AVTP) |
| 2 | P\_I | PTP Sync Injection (gPTP) |
| 3 | M\_F | MAC Flooding |
| 4 | C\_D | CAN DoS |
| 5 | C\_R | CAN Replay |
| 6 | Unknown | 낮은 confidence 미지 공격 후보 (추론 전용) |

**기존 TOW-IDS 대비 주요 개선 사항**

| 영역 | 기존 | 이 프로젝트 |
|------|------|------------|
| 데이터 계약 | 기본 NPZ 로드/저장 | dtype·shape·값범위·pcap_id 전수 검증 |
| 데이터 분할 | sample-level stratified | pcap_id 캡처 그룹 단위 누수 안전 분할 |
| 공격 분류 손실 | CrossEntropy | Focal Loss (γ=2, 클래스 가중치) |
| 이상 탐지 게이트 | 없음 | CAE (Denoising) + τ = μ+2σ |
| 제로데이 평가 | 없음 | LOAO 5-fold × 5-seed 프로토콜 |
| 추론 배치화 | 샘플별 반복 | anomalous 배치 1회 GPU 추론 |
| 테스트 | 3개 | 59개 (io, 전처리, split, 모델, 학습, LOAO) |

---

## 2. 전체 아키텍처 흐름

```
PCAP 파일 (data/raw/)
    │
    │  scripts/build_tow_dataset.py
    ▼
패킷 파싱 (pcap_parser.py)
    → 64패킷 윈도우 × 64바이트 페이로드 → 64×64 grayscale 이미지
    → coif1 / db3 / rbio1.3 wavelet LL 채널 → (N, 3, 32, 32) float32
    │
    ▼
dataset_train.npz / dataset_test.npz
    │
    │  src/utils/split.py  (pcap_id 그룹 단위 train/val 분할)
    ▼
split_manifest.json  ──────────────────────────────────────────────────┐
    │                                                                   │
    │  [S1] src/train/train_s1.py                                      │
    ▼                                                                   │
DCNN (2-class) ── 이진 분류 베이스라인 (Normal vs Attack)              │
results/tables/s1_baseline.csv                                         │
    │                                                                   │
    │  [S2] src/train/train_s2.py                                      │
    ▼                                                                   │
DCNN (6-class) ── Focal Loss, 5 seed × early stopping                 │
results/checkpoints/s2_seed_*_best.pth                                │
    │                                                                   │
    │  [CAE] src/train/train_cae.py                                    │
    ▼                                                                   │
CAE (Denoising) ── Normal 전용 학습, τ = val_normal_μ + 2σ            │
results/checkpoints/cae_best.pth                                       │
results/tables/tau_values.json                                         │
    │                                                                   │
    │  [S3] 추론: src/pipeline/two_stage.py                            │
    ▼                                                                   │
입력 이미지                                                             │
    │                                                                   │
    ├─ CAE 재구성 ──→ MSE ≤ τ ──────────────────→ Normal (종료)       │
    │                                                                   │
    └─ MSE > τ ──→ DCNN 6-class 분류                                   │
                        │                                               │
                        ├─ max_prob ≥ thr ──→ F_I / P_I / M_F / C_D / C_R
                        │                                               │
                        └─ max_prob < thr ──→ Unknown (제로데이 후보)  │
                                                                        │
    [Phase 4] experiments/leave_one_out.py ◄───────────────────────────┘
    LOAO 5-fold × 5-seed: 공격 1종 제외 후 제로데이 탐지율 측정
    results/tables/loao_per_fold.csv
    results/tables/loao_summary.csv
```

---

## 3. 디렉터리 구조

```
.
├── configs/
│   ├── cae.yaml           # CAE 하이퍼파라미터 (latent_dim, lr, epochs, noise_std 등)
│   ├── experiment.yaml    # 데이터 경로, use_cae 스위치, conf_thr, output_dir
│   ├── model.yaml         # DCNN 구조 (num_classes, dropout, img_size)
│   ├── preprocess.yaml    # PCAP 이미징·wavelet 전처리 파라미터
│   └── train.yaml         # 학습률, 배치 크기, epochs, patience, seeds
│
├── data/
│   ├── raw/               # 원본 PCAP + 레이블 CSV (git 미포함, 별도 공유)
│   └── processed/         # 생성된 NPZ, manifest, normal_only_idx (git 미포함)
│
├── experiments/
│   └── leave_one_out.py   # Phase 4: LOAO 5-fold 제로데이 평가 하네스
│
├── results/
│   ├── checkpoints/       # cae_best.pth, s2_seed_{0..4}_best.pth (git 미포함)
│   ├── figures/           # CM PNG, MSE 히스토그램, ROC, LOAO bar chart 등
│   └── tables/            # CSV 결과 테이블, tau_values.json, comparison_table
│
├── scripts/
│   ├── build_tow_dataset.py    # 실제 TOW-IDS PCAP/CSV → train/test NPZ 생성
│   ├── comparison_table.py     # S1/S2/S3 3종 비교표 생성 (CSV + Markdown)
│   ├── make_stub_dataset.py    # 실제 데이터 없이 smoke 테스트용 stub NPZ 생성
│   ├── make_timeline.py        # 프로젝트 타임라인 그림 생성
│   ├── plot_confusion_matrix.py # S2 6-class + S3 7-class CM 확정본 PNG
│   ├── plot_loao_bar.py        # LOAO 탐지율 bar chart (공격별 CAE recall / Unknown rate)
│   ├── plot_unknown_case.py    # Unknown case 4-panel (입력·재구성·오차맵·MSE 히스토그램)
│   ├── run.sh                  # ← 전체 파이프라인 one-liner 실행 스크립트
│   └── run_loao.sh             # LOAO + 비교표 + 시각화 단독 실행
│
├── src/
│   ├── data/
│   │   └── split.py            # 하위호환용 split re-export
│   ├── models/
│   │   ├── cae.py              # Convolutional Autoencoder (input_shape 완전 파라미터화)
│   │   └── dcnn.py             # TOW-IDS DCNN (SepConv BlockA/B/C)
│   ├── pipeline/
│   │   └── two_stage.py        # TwoStagePipeline / TwoStageIDS (CAE gate + S2)
│   ├── preprocessing/
│   │   ├── build_dataset.py    # 전처리 오케스트레이터
│   │   ├── imaging.py          # 패킷 window → grayscale 이미지
│   │   ├── pcap_parser.py      # PCAP/PCAPNG 파싱
│   │   └── wavelet.py          # 3-wavelet LL 채널 변환
│   ├── train/
│   │   ├── common.py           # DataLoader, device, EarlyStopping 공통 코드
│   │   ├── train_cae.py        # CAE 학습 + tau 산출 + 시각화
│   │   ├── train_s1.py         # Phase 1: 이진 분류 베이스라인
│   │   └── train_s2.py         # Phase 2: 6-class Focal Loss 분류
│   └── utils/
│       ├── focal_loss.py       # Focal Loss (Lin et al., 2017)
│       ├── io.py               # 데이터 계약: save/load/validate_schema
│       ├── metrics.py          # compute_binary / multiclass / loao metrics
│       ├── seed.py             # 재현성 seed 고정
│       └── split.py            # pcap_id 그룹 단위 누수 안전 train/val 분할
│
├── tests/
│   ├── test_io.py              # 데이터 계약 검증 (17개)
│   ├── test_loao.py            # LOAO 하네스 + metrics 검증 (11개)
│   ├── test_preprocessing.py   # 전처리 파이프라인 (7개)
│   ├── test_split_pipeline.py  # split + 2단계 파이프라인 (13개)
│   ├── test_training_utils.py  # 모델·손실·EarlyStopping (5개)
│   └── test_two_stage.py       # TwoStagePipeline 보정·라우팅 (4개)
│
├── requirements.txt
├── DATASET.md
└── spec_phase0_to_3.docx       # 전체 설계 스펙 문서
```

---

## 4. 환경 세팅

Python **3.10 이상** 권장. CUDA 환경에서는 GPU 자동 사용, 없으면 Apple MPS → CPU 순으로 폴백합니다.

```bash
# 1. 저장소 클론
git clone https://github.com/kwonhoyeong/MS_test.git
cd MS_test

# 2. 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt

# 4. 설치 확인
python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())"
```

**pytest 별도 설치** (requirements.txt 미포함):

```bash
pip install pytest
```

---

## 5. 데이터 준비

### 5-A. 실제 TOW-IDS 데이터 사용

PCAP/CSV 파일을 `data/raw/`에 배치합니다 (파일명 변경 금지).

```
data/raw/
├── Automotive_Ethernet_with_Attack_original_10_17_19_50_training.pcap
├── Automotive_Ethernet_with_Attack_original_10_17_20_04_test.pcap
├── y_train.csv
└── y_test.csv
```

- `y_train.csv` / `y_test.csv`: 헤더 없는 CSV, 3번째 열 = multiclass 레이블 (`Normal`, `F_I`, `P_I`, `M_F`, `C_D`, `C_R`)
- 각 CSV 행 ↔ PCAP 패킷이 1:1 정렬되어 있어야 합니다.

### 5-B. Stub 데이터로 smoke 테스트 (실제 데이터 없을 때)

```bash
python scripts/make_stub_dataset.py
```

N=200 랜덤 샘플 stub NPZ(`data/processed/dataset_v0.npz`)를 생성합니다. 파이프라인 동작 확인용이며 모델 학습에는 사용하지 마세요.

---

## 6. 전처리 실행

```bash
python scripts/build_tow_dataset.py
```

내부 처리 순서:

1. PCAP 파싱 → 64패킷 윈도우, 64바이트 페이로드 truncation/padding
2. 64×64 grayscale 이미지 생성
3. `coif1` / `db3` / `rbio1.3` 웨이블릿 LL 서브밴드 → (N, 3, 32, 32) float32 텐서
4. `dataset_train.npz` / `dataset_test.npz` 저장 (meta.json sidecar 포함)

처리 완료 후 예상 파일:

```
data/processed/
├── dataset_train.npz         # (N_train, 3, 32, 32)
├── dataset_train.meta.json   # wavelet, img_size, label_map 등
├── dataset_test.npz          # (N_test, 3, 32, 32)
└── dataset_test.meta.json
```

부분 처리로 구조 먼저 확인하려면:

```bash
python scripts/build_tow_dataset.py --max-packets 200000
```

---

## 7. 실험 실행 순서 (전체)

### 빠른 실행 — one-liner

```bash
# 전체 파이프라인 (전처리 → 학습 → LOAO → 시각화)
bash scripts/run.sh all

# 빠른 smoke test (각 단계 1 epoch, seed 0)
bash scripts/run.sh smoke
```

### 단계별 실행

#### Step 1. Train/Val 분할 생성

```bash
python -m src.utils.split \
  --train-npz data/processed/dataset_train.npz \
  --test-npz  data/processed/dataset_test.npz \
  --out       data/processed/split_manifest.json \
  --val-ratio 0.10 \
  --seed      42
```

산출물:
- `data/processed/split_manifest.json` — frozen train/val/test 인덱스 + sha256
- `data/processed/normal_only_idx.npy` — CAE 학습용 Normal 인덱스

`split_strategy`가 `pcap_group_stratified`인지 확인:

```bash
python -c "import json; print(json.load(open('data/processed/split_manifest.json'))['split_strategy'])"
```

#### Step 2. S1 — 이진 분류 베이스라인 학습

```bash
python -m src.train.train_s1          # 전체 (100 epoch × 5 seed)
python -m src.train.train_s1 --smoke  # smoke (1 epoch, seed 0)
```

산출물: `results/tables/s1_baseline.csv`

#### Step 3. S2 — 6-class Focal Loss 분류 학습

```bash
python -m src.train.train_s2          # 전체
python -m src.train.train_s2 --smoke  # smoke
```

산출물:
- `results/checkpoints/s2_seed_{0..4}_best.pth`
- `results/tables/s2_summary_focal.csv`, `s2_per_class_focal.csv`
- `results/figures/cm_s2_norm.png`, `cm_s2_raw.png`

#### Step 4. CAE 학습

```bash
python -m src.train.train_cae          # 전체 (150 epoch)
python -m src.train.train_cae --smoke  # smoke
```

산출물:
- `results/checkpoints/cae_best.pth`
- `results/tables/tau_values.json`, `tau_sensitivity.csv`
- `results/figures/mse_histogram.png`, `roc_cae.png`

#### Step 5. LOAO 제로데이 평가

```bash
bash scripts/run_loao.sh          # LOAO + 비교표 + 시각화 전체
bash scripts/run_loao.sh --smoke  # smoke (F_I fold, seed 0, 1 epoch)

# 또는 직접:
python -m experiments.leave_one_out          # 5 fold × 5 seed
python -m experiments.leave_one_out --smoke  # smoke
```

산출물:
- `results/tables/loao_per_fold.csv` — fold×seed 전체 raw 결과
- `results/tables/loao_summary.csv` — 5-fold mean±std

#### Step 6. 시각화

```bash
python scripts/plot_loao_bar.py        # LOAO 탐지율 bar chart
python scripts/plot_unknown_case.py    # Unknown case 4-panel
python scripts/plot_confusion_matrix.py  # S2 6-class + S3 7-class CM
python scripts/comparison_table.py    # S1/S2/S3 3종 비교표
```

산출물:
- `results/figures/loao_bar_chart.png`
- `results/figures/unknown_case_4panel.png`
- `results/figures/cm_s2_6class.png`, `cm_s3_7class.png`
- `results/tables/comparison_table.csv`, `comparison_table.md`

#### 코드 회귀 테스트

```bash
pytest tests/ -q   # 59 passed 기대
```

---

## 8. 현재 실험 결과 요약

> 데이터셋: TOW-IDS PCAP (train 18,808 / test 12,368 샘플, pcap_id 그룹 분할)  
> 하드웨어: Apple M-series MPS

### S1 — 이진 분류 베이스라인 (5 seed 평균)

| 지표 | mean | std |
|------|------|-----|
| Accuracy | **0.893** | 0.031 |
| F1 (binary) | **0.868** | 0.041 |
| FPR (Normal) | 0.025 | 0.009 |
| AUC-ROC | 0.912 | 0.042 |

### S2 — 6-class Focal Loss DCNN (5 seed 평균)

| 지표 | mean | std |
|------|------|-----|
| Accuracy | 0.919 | 0.033 |
| macro-F1 | **0.891** | 0.056 |
| weighted-F1 | 0.913 | 0.048 |

**Per-class recall (5 seed 평균)**

| 클래스 | Recall | 비고 |
|--------|--------|------|
| Normal | 0.934 | |
| F\_I   | 0.846 | Frame Injection |
| P\_I   | **0.998** | PTP Sync |
| M\_F   | 0.986 | MAC Flooding |
| C\_D   | **0.710** | ⚠ 최저 — std 큼 |
| C\_R   | **0.966** | CAN Replay |

### CAE — 이상 탐지 게이트 (seed=42 고정)

| 지표 | 값 | 비고 |
|------|-----|------|
| ROC-AUC (val) | **0.940** | binary: Normal vs 전체 공격 |
| Normal FPR (τ\_2σ) | **4.09%** | headline threshold |
| Attack TPR (τ\_2σ) | 63.2% | |
| τ\_2σ | 0.003378 | μ + 2σ of val-normal MSE |

### LOAO — 제로데이 탐지 (smoke: F\_I fold, seed 0, 1 epoch)

| 지표 | 값 | 설명 |
|------|-----|------|
| CAE anomaly recall | 0.260 | P(MSE > τ \| F\_I test) |
| End-to-end Unknown rate | 0.121 | P(Unknown \| F\_I test) |
| Normal FPR | 0.097 | P(MSE > τ \| Normal test) |

> ⚠ 위 LOAO 값은 1 epoch smoke 결과입니다. 전체 실험(`bash scripts/run.sh loao`) 후 갱신 필요.

---

## 9. 한계 및 향후 연구

### 현재 한계

**이미지 해상도 차이**
- 원본 TOW-IDS 논문: 452×452 이미지 / 이 구현: 32×32
- 64패킷 윈도우 × 64바이트 페이로드 → 64×64 grayscale, 웨이블릿 후 32×32로 축소
- 고해상도 대비 세밀한 공격 패턴 표현력 손실 가능

**C\_D 분류 불안정**
- C\_D (CAN DoS) recall std가 타 클래스 대비 매우 큼 (seed 의존성 높음)
- 원인: CAN 패킷을 Automotive Ethernet으로 캡슐화한 트래픽의 wavelet 이미지가 일부 정상 트래픽 패턴과 유사

**CAN 캡슐화 트래픽의 wavelet 유사성**
- C\_D / C\_R은 CAN 페이로드를 Ethernet 프레임에 캡슐화하여 전송
- 동일 페이로드 구조가 wavelet 도메인에서 Normal과 겹치는 영역 발생
- 더 긴 윈도우 또는 추가 특징 채널로 개선 가능

**LOAO 제로데이 탐지율**
- 1 epoch smoke 기준 CAE recall 26%, Unknown rate 12%로 낮음
- 전체 학습 및 τ 재보정 후 재측정 필요

### 향후 연구 방향

1. **해상도 확장**: 64×64 또는 128×128 이미지 직접 사용 (DCNN 입력 크기 조정)
2. **C\_D 개선**: C\_D에 특화된 특징(패킷 간격, CAN ID 분포) 추가
3. **τ 재보정**: LOAO fold별 val-normal로 fold-specific τ 계산
4. **비지도 이상 탐지 강화**: VAE 또는 Flow 기반 이상 탐지기로 CAE 대체
5. **실시간 추론**: TorchScript/ONNX 변환 후 edge device 배포

---

## 10. 주요 설정 파일

### `configs/experiment.yaml`

```yaml
experiment:
  train_npz_path: data/processed/dataset_train.npz
  test_npz_path:  data/processed/dataset_test.npz
  manifest_path:  data/processed/split_manifest.json
  use_cae: false    # true → CAE 게이트 활성화 / false → S2 단독
  conf_thr: 0.5     # Stage 2 Unknown 판단 임계값
  output_dir: results/
```

### `configs/train.yaml`

```yaml
train:
  lr: 1.0e-3
  batch_size: 32
  epochs: 100
  patience: 5
  seeds: [0, 1, 2, 3, 4]
```

### `configs/cae.yaml`

```yaml
cae:
  latent_dim: 128
  lr: 1.0e-3
  batch_size: 64
  epochs: 150
  patience: 12
  lr_patience: 5
  lr_factor: 0.5
  noise_std: 0.05   # Denoising CAE: 학습 시 입력에 Gaussian 노이즈 추가
```

---

## 11. 재현성

모든 실험 결과는 아래 조건 하에 재현 가능합니다.

- **Seed**: S1/S2 `seeds: [0,1,2,3,4]`, CAE `seed=42` 고정
- **Split**: `split_manifest.json` — sha256 기록, 변경 금지
- **Checkpoint**: `val_macro_f1_best` 기준 자동 선택 (test 점수로 checkpoint 선택 금지)
- **환경**: `requirements.txt` 고정 버전, `torch.backends.cudnn.deterministic=True`

> ⚠ `split_manifest.json`을 교체하거나 `dataset_train.npz`를 재생성하면  
> S1·S2·CAE를 전부 재학습해야 split 일관성이 유지됩니다.

---

## References

1. M. L. Han, B. I. Kwak, and H. K. Kim, "TOW-IDS: Intrusion Detection System Based on Three Overlapped Wavelets for Automotive Ethernet," *IEEE Trans. Inf. Forensics Secur.*, 2023. https://doi.org/10.1109/TIFS.2022.3221893

2. L. F. Marques da Luz, P. F. de Araujo-Filho, and D. R. Campelo, "Multi-stage Deep Learning-based Intrusion Detection System for Automotive Ethernet Networks," *Ad Hoc Networks*, 2024. https://doi.org/10.1016/j.adhoc.2024.103548

3. S. Jeong, H. K. Kim, M. L. Han, and B. I. Kwak, "AERO: Automotive Ethernet Real-Time Observer for Anomaly Detection in In-Vehicle Networks," *IEEE Trans. Ind. Informat.*, 2024. https://doi.org/10.1109/TII.2023.3324949

4. M. S. G. A. Leandro et al., "SeqWatch: Unsupervised Sequence-based Intrusion Detection System for Automotive Ethernet," *SBRC*, 2025. https://doi.org/10.5753/sbrc.2025.5949

5. T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal Loss for Dense Object Detection," *ICCV*, 2017. https://doi.org/10.1109/ICCV.2017.324

6. F. Chollet, "Xception: Deep Learning with Depthwise Separable Convolutions," *CVPR*, 2017. https://openaccess.thecvf.com/content_cvpr_2017/html/Chollet_Xception_Deep_Learning_CVPR_2017_paper.html

7. P. Vincent, H. Larochelle, Y. Bengio, and P.-A. Manzagol, "Extracting and Composing Robust Features with Denoising Autoencoders," *ICML*, 2008. https://doi.org/10.1145/1390156.1390294

8. D. Hendrycks and K. Gimpel, "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks," *ICLR*, 2017. https://openreview.net/forum?id=Hkg4TI9xl

---

## Contact

버그 제보·실행 문의: [GitHub Issues](https://github.com/kwonhoyeong/MS_test/issues)

## License

별도 오픈소스 라이선스 미적용. 사용·재배포 시 저장소 소유자에게 문의하세요.
