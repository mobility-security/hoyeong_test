# TOW-IDS Extension: 2-Stage Automotive Ethernet IDS

차량용 이더넷 네트워크에서 발생하는 사이버 공격을 탐지·분류하는 딥러닝 기반 2단계 침입 탐지 시스템입니다. [TOW-IDS (Han et al., 2023)](#references) 의 3-wavelet 이미징 파이프라인을 기반으로, CAE 이상 탐지 게이트와 제로데이 평가 프로토콜(LOAO)을 추가 구현했습니다.

> **현재 상태 (2026-06-15):** s2_recovery 라우팅 + fold-specific τ 재보정이 적용된
> 최신 full retraining + 5×5 LOAO 결과입니다. train **8,443** / val **941** /
> frozen test **6,185**, 6-class val 완전 커버. 하드웨어: Apple MPS.

**현재 결과 요약 (s2_recovery 라우팅, 64×64 v2)**

| 모델 | 주요 지표 | 현재 (s2_recovery) | 64×64 v1 (이전) | 32×32 (참고) |
|------|----------|--------------------|----------------|-------------|
| S1 (이진 DCNN) | Accuracy / F1 | **0.9791 / 0.9776** | 0.9768 / 0.9752 | 0.9636 / 0.9593 |
| S2 (6-class, Focal γ=3.0) | macro-F1 | **0.9343** | 0.8753 | 0.8500 |
| S3 (CAE + S2, s2_recovery) | Accuracy / macro-F1 | **0.9602 / 0.9345** | — | — |
| LOAO (5-fold × 5-seed) | 평균 zero-day 탐지율 | **23.6%** | 39.1% | 34.7% |

> 해상도별 상세 비교: [Section 9](#9-이미지-해상도-비교-실험-3232-vs-6464) · 상세 수치: [Section 8](#8-실험-결과-요약) · `results/tables/comparison_table.md`

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [전체 아키텍처 흐름](#2-전체-아키텍처-흐름)
3. [디렉터리 구조](#3-디렉터리-구조)
4. [환경 세팅](#4-환경-세팅)
5. [데이터 준비](#5-데이터-준비)
6. [전처리 실행](#6-전처리-실행)
7. [실험 실행 순서 (전체)](#7-실험-실행-순서-전체)
8. [현재 실험 결과 요약](#8-실험-결과-요약)
9. [이미지 해상도 비교 실험 (32×32 vs 64×64)](#9-이미지-해상도-비교-실험-3232-vs-6464)
10. [한계 및 향후 연구](#10-한계-및-향후-연구)
11. [주요 설정 파일](#11-주요-설정-파일)
12. [재현성](#12-재현성)
13. [References](#references)

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
| 데이터 계약 | 기본 NPZ 로드/저장 | dtype·shape·값범위·캡처/패킷 provenance 전수 검증 |
| 데이터 분할 | sample-level stratified | 다중 캡처 group split 또는 단일 캡처 시간 구간 split + guard gap |
| 공격 분류 손실 | CrossEntropy | Focal Loss (γ=2, 클래스 가중치) |
| 이상 탐지 게이트 | 없음 | CAE (Denoising) + τ = μ+2σ |
| 제로데이 평가 | 없음 | LOAO 5-fold × 5-seed 프로토콜 |
| 추론 배치화 | 샘플별 반복 | anomalous 배치 1회 GPU 추론 |
| 테스트 | 3개 | 74개 (io, provenance, 전처리, split, 모델, LOAO, 회귀 검증) |

---

## 2. 전체 아키텍처 흐름

```
PCAP 파일 (data/raw/)
    │
    │  scripts/build_tow_dataset.py
    ▼
패킷 파싱 (pcap_parser.py)
    → 128패킷 윈도우 × 128바이트 페이로드 → 128×128 grayscale 이미지
    → coif1 / db3 / rbio1.3 wavelet LL 채널 (level-1) → (N, 3, 64, 64) float32
    │
    ▼
dataset_train.npz / dataset_test.npz
    │
    │  src/utils/split.py  (capture 그룹 또는 시간 구간 train/val 분할)
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
    └─ [s2_recovery 모드] ──────────────────────────────────────────│
       │  anomalous = (CAE MSE > tau) OR (S2 예측 ≠ Normal)         │
       │                                                             │
       ├─ anomalous=False ──→ Normal (즉시 종료)                    │
       └─ anomalous=True  ──→ DCNN 6-class                         │
                               ├─ max_prob ≥ thr ──→ 공격 클래스     │
                               └─ max_prob < thr ──→ Unknown          │
                                                                        │
    [Phase 4] experiments/leave_one_out.py ◄───────────────────────────┘
    LOAO 5-fold × 5-seed: 공격 1종 제외 후 제로데이 탐지율 측정
    results/tables/loao_per_fold.csv
    results/tables/loao_summary.csv
```

---

## 3. 디렉터리 구조

각 주요 디렉터리에는 역할, 주요 파일, 입출력과 실행 방법을 설명하는 `README.md`가 있습니다.

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
├── docs/
│   └── presentation/      # 발표 구성과 리허설 체크리스트
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
│       ├── benchmark.py        # 동기화된 반복 추론 latency 측정
│       ├── config.py           # 실험 설정 + runtime 경로/threshold 해석
│       ├── focal_loss.py       # Focal Loss (Lin et al., 2017)
│       ├── io.py               # 데이터 계약: tensor + packet provenance 검증
│       ├── metrics.py          # compute_binary / multiclass / loao metrics
│       ├── seed.py             # 재현성 seed 고정
│       └── split.py            # capture 그룹/시간 구간 누수 안전 train/val 분할
│
├── tests/
│   ├── test_io.py              # 데이터 계약 검증 (17개)
│   ├── test_loao.py            # LOAO 하네스 + metrics 검증 (11개)
│   ├── test_preprocessing.py   # 전처리 파이프라인 (7개)
│   ├── test_split_pipeline.py  # split + 2단계 파이프라인 (13개)
│   ├── test_training_utils.py  # 모델·손실·EarlyStopping (5개)
│   ├── test_two_stage.py       # TwoStagePipeline 보정·라우팅
│   └── test_review_fixes.py    # smoke 격리, 지표, benchmark 회귀 검증
│
├── requirements.txt
├── requirements-dev.txt
├── requirements.lock.txt
├── CONTRIBUTING.md
├── PLAN.md
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
pip install -r requirements-dev.txt

# 4. 설치 확인
python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())"
```

운영 의존성만 필요하면 `requirements.txt`, 테스트까지 실행하려면
`requirements-dev.txt`를 사용합니다.

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

기본 명령은 합성 stub NPZ(`data/processed/dataset_v0.npz`)를 생성합니다. 전체 smoke
실행은 서로 다른 capture group을 가진 train/test stub을 `data/smoke/`에 만들고,
checkpoint와 표, 그림은 `results/smoke/`에만 기록합니다. 파이프라인 동작 확인용이며
모델 성능 평가에는 사용하지 마세요.

---

## 6. 전처리 실행

```bash
python scripts/build_tow_dataset.py
```

내부 처리 순서:

1. PCAP 파싱 → 128패킷 윈도우, 128바이트 페이로드 truncation/padding
2. 128×128 grayscale 이미지 생성
3. `coif1` / `db3` / `rbio1.3` 웨이블릿 LL 서브밴드 (level-1) → (N, 3, 64, 64) float32 텐서
4. `dataset_train.npz` / `dataset_test.npz` 저장
   (`pcap_id`, `packet_start`, `packet_end`, meta.json sidecar 포함)

처리 완료 후 예상 파일:

```
data/processed/
├── dataset_train.npz         # (N_train, 3, 64, 64)
├── dataset_train.meta.json   # wavelet, img_size, label_map 등
├── dataset_test.npz          # (N_test, 3, 64, 64)
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
  --guard-gap-packets 64 \
  --seed      42
```

산출물:
- `data/processed/split_manifest.json` — frozen train/val/test 인덱스 + sha256
- `data/processed/normal_only_idx.npy` — CAE 학습용 Normal 인덱스

여러 캡처가 있으면 `capture_group_stratified`, 단일 캡처면
`temporal_contiguous_block`을 사용합니다. 실제 TOW-IDS 기본 입력은 단일 캡처이므로
validation은 전체 시간축의 마지막 연속 블록이며, 경계에서 guard gap만큼 train
윈도우를 제거합니다. 이 데이터의 시간순 클래스 배치 특성상 validation에는 일부
클래스만 존재할 수 있으며, early stopping 지표는 실제 존재 클래스에 대해서만 계산합니다.

```bash
python -c "import json; m=json.load(open('data/processed/split_manifest.json')); print(m['split_strategy'], m['guard_removed_samples'])"
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
python -m src.train.train_cae          # 전체 (최대 200 epoch, patience=15)
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
python scripts/plot_unknown_case.py    # 실제 held-out Unknown case 4-panel
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
pytest tests/ -q
```

---

## 8. 실험 결과 요약

> **하드웨어**: Apple MPS · **라우팅**: `s2_recovery` (CAE MSE > τ **OR** S2 예측 ≠ Normal)
> **재현 조건**: `split_manifest.json` 고정, seeds=[0,1,2,3,4], val_macro_f1 기준 checkpoint 선택
> **CAE**: v2 — 내부 32×32 다운샘플 + detail energy 6ch, latent_dim=32, epochs=200, noise_std=0.10
> **분할**: train 8,443 / val 941 / frozen test 6,185, `class_temporal_tail` 전략으로 val 6-class 완전 커버

---

### S1 — 이진 분류 베이스라인 (Normal vs. Attack)

| Seed | Accuracy | F1 (binary) | FPR | FNR | AUC-ROC | Epochs |
|------|----------|-------------|-----|-----|---------|--------|
| 0 | 0.9827 | 0.9815 | 1.64% | 1.84% | 0.9969 | 20 |
| 1 | 0.9654 | 0.9631 | 3.55% | 3.36% | 0.9898 | 15 |
| 2 | 0.9880 | 0.9872 | 1.33% | 1.04% | 0.9969 | 24 |
| 3 | 0.9759 | 0.9742 | 2.15% | 2.70% | 0.9961 | 16 |
| 4 | 0.9833 | 0.9821 | 1.30% | 2.08% | 0.9950 | 19 |
| **mean** | **0.9791** | **0.9776** | **2.00%** | **2.20%** | **0.9949** | 18.8 |
| std | 0.0088 | 0.0094 | 0.93% | 0.88% | 0.0030 | 3.6 |

---

### S2 — 6-class Focal Loss DCNN (γ=3.0, M\_F weight ×1.5)

**5 seed 요약**

| Seed | Accuracy | macro-F1 | weighted-F1 | val_macro_F1 (best) |
|------|----------|----------|-------------|---------------------|
| 0 | 0.9496 | 0.9147 | 0.9506 | 0.9660 |
| 1 | 0.9601 | 0.9324 | 0.9603 | 0.9796 |
| 2 | 0.9673 | 0.9402 | 0.9672 | 0.9864 |
| 3 | 0.9659 | 0.9424 | 0.9663 | 0.9799 |
| 4 | 0.9664 | 0.9419 | 0.9668 | 0.9865 |
| **mean** | **0.9618** | **0.9343** | **0.9622** | **0.9797** |
| std | 0.0074 | 0.0117 | 0.0071 | 0.0083 |

**클래스별 성능 (5 seed 평균, frozen test set)**

| 클래스 | Support | Precision | Recall | F1 | Recall std | 비고 |
|--------|---------|-----------|--------|----|-----------|------|
| Normal | 3,297 | 0.9896 | 0.9812 | 0.9854 | 0.006 | 안정 |
| F\_I | 411 | 0.8819 | 0.9353 | 0.9073 | 0.015 | AVTP 주입 |
| P\_I | 596 | 0.9993 | **1.0000** | 0.9997 | 0.000 | 완벽 탐지 (단일 캡처) |
| M\_F | 397 | 0.7946 | 0.8766 | 0.8302 | 0.056 | ⚠ 낮은 Precision |
| C\_D | 678 | 0.9191 | 0.8622 | 0.8880 | 0.067 | seed 의존성 있음 |
| C\_R | 806 | 0.9970 | 0.9938 | 0.9954 | 0.010 | 안정 |

**seed별 per-class F1 (frozen test)**

| Seed | Normal | F\_I | P\_I | M\_F | C\_D | C\_R |
|------|--------|------|------|------|------|------|
| 0 | 0.9817 | 0.8822 | 0.9992 | 0.7924 | 0.8372 | 0.9956 |
| 1 | 0.9847 | 0.9214 | 1.0000 | 0.8216 | 0.8805 | 0.9862 |
| 2 | 0.9889 | 0.9238 | 1.0000 | 0.8241 | 0.9042 | 1.0000 |
| 3 | 0.9847 | 0.8976 | 1.0000 | 0.8617 | 0.9150 | 0.9957 |
| 4 | 0.9869 | 0.9117 | 0.9992 | 0.8510 | 0.9030 | 0.9994 |

---

### CAE — 이상 탐지 게이트 (Denoising CAE v2, seed=42)

> v2: 64×64 입력 → 내부 32×32 bilinear 다운샘플 + detail energy concat(6ch) → 4블록 인코더(32→64→128→256) → latent_dim=32 → MSE는 32×32 스케일에서 계산

| 파라미터 | 값 |
|----------|----|
| μ (val-Normal MSE 평균) | 0.001810 |
| σ (표준편차) | 0.000529 |
| **τ\_2σ (headline)** | **0.002868** |
| τ\_3σ | 0.003397 |
| p95 | 0.002891 |
| p99 | 0.003834 |
| 파라미터 수 | 845,443 |

> ⚠ CAE는 s2_recovery 라우팅에서 보조 게이트로 동작합니다. 공격 탐지의 주요 경로는
> S2 분류기이며, CAE는 제로데이(LOAO) 탐지에만 독립적인 기여를 합니다.
> CAE의 클래스별 실제 TPR은 M\_F(9.6%), C\_D(4.4%), C\_R(3.6%)로 낮으며
> P\_I(100%), F\_I(77.6%)에서만 유효합니다 — 단일 캡처로 인한 구조적 한계.

---

### LOAO — 제로데이 탐지 (5-fold × 5-seed, s2_recovery)

각 fold에서 공격 1종을 학습에서 제외 후 S3(s2_recovery, CAE+S2 5-class)로 Unknown 탐지율 측정.
CAE checkpoint는 전 fold 공유(τ=0.002868, fold-specific val-Normal 재보정), S2는 fold×seed마다 재학습 (γ=3.0, conf_thr=0.5).

**fold별 결과 (5-seed 평균)**

| Excluded | n\_test | CAE Recall | MSE ROC-AUC | Unknown Rate (mean±std) | Normal FPR | Known Acc | Known macro-F1 |
|----------|---------|-----------|-------------|------------------------|-----------|----------|----------------|
| F\_I | 411 | **77.62%** | 0.938 | 34.79% ± 12.27% | 1.42% | 92.41% | 89.63% |
| P\_I | 596 | **100.00%** | 0.987 | 55.60% ± 24.97% | 2.66% | 92.80% | 90.85% |
| M\_F | 397 | 9.57% | 0.623 | 13.85% ± 12.42% | 3.81% | 95.87% | 96.81% |
| C\_D | 678 | 4.42% | 0.599 | 1.62% ± 1.15% | 4.48% | 96.06% | 97.19% |
| C\_R | 806 | 3.60% | 0.577 | 12.06% ± 6.01% | 3.69% | 92.69% | 91.52% |
| **Grand Mean** | — | **39.04%** | **0.745** | **23.59% ± 21.57%** | **3.21%** | **93.97%** | **93.20%** |

**전체 25회 raw 결과 (fold × seed)**

| Fold | Seed | CAE Recall | Unknown Rate | Normal FPR | Known Acc | Known macro-F1 | conf\_thr |
|------|------|-----------|--------------|-----------|----------|----------------|---------|
| F\_I | 0 | 77.6% | 39.9% | 1.49% | 92.1% | 88.4% | 0.500 |
| F\_I | 1 | 77.6% | 34.1% | 1.94% | 92.7% | 90.1% | 0.500 |
| F\_I | 2 | 77.6% | 13.9% | 0.55% | 91.5% | 87.5% | 0.500 |
| F\_I | 3 | 77.6% | 42.3% | 1.70% | 93.0% | 90.6% | 0.500 |
| F\_I | 4 | 77.6% | 43.8% | 1.43% | 92.8% | 89.6% | 0.500 |
| P\_I | 0 | 100.0% | 19.3% | 2.97% | 93.4% | 92.5% | 0.500 |
| P\_I | 1 | 100.0% | 49.2% | 2.88% | 92.0% | 89.5% | 0.500 |
| P\_I | 2 | 100.0% | 70.5% | 3.12% | 91.7% | 88.5% | 0.500 |
| P\_I | 3 | 100.0% | 53.4% | 1.49% | 93.4% | 91.5% | 0.500 |
| P\_I | 4 | 100.0% | 85.7% | 2.82% | 93.4% | 92.3% | 0.500 |
| M\_F | 0 | 9.6% | 6.8% | 3.61% | 95.7% | 96.6% | 0.500 |
| M\_F | 1 | 9.6% | 33.8% | 3.73% | 95.4% | 95.8% | 0.500 |
| M\_F | 2 | 9.6% | 18.1% | 4.12% | 96.5% | 97.7% | 0.500 |
| M\_F | 3 | 9.6% | 7.1% | 3.64% | 96.6% | 97.7% | 0.500 |
| M\_F | 4 | 9.6% | 3.5% | 3.94% | 95.4% | 96.2% | 0.500 |
| C\_D | 0 | 4.4% | 2.8% | 4.22% | 95.6% | 97.0% | 0.500 |
| C\_D | 1 | 4.4% | 3.0% | 3.70% | 96.3% | 97.5% | 0.500 |
| C\_D | 2 | 4.4% | 0.6% | 6.43% | 96.5% | 97.4% | 0.500 |
| C\_D | 3 | 4.4% | 0.9% | 4.85% | 95.5% | 96.8% | 0.500 |
| C\_D | 4 | 4.4% | 0.9% | 3.22% | 96.4% | 97.3% | 0.500 |
| C\_R | 0 | 3.6% | 12.2% | 3.31% | 91.0% | 89.5% | 0.500 |
| C\_R | 1 | 3.6% | 20.5% | 2.94% | 93.2% | 91.9% | 0.500 |
| C\_R | 2 | 3.6% | 12.2% | 6.10% | 93.1% | 92.6% | 0.500 |
| C\_R | 3 | 3.6% | 3.5% | 4.06% | 93.4% | 91.7% | 0.500 |
| C\_R | 4 | 3.6% | 12.0% | 2.03% | 93.0% | 91.8% | 0.500 |

**해석**

- **F\_I (AVTP Frame Injection)**: CAE recall 77.6%, LOAO unknown_rate 34.8%(seed 2=13.9% 이상치). s2_recovery에서 known_macro_f1이 89.6%로 대폭 향상. AVTP 패킷의 wavelet 이미지가 정상과 명확히 구분되어 CAE+S2 모두 유효.
- **P\_I (PTP Sync Injection)**: CAE recall 100%. LOAO unknown_rate 55.6%, seed 0(19.3%)이 이상치 — S2가 P\_I를 기존 5-class 중 하나로 분류. known_macro_f1 90.9%로 안정. P\_I는 단일 캡처 패턴으로 MSE 분포 폭이 매우 좁음(std=0.00044).
- **M\_F (MAC Flooding)**: CAE recall 9.6%로 낮음. s2_recovery에서 S2가 M\_F 제로데이 샘플을 일부 Unknown으로 분류 → unknown_rate 13.9%. known_macro_f1은 96.8%로 매우 높음(5-class S2가 나머지 클래스를 잘 분류).
- **C\_D (CAN DoS)**: CAE recall 4.4%. C\_D MSE가 Normal과 겹쳐 CAE 탐지 불가. unknown_rate 1.6%로 낮음 — S2가 C\_D를 기존 클래스로 고confidence 분류. known_macro_f1 97.2%로 가장 높음.
- **C\_R (CAN Replay)**: CAE recall 3.6%. unknown_rate 12.1%(seed 3=3.5% 이상치). known_macro_f1 91.5%.
- **Normal FPR 3.21%**: strict_cascade(7.74%) 대비 크게 감소. s2_recovery에서 CAE FPR(7.67%)이 S2의 낮은 FPR(~2%)에 의해 보정됨.

---

### 모델별 종합 비교 (5 seed, frozen test set)

| 모델 | Accuracy | macro-F1 | weighted-F1 | Normal FPR | LOAO Unknown Rate | Coverage | 추론(ms/batch) |
|------|----------|----------|-------------|-----------|------------------|---------|--------------|
| S1 (이진 DCNN) | 0.9791 ± 0.009 | 0.9776 ± 0.009 | 0.9791 ± 0.009 | 2.00% ± 0.93% | — | 100% | 0.037 ± 0.002 |
| S2 (6-class Focal γ=3.0) | 0.9618 ± 0.007 | 0.9343 ± 0.012 | 0.9622 ± 0.007 | 1.88% ± 0.56% | — | 100% | 0.036 ± 0.001 |
| CAE v2 (단독) | — | — | — | 7.67% (val, τ\_2σ) | — | — | — |
| **S3 (CAE v2 + S2, s2\_recovery)** | **0.9602 ± 0.007** | **0.9345 ± 0.011** | **0.9620 ± 0.007** | **2.05% ± 0.66%** | **23.59%** | **99.69%** | **0.067 ± 0.002** |

**S3 선택적 지표 (Unknown 제외 서브셋)**

| 지표 | 값 |
|------|-----|
| Coverage (Non-Unknown 비율) | 99.69% ± 0.18% |
| Selective Accuracy | 96.32% ± 0.69% |
| Selective macro-F1 | 93.60% ± 1.09% |
| Unknown Rate (전체 test) | 0.31% ± 0.18% |
| LOAO Unknown Rate (grand mean) | **23.59%** |

**S3 클래스별 Recall (5 seed 평균, s2_recovery)**

| 클래스 | S2 단독 Recall | S3 Recall | 비고 |
|--------|--------------|----------|------|
| Normal (0) | 0.9812 ± 0.006 | 0.9795 ± 0.007 | 안정 |
| F\_I (1) | 0.9353 ± 0.015 | 0.9319 ± 0.014 | AVTP 주입 |
| P\_I (2) | **1.0000** ± 0.000 | **1.0000** ± 0.000 | 단일 캡처 패턴 |
| M\_F (3) | 0.8766 ± 0.056 | 0.8705 ± 0.057 | MAC Flooding |
| C\_D (4) | 0.8622 ± 0.067 | 0.8614 ± 0.067 | CAN DoS |
| C\_R (5) | 0.9938 ± 0.010 | 0.9938 ± 0.010 | CAN Replay |

**추론 지연 상세 (MPS, batch_size=64)**

| 모델 | mean (ms) | std (ms) | p50 (ms) | p95 (ms) | samples/sec |
|------|----------|----------|---------|---------|------------|
| S1 | 0.0369 | 0.0022 | 0.0295 | 0.0541 | 27,179 |
| S2 | 0.0357 | 0.0014 | 0.0298 | 0.0505 | 28,043 |
| S3 (CAE+S2, s2\_recovery) | 0.0672 | 0.0018 | 0.0671 | 0.0685 | 14,896 |

> 상세 수치: `results/tables/comparison_table.md` / `results/tables/comparison_table_detailed.csv`

**전체 실행 재현**

```bash
bash scripts/run.sh all --skip-preprocess   # checkpoint 있을 때
bash scripts/run.sh all                     # 전처리부터 전부
```

---

## 9. 이미지 해상도 비교 실험 (32×32 vs 64×64)

> **32×32**: 64-packet window, 64-byte payload → wavelet level-1 → (3, 32, 32) — train 18,809 / val 1,881 / test 12,369
> **64×64 v1**: 128-packet window, 128-byte payload → wavelet level-1 → (3, 64, 64), CAE 직접 64×64 학습
> **64×64 v2**: 동일 이미지, CAE 내부 32×32 다운샘플 + detail energy 6ch (현재) — train 8,463 / val 941 / test 6,185

### S1 — 이진 분류

| 지표 | 32×32 | 64×64 v2 (현재) |
|------|-------|----------------|
| Accuracy | 0.9636 ± 0.015 | **0.9797** ± 0.009 |
| F1 (binary) | 0.9593 ± 0.017 | **0.9782** ± 0.010 |
| FPR | 3.55% | **1.75%** |
| FNR | 3.75% | **2.36%** |
| AUC-ROC | 0.9928 | **0.9955** |

### S2 — 6-class Focal Loss

| 지표 | 32×32 (γ=2.0) | 64×64 v2 (γ=3.0, M\_F×1.5) |
|------|--------------|---------------------------|
| Accuracy | 0.9136 ± 0.011 | **0.9410** ± 0.022 |
| macro-F1 | 0.8500 ± 0.014 | **0.9019** ± 0.037 |
| weighted-F1 | 0.9168 ± 0.010 | **0.9414** ± 0.023 |

### CAE — 이상 탐지

| 지표 | 32×32 | 64×64 v1 (직접) | 64×64 v2 (내부 32×32) |
|------|-------|----------------|----------------------|
| ROC-AUC | 0.9944 | 0.9439 | **0.9950** |
| τ\_2σ | 0.005389 | 0.008393 | 0.002849 |
| Normal FPR (τ\_2σ) | 4.08% | 4.92% | **4.10%** |
| Attack TPR (τ\_2σ) | 99.82% | 33.70% | **99.76%** |
| 파라미터 수 | 713,475 | 2,292,483 | **845,443** |

> **v2 설계 요점**: 64×64 입력을 내부에서 32×32로 bilinear 다운샘플 후 detail energy 채널(3ch) concat → 6ch 입력. MSE는 32×32 스케일에서 계산해 τ 보정과 일관성 유지. 공간 복잡도 감소 + 이상 신호 강화로 v1 대비 Attack TPR을 33.7%→99.76%로 복원.

### LOAO — 제로데이 탐지 (5-fold × 5-seed)

| Fold | 32×32 CAE recall | 32×32 Unknown rate | 64×64 v1 CAE recall | 64×64 v1 Unknown rate | 64×64 v2 CAE recall | 64×64 v2 Unknown rate (s2\_recovery) |
|------|-----------------|-------------------|--------------------|-----------------------|--------------------|------------------------------------|
| F\_I | 66.7% | 63.2% | 67.2% | 62.4% | **77.6%** | 34.8% |
| P\_I | 99.8% | 61.5% | 40.9% | 58.9% | **100.0%** | 55.6% |
| M\_F | 3.5% | 29.7% | 5.3% | 22.5% | 9.6% | 13.9% |
| C\_D | 2.2% | 10.0% | 2.1% | 20.5% | 4.4% | 1.6% |
| C\_R | 2.6% | 8.9% | 1.0% | 31.2% | 3.6% | 12.1% |
| **Grand Mean** | 34.9% | 34.7% | 23.3% | 39.1% | **39.0%** | **23.6%** |

> **주의**: s2\_recovery 모드에서 LOAO unknown_rate가 이전 strict_cascade(76.5%)보다 낮아진 이유는,
> S2가 모든 샘플을 분류하면서 제로데이 샘플도 기존 5-class 중 하나로 높은 confidence로 분류하기 때문.
> 반면 known_macro_f1이 93.2%로 대폭 향상 — 제로데이 탐지 vs. 기존 공격 분류 간 트레이드오프.

### 해석

- **S1/S2**: 64×64가 모두 개선. 윈도우 하나당 더 많은 패킷(64→128)을 담아 트래픽 패턴이 풍부하게 인코딩됨. γ=3.0+M\_F weight 증가로 S2 macro-F1이 32×32(0.85) 대비도 대폭 향상(0.93).
- **CAE v1→v2**: 직접 64×64 학습(v1)은 파라미터 3배 증가 + 학습 샘플 절반 감소로 공격도 잘 복원해 Attack TPR 급락. v2는 내부 32×32 다운샘플 + detail energy 6ch로 이 문제를 해결.
- **s2\_recovery 라우팅**: S3 pipeline macro-F1이 0.480(strict)→**0.934**(recovery)로 향상. M\_F/C\_D/C\_R 탐지율이 3~8%→87~99%로 복원. Normal FPR은 2.05%로 낮게 유지.
- **LOAO unknown_rate 감소**: s2\_recovery에서 S2가 제로데이를 알려진 클래스로 흡수 → known_macro_f1 향상, unknown_rate 하락. 제로데이 탐지율과 known 분류율의 명시적 트레이드오프.

---

## 10. 한계 및 향후 연구

### 현재 한계

**CAE 클래스별 탐지 한계**
- CAE의 공격별 실제 탐지율 (frozen test, τ=0.002868):

  | 공격 | TPR@τ | MSE ROC-AUC | 원인 |
  |------|-------|-------------|------|
  | F\_I | 77.6% | 0.938 | AVTP 주입 — wavelet 복잡도 높아 탐지 가능 |
  | P\_I | **100.0%** | 0.987 | PTP Sync — 단일 캡처, 매우 일관된 고MSE |
  | M\_F | 9.6% | 0.623 | MAC Flooding — Normal MSE보다 낮음 (-23%) |
  | C\_D | 4.4% | 0.599 | CAN DoS — Normal MSE와 거의 동일 (-3%) |
  | C\_R | 3.6% | 0.577 | CAN Replay — Normal MSE보다 낮음 (-1%) |
  | **전체 평균** | **39.0%** | **0.745** | |

- M\_F / C\_D / C\_R은 CAN 캡슐화 트래픽이 wavelet 도메인에서 Normal보다 **단순**해 CAE 복원이 더 잘 됨. τ를 아무리 낮춰도 이 클래스는 탐지 불가 — 구조적 한계.
- **s2\_recovery** 라우팅으로 S2가 이 클래스들을 직접 분류해 보완 (M\_F 87%, C\_D 86%, C\_R 99% recall 달성).

**단일 캡처 기반 — 일반화 미검증 (M5)**
- train/test 데이터가 각각 PCAP 파일 1개에서 추출되었습니다 (pcap\_id 단일).
- 특히 P\_I: S2 confidence 최솟값 0.9997, CAE MSE std=0.00044로 모든 샘플이 거의 동일한 패턴 → 100% recall은 단일 캡처 패턴 암기에 가까움.
- 다른 차량, 다른 시간대, 다른 환경에서의 일반화 성능은 검증되지 않았습니다.
- 이는 **원본 TOW-IDS 데이터셋 자체가 단일 캡처**이기 때문에 발생하는 출처 제약입니다.
- 추가 캡처를 확보한다면 cross-capture 평가로 강화할 수 있으며, 현재로서는 이 한계를 명시합니다.

**이미지 해상도**
- 원본 TOW-IDS 논문: 452×452 이미지 / 현재 구현: 64×64 (128-packet × 128-byte window)
- 세밀한 공격 패턴 표현력은 고해상도에 미치지 못할 수 있으나 현재 데이터셋 규모에서는 충분

**LOAO Unknown Rate의 s2_recovery 트레이드오프**
- strict_cascade에서는 Unknown Rate가 높았으나(76.5%) known 분류 성능이 낮았음(macro-F1 49.0%)
- s2_recovery에서는 Unknown Rate가 낮아졌으나(23.6%) known 분류 성능이 대폭 향상(macro-F1 93.2%)
- 이는 설계 트레이드오프: 제로데이 Unknown 예측을 우선할지 vs. known 공격의 정확한 분류를 우선할지

**CAN 캡슐화 트래픽의 wavelet 유사성**
- C\_D / C\_R은 CAN 페이로드를 Ethernet 프레임에 캡슐화하여 전송
- wavelet 도메인에서 Normal과 겹치는 영역 발생 → CAE recall 4.4% / 3.6%로 낮음
- s2\_recovery에서 S2가 직접 분류하여 C\_D 86.1%, C\_R 99.4% recall로 보완

**P\_I 100% recall 신뢰성**
- P\_I는 단일 캡처 기반으로 모든 샘플이 거의 동일한 wavelet 이미지를 생성
- S2 confidence 최솟값 0.9997, CAE MSE std=0.00044 — 과적합이 아닌 데이터 다양성 부족
- 새로운 P\_I 공격 구현에 대한 일반화 미보장

### 향후 연구 방향

1. **LOAO 제로데이 Unknown Rate 향상**: conf_thr 상향(0.7~0.9) 조정으로 Unknown 범위 확장 — 탐지율 vs. FPR 트레이드오프
2. **C\_D/C\_R CAE 탐지**: 패킷 간격·CAN ID 분포 추가 특징, 또는 VAE/Flow 기반 이상 탐지기로 CAE 대체
3. **P\_I 일반화 검증**: 추가 P\_I 캡처 확보 후 cross-capture 평가
4. **다중 캡처 데이터 수집**: 단일 캡처 한계 극복을 위한 다양한 환경 PCAP 추가
5. **실시간 추론**: TorchScript/ONNX 변환 후 edge device 배포 (현재 14,896 samples/sec)

---

## 11. 주요 설정 파일

### `configs/experiment.yaml`

```yaml
experiment:
  train_npz_path: data/processed/dataset_train.npz
  test_npz_path:  data/processed/dataset_test.npz
  manifest_path:  data/processed/split_manifest.json
  use_cae: true        # true → CAE 게이트 활성화 / false → S2 단독
  routing_mode: s2_recovery  # strict_cascade | s2_recovery
  conf_thr: 0.5
  conf_thr_source: fixed   # fixed | artifact
  conf_thr_artifact: results/tables/conf_threshold.json
  loao_conf_thr_mode: validation
  target_known_reject_rate: 0.05
  benchmark_batch_size: 64
  benchmark_warmup: 2
  benchmark_repeats: 5
  output_dir: results/
```

### `configs/train.yaml`

```yaml
train:
  lr: 1.0e-3
  batch_size: 32
  epochs: 150
  patience: 10
  seeds: [0, 1, 2, 3, 4]
```

### `configs/cae.yaml`

```yaml
cae:
  latent_dim: 32        # 강한 정보 병목
  lr: 1.0e-3
  batch_size: 64
  epochs: 200
  patience: 15
  lr_patience: 5
  lr_factor: 0.5
  noise_std: 0.10       # 공격적 denoising
  cae_input_size: 32    # 내부 처리 해상도 (64×64 입력 → 32×32 다운샘플)
  use_detail_channels: true  # detail energy 6ch 입력
```

---

## 12. 재현성

모든 실험 결과는 아래 조건 하에 재현 가능합니다.

- **Seed**: S1/S2 `seeds: [0,1,2,3,4]`, CAE `seed=42` 고정
- **Split**: `split_manifest.json` — sha256 기록, 변경 금지
- **Checkpoint**: `val_macro_f1_best` 기준 자동 선택 (test 점수로 checkpoint 선택 금지)
- **환경**: 재현 실행에는 `requirements.lock.txt` 사용,
  deterministic algorithm과 고정 hash seed 적용

> ⚠ `split_manifest.json`을 교체하거나 `dataset_train.npz`를 재생성하면  
> S1·S2·CAE를 전부 재학습해야 split 일관성이 유지됩니다.

---

## 13. References

1. M. L. Han, B. I. Kwak, and H. K. Kim, "TOW-IDS: Intrusion Detection System Based on Three Overlapped Wavelets for Automotive Ethernet," *IEEE Trans. Inf. Forensics Secur.*, 2023. https://doi.org/10.1109/TIFS.2022.3221893

2. L. F. Marques da Luz, P. F. de Araujo-Filho, and D. R. Campelo, "Multi-stage Deep Learning-based Intrusion Detection System for Automotive Ethernet Networks," *Ad Hoc Networks*, 2024. https://doi.org/10.1016/j.adhoc.2024.103548

3. S. Jeong, H. K. Kim, M. L. Han, and B. I. Kwak, "AERO: Automotive Ethernet Real-Time Observer for Anomaly Detection in In-Vehicle Networks," *IEEE Trans. Ind. Informat.*, 2024. https://doi.org/10.1109/TII.2023.3324949

4. M. S. G. A. Leandro et al., "SeqWatch: Unsupervised Sequence-based Intrusion Detection System for Automotive Ethernet," *SBRC*, 2025. https://doi.org/10.5753/sbrc.2025.5949

5. T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal Loss for Dense Object Detection," *ICCV*, 2017. https://doi.org/10.1109/ICCV.2017.324

6. F. Chollet, "Xception: Deep Learning with Depthwise Separable Convolutions," *CVPR*, 2017. https://openaccess.thecvf.com/content_cvpr_2017/html/Chollet_Xception_Deep_Learning_CVPR_2017_paper.html

7. P. Vincent, H. Larochelle, Y. Bengio, and P.-A. Manzagol, "Extracting and Composing Robust Features with Denoising Autoencoders," *ICML*, 2008. https://doi.org/10.1145/1390156.1390294

8. D. Hendrycks and K. Gimpel, "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks," *ICLR*, 2017. https://openreview.net/forum?id=Hkg4TI9xl

9. S. Jeong et al., Automotive Ethernet IDS base CNN study, *Vehicular Communications*, 2021.

10. Alkhatib et al., autoencoder-based Automotive Ethernet binary IDS, arXiv:2202.00045, 2022.

11. Related open-set/two-stage IDS studies, arXiv:2408.08433 and arXiv:2403.04193.

---

## Contact

버그 제보·실행 문의: [GitHub Issues](https://github.com/kwonhoyeong/MS_test/issues)

## License

별도 오픈소스 라이선스 미적용. 사용·재배포 시 저장소 소유자에게 문의하세요.
