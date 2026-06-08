# Automotive Ethernet IDS

차량용 이더넷 네트워크에서 발생하는 사이버 공격을 탐지·분류하는 딥러닝 기반 침입 탐지 시스템(IDS)입니다.
PCAP에서 생성한 웨이블릿 이미지를 입력으로 받아 **2단계 게이트 파이프라인**으로 공격 유무와 유형을 분류합니다.

> **현재 상태:** Phase 0~3의 전처리, 모델 학습, 누수 안전 분할, 2단계 추론
> 파이프라인까지 구현되어 있습니다. 실제 TOW-IDS PCAP/CSV에서
> `dataset_train.npz`와 `dataset_test.npz`를 생성하는 코드가 연결되어 있으며,
> Phase 4 LOAO 평가는 의도적으로 시그니처만 존재합니다.

> **결과 해석 주의:** 현재 `split_manifest.json`은 `pcap_id` 그룹 단위로
> 재생성되었지만, 기존 figures/tables/checkpoints는 이전 split에서 생성된 항목이
> 포함될 수 있습니다. 최종 성능 보고 전에는 새 manifest 기준으로 S1, S2, CAE를
> 다시 학습해야 합니다.

---

## 기존 저장소 대비 최신화 내용

비교 기준은 `mobility-security/hoyeong`의 기존 `main` 브랜치입니다.

| 영역 | 기존 저장소 | 현재 프로젝트 |
|------|-------------|---------------|
| 데이터 준비 | `dataset_v0.npz` stub 중심 | 실제 PCAP/CSV 파싱, 이미징, 3-wavelet 변환 및 train/test NPZ 생성 |
| 데이터 계약 | 기본 NPZ 로드/저장 | dtype, shape, 값 범위, metadata, `pcap_id` 검증과 sidecar metadata 지원 |
| 데이터 분할 | sample-level stratified split | `pcap_id` capture group 단위 train/validation 분할, frozen test 전체 사용 |
| 누수 방지 | 동일 capture가 train/validation에 섞일 가능성 | 현재 manifest 기준 train/validation group 교집합 0 |
| 2단계 추론 | Stage-2를 샘플별 반복 호출 | anomalous batch를 한 번에 GPU/CPU 추론하고 공통 라우팅 로직 사용 |
| confidence 보정 | threshold마다 모델 재추론 | validation 추론 1회 결과를 모든 threshold 후보에 재사용 |
| 학습 공통 코드 | S1/S2별 DataLoader와 early stopping 중복 | `src/train/common.py`로 통합, best weights를 CPU에 보관 |
| 오류 처리 | 일부 `assert`, 암묵적 dtype 변환, 제한적 shape 검사 | 명시적 예외, 원본 dtype 검증, 모델·loss·checkpoint 입력 계약 강화 |
| 체크포인트 보안 | 일반 `torch.load()` | state-dict checkpoint를 `weights_only=True`로 로드 |
| 테스트 | `test_two_stage.py` 중심 3개 | I/O, 전처리, split, 모델, 학습 유틸리티 포함 45개 테스트 |

현재 누수 안전 manifest 크기는 train 17,058개, validation 1,750개,
frozen test 12,368개이며 train/validation `pcap_id` 교집합은 0입니다.

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
| 6 | Unknown | 낮은 confidence로 분류된 미지 공격 후보 (추론 전용) |

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
| Phase 4 | LOAO 평가 | Leave-One-Attack-Out 제로데이 평가 (stub) |

---

## 프로젝트 구조

```
.
├── configs/
│   ├── cae.yaml          # CAE 하이퍼파라미터
│   ├── experiment.yaml   # 데이터 경로, use_cae 스위치 등
│   ├── model.yaml        # DCNN 구조 설정
│   ├── preprocess.yaml   # PCAP 이미징·wavelet 전처리 설정
│   └── train.yaml        # 학습률, 배치 크기, seeds 등
├── data/
│   ├── raw/              # 원본 PCAP + 레이블 CSV (git 제외 — 별도 공유)
│   └── processed/        # dataset_v0.npz, split_manifest.json (git 제외 — 별도 공유)
├── experiments/
│   └── leave_one_out.py  # Phase 4 LOAO 평가 (stub)
├── results/
│   ├── checkpoints/      # cae_best.pth, s2_seed_*_best.pth (git 제외)
│   ├── figures/          # 혼동 행렬, MSE 히스토그램, ROC 곡선
│   └── tables/           # CSV 결과 테이블, tau_values.json
├── scripts/
│   ├── build_tow_dataset.py  # 실제 TOW-IDS train/test 데이터 생성
│   ├── make_stub_dataset.py  # 실제 데이터 없이 스모크 테스트용 stub 생성
│   └── make_timeline.py      # 프로젝트 타임라인 그림 생성
├── src/
│   ├── data/
│   │   └── split.py      # 기존 import 호환용 split re-export
│   ├── models/
│   │   ├── dcnn.py       # TOW-IDS DCNN (SepConv 기반)
│   │   └── cae.py        # Convolutional Autoencoder
│   ├── pipeline/
│   │   └── two_stage.py  # TwoStagePipeline (CAE + S2 연결)
│   ├── train/
│   │   ├── common.py     # DataLoader, device, early stopping 공통 코드
│   │   ├── train_s1.py   # Phase 1: 이진 분류 학습
│   │   ├── train_s2.py   # Phase 2: 6-class 분류 학습
│   │   └── train_cae.py  # Phase 3: CAE 학습 + tau 계산
│   ├── preprocessing/
│   │   ├── pcap_parser.py    # PCAP/PCAPNG 파싱
│   │   ├── imaging.py        # 패킷 window 이미지 생성
│   │   ├── wavelet.py        # 3-wavelet LL 채널 변환
│   │   └── build_dataset.py  # manifest 기반 전처리 오케스트레이터
│   └── utils/
│       ├── focal_loss.py  # S2 focal loss
│       ├── io.py          # 데이터 로드/저장
│       ├── metrics.py     # 평가 지표
│       ├── seed.py        # 재현성 seed 고정
│       └── split.py       # 시간적 누수 방지 train/val/test 분할
├── tests/
│   ├── test_io.py
│   ├── test_preprocessing.py
│   ├── test_split_pipeline.py
│   ├── test_training_utils.py
│   └── test_two_stage.py
├── requirements.txt
└── spec_phase0_to_3.docx  # 전체 설계 스펙 문서
```

---

## 데이터셋에서 최종 테스트까지

아래 순서는 **이 저장소를 처음 받은 사람이 원본 TOW-IDS 데이터셋만 가진 상태**에서
전처리, 학습, 검증셋 기반 모델 선택, frozen test 평가까지 진행하는 전체 절차입니다.
모든 명령은 프로젝트 루트에서 실행합니다.

### 전체 흐름

```text
원본 PCAP/CSV 4개
  → dataset_train.npz / dataset_test.npz 생성
  → pcap_id 기반 train/validation 분할
  → 코드 회귀 테스트와 1 epoch 스모크 실행
  → S2 공격 분류기 학습
  → CAE 이상 탐지기 학습 및 tau 산출
  → validation 점수로 S2 checkpoint와 confidence threshold 선택
  → dataset_test.npz로 CAE+S2 최종 파이프라인 평가
```

Phase 1(S1) 이진 분류는 성능 비교용 베이스라인이며 최종 CAE+S2 파이프라인의 필수 요소는
아닙니다. 최종 파이프라인을 재현하려면 S2와 CAE를 모두 학습해야 합니다.

### 1. 저장소와 Python 환경 준비

Python **3.10 이상**을 사용합니다. CUDA를 사용할 수 있는 PyTorch 환경을 권장하며,
CUDA가 없으면 CPU로 자동 실행되지만 전체 학습은 매우 느릴 수 있습니다.

```bash
git clone https://github.com/mobility-security/hoyeong_test.git
cd hoyeong_test

python3 -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch가 인식한 장치를 확인합니다.

```bash
python -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available())"
```

### 2. 원본 데이터 4개 배치

PCAP/CSV는 용량과 배포 제약 때문에 Git에 포함되지 않습니다. 보유한 데이터셋에서 아래
파일명을 확인한 뒤 **이름을 바꾸지 말고** `data/raw/`에 배치합니다.

```text
data/raw/
├── Automotive_Ethernet_with_Attack_original_10_17_19_50_training.pcap
├── Automotive_Ethernet_with_Attack_original_10_17_20_04_test.pcap
├── y_train.csv
└── y_test.csv
```

`y_train.csv`와 `y_test.csv`는 헤더가 없는 CSV이며, 세 번째 열의 multiclass 레이블이
`Normal`, `F_I`, `P_I`, `M_F`, `C_D`, `C_R` 중 하나여야 합니다. 각 CSV 행은 대응하는
PCAP 패킷과 같은 순서로 1:1 정렬되어 있어야 합니다.

### 3. PCAP/CSV를 학습용 NPZ로 전처리

```bash
python scripts/build_tow_dataset.py
```

기본 전처리는 64개 패킷을 64바이트씩 잘라 `64x64` 이미지로 만든 후,
`coif1`, `db3`, `rbio1.3` LL 채널을 사용해 `(N, 3, 32, 32)` 텐서를 생성합니다.
실행이 끝나면 다음 파일이 있어야 합니다.

```text
data/processed/
├── dataset_train.npz
├── dataset_train.meta.json
├── dataset_test.npz
└── dataset_test.meta.json
```

먼저 일부 패킷만 처리해 파일 구조를 확인하려면 아래 명령을 사용할 수 있습니다. 이 결과는
최종 학습용이 아니므로 확인 후 반드시 위의 전체 빌드 명령을 다시 실행합니다.

```bash
python scripts/build_tow_dataset.py --max-packets 200000
```

### 4. 생성된 NPZ 스키마 검증

`load_dataset()`은 dtype, shape, 값 범위, metadata와 `pcap_id`를 검증하며 문제가 있으면
즉시 예외를 발생시킵니다.

```bash
python - <<'PY'
from src.utils.io import load_dataset

for path in (
    "data/processed/dataset_train.npz",
    "data/processed/dataset_test.npz",
):
    X, y, meta, pcap_id = load_dataset(path, with_pcap_id=True)
    print(path)
    print("  X:", X.shape, X.dtype, f"range=({X.min():.4f}, {X.max():.4f})")
    print("  y:", y.shape, y.dtype, "labels=", sorted(set(y.tolist())))
    print("  pcap groups:", len(set(pcap_id.tolist())) if pcap_id is not None else None)
    print("  wavelets:", meta["wavelets"])
PY
```

기대 스키마는 다음과 같습니다.

- `X`: `float32`, `(N, 3, 32, 32)`, 값 범위 `[0, 1]`
- `y`: `int64`, `(N,)`, 레이블 `{0, 1, 2, 3, 4, 5}`
- `meta_json`: wavelet, 이미지 크기, label map 등의 JSON metadata
- `pcap_id`: `int64`, `(N,)`, capture block 단위 누수 방지 분할에 사용

### 5. Train/validation 분할 생성

`dataset_test.npz`는 전체를 frozen test로 보존하고, `dataset_train.npz`만 `pcap_id`
그룹 단위로 train/validation에 나눕니다.

```bash
python -m src.utils.split \
  --train-npz data/processed/dataset_train.npz \
  --test-npz data/processed/dataset_test.npz \
  --out data/processed/split_manifest.json \
  --val-ratio 0.10 \
  --seed 42
```

생성 결과:

- `data/processed/split_manifest.json`: 고정된 train/validation/frozen-test 인덱스와 hash
- `data/processed/normal_only_idx.npy`: CAE 학습에 쓰는 train-normal 인덱스

`split_manifest.json`의 `split_strategy`가 `pcap_group_stratified`인지 확인합니다.

```bash
python -c "import json; print(json.load(open('data/processed/split_manifest.json'))['split_strategy'])"
```

`pcap_id`가 없다는 오류가 나면 NPZ를 수정하거나 3단계의 전처리를 다시 실행하세요.
실제 실험에서 `--allow-unsafe-fallback`으로 이 검증을 우회하면 안 됩니다.

### 6. 코드 회귀 테스트

학습 전에 설치와 주요 코드 계약이 정상인지 확인합니다. `tests/`는 원본 데이터셋을
직접 읽지 않으므로 데이터 용량과 관계없이 실행할 수 있습니다.

```bash
pytest tests/ -q
```

정상 기준: **45 passed**

### 7. 1 epoch 스모크 실행(선택)

전체 학습 전에 데이터 로딩, GPU/CPU 전송, checkpoint 저장 경로를 1 epoch로 확인합니다.
`--smoke`도 `results/`에 파일을 저장하므로 **전체 학습 전에만** 실행하세요. 이후의 전체 학습이
스모크 결과를 덮어씁니다.

```bash
python -m src.train.train_s1 --smoke
python -m src.train.train_s2 --smoke
python -m src.train.train_cae --smoke
```

### 8. 전체 학습

먼저 `configs/train.yaml`, `configs/model.yaml`, `configs/cae.yaml`을 확인합니다. 최종 실험을
시작한 후에는 같은 실험의 설정과 `split_manifest.json`을 변경하지 않습니다.

```bash
# 선택: Normal vs Attack 이진 분류 베이스라인
python -m src.train.train_s1

# 필수: 6-class S2 분류기, 5개 seed의 best checkpoint 저장
python -m src.train.train_s2

# 필수: train-normal로 CAE 학습, validation-normal로 tau 산출
python -m src.train.train_cae
```

주요 산출물:

| 단계 | 산출물 |
|------|--------|
| S1 | `results/tables/s1_baseline.csv` |
| S2 | `results/checkpoints/s2_seed_<seed>_best.pth` |
| S2 | `results/tables/s2_summary_focal.csv`, `s2_per_class_focal.csv` |
| S2 | `results/figures/cm_s2_norm.png`, `cm_s2_raw.png` |
| CAE | `results/checkpoints/cae_best.pth` |
| CAE | `results/tables/tau_values.json`, `tau_sensitivity.csv` |
| CAE | `results/figures/mse_histogram.png`, `roc_cae.png` |

S1과 S2 학습 스크립트는 현재 구현상 seed별 학습이 끝날 때 frozen-test 지표를 자동으로
산출합니다. 이 지표는 보고에만 사용하고, S2 checkpoint는 반드시 test 점수가 아닌
`val_macro_f1_best`가 가장 높은 seed로 선택해야 합니다. test 점수를 보고 seed, 설정,
threshold를 다시 선택하면 데이터 누수가 됩니다.

### 9. CAE+S2 최종 frozen-test 평가

아래 명령은 S2 요약 테이블에서 validation macro-F1이 가장 높은 checkpoint를 자동 선택하고,
validation split만 사용해 confidence threshold를 보정한 뒤 `dataset_test.npz`를 배치 단위로
평가합니다. 최종 평가 전에 8단계의 S2와 CAE 전체 학습이 완료되어 있어야 합니다.

```bash
python - <<'PY'
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.pipeline.two_stage import TwoStagePipeline
from src.train.common import load_manifest
from src.utils.io import LABEL_NAMES, UNKNOWN_LABEL, load_dataset

cfg = OmegaConf.load("configs/experiment.yaml").experiment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train, y_train, _ = load_dataset(cfg.train_npz_path)
X_test, y_test, _ = load_dataset(cfg.test_npz_path)
manifest = load_manifest(cfg.manifest_path, len(X_train), len(X_test))

summary = pd.read_csv("results/tables/s2_summary_focal.csv")
seed_number = pd.to_numeric(summary["seed"], errors="coerce")
seed_rows = summary.loc[seed_number.notna()].copy()
if seed_rows.empty:
    raise RuntimeError("S2 seed checkpoint가 없습니다. train_s2를 먼저 실행하세요.")
best = seed_rows.loc[seed_rows["val_macro_f1_best"].astype(float).idxmax()]
s2_checkpoint = Path(best["checkpoint_path"])

pipeline = TwoStagePipeline.from_checkpoints(
    cae_ckpt_path="results/checkpoints/cae_best.pth",
    s2_model=s2_checkpoint,
    tau_json_path="results/tables/tau_values.json",
    conf_thr=float(cfg.conf_thr),
    use_cae=True,
    device=device,
)

val_idx = np.asarray(manifest["val_idx"], dtype=np.int64)
conf_thr = pipeline.calibrate_conf_thr(
    X_train[val_idx], y_train[val_idx], device=device, target_fpr=0.05
)

pred = []
batch_size = 256
for start in range(0, len(X_test), batch_size):
    xb = torch.from_numpy(X_test[start:start + batch_size]).to(device)
    pred.extend(pipeline.predict(xb)["class_id"])
pred = np.asarray(pred, dtype=np.int64)

accuracy = float(accuracy_score(y_test, pred))
unknown_rate = float(np.mean(pred == UNKNOWN_LABEL))
report = classification_report(
    y_test,
    pred,
    labels=list(range(len(LABEL_NAMES))),
    target_names=LABEL_NAMES,
    digits=4,
    zero_division=0,
)
cm = confusion_matrix(
    y_test, pred, labels=list(range(UNKNOWN_LABEL + 1))
)[:len(LABEL_NAMES), :]

out_dir = Path("results/tables")
out_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"y_true": y_test, "y_pred": pred}).to_csv(
    out_dir / "two_stage_test_predictions.csv", index=False
)
(out_dir / "two_stage_test_summary.json").write_text(
    json.dumps(
        {
            "s2_checkpoint": str(s2_checkpoint),
            "conf_thr": conf_thr,
            "accuracy": accuracy,
            "unknown_rate": unknown_rate,
        },
        indent=2,
    ),
    encoding="utf-8",
)

print("device:", device)
print("selected S2 checkpoint:", s2_checkpoint)
print("calibrated conf_thr:", conf_thr)
print("test accuracy:", accuracy)
print("unknown rate:", unknown_rate)
print(report)
print("confusion matrix rows=true 0..5, cols=pred 0..6(Unknown):\n", cm)
PY
```

최종 산출물:

- `results/tables/two_stage_test_predictions.csv`: 샘플별 정답/예측 클래스 ID
- `results/tables/two_stage_test_summary.json`: 선택된 checkpoint, confidence threshold, accuracy, Unknown 비율
- 터미널 출력: 6개 기지 클래스 분류 보고와 Unknown 열을 포함한 `6x7` confusion matrix

`dataset_test.npz`의 label은 지표 산출에만 사용하고 threshold 보정이나 checkpoint 선택에는
사용하지 않습니다. test 결과를 확인한 뒤 설정을 바꾸고 동일한 test set을 반복 평가하면
test set에 간접적으로 과적합됩니다.

### 문제 해결

| 증상 | 확인할 내용 |
|------|---------------|
| `FileNotFoundError` | 2단계의 파일명과 `data/raw/` 경로를 확인 |
| CSV 레이블 또는 패킷 수 불일치 | CSV 3번째 열과 PCAP 패킷이 1:1 정렬되었는지 확인 |
| `pcap_id is required` | stub/legacy NPZ가 아닌지 확인하고 `build_tow_dataset.py`로 재생성 |
| CUDA out of memory | `configs/train.yaml` 또는 `configs/cae.yaml`의 `batch_size` 축소 |
| checkpoint shape 오류 | 현재 NPZ와 같은 전처리/모델 설정으로 S2와 CAE를 재학습 |

---

## 주요 설정 파일

### `configs/experiment.yaml`

```yaml
experiment:
  train_npz_path: data/processed/dataset_train.npz
  test_npz_path: data/processed/dataset_test.npz
  manifest_path: data/processed/split_manifest.json
  use_cae: false   # true → Stage 1 CAE 게이트 활성화 / false → S2 단독 모드
  conf_thr: 0.5    # Stage 2 Unknown 판단 임계값
  output_dir: results/
```

위 `use_cae` 기본값은 S2-only 비교 실험용입니다. 9단계의 최종 CAE+S2 평가 예제는
`TwoStagePipeline.from_checkpoints(..., use_cae=True)`를 명시하므로 CAE 게이트를 활성화합니다.

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
- S1 / S2 / S3: 동일한 `split_manifest.json`의 train/validation과 별도 frozen test 사용
- S2: seed별 validation macro-F1 best checkpoint 저장
- CAE: `seed=42` 고정 (한 번만 학습 후 모든 fold에서 재사용)

---

## 주의사항

- 최종 실험을 시작한 뒤에는 `data/processed/split_manifest.json`을 수정하지 마세요.
  S1·S2·S3가 동일한 train/validation split과 frozen test를 공유해야 합니다.
- 실제 NPZ로 교체한 뒤에는 기존 stub manifest를 재사용하지 말고 실제 데이터 기준으로 새 manifest를 생성하세요.
- 데이터 분할은 `pcap_id` 기반 PCAP 블록 분할을 기본으로 사용합니다. sample-level
  fallback은 누수 위험이 있으므로 명시적 옵션 없이는 실행되지 않습니다.
- `configs/experiment.yaml`의 `use_cae`는 CAE 검증 결과에 따라 수동으로 결정합니다. `false`이면 S2-only 모드입니다.

## References

이 프로젝트의 구조와 평가 방법은 다음 연구를 참고했습니다.

1. M. L. Han, B. I. Kwak, and H. K. Kim, ["TOW-IDS: Intrusion Detection System Based on Three Overlapped Wavelets for Automotive Ethernet"](https://doi.org/10.1109/TIFS.2022.3221893), *IEEE Transactions on Information Forensics and Security*, 2023.
2. L. F. Marques da Luz, P. F. de Araujo-Filho, and D. R. Campelo, ["Multi-stage Deep Learning-based Intrusion Detection System for Automotive Ethernet Networks"](https://doi.org/10.1016/j.adhoc.2024.103548), *Ad Hoc Networks*, 2024.
3. S. Jeong, H. K. Kim, M. L. Han, and B. I. Kwak, ["AERO: Automotive Ethernet Real-Time Observer for Anomaly Detection in In-Vehicle Networks"](https://doi.org/10.1109/TII.2023.3324949), *IEEE Transactions on Industrial Informatics*, 2024.
4. M. S. G. A. Leandro et al., ["SeqWatch: Unsupervised Sequence-based Intrusion Detection System for Automotive Ethernet"](https://doi.org/10.5753/sbrc.2025.5949), *SBRC*, 2025.
5. F. Chollet, ["Xception: Deep Learning with Depthwise Separable Convolutions"](https://openaccess.thecvf.com/content_cvpr_2017/html/Chollet_Xception_Deep_Learning_CVPR_2017_paper.html), *CVPR*, 2017.
6. K. He et al., ["Deep Residual Learning for Image Recognition"](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html), *CVPR*, 2016.
7. P. Vincent et al., ["Extracting and Composing Robust Features with Denoising Autoencoders"](https://doi.org/10.1145/1390156.1390294), *ICML*, 2008.
8. D. Hendrycks and K. Gimpel, ["A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks"](https://openreview.net/forum?id=Hkg4TI9xl), *ICLR*, 2017.

## Contact

버그 제보, 실행 문의 및 개선 제안은
[GitHub Issues](https://github.com/mobility-security/hoyeong_test/issues)에 등록해 주세요.

## Acknowledgements

Automotive Ethernet IDS 연구 기반을 제공한 TOW-IDS 저자들과 데이터 전처리 및 평가를 함께 진행하는 프로젝트 팀원들에게 감사드립니다.

## License

현재 이 저장소에는 별도의 오픈소스 라이선스가 적용되어 있지 않습니다. 사용 또는 재배포가 필요한 경우 저장소 소유자에게 문의해 주세요.
