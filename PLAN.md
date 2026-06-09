# TOW-IDS 확장: 2단계 IDS 파이프라인 (CAE Zero-day + 6-class) 구현 계획서

## 프로젝트 한눈에 보기

- **목표 (핵심 기여)**: **supervised multiclass 공격유형 분류와 reconstruction-error(MSE) 기반 zero-day 탐지를 단일 2단계 파이프라인으로 통합**하고, **leave-one-attack-out(LOAO)로 TOW-IDS 데이터셋에서 zero-day 일반화를 정량화**한다. 구체적으로 TOW-IDS(이진 분류)를 확장하여 (1) 어떤 공격인지 분류하는 멀티클래스 분류기(Stage-2)와 (2) 학습에 없던 Zero-day 공격을 탐지하는 CAE 이상탐지 게이트(Stage-1)를 결합한다. 최종 출력은 7클래스(Normal + 5공격 + Unknown).
- **핵심 산출물**: 3종 비교표(베이스라인 binary vs 6-class vs 2단계), 6-class Confusion Matrix, Leave-one-attack-out Zero-day 탐지율 그래프, Unknown Attack 탐지 사례 시각화, 발표 슬라이드.
- **성공 기준**: 베이스라인 재현 Acc/F1 ≥ 0.985(목표 ~0.99 = 논문 0.9965/0.9974), 6-class macro-F1 ≥ 0.95, LOAO 5-fold 평균 Zero-day 탐지율 ≥ 0.80(목표 ≥ 0.90), Normal FPR ≤ 5%.
- **기간/팀**: 13일, 4인(A 전처리 / B 분류기 / C Autoencoder / D 실험·발표).

---

## 차별점 & 관련연구 (Novelty & Related Work)

> **방어 가능한 novelty (헤드라인 — 발표·보고서에 그대로 사용)**:
> *"우리가 아는 한, supervised multiclass 공격유형 분류와 reconstruction-error 기반 zero-day/unknown 탐지를 **단일 2단계 파이프라인으로 통합**하고, **leave-one-attack-out 프로토콜로 TOW-IDS에서 zero-day 일반화를 정량화한 첫 시도**이다."*

⚠️ **"TOW-IDS에서 multiclass 최초" 또는 "zero-day 최초"라고 주장하지 말 것.** 두 축은 이미 선행연구가 점유했다(아래 표). 기여는 **통합(unification) + LOAO 정량화**로 좁혀야 방어된다.

| 연구 | Multiclass | Zero-day/Unknown | 차별 포인트 |
|---|---|---|---|
| **TOW-IDS** (Han/Kwak/Kim, IEEE TIFS 2023) | ❌ binary | ❌ | 베이스 모델·데이터셋. 우리가 마지막 레이어 Sigmoid→Softmax로 확장 |
| **da Luz et al.** (Ad Hoc Networks 2024) | ✅ (RF gate + Pruned CNN) | ❌ supervised-only | TOW-IDS에서 multiclass 이미 수행 → "unknown 못 잡음"으로 차별. 공식 코드 = `luigiluz/automotive-ids-evaluation-framework` (MIT) |
| **AERO** (Jeong et al., IEEE TII 2024) | ❌ | ✅ unsupervised anomaly | TOW-IDS 저자 동일그룹. "공격유형 분류 없음"으로 차별 |
| **SeqWatch** (UFPE 그룹, SBRC 2025) | ❌ | ✅ unsupervised zero-day | da Luz 동일그룹의 zero-day 연구. **같은 그룹조차 multiclass·zero-day를 별도 파이프라인으로 가짐** → 우리의 통합 간극을 정확히 드러냄 |
| **본 프로젝트 (2단계 파이프라인)** | ✅ 6-class | ✅ Unknown (CAE) | **통합 + LOAO 정량화 = 유일한 차별점** |

**반드시 인용·차별화할 must-cite 세트**: ① TOW-IDS (TIFS 2023, base) ② da Luz (Ad Hoc Networks 2024, multiclass 베이스라인) ③ AERO (TII 2024, anomaly-only) ④ **SeqWatch (SBRC 2025, zero-day-only — 누락 시 novelty 붕괴)** ⑤ Jeong et al. (Vehicular Communications 2021, AEID base CNN) ⑥ Alkhatib et al. (arXiv 2202.00045, AE 기반 binary). 추가로 open-set/2단계 IDS 선행(arXiv 2408.08433, 2403.04193)을 인지하고 있음을 명시 — "아키텍처 패턴 자체는 추상적으로 새롭지 않으나, TOW-IDS에서의 통합·정량화가 기여"라고 정직하게 한정한다.

> ⚠️ **검증 필요 (confidence medium)**: SeqWatch가 실제로 TOW-IDS 데이터셋을 사용하는지는 미확정("two public datasets"로만 확인). novelty 문장 확정 전 SeqWatch(SBRC 2025) 원문으로 데이터셋을 직접 확인할 것 (Day 1 작업, R15).

---

## 전체 아키텍처 / 데이터 흐름

```
[PCAP + CSV labels]
      │  (Team A: src/preprocessing)
      ▼
패킷 payload bytes(0-255) ─► 고정길이 패딩/절단 ─► N개 패킷을 NxN 이미지로 ─► /255 정규화([0,1])
      │
      ▼
3-wavelet 2D-DWT(coif1 / db3 / rbio1.3), 각 LL(approximation) sub-band만 유지 ─► np.stack(axis=-1)
      │
      ▼
  ┌──────────────────────────────────────────────┐
  │  CONTRACT: dataset.npz  X(N,3,H,W) f32 [0,1]  │  ◄── 팀 간 단일 인터페이스
  │           y(N,) int64 {0..5} + meta.json      │
  └──────────────────────────────────────────────┘
      │                              │
      │ Normal-only subset           │ 전체(라벨 포함)
      ▼                              ▼
[Stage-1 CAE (Team C)]        [Stage-2 6-class Softmax (Team B)]
정상만 학습, MSE 재구성오차    SeparableConv2D DCNN, 마지막 레이어
tau = mu + 2*sigma            Sigmoid→Softmax(6)
      │
      ▼  (Team C: src/pipeline/two_stage.py — gated cascade)
  MSE(x) <= tau ? ──► Normal (종료)
        │ else
        ▼
  Stage-2 softmax ─► max_prob < conf_thr ? ──► Unknown
                          │ else
                          ▼
                   5공격 중 하나
      ▼
최종 7클래스: {Normal, F_I, P_I, M_F, C_D, C_R, Unknown}
```

### 핵심 인터페이스 / 계약 (KEY CONTRACT) — 병렬 작업의 전제

**`dataset.npz` 스키마 (src/utils/io.py가 강제)**:

| 항목 | 규격 |
|---|---|
| `X` | `np.float32`, shape `(N, 3, H, W)`, 값 범위 `[0,1]` (채널 = coif1, db3, rbio1.3 LL) |
| `y` | `np.int64`, shape `(N,)`, 값 `{0..5}` |
| `meta.json` (npz 내부 또는 동봉) | `{wavelets:["coif1","db3","rbio1.3"], H, W, mode:"periodization", norm:{method,stats}, label_map, seed, git_sha, created_at}` |

**라벨 맵 (전원 동일하게 사용)**: `0=Normal, 1=F_I(Frame Injection), 2=P_I(PTP Sync), 3=M_F(Switch/MAC Flooding), 4=C_D(CAN DoS), 5=C_R(CAN Replay)`. Unknown은 **학습 클래스가 아니라 추론 시점 결정**(CAE 게이트 + softmax 신뢰도 임계).

**병렬화 핵심**: A가 실제 PCAP 파이프라인을 완성하기 전(Day1), `scripts/make_stub_dataset.py`로 위 스키마를 만족하는 더미 `dataset_v0.npz`(예 N=200, 3×64×64, 랜덤)를 먼저 발행한다. B/C/D는 이 스텁에 대해 코드를 작성·테스트하고, A의 실데이터로 교체만 하면 된다. **스키마(io.py)가 유일한 공유 결합점**이며, 이를 변경할 때만 팀 합의가 필요하다.

---

## 단계별 계획 (PHASES)

> 프레임워크 결정: **PyTorch** (로컬에 torch 2.10.0 설치됨, CAE 커스텀 루프·LOAO 재학습·head 교체가 단순). 논문은 Keras지만 기여는 아키텍처+전처리이므로 프레임워크는 결과에 영향 없음. 로컬 = CPU 스모크 테스트 전용, 실제 학습 = Colab T4 / Kaggle P100 / 랩 GPU.

---

### Phase 0 — 환경 세팅 + 데이터 확보 + 계약 동결 (Day 1-2)

**목표**: 레포·환경·데이터셋·.npz 계약을 동시에 확정하여 Day3부터 4인 완전 병렬화 가능 상태 만들기.

**작업 목록**:
- [ ] (전원) `git init -b main`, `.gitignore` 작성 (`data/raw/`, `*.pcap`, `*.npz`, `.venv/`, `__pycache__/`, `runs/`, `*.pth`, `wandb/`)
- [ ] (전원) `python3.10 -m venv .venv` + `requirements.txt` 작성 후 설치, `pip freeze > requirements.lock.txt` 커밋
- [ ] (전원) 레포 레이아웃 생성: `data/{raw,interim,processed} src/{preprocessing,models,pipeline,utils} configs notebooks experiments results/{figures,tables,checkpoints} tests scripts`
- [ ] (D) `src/utils/io.py`: `save_dataset/load_dataset/validate_schema` 구현 (dtype·shape·range·label 강제, 위반 시 raise)
- [ ] (D) `scripts/make_stub_dataset.py` 작성 → `dataset_v0.npz`(N=200, 3×64×64) 생성 후 공유 Drive 업로드 + DATASET.md에 sha256 기록
- [ ] (D) `src/utils/seed.py` (numpy/torch/PYTHONHASHSEED 고정), `src/utils/metrics.py` 스텁
- [ ] (A) **TOW-IDS 데이터셋 다운로드** (HCRL Dropbox 무로그인 링크 우선, IEEE DataPort DOI `10.21227/bz0w-zc12` 백업). zip ~197MB → 압축 해제, **DOI/AVTP+gPTP+CAN(UDP)+5공격 존재 확인** (Jeong et al. AVTP-only 데이터셋과 혼동 금지)
- [ ] (A) 파일 매니페스트 작성: 각 pcap → {Normal | 5공격 중 하나} + 프로토콜 + 패킷 수. **per-packet 라벨이 binary(Normal/Abnormal)뿐인지, 6-class 라벨을 capture/scenario에서 유도해야 하는지 확정**
- [ ] (A) **팀 보유 논문 PDF의 Section III에서 정확한 이미지 N×M·패킷 윈도우·정규화 방식 확인** (공개 소스로는 paywall, 확인 불가)
- [ ] (B/C/D) main에서 feature 브랜치 분기: `feat/preprocess`(A), `feat/clf-6class`(B), `feat/cae`(C), `feat/experiments`(D)

**사용 기술/라이브러리**: git, venv/pip, `torch torchvision numpy pandas scikit-learn PyWavelets dpkt scapy opencv-python-headless matplotlib seaborn tensorboard omegaconf tqdm pyyaml`, hashlib(sha256)

**담당자**: 전원(레포·env), D(io.py·stub·utils), A(데이터·매니페스트·논문 확인)

**산출물**: 동작하는 venv + lock 파일, 레포 트리, `src/utils/io.py`, `dataset_v0.npz` + DATASET.md, 데이터 매니페스트 CSV

**완료 기준 (DoD)**:
- `python -c "import torch,pywt,dpkt,cv2,sklearn,seaborn,omegaconf"` 무오류
- `load_dataset(stub)` → X `(200,3,64,64)` f32 [0,1], y int64 {0..5}; 손상된 배열엔 `validate_schema` raise
- 압축 해제된 데이터셋이 bz0w-zc12(5공격·gPTP·CAN-UDP 포함) 확인됨
- 매니페스트가 모든 pcap → 클래스·패킷수 매핑

**의존성**: 없음(시작점). 단, A의 데이터 확보는 외부 다운로드에 의존(리스크 낮음, ~200MB).

---

### Phase 1 — 전처리 파이프라인 + 베이스라인 binary 재현 (Day 3-4)

**목표**: 실제 PCAP → 3채널 wavelet 텐서 파이프라인 완성(`dataset.npz` v1 발행) **그리고** 원본 TOW-IDS binary DCNN을 재현하여 ~0.99 확인(이후 변경의 기준선 고정).

**작업 목록 (전처리 — A)**:
- [ ] dpkt(고속) 또는 scapy로 pcap 파싱: 패킷별 payload bytes → int list(0-255). Raw 레이어 없는 제어 프레임(일부 gPTP)은 `bytes(packet)` 전체 프레임 사용 고려. **`enumerate(packets)` 동일 인덱스로 라벨 정렬**, `len(packets)==len(labels)` assert
- [ ] 이미지 차원 결정 — **권장: 고정 64×64 슬라이딩 윈도우** (payload를 64바이트로 패딩/절단, 연속 64패킷을 64×64로 stack, 마지막 부분 그룹은 zero-padding). 대안: repo 방식 길이 버킷 {32,60,116,228,452}. **선택을 meta에 기록**
- [ ] `/255.0` 정규화 (wavelet **이전**), float32
- [ ] 3-wavelet level-1 `pywt.dwt2(img, w, mode='periodization')[0]` (LL만) for w in `['coif1','db3','rbio1.3']` → `np.stack(axis=-1)`. **세 wavelet 모두 동일 mode 사용**, stack 전 세 LL shape 동일 assert (periodization 시 64×64→32×32×3)
- [ ] **gotcha 가드**: rbio1.3는 biorthogonal이라 LL 크기/값 범위가 다를 수 있음 → 필요 시 세 LL을 공통 min(H,W)로 crop 후 stack; DWT 후 값이 [0,1] 초과 가능하므로 채널별 min-max 재정규화 또는 모델 BatchNorm 의존
- [ ] 5샘플 채널별 imshow 검증(공격 vs 정상 시각적 차이), 이미지당 라벨 CSV 정렬 → `save_dataset()`로 `dataset.npz` v1 발행

**작업 목록 (베이스라인 binary — B)**:
- [ ] 커스텀 DCNN 재현(PyTorch): `conv_block_a(256)` = SeparableConv2D(256,k=3,stride=3,relu,same)→BatchNorm→MaxPool(2,2); `conv_block_b(256)` ×5 = SeparableConv2D→BatchNorm; `conv_block_c` = SeparableConv2D(512,k=3,stride=3)→GlobalAvgPool→Dense(256,relu)→Dense(64,relu)→Dropout(0.5)→**Dense(2, sigmoid)**
- [ ] compile: Adam(lr=1e-3), binary 손실, train_test_split(test_size=0.2, random_state=42), batch=32, epochs~100(EarlyStopping patience=5 restore_best 권장 hardened variant)
- [ ] frozen test에서 1회 평가, 5 seeds(0-4) 실행
- [ ] **참고**: 'ResNet'은 실제 skip-connection 없는 depthwise-separable DCNN. 채점자가 literal residual 요구 시에만 functional API로 Add() 추가(residual block 내부는 stride=1로 shape 정합 유지). 기본은 충실 재현.

**사용 기술/라이브러리**: dpkt/scapy, PyWavelets(`pywt.dwt2`), numpy, opencv(resize), PyTorch, sklearn(LabelEncoder, train_test_split, metrics)

**담당자**: A(전처리·npz 발행), B(베이스라인 DCNN)

**산출물**: `src/preprocessing/*` + `dataset.npz` v1, `src/models/dcnn.py`, 베이스라인 binary 결과(Acc/F1/FPR/AUC mean±std)

**완료 기준 (DoD)**:
- npz가 `validate_schema` 통과, 동일 seed 재실행 시 X byte-identical(`np.array_equal`)
- per-image 출력 shape `(H,W,3)`, 세 채널 동일 spatial dim, `np.isfinite().all()`
- `model.summary()`가 Dense(2)로 끝나는 7 conv block; binary test Acc ≥ 0.985, F1 ≥ 0.985 (목표 ~0.99)

**의존성**: A의 npz 파이프라인은 Phase 0(데이터·io.py)에 의존. B는 **스텁으로 Day2 착수 가능**, 실데이터 npz로 교체 후 본 결과 산출.

---

### Phase 2 — 6-class Softmax 분류기 + 누수 안전 split (Day 5-6)

**목표**: 베이스라인 head를 Softmax(6)로 교체한 standalone 멀티클래스 분류기(S2) 완성 + Confusion Matrix. **그리고** 시간 누수 없는 train/val/test split 매니페스트 동결(전 시스템·전 seed 재사용).

**작업 목록 (분류기 — B)**:
- [ ] `conv_block_c` 마지막 레이어만 교체: `Dense(2,sigmoid)` → **`Dense(6, softmax)`**, 손실 binary → **CrossEntropy(클래스 가중)**. 앞단 Dense(256,relu)→Dense(64,relu)→Dropout(0.5) 유지
- [ ] 라벨: 매니페스트 기반 6-class(per-attack) 재구성, LabelEncoder + to_categorical(num_classes=6)
- [ ] **클래스 불균형 처리**: `np.bincount` 측정(Normal·flooding 多, PTP Sync·Frame Injection 少) → `compute_class_weight('balanced')` 적용; 필요 시 focal loss(gamma=2.0). **split 이후에만** 가중/오버샘플
- [ ] 5 seeds, frozen test에서 `classification_report(digits=4)` + `confusion_matrix(normalize='true')` → seaborn heatmap(정규화·raw 2종)
- [ ] **Unknown 결정**: Option A(권장) — S2는 5공격(또는 Normal 포함 6)만 학습, Unknown은 추론 시 max-softmax < tau로 산출. Option B(시간 허용 시) — held-out으로 Unknown 노드 학습. 사용한 방식 명시

**작업 목록 (누수 안전 split — D)**:
- [ ] **split 단위 = 연속 PCAP segment/time block** (인접 이미지가 패킷 공유 → random split은 누수, near-perfect 부풀림)
- [ ] 클래스별 capture 순서 정렬 → 연속 chunk로 절단 → chunk 단위 70/15/15 배정, train/val/test 사이 **guard gap ≥ 윈도우 크기(예 ≥64패킷)** 삽입
- [ ] stratify(클래스별 동일 비율), split 인덱스를 `manifest.json/npz`로 동결 → S1/S2/S3·전 seed 동일 test 재사용
- [ ] CAE용 **Normal-only train/val subset** 분리 추출

**사용 기술/라이브러리**: PyTorch, sklearn(class_weight, classification_report, confusion_matrix, f1_score macro/weighted, ConfusionMatrixDisplay), `CategoricalFocalCrossentropy`, seaborn/matplotlib, numpy/pandas, json/npz

**담당자**: B(6-class 모델·CM), D(split 매니페스트)

**산출물**: `src/models/dcnn.py`(softmax head), 6-class Confusion Matrix PNG(정규화·raw), per-class P/R/F1 표(mean±std), 동결된 split manifest, Normal-only subset

**완료 기준 (DoD)**:
- 출력 Dense(6,softmax), `predict` shape `(batch,6)` 합=1; 손실 CrossEntropy 감소
- split 간 패킷/이미지 **교집합 0** assert, 클래스 비율 ±1%, 동일 seed 동일 split 재현
- 6-class macro-F1 ≥ 0.95(목표 ≥ 0.97), 가능하면 클래스별 recall ≥ 0.90; CM PNG 산출

**의존성**: B는 Phase1 베이스라인 backbone에 의존(head만 교체). D의 split은 A의 실 npz(capture 순서 정보) 필요 — 그 전엔 스텁으로 로직 개발. **Phase3 CAE는 D의 Normal-only subset 정의에 의존.**

---

### Phase 3 — CAE Stage-1 + 임계값 + 파이프라인 결합 (Day 7-9)

**목표**: 정상만 학습하는 CAE 구현, `tau = mu + 2*sigma` 임계 설정, Stage-1→Stage-2 gated cascade 결합. **Day 9에 비상 플랜 결정 게이트.**

**작업 목록 (CAE — C)**:
- [ ] 대칭 Conv2D AE 구현: Encoder = `Input(H,W,3)`→ Conv2D(f,3,stride=2,same)+BN+ReLU 블록 ×3-4(f=32→64→128(→256))→Flatten→Dense(latent=128). Decoder = Dense+Reshape→Conv2DTranspose(stride=2)+BN+ReLU 미러→최종 Conv2DTranspose(3,sigmoid). **입력 dim이 홀수/2의 거듭제곱 아니면 마지막에 Cropping2D/Resizing으로 output==input 보장**. CAE 입력 shape는 런타임 `X.shape[1:]`에서 파라미터화
- [ ] 학습: Normal-train만, loss=MSE, Adam(1e-3), batch 64-128, epochs~150, `EarlyStopping(val_loss, patience=12, restore_best)` + `ReduceLROnPlateau(patience=5,factor=0.5)`. identity-mapping 과적합 방지 위해 입력 GaussianNoise(denoising) 옵션
- [ ] **임계값 (validation-normal 기준, training 아님)**: `err = mean((x̂-x)^2, axis=(1,2,3))` per-image MSE; `mu=err.mean(); sigma=err.std(ddof=1); tau2=mu+2σ`. 추가로 `tau3, p95, p99`도 계산. 우편향 대비 `np.log(err)` 변환 검토
- [ ] **분리도 sanity**: normal vs known-attack MSE 히스토그램 + tau 수직선, ROC-AUC 보고(목표 ≥ 0.90)
- [ ] 임계 민감도 분석표: mu+2σ / mu+3σ / p95 / p99 / Youden's J(혼합 val에서 `roc_curve`, J=tpr-fpr, oracle 상한으로만 보고). **headline = mu+2σ**

**작업 목록 (파이프라인 결합 — C)**:
- [ ] `src/pipeline/two_stage.py`: `predict(x)` = `mse<=tau → 'Normal'(stop)`; else `probs=stage2.predict(x)`; `probs.max()<conf_thr(예 0.5) → 'Unknown'`; else `class_names[argmax]`. conf_thr은 val에서 calibrate
- [ ] 단계별 결정 로깅(Unknown 시각화 산출물용)

**🚨 Day 9 비상 플랜 결정 게이트**:
- **PASS 조건**: val-normal FPR ≤ 5%, MSE ROC-AUC ≥ 0.75(목표 ≥ 0.90), fold 간 tau 안정
- **FAIL 시(AUC<0.75 또는 val FPR>15% 또는 tau 불안정)**: Stage-1 포기, **S2(6-class)만으로 발표**. CAE는 "향후 연구" 슬라이드 1장. 단계가 config 분리되어 있어 `pipeline.use_cae=false`, `model.head=softmax` 플래그 전환만으로 처리(코드 재작성 불필요)

**사용 기술/라이브러리**: PyTorch(Conv2D/Conv2DTranspose/BatchNorm/Cropping2D), callbacks(EarlyStopping, ReduceLROnPlateau), numpy(mean/std ddof=1/percentile), sklearn(roc_curve, roc_auc_score, f1_score), matplotlib(hist, axvline)

**담당자**: C(CAE·임계·결합), D(비상 플랜 판정 데이터 제공)

**산출물**: `src/models/cae.py`, `src/pipeline/two_stage.py`, 임계값(mu,σ,tau 4종) + 히스토그램·ROC, 민감도 분석표, Day9 판정 결과

**완료 기준 (DoD)**:
- `model.summary()` decoder output shape == input H×W×3, forward pass 동형 출력, params < 5M
- val_loss 수렴, train/val MSE 격차 작음(과적합 아님), 정상 재구성 시각적 유사
- tau 4종 출력, 정상 MSE는 tau 아래·known-attack은 위 분포, ROC-AUC 보고
- 파이프라인이 혼합 test에서 7라벨 산출, 정상은 대부분 Stage-1에서 종료

**의존성**: **임계 경로의 핵심.** C는 D의 Normal-only subset(Phase2) + A의 실 npz에 의존. 파이프라인 결합은 B의 학습된 S2 모델에 의존. **CAE는 LOAO에서 1회 학습 후 전 fold 재사용**(정상만 학습하므로 fold 무관) — 컴퓨트 절약.

---

### Phase 4 — 실험·검증: 3종 비교 + LOAO Zero-day + 시각화 (Day 10-11)

**목표**: S1/S2/S3를 **동일 frozen test**에서 평가, LOAO Zero-day 실험, 4대 산출물 + 통계 엄밀성(seeds·error bar) 확보.

**작업 목록 (실험 — D, C/B 지원)**:
- [ ] **3종 비교표** (동일 manifest, mean±std over 5 seeds): 열 = {Accuracy, binary-F1(S1만), macro-F1, weighted-F1, per-attack recall 요약, Normal FPR, zero-day 탐지율, inference latency/img}; 행 = S1 baseline / S2 6-class / S3 2-stage. 각 시스템 capability(multiclass·zero-day yes/no) 표기. `pandas.to_markdown()/to_latex()`
- [ ] **6-class(및 S3의 7-class) Confusion Matrix** 재생성·확정
- [ ] **LOAO Zero-day 프로토콜** (`experiments/leave_one_out.py`, config matrix):
  - for a in {F_I,P_I,M_F,C_D,C_R}: **(1) CAE = Normal-only로 학습(전 fold 동일, 1회 학습 후 재사용)**; **(2) Stage-2 = Normal + 나머지 4공격으로 학습, a 제외**(a에 대한 클래스 없음); **(3) test = held-out a 샘플**: `detection_rate = mean( mse(a)>tau AND stage2→Unknown )`; Normal FPR(≤5%), a-vs-normal MSE ROC-AUC 기록
  - fold당 5 seeds, 5-fold 평균
  - **두 지표 분리 보고**: ① CAE anomaly recall = `P(MSE>tau | a)` ② end-to-end Unknown rate = `P(flagged AND Unknown | a)`. (CAE가 잡았는데 S2가 known으로 오배정하면 Unknown 로직 실패 — 원인 분리)
- [ ] **Zero-day 탐지율 그래프**: x=5공격, y=탐지율, error bar=seed std, fold 평균 점선 (`matplotlib.bar(yerr=std)`)
- [ ] **Unknown Attack 사례 시각화**: held-out 샘플(예 C_R) 4패널 — (a)입력 3채널 이미지 (b)CAE 재구성 (c)per-pixel error heatmap `(x-x̂)^2` (d)MSE vs tau 히스토그램 + softmax confidence bar(Unknown 임계 미달). 캡션: "모델이 C_R 미학습 → 높은 재구성오차 → Unknown 라우팅". 대비용 Normal 저오차 사례 1개
- [ ] **통계 하네스**: seeds [0,1,2,3,4], per-seed 지표 CSV 로깅, 집계기로 mean/std/95%CI(=1.96σ/√n). 모든 figure/table은 CSV에서 재생성 가능

**사용 기술/라이브러리**: PyTorch, sklearn.metrics(classification_report, confusion_matrix, roc_auc_score, f1_score), seaborn/matplotlib(heatmap, bar yerr, imshow, errorbar), pandas, OmegaConf(config matrix)

**담당자**: D(실험 총괄·표·그래프·통계), C(CAE·파이프라인 추론 지원), B(S2 LOAO 재학습 지원)

**산출물**: 3종 비교표(md+latex), 6/7-class CM PNG, LOAO 탐지율 bar chart PNG, Unknown 사례 4패널 PNG, per-seed 지표 CSV, LOAO per-fold 표(탐지율/FPR/AUC mean±std)

**완료 기준 (DoD)**:
- S1/S2/S3 **byte-identical test set**에서 평가, 모든 셀 mean±std
- LOAO 5-fold 평균 zero-day 탐지율 ≥ 0.80(목표 ≥ 0.90), 모든 fold Normal FPR ≤ 5%
- CAE는 LOAO에서 1회만 학습·재사용(문서화), 두 지표 분리 보고
- 모든 figure가 로깅 CSV에서 재생성 가능, LOAO 그래프·비교표에 error bar 존재

**의존성**: Phase4는 S1(Phase1)·S2(Phase2)·CAE+파이프라인(Phase3) 전부에 의존. **임계 경로의 끝단.** Day9 FAIL 시 LOAO·S3 컬럼은 N/A 처리하고 S2 단독 결과로 축소.

---

### Phase 5 — 발표 자료 + 리허설 (Day 12-13)

**목표**: 구현+실험 결과를 발표 슬라이드로 정리, 리허설로 발표 완성도 확보.

**작업 목록 (전원)**:
- [ ] 스토리라인: 문제(TOW-IDS 한계 2가지) → **관련연구·차별점(da Luz/AERO/SeqWatch 대비 "통합 + LOAO 정량화" novelty 명시)** → 제안(2단계) → 아키텍처/데이터흐름 → 전처리(3-wavelet) → S2 결과(CM) → CAE/임계 방법론(히스토그램) → LOAO Zero-day 결과 → 3종 비교표 → Unknown 사례 시각화 → 한계·향후연구
- [ ] **novelty 문장은 차별점 섹션 헤드라인("통합 + LOAO 정량화")을 그대로 사용**, must-cite 세트(특히 SeqWatch) 인용 — **"최초 multiclass/zero-day" 표현 금지**(R15)
- [ ] 4대 산출물 슬라이드 삽입 + 핵심 수치 강조(macro-F1, zero-day 탐지율, FPR)
- [ ] (Day9 FAIL 시) CAE "향후 연구" 슬라이드 1장으로 정직하게 전환
- [ ] 데모/코드 워크스루 준비, 역할 분담 발표, 2회 이상 리허설(시간 측정)
- [ ] README·CONTRIBUTING·DATASET.md 최종 정리, 결과 재현 명령 문서화

**사용 기술/라이브러리**: 슬라이드 도구, matplotlib/seaborn(figure 마감)

**담당자**: 전원(D 총괄)

**산출물**: 발표 슬라이드, 최종 README, 재현 가능한 실험 명령

**완료 기준 (DoD)**: 슬라이드에 4대 산출물 모두 포함, 리허설 시간 내 완료, 신규 클론에서 결과 재생성 명령 동작

**의존성**: Phase4 결과에 의존.

---

## 의존성 & 임계 경로 (Critical Path)

```
Phase0(env+데이터+io.py계약) ──► Phase1(전처리 npz v1 + 베이스라인 S1)
        │                              │
        │  stub으로 병렬 시작           ├──► Phase2(6-class S2 + 누수안전 split + Normal-only subset)
        │                              │              │
        └──────────────────────────────┘              ▼
                                              Phase3(CAE + tau + 파이프라인) ──[Day9 게이트]──► Phase4(실험·LOAO) ──► Phase5(발표)
```

**임계 경로**: `데이터 확보(A) → 전처리 npz(A) → Normal-only subset(D, Phase2) → CAE+임계+파이프라인(C, Phase3) → LOAO 실험(D, Phase4) → 발표(Phase5)`. **CAE 임계 안정성이 최대 게이트** (Day9 비상 플랜 분기점).

**무엇이 무엇을 막는가**:
- A의 **실 npz**가 늦으면 B/C/D의 *본* 결과가 막힘 → **stub(Day1)로 코드는 병렬 진행**, 실데이터 교체만 대기.
- D의 **Normal-only subset 정의(Phase2)**가 C의 CAE 학습을 막음 → Phase2 split을 Phase3보다 먼저 동결.
- C의 **파이프라인 결합**은 B의 학습된 S2를 요구 → Phase2 S2 완료 후 Phase3 cascade.
- **io.py 스키마**는 유일한 공유 결합점 — 변경 시 전원 합의.

**병렬 가능**: Phase1의 전처리(A)와 베이스라인 DCNN(B, stub) 동시 진행 가능. Phase2의 6-class(B)와 split 매니페스트(D) 동시 진행. CAE 학습(C)은 GPU에서 백그라운드, 동안 D는 실험 하네스 구축.

---

## 리스크 & 대응 (Risk Register)

| # | 리스크 | 심각도 | 대응 |
|---|---|---|---|
| R1 | 데이터셋 다운로드 지연/혼동 (Jeong AVTP-only와 혼동) | 낮 | ~197MB 소형(수 분). HCRL Dropbox 무로그인 우선 + IEEE 계정 병행. **DOI bz0w-zc12 + gPTP/CAN-UDP/5공격 확인** 필수 |
| R2 | per-packet 라벨이 binary뿐 → 6-class 라벨 부재 | 중 | capture/scenario별로 attack-type 유도. Day1-2 매니페스트로 사전 검증 |
| R3 | 정확한 이미지 N×M·윈도우·정규화가 paper paywall로 미확인 | 중 | 팀 보유 PDF Section III 직독. 그 전엔 고정 64×64(권장)로 진행 — 파이프라인은 size-agnostic하므로 정확성 무관, H/W만 변동 |
| R4 | 3 wavelet LL 크기/값 범위 불일치로 np.stack 실패 (rbio1.3 biorthogonal) | 중 | 단일 `WAVELET_MODE='periodization'` + 단일 IMG_SIZE 중앙화; stack 전 공통 min(H,W) crop·shape assert; 32/60/116/228/452 버킷별 unit test |
| R5 | **시간 누수**(인접 윈도우 패킷 공유) → 전 지표 near-perfect 부풀림 | 높 | random split 금지. 연속 time/PCAP block 단위 split + guard gap ≥ 윈도우. 패킷 교집합 0 assert, 단일 frozen manifest 전 시스템 재사용 |
| R6 | **CAE 임계(mu+2σ) 불안정** / normal·attack MSE 분포 중첩 | 높 | k {1.5,2,2.5,3} sweep, FPR 예산 ≤5%, ROC-AUC 모니터. log 변환/percentile 대안. **Day9 AUC<0.75 or FPR>15% → 비상 플랜(S2-only)** |
| R7 | 클래스 불균형(Normal≫각 공격, flooding 多/PTP少) → accuracy 무의미 | 높 | headline = **macro-F1**, class_weight('balanced')/focal(γ=2), stratified split, row-normalized CM, per-class recall 보고 |
| R8 | CAE identity-mapping 과적합 → 공격도 저오차로 재구성, anomaly recall 붕괴 | 중 | tight bottleneck(압축비 20-100x), denoising(GaussianNoise)/Dropout, shallow 유지, 히스토그램으로 분리 확인 |
| R9 | Zero-day가 CAE는 잡았으나 S2가 known으로 오배정 → Unknown rate 낮음 | 중 | max-softmax confidence 게이트(val calibrate), CAE recall과 end-to-end Unknown rate **분리 보고**로 원인 가시화 |
| R10 | 베이스라인 ~0.99 재현 실패(원본 Keras/하이퍼파라미터 부재) | 중 | community repo(LokeshNaganaboina/Tow-IDS)를 구현 기준으로, Acc ≥0.985면 '재현' 인정. 자체 하이퍼파라미터 명시 보고 |
| R11 | 로컬 GPU 없음 → CPU 학습 정체 | 높 | Day1부터 Colab T4/Kaggle P100/랩 GPU 의무화. 로컬은 1-epoch 스텁 스모크 전용. wavelet 이미지 .npy 캐싱으로 재계산 방지. 3×64×64 batch128에서 <4GB VRAM |
| R12 | 단일 seed 결과 신뢰 부족 | 중 | seeds [0-4](최소 3) 전 모델, mean±std/95%CI, LOAO·비교표 error bar |
| R13 | Unknown 학습 라벨 부재로 naive 6-node softmax가 Unknown 예측 불가 | 중 | Option A: CAE OOD 게이트 + max-softmax 임계로 추론 시 산출. Option B(시간 허용): held-out으로 Unknown 노드 학습. LOAO가 정직한 평가 |
| R14 | 직렬 의존(A→B→C)으로 3인 대기 | 중 | Day1 stub 발행로 B/C/D 즉시 병렬. io.py 스키마만 공유 결합 |
| R15 | **Novelty 붕괴** — da Luz(multiclass)·AERO/SeqWatch(zero-day)가 각 축 선점 → "최초 multiclass/zero-day" 주장이 Q&A에서 반박됨 | 높 | 주장을 **"통합 + LOAO 정량화"로 좁힘**(차별점 섹션 헤드라인). must-cite 세트(특히 **SeqWatch**) 인용. Day1에 SeqWatch의 TOW-IDS 사용 여부 확인 |
| R16 | (옵션) da Luz 외부 베이스라인(`repo_eval`) 재사용 시 **feature space 비호환** — repo_eval은 wavelet이 아닌 nibble image `(1,44,116)` → 사전학습 가중치가 팀 wavelet 입력과 호환 불가 | 중 | repo_eval은 **별도 격리 외부 베이스라인**으로만 사용(가중치 그대로). 팀 S1/S2/S3는 wavelet 입력으로 통일. 비교 시 "입력 feature가 다름"을 한계로 명시 |

**비상 플랜 통합**: Day9 게이트에서 CAE 실패 시 → 단계가 config 분리(`pipeline.use_cae=false`)되어 있어 코드 재작성 없이 **S2 6-class 단독 발표 + CAE 향후연구 슬라이드 1장**으로 전환. S2는 동일 npz로 독립 동작하므로 영향 격리됨.

---

## 실험·평가 계획

### 3종 시스템 비교 (동일 frozen test, 5 seeds, mean±std)
- **S1 — 베이스라인 binary TOW-IDS** (Sigmoid → Normal/Attack)
- **S2 — standalone 6-class Softmax 분류기** (Normal 포함 6-class 권장, 독립 비교 가능)
- **S3 — 2단계 CAE→6-class 파이프라인** (7-class 출력)
- **(옵션) S2′ — da Luz 외부 베이스라인**: `luigiluz/automotive-ids-evaluation-framework`(MIT)의 사전학습 multiclass 가중치를 재학습 없이 로드해 외부 비교군으로 추가. **단 입력이 nibble image `(1,44,116)`로 다름 → "동일 입력 통제 아님"을 한계로 명시**(R16). 시간 여유 시에만 — 우리 차별점(통합+LOAO)은 S2′ 없이도 S1/S2/S3로 충분히 입증됨.

### 단계별 지표
- **Binary (S1)**: Accuracy, F1, FPR=FP/(FP+TN), FNR, ROC-AUC
- **Multiclass (S2/S3)**: per-class Precision/Recall/F1, **macro-F1(headline)**, weighted-F1, Accuracy, 6×6(또는 7×7) Confusion Matrix(정규화·raw)
- **Zero-day (LOAO)**: held-out 공격 탐지율/recall, Normal FPR, MSE ROC-AUC, 5-fold mean±std

### Leave-one-attack-out (LOAO) 프로토콜 — 정확한 정의
> **핵심 정정**: CAE는 **항상 정상 트래픽만 학습**한다. 따라서 "공격 k를 CAE 학습에서 제외"는 **무의미한 no-op**(CAE 학습셋엔 애초에 공격이 없음). LOAO는 **Stage-2 학습 데이터**와 **test 시점 zero-day 대상**만 바꾼다.

각 fold k ∈ {F_I, P_I, M_F, C_D, C_R}:
1. **CAE**: Normal-only로 학습 — **모든 fold 동일**, 따라서 **1회 학습 후 전 fold 재사용**(컴퓨트 절약·문서화).
2. **Stage-2**: Normal + 나머지 **4공격**으로 학습, **공격 k 제외**(k에 대한 softmax 노드 없음).
3. **Test**: held-out 공격 k 샘플 투입 → 기대: 높은 MSE(CAE가 anomaly로 포착) **AND** S2가 미학습이므로 Unknown(낮은 max-softmax) 산출.
4. **두 지표 분리 측정**:
   - ① **CAE anomaly recall** = `P(MSE > tau | attack k)` — CAE가 미지 공격을 잡는가
   - ② **end-to-end Unknown rate** = `P(flagged anomalous AND classified Unknown | attack k)` — 최종 Unknown 라우팅 성공률
5. **Normal FPR** = `P(MSE > tau | normal)` (tau 규칙으로 ~2-5% 고정)도 매 fold 기록.
6. fold당 5 seeds, 5-fold 평균 = zero-day 일반화 능력.

### Target 수치
| 항목 | 최소 | 목표 |
|---|---|---|
| 베이스라인 재현 Acc/F1 | ≥ 0.985 | ~0.99 (논문 0.9965/0.9974) |
| 6-class macro-F1 | ≥ 0.95 | ≥ 0.97 |
| LOAO 5-fold 평균 zero-day 탐지율 | ≥ 0.80 | ≥ 0.90 |
| Normal FPR (mu+2σ) | ≤ 5% (전 fold) | — |
| MSE ROC-AUC (normal vs known-attack) | ≥ 0.90 (val) | — |

### 데이터 스누핑 방지
tau·k·epochs·architecture·conf_thr는 **validation에서만** 동결. test set은 모든 결정 확정 후 **시스템당 1회만** 평가.

---

## 즉시 시작할 첫 작업 (Day 1 Kickoff Checklist)

- [ ] (전원) `cd /home/s3min/s3min-projects/mobility && git init -b main`, `.gitignore` 작성, 레포 레이아웃 생성, 첫 커밋
- [ ] (전원) `python3.10 -m venv .venv` + `requirements.txt` 설치 + `pip freeze > requirements.lock.txt`. 검증: `python -c "import torch,pywt,dpkt,cv2,sklearn,seaborn,omegaconf"`
- [ ] (D) `src/utils/io.py`(save/load/validate_schema) + `scripts/make_stub_dataset.py` → `dataset_v0.npz`(200,3,64,64) 발행, Drive 업로드, DATASET.md에 sha256
- [ ] (A) **TOW-IDS 데이터셋 다운로드 시작** (HCRL Dropbox), 압축 해제, **bz0w-zc12 + 5공격/gPTP/CAN-UDP 존재 확인**
- [ ] (A) 데이터 매니페스트 작성 착수 + **팀 PDF에서 이미지 N×M·정규화 확인**
- [ ] (B/C/D) feature 브랜치 분기, stub으로 모델/CAE/실험 코드 골격 작성 시작
- [ ] (전원) GPU 런타임 확보(Colab/Kaggle/랩) — 로컬은 GPU 없음, CPU 스모크 전용임을 합의
- [ ] (D) `configs/{preprocess,model,cae,experiment}.yaml` 골격 + seed.py + metrics.py 스텁
- [ ] (D) **관련연구 검증 (R15)**: SeqWatch(SBRC 2025)·da Luz(Ad Hoc Networks 2024)·AERO(TII 2024) 원문 확보 → **SeqWatch의 TOW-IDS 데이터셋 사용 여부 확인** → must-cite 세트·차별점 헤드라인 문장 확정

## 미해결 질문 / 조기 확인 필요 (Day 1-2 내 해소)

1. **이미지 차원·윈도우·정규화**: 원본 TOW-IDS Section III의 정확한 N×M, 패킷 윈도우/stride(overlap 여부), 정규화(/255 vs min-max vs z-score), 헤더 포함 여부 — **팀 보유 PDF 직독으로 확정**(공개 소스 paywall). 윈도우 overlap 여부는 R5 guard-gap 크기를 결정.
2. **라벨 스키마**: 공식 CSV가 6-class attack-type 컬럼을 갖는가, 아니면 capture/scenario에서 유도해야 하는가 (R2) — 다운로드 후 즉시 확인.
3. **S2에 Normal 포함 여부**: standalone 6-class에 Normal을 넣을지(6×6 CM, S3와 비교 용이) 또는 5공격만(Normal은 CAE 게이트 전담)·5×5 CM. → **Normal 포함 6-class 권장**.
4. **Unknown 결정 규칙**: known 데이터 추론 시 Unknown을 순수 CAE 게이트로만 할지, softmax confidence 임계도 병용할지 — val에서 단일 규칙 고정.
5. **CAE 입력**: 3채널 wavelet 텐서(HxWx3) 사용 확정(일관성 권장).
6. **Normal 샘플 수**: 70/15/15 Normal-only split + 안정적 mu/sigma 추정에 충분한가 — 다운로드 후 카운트.
7. **GPU 접근**: 랩 GPU 유무에 따라 Docker 이미지 작성 가치 결정. config 도구는 **OmegaConf**(경량)로 통일, W&B는 opt-in(기본 TensorBoard).
8. **차별점 확정 (R15)**: SeqWatch가 TOW-IDS를 쓰는지 확인 → 헤드라인 novelty 문장("통합 + LOAO 정량화") 최종 고정. da Luz 외부 베이스라인(S2′, repo_eval) 포함 여부 결정(옵션, R16 feature 비호환 한계 인지).
