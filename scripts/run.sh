#!/usr/bin/env bash
# =============================================================================
# run.sh  —  TOW-IDS Extension 전체 파이프라인 실행 스크립트
#
# 사용법:
#   bash scripts/run.sh all        # 전체 파이프라인 (전처리 → 학습 → LOAO → 시각화)
#   bash scripts/run.sh preprocess # 전처리만 (NPZ 생성 + split manifest)
#   bash scripts/run.sh train      # S1 + S2 + CAE 학습
#   bash scripts/run.sh loao       # LOAO 실험 + 비교표 + 시각화
#   bash scripts/run.sh visualize  # 시각화 4종 (모델/LOAO 체크포인트 필요)
#   bash scripts/run.sh smoke      # 전체 smoke test (각 단계 1 epoch, seed 0)
#
# 옵션:
#   --skip-preprocess  'all' 실행 시 전처리 건너뜀 (NPZ 이미 있을 때)
#   --skip-train       'all' 실행 시 학습 건너뜀 (checkpoint 이미 있을 때)
#
# 예시:
#   bash scripts/run.sh all --skip-preprocess   # NPZ 있을 때 학습부터
#   bash scripts/run.sh all --skip-train        # checkpoint 있을 때 LOAO부터
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Python 실행 파일 자동 탐지 (venv 우선)
if [ -x "$ROOT/.venv/bin/python" ]; then
    PY="$ROOT/.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PY="python3"
else
    echo "[ERROR] Python을 찾을 수 없습니다. .venv를 생성하거나 python3를 설치하세요."
    exit 1
fi

# ---------------------------------------------------------------------------
# 유틸리티 함수
# ---------------------------------------------------------------------------

_step() {
    echo ""
    echo "============================================="
    echo "  $*"
    echo "============================================="
}

_ok() {
    echo "  [OK] $*"
}

_warn() {
    echo "  [WARN] $*"
}

_check_file() {
    if [ -f "$1" ]; then
        _ok "확인: $1"
    else
        echo "  [FAIL] 파일 없음: $1"
        exit 1
    fi
}

_check_dir_nonempty() {
    if ls "$1"/*.pth &>/dev/null 2>&1; then
        _ok "확인: $1/*.pth"
    else
        echo "  [FAIL] 체크포인트 없음: $1"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# 개별 단계 함수
# ---------------------------------------------------------------------------

# ── 전처리 ──────────────────────────────────────────────────────────────────

step_preprocess() {
    local SMOKE="${1:-}"

    _step "전처리 — NPZ 생성 + split manifest"

    # NPZ 생성
    if [ -n "$SMOKE" ]; then
        _warn "Smoke 모드: stub dataset 생성"
        "$PY" scripts/make_stub_dataset.py
        # smoke용 train/test NPZ가 없으면 stub을 복사해 임시 사용
        if [ ! -f "data/processed/dataset_train.npz" ]; then
            cp data/processed/dataset_v0.npz data/processed/dataset_train.npz
        fi
        if [ ! -f "data/processed/dataset_test.npz" ]; then
            cp data/processed/dataset_v0.npz data/processed/dataset_test.npz
        fi
    else
        if [ ! -f "data/raw/y_train.csv" ] || [ ! -f "data/raw/y_test.csv" ]; then
            echo ""
            echo "[ERROR] data/raw/ 에 PCAP/CSV 파일이 없습니다."
            echo "        실제 파일을 배치하거나 smoke 테스트용 stub을 사용하세요:"
            echo "        bash scripts/run.sh smoke"
            exit 1
        fi
        "$PY" scripts/build_tow_dataset.py
    fi

    _check_file "data/processed/dataset_train.npz"
    _check_file "data/processed/dataset_test.npz"

    # split manifest 생성
    _step "Split manifest 생성 (pcap_id 그룹 단위)"
    if [ -n "$SMOKE" ]; then
        "$PY" -m src.utils.split \
          --train-npz data/processed/dataset_train.npz \
          --test-npz  data/processed/dataset_test.npz \
          --out       data/processed/split_manifest.json \
          --val-ratio 0.10 \
          --seed 42 \
          --allow-unsafe-fallback
    else
        "$PY" -m src.utils.split \
          --train-npz data/processed/dataset_train.npz \
          --test-npz  data/processed/dataset_test.npz \
          --out       data/processed/split_manifest.json \
          --val-ratio 0.10 \
          --seed 42
    fi

    _check_file "data/processed/split_manifest.json"
    _ok "전처리 완료."
}

# ── 학습 ────────────────────────────────────────────────────────────────────

step_train() {
    local SMOKE="${1:-}"
    local SMOKE_FLAG=""
    [ -n "$SMOKE" ] && SMOKE_FLAG="--smoke"

    _step "S1 — 이진 분류 베이스라인 학습"
    "$PY" -m src.train.train_s1 $SMOKE_FLAG
    _check_file "results/tables/s1_baseline.csv"

    _step "S2 — 6-class Focal Loss 분류 학습"
    "$PY" -m src.train.train_s2 $SMOKE_FLAG
    _check_file "results/checkpoints/s2_seed_0_best.pth"
    _check_file "results/tables/s2_summary_focal.csv"

    _step "CAE — 정상 트래픽 재구성 + tau 산출"
    "$PY" -m src.train.train_cae $SMOKE_FLAG
    _check_file "results/checkpoints/cae_best.pth"
    _check_file "results/tables/tau_values.json"

    _ok "학습 완료."
}

# ── LOAO ────────────────────────────────────────────────────────────────────

step_loao() {
    local SMOKE="${1:-}"
    local SMOKE_FLAG=""
    [ -n "$SMOKE" ] && SMOKE_FLAG="--smoke"

    _step "LOAO 제로데이 평가 (5 fold × 5 seed)"
    "$PY" -m experiments.leave_one_out $SMOKE_FLAG
    _check_file "results/tables/loao_per_fold.csv"
    _check_file "results/tables/loao_summary.csv"

    _step "3종 비교표 생성 (S1 / S2 / S3)"
    "$PY" scripts/comparison_table.py $SMOKE_FLAG
    _check_file "results/tables/comparison_table.csv"
    _check_file "results/tables/comparison_table.md"

    _ok "LOAO 완료."
}

# ── 시각화 ──────────────────────────────────────────────────────────────────

step_visualize() {
    _step "시각화 — LOAO bar chart"
    "$PY" scripts/plot_loao_bar.py
    _check_file "results/figures/loao_bar_chart.png"

    _step "시각화 — Unknown case 4-panel"
    "$PY" scripts/plot_unknown_case.py
    _check_file "results/figures/unknown_case_4panel.png"

    _step "시각화 — S2/S3 Confusion Matrix"
    "$PY" scripts/plot_confusion_matrix.py
    _check_file "results/figures/cm_s2_6class.png"
    _check_file "results/figures/cm_s3_7class.png"

    _ok "시각화 완료."
}

# ── pytest ──────────────────────────────────────────────────────────────────

step_test() {
    _step "pytest — 전체 테스트 (59 tests 기대)"
    "$PY" -m pytest tests/ -v
    _ok "테스트 완료."
}

# ---------------------------------------------------------------------------
# 결과 요약 출력
# ---------------------------------------------------------------------------

_print_summary() {
    echo ""
    echo "============================================="
    echo "  완료. 주요 산출물 목록"
    echo "============================================="
    echo ""
    echo "  [데이터]"
    [ -f "data/processed/dataset_train.npz"    ] && _ok "data/processed/dataset_train.npz"
    [ -f "data/processed/split_manifest.json"  ] && _ok "data/processed/split_manifest.json"
    echo ""
    echo "  [체크포인트]"
    [ -f "results/checkpoints/s2_seed_0_best.pth" ] && _ok "results/checkpoints/s2_seed_*.pth"
    [ -f "results/checkpoints/cae_best.pth"        ] && _ok "results/checkpoints/cae_best.pth"
    echo ""
    echo "  [테이블]"
    for f in \
        results/tables/s1_baseline.csv \
        results/tables/s2_summary_focal.csv \
        results/tables/tau_values.json \
        results/tables/loao_per_fold.csv \
        results/tables/loao_summary.csv \
        results/tables/comparison_table.csv \
        results/tables/comparison_table.md; do
        [ -f "$f" ] && _ok "$f"
    done
    echo ""
    echo "  [그림]"
    for f in \
        results/figures/cm_s2_norm.png \
        results/figures/cm_s2_6class.png \
        results/figures/cm_s3_7class.png \
        results/figures/mse_histogram.png \
        results/figures/roc_cae.png \
        results/figures/loao_bar_chart.png \
        results/figures/unknown_case_4panel.png; do
        [ -f "$f" ] && _ok "$f"
    done
    echo ""
}

# ---------------------------------------------------------------------------
# 커맨드 파싱
# ---------------------------------------------------------------------------

CMD="${1:-help}"
SKIP_PREPROCESS=false
SKIP_TRAIN=false

shift || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-preprocess) SKIP_PREPROCESS=true ;;
        --skip-train)      SKIP_TRAIN=true ;;
        *) echo "[WARN] 알 수 없는 옵션: $1" ;;
    esac
    shift
done

# ---------------------------------------------------------------------------
# 커맨드 실행
# ---------------------------------------------------------------------------

case "$CMD" in

    all)
        _step "전체 파이프라인 시작"
        if [ "$SKIP_PREPROCESS" = false ]; then
            step_preprocess ""
        else
            _warn "--skip-preprocess: 전처리 건너뜀"
            _check_file "data/processed/dataset_train.npz"
            _check_file "data/processed/split_manifest.json"
        fi
        if [ "$SKIP_TRAIN" = false ]; then
            step_train ""
        else
            _warn "--skip-train: 학습 건너뜀"
            _check_file "results/checkpoints/cae_best.pth"
        fi
        step_loao ""
        step_visualize
        _print_summary
        ;;

    preprocess)
        step_preprocess ""
        ;;

    train)
        _check_file "data/processed/split_manifest.json"
        step_train ""
        ;;

    loao)
        _check_file "results/checkpoints/cae_best.pth"
        _check_file "results/checkpoints/s2_seed_0_best.pth"
        step_loao ""
        step_visualize
        ;;

    visualize)
        step_visualize
        ;;

    smoke)
        _step "Smoke test 시작 (1 epoch, seed 0)"
        step_preprocess "smoke"
        step_train "smoke"
        step_loao "smoke"
        step_visualize
        step_test
        _print_summary
        ;;

    test)
        step_test
        ;;

    help|--help|-h|*)
        echo ""
        echo "사용법: bash scripts/run.sh <command> [옵션]"
        echo ""
        echo "Commands:"
        echo "  all        전체 파이프라인 (전처리 → S1+S2+CAE 학습 → LOAO → 시각화)"
        echo "  preprocess 전처리만 (NPZ 생성 + split manifest)"
        echo "  train      S1 + S2 + CAE 학습"
        echo "  loao       LOAO 실험 + 3종 비교표 + 시각화"
        echo "  visualize  시각화 4종 (체크포인트 필요)"
        echo "  smoke      전체 smoke test (1 epoch, seed 0)"
        echo "  test       pytest tests/ 실행"
        echo ""
        echo "Options (all 전용):"
        echo "  --skip-preprocess  NPZ/manifest 이미 있을 때 전처리 건너뜀"
        echo "  --skip-train       checkpoint 이미 있을 때 학습 건너뜀"
        echo ""
        echo "예시:"
        echo "  bash scripts/run.sh all                     # 처음부터 전체 실행"
        echo "  bash scripts/run.sh all --skip-preprocess   # NPZ 있을 때"
        echo "  bash scripts/run.sh all --skip-train        # checkpoint 있을 때"
        echo "  bash scripts/run.sh smoke                   # 빠른 동작 확인"
        echo "  bash scripts/run.sh loao                    # LOAO만 재실행"
        echo ""
        [ "$CMD" = "help" ] || [ "$CMD" = "--help" ] || [ "$CMD" = "-h" ] && exit 0
        exit 1
        ;;
esac
