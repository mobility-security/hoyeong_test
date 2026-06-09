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
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/ms-test-matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TMPDIR:-/tmp}/ms-test-cache}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TRAIN_NPZ="${TOW_IDS_TRAIN_NPZ:-data/processed/dataset_train.npz}"
TEST_NPZ="${TOW_IDS_TEST_NPZ:-data/processed/dataset_test.npz}"
MANIFEST="${TOW_IDS_MANIFEST:-data/processed/split_manifest.json}"
OUTPUT_DIR="${TOW_IDS_OUTPUT_DIR:-results}"

# Python 실행 파일 자동 탐지 (명시적 override → venv → system)
if [ -n "${TOW_IDS_PYTHON:-}" ]; then
    PY="$TOW_IDS_PYTHON"
elif [ -x "$ROOT/.venv/bin/python" ]; then
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

_use_cae() {
    "$PY" -c 'from src.utils.config import load_experiment_config; print("true" if bool(load_experiment_config().use_cae) else "false")'
}

_validate_data_contract() {
    "$PY" -c 'from src.utils.config import load_experiment_config; from src.utils.io import load_dataset; from src.train.common import load_manifest; c=load_experiment_config(); xt,_,_=load_dataset(c.train_npz_path); xv,_,_=load_dataset(c.test_npz_path); load_manifest(c.manifest_path,len(xt),len(xv),train_npz_path=c.train_npz_path,test_npz_path=c.test_npz_path); print("  [OK] dataset/manifest provenance")'
}

_check_training_artifacts() {
    "$PY" scripts/validate_artifacts.py
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
        _warn "Smoke 모드: 격리된 train/test stub dataset 생성"
        "$PY" scripts/make_stub_dataset.py \
          --out-dir data/smoke --smoke-layout --force
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

    _check_file "$TRAIN_NPZ"
    _check_file "$TEST_NPZ"

    # split manifest 생성
    _step "Split manifest 생성 (capture/시간 구간 누수 방지)"
    if [ -n "$SMOKE" ]; then
        "$PY" -m src.utils.split \
          --train-npz "$TRAIN_NPZ" \
          --test-npz  "$TEST_NPZ" \
          --out       "$MANIFEST" \
          --val-ratio 0.10 \
          --seed 42 \
          --guard-gap-packets 0
    else
        "$PY" -m src.utils.split \
          --train-npz "$TRAIN_NPZ" \
          --test-npz  "$TEST_NPZ" \
          --out       "$MANIFEST" \
          --val-ratio 0.10 \
          --guard-gap-packets 64 \
          --seed 42
    fi

    _check_file "$MANIFEST"
    _ok "전처리 완료."
}

# ── 학습 ────────────────────────────────────────────────────────────────────

step_train() {
    local SMOKE="${1:-}"
    local SMOKE_FLAG=""
    [ -n "$SMOKE" ] && SMOKE_FLAG="--smoke"

    _step "S1 — 이진 분류 베이스라인 학습"
    "$PY" -m src.train.train_s1 $SMOKE_FLAG
    _check_file "$OUTPUT_DIR/tables/s1_baseline.csv"

    _step "S2 — 6-class Focal Loss 분류 학습"
    "$PY" -m src.train.train_s2 $SMOKE_FLAG
    _check_file "$OUTPUT_DIR/checkpoints/s2_seed_0_best.pth"
    _check_file "$OUTPUT_DIR/tables/s2_summary_focal.csv"

    if [ "$(_use_cae)" = "true" ]; then
        _step "CAE — 정상 트래픽 재구성 + tau 산출"
        "$PY" -m src.train.train_cae $SMOKE_FLAG
        _check_file "$OUTPUT_DIR/checkpoints/cae_best.pth"
        _check_file "$OUTPUT_DIR/tables/tau_values.json"
    else
        _warn "experiment.use_cae=false: CAE 학습을 건너뜀"
    fi

    _ok "학습 완료."
}

# ── LOAO ────────────────────────────────────────────────────────────────────

step_loao() {
    local SMOKE="${1:-}"
    local SMOKE_FLAG=""
    [ -n "$SMOKE" ] && SMOKE_FLAG="--smoke"

    _step "LOAO 제로데이 평가 (5 fold × 5 seed)"
    "$PY" -m experiments.leave_one_out $SMOKE_FLAG
    _check_file "$OUTPUT_DIR/tables/loao_per_fold.csv"
    _check_file "$OUTPUT_DIR/tables/loao_summary.csv"

    _step "비교표 생성 (활성화된 S1 / S2 / S3)"
    "$PY" scripts/comparison_table.py $SMOKE_FLAG
    _check_file "$OUTPUT_DIR/tables/comparison_table.csv"
    _check_file "$OUTPUT_DIR/tables/comparison_table.md"

    _ok "LOAO 완료."
}

# ── 시각화 ──────────────────────────────────────────────────────────────────

step_visualize() {
    _step "시각화 — LOAO bar chart"
    "$PY" scripts/plot_loao_bar.py
    _check_file "$OUTPUT_DIR/figures/loao_bar_chart.png"

    if [ "$(_use_cae)" = "true" ]; then
        _step "시각화 — Unknown case 4-panel"
        "$PY" scripts/plot_unknown_case.py
        _check_file "$OUTPUT_DIR/figures/unknown_case_4panel.png"
    else
        _warn "experiment.use_cae=false: CAE Unknown 사례 그림을 건너뜀"
    fi

    _step "시각화 — S2/S3 Confusion Matrix"
    "$PY" scripts/plot_confusion_matrix.py
    _check_file "$OUTPUT_DIR/figures/cm_s2_6class.png"
    if [ "$(_use_cae)" = "true" ]; then
        _check_file "$OUTPUT_DIR/figures/cm_s3_7class.png"
    fi

    _ok "시각화 완료."
}

# ── pytest ──────────────────────────────────────────────────────────────────

step_test() {
    _step "pytest — 전체 테스트"
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
    [ -f "$TRAIN_NPZ" ] && _ok "$TRAIN_NPZ"
    [ -f "$MANIFEST"  ] && _ok "$MANIFEST"
    echo ""
    echo "  [체크포인트]"
    [ -f "$OUTPUT_DIR/checkpoints/s2_seed_0_best.pth" ] && _ok "$OUTPUT_DIR/checkpoints/s2_seed_*.pth"
    [ -f "$OUTPUT_DIR/checkpoints/cae_best.pth"        ] && _ok "$OUTPUT_DIR/checkpoints/cae_best.pth"
    echo ""
    echo "  [테이블]"
    for f in \
        "$OUTPUT_DIR/tables/s1_baseline.csv" \
        "$OUTPUT_DIR/tables/s2_summary_focal.csv" \
        "$OUTPUT_DIR/tables/tau_values.json" \
        "$OUTPUT_DIR/tables/loao_per_fold.csv" \
        "$OUTPUT_DIR/tables/loao_summary.csv" \
        "$OUTPUT_DIR/tables/comparison_table.csv" \
        "$OUTPUT_DIR/tables/comparison_table.md"; do
        [ -f "$f" ] && _ok "$f"
    done
    echo ""
    echo "  [그림]"
    for f in \
        "$OUTPUT_DIR/figures/cm_s2_norm.png" \
        "$OUTPUT_DIR/figures/cm_s2_6class.png" \
        "$OUTPUT_DIR/figures/cm_s3_7class.png" \
        "$OUTPUT_DIR/figures/mse_histogram.png" \
        "$OUTPUT_DIR/figures/roc_cae.png" \
        "$OUTPUT_DIR/figures/loao_bar_chart.png" \
        "$OUTPUT_DIR/figures/unknown_case_4panel.png"; do
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
            _check_file "$TRAIN_NPZ"
            _check_file "$MANIFEST"
            _validate_data_contract
        fi
        if [ "$SKIP_TRAIN" = false ]; then
            step_train ""
        else
            _warn "--skip-train: 학습 건너뜀"
            _check_training_artifacts
        fi
        step_loao ""
        step_visualize
        step_test
        _print_summary
        ;;

    preprocess)
        step_preprocess ""
        ;;

    train)
        _check_file "$MANIFEST"
        step_train ""
        ;;

    loao)
        _check_training_artifacts
        step_loao ""
        step_visualize
        ;;

    visualize)
        step_visualize
        ;;

    smoke)
        _step "Smoke test 시작 (1 epoch, seed 0)"
        TRAIN_NPZ="data/smoke/dataset_train.npz"
        TEST_NPZ="data/smoke/dataset_test.npz"
        MANIFEST="data/smoke/split_manifest.json"
        OUTPUT_DIR="results/smoke"
        export TOW_IDS_TRAIN_NPZ="$TRAIN_NPZ"
        export TOW_IDS_TEST_NPZ="$TEST_NPZ"
        export TOW_IDS_MANIFEST="$MANIFEST"
        export TOW_IDS_OUTPUT_DIR="$OUTPUT_DIR"
        export TOW_IDS_CONF_THR_ARTIFACT="$OUTPUT_DIR/tables/conf_threshold.json"
        export TOW_IDS_USE_CAE=true
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
