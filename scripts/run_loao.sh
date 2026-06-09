#!/usr/bin/env bash
# Run the full LOAO experiment pipeline.
# Usage: bash scripts/run_loao.sh [--smoke]
#
# --smoke: 1 fold × 1 seed × 1 epoch (fast sanity check)

set -euo pipefail
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/ms-test-matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TMPDIR:-/tmp}/ms-test-cache}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${ROOT}/.venv/bin/python"
if [ -n "${TOW_IDS_PYTHON:-}" ]; then
    PYTHON="$TOW_IDS_PYTHON"
elif [ ! -x "$PYTHON" ]; then
    PYTHON="python3"
fi

SMOKE=""
for arg in "$@"; do
    if [ "$arg" = "--smoke" ]; then
        SMOKE="--smoke"
    fi
done

if [ -n "$SMOKE" ]; then
    export TOW_IDS_TRAIN_NPZ="data/smoke/dataset_train.npz"
    export TOW_IDS_TEST_NPZ="data/smoke/dataset_test.npz"
    export TOW_IDS_MANIFEST="data/smoke/split_manifest.json"
    export TOW_IDS_OUTPUT_DIR="results/smoke"
    export TOW_IDS_CONF_THR_ARTIFACT="results/smoke/tables/conf_threshold.json"
    export TOW_IDS_USE_CAE=true
    "$PYTHON" scripts/make_stub_dataset.py --out-dir data/smoke --smoke-layout --force
    "$PYTHON" -m src.utils.split \
      --train-npz "$TOW_IDS_TRAIN_NPZ" --test-npz "$TOW_IDS_TEST_NPZ" \
      --out "$TOW_IDS_MANIFEST" --val-ratio 0.10 --seed 42
    "$PYTHON" -m src.train.train_s1 --smoke
    "$PYTHON" -m src.train.train_s2 --smoke
    "$PYTHON" -m src.train.train_cae --smoke
fi

echo "=========================================="
echo " LOAO Experiment"
echo "=========================================="

VALIDATE_FLAG=""
[ -n "$SMOKE" ] && VALIDATE_FLAG="--smoke"
"$PYTHON" scripts/validate_artifacts.py $VALIDATE_FLAG
USE_CAE="$("$PYTHON" -c 'from src.utils.config import load_experiment_config; print("true" if bool(load_experiment_config().use_cae) else "false")')"

# Step 1: Run LOAO experiment
echo ""
echo "[1/4] Running LOAO experiment..."
"$PYTHON" -m experiments.leave_one_out $SMOKE

# Step 2: Comparison table
echo ""
echo "[2/4] Building comparison table..."
"$PYTHON" scripts/comparison_table.py $SMOKE

# Step 3: Bar chart
echo ""
echo "[3/4] Plotting LOAO bar chart..."
"$PYTHON" scripts/plot_loao_bar.py

# Step 4: Unknown case visualization
echo ""
if [ "$USE_CAE" = "true" ]; then
    echo "[4/4] Plotting unknown case 4-panel..."
    "$PYTHON" scripts/plot_unknown_case.py
else
    echo "[4/4] CAE disabled; skipping reconstruction figure."
fi

echo ""
echo "=========================================="
echo " All done."
echo " Results:"
OUT_DIR="${TOW_IDS_OUTPUT_DIR:-results}"
echo "   $OUT_DIR/tables/loao_per_fold.csv"
echo "   $OUT_DIR/tables/loao_summary.csv"
echo "   $OUT_DIR/tables/comparison_table.csv"
echo "   $OUT_DIR/tables/comparison_table.md"
echo "   $OUT_DIR/figures/loao_bar_chart.png"
if [ "$USE_CAE" = "true" ]; then
    echo "   $OUT_DIR/figures/unknown_case_4panel.png"
fi
echo "=========================================="
