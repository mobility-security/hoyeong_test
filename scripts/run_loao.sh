#!/usr/bin/env bash
# Run the full LOAO experiment pipeline.
# Usage: bash scripts/run_loao.sh [--smoke]
#
# --smoke: 1 fold × 1 seed × 1 epoch (fast sanity check)

set -euo pipefail

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
fi

echo "=========================================="
echo " LOAO Experiment"
echo "=========================================="

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
echo "[4/4] Plotting unknown case 4-panel..."
"$PYTHON" scripts/plot_unknown_case.py

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
echo "   $OUT_DIR/figures/unknown_case_4panel.png"
echo "=========================================="
