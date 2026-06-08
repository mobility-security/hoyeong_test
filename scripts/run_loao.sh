#!/usr/bin/env bash
# Run the full LOAO experiment pipeline.
# Usage: bash scripts/run_loao.sh [--smoke]
#
# --smoke: 1 fold × 1 seed × 1 epoch (fast sanity check)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${ROOT}/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    PYTHON="python3"
fi

SMOKE=""
for arg in "$@"; do
    if [ "$arg" = "--smoke" ]; then
        SMOKE="--smoke"
    fi
done

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
echo "   results/tables/loao_per_fold.csv"
echo "   results/tables/loao_summary.csv"
echo "   results/tables/comparison_table.csv"
echo "   results/tables/comparison_table.md"
echo "   results/figures/loao_bar_chart.png"
echo "   results/figures/unknown_case_4panel.png"
echo "=========================================="
