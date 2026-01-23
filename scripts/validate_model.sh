#!/bin/bash

# validation_pipeline.sh
# Runs the full Model Validation Suite: Comparison -> Plotting -> History Tracking
# Creates both 'latest' output and a dated archive.

# Exit on error
set -e

# Get the directory where this script is located
SCRIPT_DIR=$(dirname "$0")

# Define output paths
BASE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
REPORTS_DIR="$BASE_DIR/reports"
CSV_FILE="$REPORTS_DIR/forecast_comparison_latest.csv"

# Define Date for Archive
DATE_STAMP=$(date +%Y-%m-%d)
ARCHIVE_DIR="$REPORTS_DIR/$DATE_STAMP"
LOG_DIR="$BASE_DIR/logs"
LOG_FILE="$LOG_DIR/validation_$DATE_STAMP.log"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Redirect stdout and stderr to both console and log file
exec > >(tee -a "$LOG_FILE") 2>&1

# Activate the virtual environment
source "$BASE_DIR/venv/bin/activate"

echo "=================================================="
echo "   Show Me Fire - Model Validation Pipeline"
echo "=================================================="
echo "Date: $(date)"
echo "Base Directory: $BASE_DIR"
echo "Reports Output: $REPORTS_DIR"
echo "Archive Output: $ARCHIVE_DIR"
echo "Log File: $LOG_FILE"
echo "--------------------------------------------------"

# 1. Compare Forecasts vs Observations
echo ""
echo "[Step 1/4] Running comparison (Forecast vs Observations)..."
echo "   - Comparing Temp, RH, Fuel Moisture, and Wind Speed"
# Ensure reports dir exists
mkdir -p "$REPORTS_DIR"
python3 "$SCRIPT_DIR/compare_forecasts.py" --output "$CSV_FILE"

# 2. Generate Visualizations
echo ""
echo "[Step 2/4] Generating performance plots..."
echo "   - Creating Scatter Plots, Error Histograms, and Categorical Heatmap"
python3 "$SCRIPT_DIR/generate_performance_plots.py" "$CSV_FILE"

# 3. Update History
echo ""
echo "[Step 3/4] Updating rolling history..."
echo "   - Tracking 7-day and 30-day trends"
python3 "$SCRIPT_DIR/update_validation_history.py" --input "$CSV_FILE"

# 4. Archive Results
echo ""
echo "[Step 4/4] Archiving results to $DATE_STAMP folder..."
mkdir -p "$ARCHIVE_DIR/plots"

# Copy CSV
echo "   - Copying comparison data..."
cp "$CSV_FILE" "$ARCHIVE_DIR/forecast_comparison.csv"

# Copy Plots
echo "   - Copying visualizations..."
if [ -d "$REPORTS_DIR/plots" ]; then
    cp "$REPORTS_DIR/plots/"*.png "$ARCHIVE_DIR/plots/"
else
    echo "Warning: No plots found to archive."
fi

echo ""
echo "=================================================="
echo "SUCCESS"
echo "--------------------------------------------------"
echo "Latest:   $REPORTS_DIR/forecast_comparison_latest.csv"
echo "          $REPORTS_DIR/plots/"
echo "Archived: $ARCHIVE_DIR/"
echo "=================================================="
