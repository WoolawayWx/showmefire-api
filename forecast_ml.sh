#!/bin/bash

# ==============================================================================
# Daily ML Model Pipeline: Fetch -> Verify -> Train
# ==============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
PYTHON="$PROJECT_DIR/venv/bin/python"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/training_$(date +%Y%m%d).log"
API_URL="https://api.showmefire.org/api/historical/archive/save?days_back=1"

# The actual path to your script
VERIFY_SCRIPT="$PROJECT_DIR/forecast/forecastverification.py"

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

# This ensures Python can see modules in the root folder while running inside /forecast/
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "------------------------------------------------------------"
echo "PIPELINE STARTED: $(date)"
echo "------------------------------------------------------------"

# 1. DATA INGESTION
echo "[1/3] Fetching new data from Synoptic API..."
curl -f -X POST "$API_URL"

# 2. FORECAST VERIFICATION
MODEL_RUN="$(date +%Y%m%d)_12"
echo "[2/3] Verifying today's 12z forecast ($MODEL_RUN)..."
"$PYTHON" "$VERIFY_SCRIPT" --verify-forecast "$MODEL_RUN"

# 3. MODEL TRAINING
echo "[3/3] Starting ML model training..."
"$PYTHON" "$VERIFY_SCRIPT" --train

# CLEANUP
find "$LOG_DIR" -name "training_*.log" -mtime +30 -delete

echo "------------------------------------------------------------"
echo "PIPELINE COMPLETED: $(date)"
echo "------------------------------------------------------------"