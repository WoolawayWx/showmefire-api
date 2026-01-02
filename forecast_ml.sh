#!/bin/bash

# ==============================================================================
# Daily ML Model Pipeline: Fetch -> Verify -> Train
# ==============================================================================

# Exit on any error
set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
PYTHON="$PROJECT_DIR/venv/bin/python"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/training_$(date +%Y%m%d).log"
API_URL="https://api.showmefire.org/api/historical/archive/save?days_back=1"

# Create logs directory if missing
mkdir -p "$LOG_DIR"

# Ensure we are in the right directory
cd "$PROJECT_DIR"

# Start logging
exec > >(tee -a "$LOG_FILE") 2>&1

echo "------------------------------------------------------------"
echo "PIPELINE STARTED: $(date)"
echo "------------------------------------------------------------"

# 1. DATA INGESTION
echo "[1/3] Fetching new data from Synoptic API..."
if curl -f -X POST "$API_URL"; then
    echo "Successfully archived new data."
else
    echo "ERROR: Data fetch failed. Check network or API status."
    exit 1
fi

# 2. FORECAST VERIFICATION
# Targets the 12z (noon) run from the current day
MODEL_RUN="$(date +%Y%m%d)_12"
echo "[2/3] Verifying today's 12z forecast ($MODEL_RUN)..."
"$PYTHON" "$PROJECT_DIR/forecastverification.py" --verify-forecast "$MODEL_RUN"

# 3. MODEL TRAINING
echo "[3/3] Starting ML model training..."
# Note: Fixed path (removed /forecast/) as per your file structure
"$PYTHON" "$PROJECT_DIR/forecastverification.py" --train

# CLEANUP
echo "Cleaning up logs older than 30 days..."
find "$LOG_DIR" -name "training_*.log" -mtime +30 -delete

echo "------------------------------------------------------------"
echo "PIPELINE COMPLETED: $(date)"
echo "------------------------------------------------------------"