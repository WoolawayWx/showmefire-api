#!/bin/bash

# Daily ML model training script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create log directory
mkdir -p "$LOG_DIR"

# Log file
LOG_FILE="$LOG_DIR/training_$TIMESTAMP.log"

# Log everything
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================"
echo "Training Started: $(date)"
echo "========================================"

cd "$PROJECT_DIR"

source ./venv/bin/activate

python forecast/forecastverification.py

deactivate

echo "========================================"
echo "Training Completed: $(date)"
echo "========================================"

# Keep only last 30 days of logs
find "$LOG_DIR" -name "training_*.log" -mtime +30 -delete