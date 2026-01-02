#!/bin/bash

# Daily ML model training script

set -e

# Detect project directory from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

# Change to project directory
cd "$PROJECT_DIR" || exit 1

# Use Python from venv (no activation needed)
PYTHON="$PROJECT_DIR/venv/bin/python"

# Verify Python exists
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: Python not found at $PYTHON"
    exit 1
fi

# Set up PATH
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_DIR/logs"

# Log file with date
LOG_FILE="$PROJECT_DIR/logs/training_$(date +\%Y\%m\%d).log"

# Run the script with full logging
echo "========================================" >> "$LOG_FILE" 2>&1
echo "Training Started: $(date)" >> "$LOG_FILE" 2>&1
echo "========================================" >> "$LOG_FILE" 2>&1

"$PYTHON" "$PROJECT_DIR/forecast/forecastverification.py" >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

echo "========================================" >> "$LOG_FILE" 2>&1
echo "Training Completed: $(date)" >> "$LOG_FILE" 2>&1
echo "========================================" >> "$LOG_FILE" 2>&1

# Keep only last 30 days of logs
find "$PROJECT_DIR/logs" -name "training_*.log" -mtime +30 -delete

exit $EXIT_CODE