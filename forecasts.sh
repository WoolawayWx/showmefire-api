#!/bin/bash

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
LOG_FILE="$PROJECT_DIR/logs/forecast_$(date +\%Y\%m\%d).log"

# Run the script with full logging
echo "=== Starting fire danger forecast at $(date) ===" >> "$LOG_FILE" 2>&1
echo "Running from: $PROJECT_DIR" >> "$LOG_FILE" 2>&1
echo "Using Python: $PYTHON" >> "$LOG_FILE" 2>&1

"$PYTHON" "$PROJECT_DIR/forecast/forecastedfiredanger.py" >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "=== Completed successfully at $(date) ===" >> "$LOG_FILE" 2>&1
else
    echo "=== FAILED at $(date) with exit code $EXIT_CODE ===" >> "$LOG_FILE" 2>&1
fi

exit $EXIT_CODE