#!/bin/bash

# Detect project directory from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR" || exit 1

# Detect correct Python executable
if [ -f "/opt/venv/bin/python" ]; then
    PYTHON="/opt/venv/bin/python" # Docker production
elif [ -f "$PROJECT_DIR/venv/bin/python" ]; then
    PYTHON="$PROJECT_DIR/venv/bin/python" # Local development
else
    PYTHON="python" # System fallback
fi

# Verify Python works
if ! "$PYTHON" --version > /dev/null 2>&1; then
    echo "ERROR: Python not found or not working at $PYTHON"
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

"$PYTHON" "$PROJECT_DIR/forecast/DailyForecast_RRFS.py" >> "$LOG_FILE" 2>&1
EXIT_CODE=$?


if [ $EXIT_CODE -eq 0 ]; then
    echo "=== Completed successfully at $(date) ===" >> "$LOG_FILE" 2>&1
else
    echo "=== FAILED at $(date) with exit code $EXIT_CODE ===" >> "$LOG_FILE" 2>&1
fi

exit $EXIT_CODE