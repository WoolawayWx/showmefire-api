#!/bin/bash

# Secondary 9z HRRR forecast run. Mirrors forecasts.sh but pinned to the 9z
# cycle (FORECAST_CYCLE_HOUR=9), with every output suffixed _09z so it never
# collides with the 12z run's files, and with NO Discord/mobile notification
# and no PerCounty/beta steps.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR" || exit 1

if [ -f "/opt/venv/bin/python" ]; then
    PYTHON="/opt/venv/bin/python" # Docker production
elif [ -f "$PROJECT_DIR/venv/bin/python" ]; then
    PYTHON="$PROJECT_DIR/venv/bin/python" # Local development
else
    PYTHON="python" # System fallback
fi

if ! "$PYTHON" --version > /dev/null 2>&1; then
    echo "ERROR: Python not found or not working at $PYTHON"
    exit 1
fi

export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"

# Pin this whole run to the 9z HRRR cycle and suffix every generated artifact
# so it can never overwrite the operational 12z run's files.
export FORECAST_CYCLE_HOUR=9
export FORECAST_IMAGE_SUFFIX=_09z
export FORECAST_STATUS_KEY=ForecastFireDanger09z
export CDN_TEST_PREFIX=09z

mkdir -p "$PROJECT_DIR/logs"
LOG_FILE="$PROJECT_DIR/logs/forecast_09z_$(date +\%Y\%m\%d).log"

echo "=== Starting 9z fire danger forecast at $(date) ===" >> "$LOG_FILE" 2>&1
echo "Running from: $PROJECT_DIR" >> "$LOG_FILE" 2>&1
echo "Using Python: $PYTHON" >> "$LOG_FILE" 2>&1

run_step() {
    local step_name="$1"
    local script_path="$2"

    echo "=== $step_name ===" >> "$LOG_FILE" 2>&1
    "$PYTHON" "$script_path" >> "$LOG_FILE" 2>&1
    local step_exit=$?
    if [ $step_exit -ne 0 ]; then
        echo "=== $step_name FAILED with exit code $step_exit at $(date) ===" >> "$LOG_FILE" 2>&1
        return $step_exit
    fi
    return 0
}

run_step "Running 9z Daily Forecast" "$PROJECT_DIR/forecast/DailyForecast.py"
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "=== FAILED at $(date) with exit code $EXIT_CODE ===" >> "$LOG_FILE" 2>&1
    exit $EXIT_CODE
fi

run_step "Running 9z AI Text Generation" "$PROJECT_DIR/forecast/forecast_ai.py"
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "=== FAILED at $(date) with exit code $EXIT_CODE ===" >> "$LOG_FILE" 2>&1
    exit $EXIT_CODE
fi

# Comparison is best-effort: a missing same-day 12z archive should not fail
# an otherwise-successful 9z run (it exits 0 and logs when that happens).
run_step "Comparing 9z run against same-day 12z run" "$PROJECT_DIR/scripts/compare_09z_12z.py"

echo "=== Completed successfully at $(date) ===" >> "$LOG_FILE" 2>&1
exit 0
