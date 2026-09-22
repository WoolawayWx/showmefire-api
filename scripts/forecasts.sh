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

run_step "Running Daily Forecast" "$PROJECT_DIR/forecast/DailyForecast.py"
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "=== FAILED at $(date) with exit code $EXIT_CODE ===" >> "$LOG_FILE" 2>&1
    exit $EXIT_CODE
fi

# The 09z comparison must run after this operational 12z archive exists.
# It is observational/reporting-only and must not block forecast publication.
run_step "Comparing 9z run against same-day 12z run" "$PROJECT_DIR/scripts/compare_09z_12z.py" || \
  echo "WARNING: 9z comparison generation failed at $(date)" >> "$LOG_FILE" 2>&1

run_step "Running AI Text Generation" "$PROJECT_DIR/forecast/forecast_ai.py"
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "=== FAILED at $(date) with exit code $EXIT_CODE ===" >> "$LOG_FILE" 2>&1
    exit $EXIT_CODE
fi

# Per-county maps are a supplementary product, not the core forecast. A
# failure here must not suppress the "forecast ready" notifications below.
run_step "Updating Per County Maps" "$PROJECT_DIR/forecast/PerCounty.py"
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "WARNING: Updating Per County Maps failed at $(date) with exit code $EXIT_CODE - continuing so forecast notifications still go out" >> "$LOG_FILE" 2>&1
fi

echo "=== Sending forecast maps to Discord ===" >> "$LOG_FILE" 2>&1
"$PYTHON" "$PROJECT_DIR/scripts/notify_forecast_complete.py" >> "$LOG_FILE" 2>&1 || \
  echo "WARNING: Forecast Discord completion notification failed at $(date)" >> "$LOG_FILE" 2>&1

if [ $EXIT_CODE -eq 0 ]; then
    echo "=== Completed successfully at $(date) ===" >> "$LOG_FILE" 2>&1
else
    echo "=== FAILED at $(date) with exit code $EXIT_CODE ===" >> "$LOG_FILE" 2>&1
fi

# Beta must never be able to touch production's own state - without these,
# DailyForecast_ModelFD.py falls back to its own unsafe defaults (writes
# under PROJECT_DIR, FORECAST_STATUS_KEY='ForecastFireDanger' - the SAME
# key the production run above just wrote - FORECAST_WRITE_DATABASE=True,
# MODEL_SHADOW_ENABLED=True), silently overwriting the operational status
# and re-triggering shadow scoring a second time every day. Same isolation
# services/forecast_jobs.py's BETA_FORECAST_ENV already applies to the
# admin-triggered beta run - kept in sync with that dict.
export FORECAST_OUTPUT_ROOT="$("$PYTHON" -c "from services.beta_products import BETA_ROOT; print(BETA_ROOT / 'forecast')")"
export FORECAST_STATUS_KEY="ForecastFireDangerBeta"
export FORECAST_WRITE_DATABASE="false"
export MODEL_SHADOW_ENABLED="false"
export uploadForecast="false"

run_step "Running BETA Forecast" "$PROJECT_DIR/forecast/DailyForecast_ModelFD.py"
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "=== FAILED at $(date) with exit code $EXIT_CODE ===" >> "$LOG_FILE" 2>&1
    exit $EXIT_CODE
fi

exit $EXIT_CODE
