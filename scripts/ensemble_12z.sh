#!/bin/bash

# Ensemble fire danger (BETA) for the 12z cycle: HREF-style multi-model plus the
# REFS/HREF ensprod probabilities and a categorical ensemble forecast.
#
# This used to be a step inside scripts/forecasts.sh, right after the daily
# forecast. The forecast now starts as soon as its HRRR hours are out (~13:15Z),
# but this needs REFS (~14:36Z) and HREF (~15:06Z), so it has its own cron entry
# at 15:00Z (see the Dockerfile). Like the old step it only reads this run's
# fuel-moisture hand-off if the forecast has produced it, takes the same
# forecast lock, and is advisory only: it never blocks or fails anything else.

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

if ! "$PYTHON" --version > /dev/null 2>&1; then
    echo "ERROR: Python not found or not working at $PYTHON"
    exit 1
fi

export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
mkdir -p "$PROJECT_DIR/logs"
LOG_FILE="$PROJECT_DIR/logs/ensemble_12z_$(date +%Y%m%d).log"

# Wait for the daily forecast (and a 09z run) to finish rather than compete with them,
# then hold the lock so maps.sh yields CPU, as it did when this ran inside forecasts.sh.
source "$SCRIPT_DIR/forecast_lock.sh"
take_forecast_lock

echo "=== Starting 12z ensemble fire danger (beta) at $(date) ===" >> "$LOG_FILE" 2>&1
echo "Running from: $PROJECT_DIR" >> "$LOG_FILE" 2>&1
echo "Using Python: $PYTHON" >> "$LOG_FILE" 2>&1

"$PYTHON" "$PROJECT_DIR/scripts/run_ensemble_fire_danger.py" >> "$LOG_FILE" 2>&1
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "WARNING: Ensemble fire danger (beta) generation failed at $(date) with exit code $EXIT_CODE" >> "$LOG_FILE" 2>&1
else
    echo "=== Completed successfully at $(date) ===" >> "$LOG_FILE" 2>&1
fi

# Advisory only: a failure is logged above and must not turn the cron job red.
exit 0
