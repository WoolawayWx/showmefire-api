#!/bin/bash

# Day 2 and Day 3 fire danger forecast (BETA) for the 12z cycle - see forecast/ExtendedForecast.py and
# docs/extended_forecast.md.
#
# Day 2 reads HRRR f28-f39 and Day 3 reads RRFS f52-f63 of the 12z run. HRRR's f39 is out about 2-3 h after the cycle
# and RRFS's f63 about 5-6 h after, so this is started once the daily forecast is done and the script itself keeps
# retrying (--wait-minutes) until each run has published the hours it needs. Advisory only: nothing depends on it, it
# holds the forecast lock only while it runs so maps.sh yields CPU, and it writes nothing Day 1 reads.
#
# Not scheduled yet. To schedule it, add to the Dockerfile cron block next to ensemble_12z:
#   echo "30 15 * * * root /bin/bash /app/scripts/run_job.sh extended_forecast /app/scripts/extended_forecast.sh >> /app/logs/cron_extended.log 2>&1"

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
mkdir -p "$PROJECT_DIR/logs"
LOG_FILE="$PROJECT_DIR/logs/extended_forecast_$(date +%Y%m%d).log"

source "$SCRIPT_DIR/forecast_lock.sh"
take_forecast_lock

echo "=== Starting Day 2/3 extended forecast (beta) at $(date) ===" >> "$LOG_FILE" 2>&1
echo "Running from: $PROJECT_DIR" >> "$LOG_FILE" 2>&1
echo "Using Python: $PYTHON" >> "$LOG_FILE" 2>&1

"$PYTHON" "$PROJECT_DIR/forecast/ExtendedForecast.py" --wait-minutes "${SMF_EXTENDED_WAIT_MINUTES:-150}" >> "$LOG_FILE" 2>&1
EXIT_CODE=$?
if [ "$EXIT_CODE" -eq 0 ]; then
    echo "=== Completed successfully at $(date) ===" >> "$LOG_FILE" 2>&1
else
    echo "=== FAILED at $(date) with exit code $EXIT_CODE ===" >> "$LOG_FILE" 2>&1
    [ -n "${JOB_SUMMARY_FILE:-}" ] && echo "extended forecast exit code $EXIT_CODE; see logs/extended_forecast_$(date +%Y%m%d).log" > "$JOB_SUMMARY_FILE"
fi
exit "$EXIT_CODE"
