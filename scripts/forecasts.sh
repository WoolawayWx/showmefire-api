#!/bin/bash

# Detect project directory from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR" || exit 1

# The forecast lock (scripts/forecast_lock.sh) is taken further down, after the
# wait for HRRR, so maps.sh keeps running while this script is only polling.

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

# Health ledger (services/forecast_health.py): stage results and Discord alerts for this run. Python steps read
# SMF_FORECAST_JOB to file their own results under it; the EXIT trap closes the run so a hung one stands out.
export SMF_FORECAST_JOB=forecasts_12z
health() { (cd "$PROJECT_DIR" && PYTHONPATH="$PROJECT_DIR" "$PYTHON" -m services.forecast_health "$@") >> "$LOG_FILE" 2>&1 || true; }
health begin
trap 'health finish' EXIT

# Run the script with full logging
echo "=== Starting fire danger forecast at $(date) ===" >> "$LOG_FILE" 2>&1
echo "Running from: $PROJECT_DIR" >> "$LOG_FILE" 2>&1
echo "Using Python: $PYTHON" >> "$LOG_FILE" 2>&1

run_step() {
    local step_name="$1"
    local script_path="$2"
    local severity="${3:-failed}"   # "degraded" for steps the run carries on without

    echo "=== $step_name ===" >> "$LOG_FILE" 2>&1
    "$PYTHON" "$script_path" >> "$LOG_FILE" 2>&1
    local step_exit=$?
    if [ $step_exit -ne 0 ]; then
        echo "=== $step_name FAILED with exit code $step_exit at $(date) ===" >> "$LOG_FILE" 2>&1
        health stage "$step_name" "$severity" "exited with code $step_exit" --log "$LOG_FILE"
        return $step_exit
    fi

    return 0
}

# Start when the 12z HRRR hours this forecast uses (f04-f15, on AWS about 1.2 h
# after the cycle, ~13:10Z) are published, not at a fixed clock time. The cron
# fires at 13:00Z and this polls; the wait also holds the start until 07:30
# Central so the morning RAWS fuel-moisture observations are in. The lag makes
# DailyForecast (and the beta run below) treat today's 12z as the current run.
export FORECAST_CYCLE_HOUR=12
export FORECAST_START_HOUR=4
export FORECAST_END_HOUR=15
export FORECAST_AVAILABILITY_LAG_HOURS=1
export FORECAST_NOT_BEFORE_LOCAL=07:30
export REQUIRE_COMPLETE_HRRR=true
# 14:30Z was the old fixed start, so it is also the latest a late HRRR run can delay this.
export HRRR_READINESS_TIMEOUT_SECONDS=5400

run_step "Waiting for complete 12z HRRR f04-f15 data" "$PROJECT_DIR/scripts/wait_for_hrrr.py" degraded
if [ $? -ne 0 ]; then
    echo "WARNING: HRRR was not confirmed complete by $(date); starting the forecast anyway, as the old fixed 14:30Z start did" >> "$LOG_FILE" 2>&1
    export REQUIRE_COMPLETE_HRRR=false
fi

# Hold an exclusive lock for the rest of the run so maps.sh skips its cycles
# instead of competing with the forecast (and the API it calls) for CPU/memory.
# It lives in the project folder so every copy of maps.sh (container or host
# cron) sees it. Taken after the wait, so a running 09z forecast (which holds
# it) just delays this one until it finishes.
source "$SCRIPT_DIR/forecast_lock.sh"
take_forecast_lock
echo "=== Holding forecast lock; map refreshes paused ===" >> "$LOG_FILE" 2>&1

run_step "Running Daily Forecast" "$PROJECT_DIR/forecast/DailyForecast.py"
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "=== FAILED at $(date) with exit code $EXIT_CODE ===" >> "$LOG_FILE" 2>&1
    exit $EXIT_CODE
fi

# While FORECAST_USE_SHARED_HRRR is off, check that the cache the hourly prefetch assembled would have matched
# the one DailyForecast just downloaded for itself. Report only: it always exits 0.
run_step "Comparing shared HRRR cache with the downloaded one" "$PROJECT_DIR/scripts/compare_hrrr_cache.py" degraded || \
  echo "WARNING: shared HRRR cache comparison failed at $(date)" >> "$LOG_FILE" 2>&1

# The ensemble fire danger (beta) step used to run here. It needs REFS/HREF,
# which publish at about 14:36Z/15:06Z, so it now has its own run at 15:00Z
# (scripts/ensemble_12z.sh) and no longer holds up this forecast's publication.

# The 09z comparison must run after this operational 12z archive exists.
# It is observational/reporting-only and must not block forecast publication.
run_step "Comparing 9z run against same-day 12z run" "$PROJECT_DIR/scripts/compare_09z_12z.py" degraded || \
  echo "WARNING: 9z comparison generation failed at $(date)" >> "$LOG_FILE" 2>&1

run_step "Running AI Text Generation" "$PROJECT_DIR/forecast/forecast_ai.py"
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "=== FAILED at $(date) with exit code $EXIT_CODE ===" >> "$LOG_FILE" 2>&1
    exit $EXIT_CODE
fi

# Per-county maps are a supplementary product, not the core forecast. A
# failure here must not suppress the "forecast ready" notifications below.
run_step "Updating Per County Maps" "$PROJECT_DIR/forecast/PerCounty.py" degraded
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "WARNING: Updating Per County Maps failed at $(date) with exit code $EXIT_CODE - continuing so forecast notifications still go out" >> "$LOG_FILE" 2>&1
fi

echo "=== Sending forecast maps to Discord ===" >> "$LOG_FILE" 2>&1
"$PYTHON" "$PROJECT_DIR/scripts/notify_forecast_complete.py" >> "$LOG_FILE" 2>&1 || \
  { echo "WARNING: Forecast Discord completion notification failed at $(date)" >> "$LOG_FILE" 2>&1
    health stage "Forecast completion notifications" degraded "notify_forecast_complete.py failed" --log "$LOG_FILE"; }

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
