#!/bin/bash

# Seven-day fire outlook worker, one tick. Cron runs this every 30 minutes (see the Dockerfile); almost every tick just
# finds the latest 00Z/12Z cycle already done, or its GEFS inputs not complete yet, and exits in seconds. The tick that finds
# a complete, unprocessed cycle does the work (about 10-15 minutes, ~2.5 GB peak; measured, see _plans/CLAUDE_PLAN.md).
#
# It does nothing unless SMF_FIRE_OUTLOOK_ENABLED=true (the staff-only pilot is not approved yet; see docs/fire_outlook_worker.md).
#
# Guards, in order:
#   1. its own lock: two ticks never run at once (a run can outlast the 30-minute interval);
#   2. the shared forecast lock, waited for at most 20 s: while the daily forecast/ensemble holds it exclusively this tick is
#      skipped (cron retries in 30 minutes, nothing queues), and while the outlook computes it holds the lock so scripts/maps.sh
#      yields CPU, exactly as the forecast does;
#   3. one processing thread, low priority, and a hard 4-hour limit (the plan's budget; exit 124 = timed out).
# The worker itself keeps the previous complete run current on any failure (services/fire_outlook/worker.py).

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

export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
mkdir -p "$PROJECT_DIR/logs"
LOG_FILE="$PROJECT_DIR/logs/fire_outlook_$(date +%Y%m%d).log"

# cron gives jobs an almost empty environment; pick up the container's settings (SMF_FIRE_OUTLOOK_ENABLED etc.) before checking them.
# shellcheck source=container_env.sh
. "$SCRIPT_DIR/container_env.sh"

case "$(printf '%s' "${SMF_FIRE_OUTLOOK_ENABLED:-false}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) ;;
    *) echo "fire outlook disabled (SMF_FIRE_OUTLOOK_ENABLED is not true)"; exit 0 ;;
esac

LOCK_DIR="$PROJECT_DIR/.locks"
mkdir -p "$LOCK_DIR" 2>/dev/null
chmod 1777 "$LOCK_DIR" 2>/dev/null || true

exec 9>"$LOCK_DIR/fire-outlook.lock"
if ! flock -n 9; then
    echo "previous fire outlook run is still active; skipping this tick"
    exit 0
fi

[ -e "$LOCK_DIR/showmefire-forecast.lock" ] || (umask 000 && : >> "$LOCK_DIR/showmefire-forecast.lock")
exec 8<>"$LOCK_DIR/showmefire-forecast.lock"
if ! flock -w 20 -x 8; then
    echo "the daily forecast holds the forecast lock; skipping this tick (retry in 30 minutes)"
    exit 0
fi

# The plan's default is ONE processing thread: keep numpy/BLAS from quietly using more.
export OMP_NUM_THREADS="${SMF_FIRE_OUTLOOK_THREADS:-1}" OPENBLAS_NUM_THREADS="${SMF_FIRE_OUTLOOK_THREADS:-1}" MKL_NUM_THREADS="${SMF_FIRE_OUTLOOK_THREADS:-1}"

echo "=== fire outlook tick at $(date -u +%Y-%m-%dT%H:%M:%SZ) (python: $PYTHON) ===" >> "$LOG_FILE" 2>&1
nice -n 10 timeout --kill-after=60 "${SMF_FIRE_OUTLOOK_TIMEOUT_SECONDS:-14400}" "$PYTHON" "$PROJECT_DIR/scripts/run_fire_outlook.py" >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

# One line for the admin Logs page (run_job.sh reads it): the worker's own status.
SUMMARY="$("$PYTHON" - <<'EOF' 2>/dev/null
import json, os
from pathlib import Path
root = Path(os.getenv("SMF_FIRE_OUTLOOK_DIR", "data/fire-outlook"))
try:
    s = json.loads((root / "status.json").read_text())
    print(f"{s.get('state')} {s.get('stage', '')} run={s.get('run_id') or '-'} cycle_age_h={s.get('cycle_age_hours')}")
except Exception:
    print("no status yet")
EOF
)"
if [ -n "${JOB_SUMMARY_FILE:-}" ]; then
    case "$EXIT_CODE" in
        0)   echo "$SUMMARY" > "$JOB_SUMMARY_FILE" ;;
        124|137) echo "fire outlook killed after the time limit / out of memory (exit $EXIT_CODE); see logs/fire_outlook_$(date +%Y%m%d).log" > "$JOB_SUMMARY_FILE" ;;
        *)   echo "fire outlook failed (exit $EXIT_CODE): $SUMMARY" > "$JOB_SUMMARY_FILE" ;;
    esac
fi
exit "$EXIT_CODE"
