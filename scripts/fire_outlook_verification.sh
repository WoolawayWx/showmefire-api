#!/bin/bash

# Daily: build the seven-day fire outlook verification records from windows that have matured (see
# scripts/export_fire_outlook_verification.py and services/fire_outlook/verification.py). Idempotent and read-only with
# respect to the outlook itself; the output folder is what gets copied to the training machine.
# Does nothing unless SMF_FIRE_OUTLOOK_ENABLED=true, like the worker.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR" || exit 1

if [ -f "/opt/venv/bin/python" ]; then
    PYTHON="/opt/venv/bin/python"
elif [ -f "$PROJECT_DIR/venv/bin/python" ]; then
    PYTHON="$PROJECT_DIR/venv/bin/python"
else
    PYTHON="python"
fi

export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
mkdir -p "$PROJECT_DIR/logs"
LOG_FILE="$PROJECT_DIR/logs/fire_outlook_verification.log"

case "$(printf '%s' "${SMF_FIRE_OUTLOOK_ENABLED:-false}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) ;;
    *) echo "fire outlook disabled (SMF_FIRE_OUTLOOK_ENABLED is not true)"; exit 0 ;;
esac

echo "=== fire outlook verification export at $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$LOG_FILE" 2>&1
nice -n 10 timeout --kill-after=60 3600 "$PYTHON" "$PROJECT_DIR/scripts/export_fire_outlook_verification.py" >> "$LOG_FILE" 2>&1
EXIT_CODE=$?
if [ "$EXIT_CODE" -ne 0 ] && [ -n "${JOB_SUMMARY_FILE:-}" ]; then
    echo "verification export failed (exit $EXIT_CODE); see logs/fire_outlook_verification.log" > "$JOB_SUMMARY_FILE"
fi
exit "$EXIT_CODE"
