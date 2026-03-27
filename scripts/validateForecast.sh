#!/usr/bin/env bash
set -euo pipefail

# Resolve paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# If script is in /app/scripts/, go up one level to /app
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

# Ensure cron can find common binaries and prefer virtualenv Python.
export PATH="/opt/venv/bin:$REPO_ROOT/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"

if [[ -x /opt/venv/bin/python ]]; then
	PYTHON="/opt/venv/bin/python"
elif [[ -x "$REPO_ROOT/venv/bin/python" ]]; then
	PYTHON="$REPO_ROOT/venv/bin/python"
else
	PYTHON="python3"
fi

if ! "$PYTHON" --version >/dev/null 2>&1; then
	echo "ERROR: Python interpreter check failed: $PYTHON" >&2
	exit 1
fi

"$PYTHON" scripts/endOfDay.py "$@"

"$PYTHON" forecast/endOfDayReport.py
"$PYTHON" forecast/endOfDayReport.py --forecast-glob "station_forecasts_beta_*.json" --report-suffix beta

TODAY_DASH=$(TZ="America/Chicago" date +%Y-%m-%d)
SUMMARY_FILE="reports/$TODAY_DASH/validation_summary.json"
SUMMARY_FILE_BETA="reports/$TODAY_DASH/validation_summary_beta.json"
VERIFICATION_CSV="reports/verification_history.csv"
VERIFICATION_CSV_BETA="reports/verification_history_beta.csv"

if [[ ! -f "$SUMMARY_FILE" ]]; then
	echo "ERROR: Missing validation summary: $SUMMARY_FILE" >&2
	exit 1
fi

if [[ ! -f "$SUMMARY_FILE_BETA" ]]; then
	echo "ERROR: Missing beta validation summary: $SUMMARY_FILE_BETA" >&2
	exit 1
fi

RECORD_COUNT=$("$PYTHON" - "$SUMMARY_FILE" <<'PY'
import json
import sys

summary_path = sys.argv[1]
with open(summary_path, 'r') as f:
    summary = json.load(f)

print(int(summary.get('record_count', 0) or 0))
PY
)

RECORD_COUNT_BETA=$("$PYTHON" - "$SUMMARY_FILE_BETA" <<'PY'
import json
import sys

summary_path = sys.argv[1]
with open(summary_path, 'r') as f:
	summary = json.load(f)

print(int(summary.get('record_count', 0) or 0))
PY
)

if [[ "$RECORD_COUNT" -le 0 ]]; then
	echo "ERROR: Validation produced zero overlapping records." >&2
	exit 1
fi

if [[ "$RECORD_COUNT_BETA" -le 0 ]]; then
	echo "ERROR: Beta validation produced zero overlapping records." >&2
	exit 1
fi

if [[ ! -f "$VERIFICATION_CSV" ]]; then
	echo "ERROR: Missing compatibility CSV: $VERIFICATION_CSV" >&2
	exit 1
fi

if [[ ! -f "$VERIFICATION_CSV_BETA" ]]; then
	echo "ERROR: Missing beta compatibility CSV: $VERIFICATION_CSV_BETA" >&2
	exit 1
fi

echo "Validation complete: record_count=$RECORD_COUNT beta_record_count=$RECORD_COUNT_BETA"
