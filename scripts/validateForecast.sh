#!/usr/bin/env bash
set -euo pipefail

# Resolve paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# If script is in /app/scripts/, go up one level to /app
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

TODAY_DASH=$(TZ="America/Chicago" date +%Y-%m-%d)
CURRENT_HOUR_CT=$(TZ="America/Chicago" date +%H)

# A scheduled current-day report is invalid until the 10:00-21:00 Central
# observation window has closed. Explicit historical reruns may pass args.
if [[ $# -eq 0 && 10#$CURRENT_HOUR_CT -lt 22 ]]; then
	echo "ERROR: Refusing current-day verification before 22:00 Central." >&2
	exit 1
fi

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

# Peak RTMA analysis for the same local date / 10:00–21:00 CT window as
# the station verification report. Failure here must not skip scoring.
"$PYTHON" -c "from maps.observed_peak_history import snapshot_observed_peak_for_date; snapshot_observed_peak_for_date('$TODAY_DASH')" \
	|| echo "WARN: Observed peak archive snapshot failed for $TODAY_DASH" >&2
"$PYTHON" -c "from services.rtma_peak import generate_rtma_peak; generate_rtma_peak('$TODAY_DASH')" \
	|| echo "WARN: RTMA peak generation failed for $TODAY_DASH" >&2

"$PYTHON" forecast/endOfDayReport.py
"$PYTHON" forecast/endOfDayReport.py --forecast-glob "station_forecasts_beta_*.json" --report-suffix beta

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
