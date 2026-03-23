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

TODAY_DASH=$(TZ="America/Chicago" date +%Y-%m-%d)
SUMMARY_FILE="reports/$TODAY_DASH/validation_summary.json"
VERIFICATION_CSV="reports/verification_history.csv"

if [[ ! -f "$SUMMARY_FILE" ]]; then
	echo "ERROR: Missing validation summary: $SUMMARY_FILE" >&2
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

if [[ "$RECORD_COUNT" -le 0 ]]; then
	echo "ERROR: Validation produced zero overlapping records." >&2
	exit 1
fi

if [[ ! -f "$VERIFICATION_CSV" ]]; then
	echo "ERROR: Missing compatibility CSV: $VERIFICATION_CSV" >&2
	exit 1
fi

echo "Validation complete: record_count=$RECORD_COUNT"
