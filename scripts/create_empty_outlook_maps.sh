#!/bin/bash

set -euo pipefail

# Usage:
#   ./scripts/create_empty_outlook_maps.sh [YYYY-MM-DD]
# If no date is provided, today's date in America/Chicago is used.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Cron has no `python` on PATH: pick the interpreter the same way the other jobs do.
if [[ -x /opt/venv/bin/python ]]; then
  PYTHON="/opt/venv/bin/python"
elif [[ -x "$PROJECT_DIR/venv/bin/python" ]]; then
  PYTHON="$PROJECT_DIR/venv/bin/python"
else
  PYTHON="python3"
fi

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

if [[ $# -gt 0 ]]; then
  VALID_DATE="$1"
else
  VALID_DATE="$("$PYTHON" - <<'PY'
from datetime import datetime
from zoneinfo import ZoneInfo
print(datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d'))
PY
)"
fi

ISSUE_TIME="$("$PYTHON" - <<'PY'
from datetime import datetime
from zoneinfo import ZoneInfo
print(datetime.now(ZoneInfo('America/Chicago')).isoformat())
PY
)"

for day in 2 3; do
  GEOJSON_PATH="$PROJECT_DIR/gis/outlook_day${day}_published.geojson"

  cat > "$GEOJSON_PATH" <<'JSON'
{
  "type": "FeatureCollection",
  "features": [],
  "outlook_text": ""
}
JSON

  echo "Created empty day ${day} published outlook: $GEOJSON_PATH"
  echo "Generating day ${day} outlook graphic..."
  "$PYTHON" "$PROJECT_DIR/maps/outlookgraphic.py" --day "$day" --valid-date "$VALID_DATE" --issue-time "$ISSUE_TIME"
done

echo "Done. Empty day 2/day 3 outlook maps generated for valid date: $VALID_DATE"

