#!/usr/bin/env bash
set -euo pipefail

# Resolve paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# If script is in /app/scripts/, go up one level to /app
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

TODAY=$(date -u +%Y%m%d)

PYTHON=${PYTHON:-python3}

"$PYTHON" scripts/endOfDay.py "$@"

"$PYTHON" scripts/compare_forecasts.py --date "$TODAY"

"$PYTHON" scripts/generate_performance_plots.py --date "$TODAY"
