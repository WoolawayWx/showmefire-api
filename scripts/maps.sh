#!/bin/bash

set -u

# Detect project directory from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR" || exit 1

# Detect correct Python executable (same convention as forecasts.sh / forecast_RRFS.sh).
# Cron does not inherit the Docker image's VIRTUAL_ENV/PATH, and in production the
# venv lives at /opt/venv, not $PROJECT_DIR/venv - so `source ./venv/bin/activate`
# silently fails there and every script below would run against a bare system
# python with none of the required packages installed.
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

# Ensure project-root imports like core.* and maps.* resolve in script mode.
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

failed_scripts=()

# Run all Python scripts in the maps folder
for script in ./maps/*.py; do
    base_script="$(basename "$script")"

    # Helper modules are imported by other scripts and should not be executed directly.
    if [[ "$base_script" == "station_danger_history.py" || "$base_script" == "realtime_geotiff.py" || "$base_script" == "dailyCapture.py" || "$base_script" == "__init__.py" ]]; then
        continue
    fi

    echo "Running $script..."
    if ! "$PYTHON" "$script"; then
        echo "[ERROR] Failed: $script"
        failed_scripts+=("$script")
    fi
done

UPLOAD_FORECAST_VALUE="${uploadForecast:-true}"
UPLOAD_FORECAST_VALUE="$(echo "$UPLOAD_FORECAST_VALUE" | tr '[:upper:]' '[:lower:]')"

if [[ "$UPLOAD_FORECAST_VALUE" == "false" || "$UPLOAD_FORECAST_VALUE" == "0" || "$UPLOAD_FORECAST_VALUE" == "no" || "$UPLOAD_FORECAST_VALUE" == "off" ]]; then
    echo "CDN upload disabled (uploadForecast=false)"
else
    echo "Uploading to CDN..."
    "$PYTHON" scripts/upload_cdn.py
fi

echo "Generating RSS feed..."
"$PYTHON" -m services.rss --add-summary

if (( ${#failed_scripts[@]} > 0 )); then
    echo ""
    echo "Map run completed with failures:"
    for failed in "${failed_scripts[@]}"; do
        echo " - $failed"
    done
    exit 1
fi