#!/bin/bash

set -u

# Detect project directory from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR" || exit 1

# Activate the virtual environment
source ./venv/bin/activate

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
    if ! python "$script"; then
        echo "[ERROR] Failed: $script"
        failed_scripts+=("$script")
    fi
done

echo "Uploading to CDN..."
python scripts/upload_cdn.py

echo "Generating RSS feed..."
python -m services.rss --add-summary

# Deactivate the virtual environment
deactivate

if (( ${#failed_scripts[@]} > 0 )); then
    echo ""
    echo "Map run completed with failures:"
    for failed in "${failed_scripts[@]}"; do
        echo " - $failed"
    done
    exit 1
fi