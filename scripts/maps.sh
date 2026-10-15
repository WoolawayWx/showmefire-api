#!/bin/bash

set -u

# Detect project directory from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR" || exit 1

# Guard against overlapping runs: this fires every 15 minutes via cron, and if
# one run (HRRR/station fetches, shapefile I/O, several PNG renders) ever runs
# long, a second overlapping instance would start and both would write the
# same images/*.png, status.json, and gis/realtime/*.tif concurrently. Skip
# this cycle entirely rather than race with a still-running one.
LOCK_FILE="${TMPDIR:-/tmp}/showmefire-maps.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "Another maps.sh run is still in progress (lock: $LOCK_FILE); skipping this cycle."
    exit 0
fi

# The daily forecast (forecasts.sh) takes priority over map refreshes: skip
# this cycle while it holds its lock, and run at low CPU priority otherwise so
# a map run that overlaps the forecast start can't starve it or the API.
FORECAST_LOCK_FILE="${TMPDIR:-/tmp}/showmefire-forecast.lock"
exec 8>"$FORECAST_LOCK_FILE"
if ! flock -n -s 8; then
    echo "Daily forecast is running (lock: $FORECAST_LOCK_FILE); skipping this cycle."
    exit 0
fi
exec 8>&-
renice -n 15 -p $$ > /dev/null 2>&1 || true

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

# Marker used to detect which images/*.png files this run's scripts actually
# touched (see MAPS_SH_NEW_IMAGES below) - set before the loop starts.
IMAGE_MARKER="$(mktemp)"
SCRIPT_LOG_DIR="$(mktemp -d)"
trap 'rm -rf "$IMAGE_MARKER" "$SCRIPT_LOG_DIR"' EXIT

# Collect every runnable script in the maps folder. Helper modules are
# imported by other scripts and should not be executed directly. The slowest
# scripts go first so they start immediately and the short ones fill in the
# gaps (shortest total wall time for a fixed number of parallel slots).
SLOW_FIRST=(advisory_category_maps.py realtimefiredanger.py windmap-fil.py rhmap-fil.py fuelmoisturemap.py outlookgraphic.py)
scripts=()
for base_script in "${SLOW_FIRST[@]}"; do
    [ -f "./maps/$base_script" ] && scripts+=("./maps/$base_script")
done
for script in ./maps/*.py; do
    base_script="$(basename "$script")"
    if [[ "$base_script" == "station_danger_history.py" || "$base_script" == "realtime_geotiff.py" || "$base_script" == "dailyCapture.py" || "$base_script" == "precipitation_graphics.py" || "$base_script" == "__init__.py" || "$base_script" == "station_plot.py" || "$base_script" == "station_history.py" || "$base_script" == "status_lock.py" || "$base_script" == "station_products.py" ]]; then
        continue
    fi
    [[ " ${SLOW_FIRST[*]} " == *" $base_script "* ]] && continue
    scripts+=("$script")
done

# Run the scripts MAPS_PARALLEL at a time (each is an independent process
# that renders its own PNG). Scripts that update the shared status.json
# serialize that read-modify-write themselves via maps/status_lock.py. Each
# script's output goes to its own log and is replayed below in order, so
# maps.log stays readable instead of interleaving.
MAPS_PARALLEL="${MAPS_PARALLEL:-3}"
run_map_script() {
    local script="$1"
    local base_script
    base_script="$(basename "$script")"
    local started=$SECONDS
    "$PYTHON" "$script" > "$SCRIPT_LOG_DIR/$base_script.log" 2>&1
    echo "$? $((SECONDS - started))" > "$SCRIPT_LOG_DIR/$base_script.rc"
}
export -f run_map_script
export PYTHON SCRIPT_LOG_DIR

run_started=$SECONDS
printf '%s\0' "${scripts[@]}" | xargs -0 -n1 -P "$MAPS_PARALLEL" bash -c 'run_map_script "$0"'

for script in "${scripts[@]}"; do
    base_script="$(basename "$script")"
    read -r rc elapsed < "$SCRIPT_LOG_DIR/$base_script.rc" || { rc=1; elapsed=0; }
    echo "Running $script..."
    cat "$SCRIPT_LOG_DIR/$base_script.log"
    echo "[timing] $base_script: ${elapsed}s"
    if [ "$rc" != "0" ]; then
        echo "[ERROR] Failed: $script"
        failed_scripts+=("$script")
    fi
done
echo "[timing] all map scripts: $((SECONDS - run_started))s (MAPS_PARALLEL=$MAPS_PARALLEL)"

UPLOAD_FORECAST_VALUE="${uploadForecast:-true}"
UPLOAD_FORECAST_VALUE="$(echo "$UPLOAD_FORECAST_VALUE" | tr '[:upper:]' '[:lower:]')"

if [[ "$UPLOAD_FORECAST_VALUE" == "false" || "$UPLOAD_FORECAST_VALUE" == "0" || "$UPLOAD_FORECAST_VALUE" == "no" || "$UPLOAD_FORECAST_VALUE" == "off" ]]; then
    echo "CDN upload disabled (uploadForecast=false)"
else
    echo "Uploading to CDN..."
    # Pass exactly the images/*.png files this run's scripts touched (by
    # mtime vs. IMAGE_MARKER) instead of relying on upload_cdn.py's hardcoded
    # list - so a new maps/*.py script's output reaches the CDN automatically
    # without anyone needing to remember to register its filename there. The
    # images/ folder also holds beta/test/precip-composite artifacts from
    # other jobs that must NOT ride along, so this is deliberately scoped to
    # "changed this cycle," not a blind glob of the whole folder.
    NEW_IMAGES="$(find images -maxdepth 1 -type f -name '*.png' -newer "$IMAGE_MARKER" 2>/dev/null)"
    MAPS_SH_NEW_IMAGES="$NEW_IMAGES" "$PYTHON" scripts/upload_cdn.py
fi

echo "Generating RSS feed..."
"$PYTHON" -m services.rss

if (( ${#failed_scripts[@]} > 0 )); then
    echo ""
    echo "Map run completed with failures:"
    for failed in "${failed_scripts[@]}"; do
        echo " - $failed"
    done
    exit 1
fi
