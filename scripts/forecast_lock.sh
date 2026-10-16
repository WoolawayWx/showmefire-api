# Sourced by the forecast scripts. take_forecast_lock holds the shared forecast
# lock exclusively on fd 8 until the calling script exits, which makes
# scripts/maps.sh skip its 15-minute cycles (and stop starting new map scripts
# mid-cycle) so the forecast gets the CPU. maps.sh only ever takes this lock
# shared and momentarily, so waiting for it here is at most a blink.
take_forecast_lock() {
    local lock_dir="$PROJECT_DIR/.locks"
    mkdir -p "$lock_dir" 2>/dev/null
    chmod 1777 "$lock_dir" 2>/dev/null || true
    [ -e "$lock_dir/showmefire-forecast.lock" ] || (umask 000 && : >> "$lock_dir/showmefire-forecast.lock")
    exec 8<>"$lock_dir/showmefire-forecast.lock"
    flock -x 8
}
