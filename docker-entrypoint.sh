#!/bin/sh
set -e

DATA_DIR="${DATA_DIR:-/app/data}"

# Ensure data dir exists and is writable
mkdir -p "$DATA_DIR"
chmod 755 "$DATA_DIR" || true

# logs/ is excluded from the build context (.dockerignore), so it won't exist
# in a fresh image. Cron jobs redirect stdout/stderr into /app/logs/*.log -
# if the directory is missing, that redirect fails before the job even runs.
mkdir -p /app/logs
chmod 755 /app/logs || true

# Initialize the sqlite DB (idempotent)
python3 - <<'PY'
try:
    from core.database import init_database
    init_database()
except Exception as e:
    import sys, traceback
    print("DB init error:", e, file=sys.stderr)
    traceback.print_exc()
PY

# Exec the main process
service cron start
exec "$@"
