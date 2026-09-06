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

# DB initialization happens in main.py's FastAPI lifespan startup (init_database()
# is idempotent), which runs on every boot since CMD always launches uvicorn - no
# separate init needed here.

# Dockerfile.dev doesn't install the cron package; only start it if present so the
# dev image doesn't crash under `set -e`.
if [ -x /etc/init.d/cron ]; then
    service cron start
else
    echo "cron not installed, skipping cron start"
fi

# Exec the main process
exec "$@"
