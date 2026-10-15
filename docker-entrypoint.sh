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

# The statewide MoDOT road archive is intentionally downloaded at runtime,
# not committed into Git or copied into the image.
if [ "${SMF_DOWNLOAD_MODOT_ROADS:-1}" = "1" ]; then
    python3 /app/scripts/download_modot_roads.py || echo "MoDOT road download unavailable; incident maps will use fallback roads"
fi

# DB initialization happens in main.py's FastAPI lifespan startup (init_database()
# is idempotent), which runs on every boot since CMD always launches uvicorn - no
# separate init needed here.

# Dockerfile.dev doesn't install the cron package; only start it if present so the
# dev image doesn't crash under `set -e`.
# The docker-compose `backend` service sets SMF_START_CRON=0 because cron jobs
# run in the separate `worker` service.
if [ "${SMF_START_CRON:-1}" != "1" ]; then
    echo "SMF_START_CRON=${SMF_START_CRON}, skipping cron start (cron runs in the worker service)"
elif [ -x /etc/init.d/cron ]; then
    service cron start
else
    echo "cron not installed, skipping cron start"
fi

# Exec the main process
exec "$@"
