#!/bin/sh
# Entrypoint for the docker-compose `worker` service: runs cron only.
set -e

mkdir -p /app/logs
chmod 755 /app/logs || true

# Cron-driven scripts call the API at http://localhost:8000, but the API lives
# in the `backend` container. Forward localhost:8000 there (restarted if it dies).
( while true; do /opt/venv/bin/python /app/scripts/localhost_forward.py; sleep 2; done ) \
    >> /app/logs/worker_forward.log 2>&1 &

exec "$@"
