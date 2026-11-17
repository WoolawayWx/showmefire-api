#!/bin/sh
# Entrypoint for the docker-compose `worker` service: runs cron only.
set -e

mkdir -p /app/logs
chmod 755 /app/logs || true

# Which cron jobs this container runs (SMF_CRON_SET): `all` (default, the old single worker), `short` (quick jobs: map
# refresh, nightly verification, blank outlook maps) or `long` (jobs that run for minutes to hours: the 12z/09z forecasts,
# ensemble fire danger, the seven-day outlook). The long set runs in its own container so a deploy that restarts the quick
# jobs never has to wait for it. The crontab files are baked into the image (see the Dockerfile); the unwanted ones are
# removed here, at container start.
case "${SMF_CRON_SET:-all}" in
    all) ;;
    short) rm -f /etc/cron.d/forecasts /etc/cron.d/ensemble12z /etc/cron.d/fireoutlook /etc/cron.d/forecasts09z ;;
    long)  rm -f /etc/cron.d/validate /etc/cron.d/createemptymaps /etc/cron.d/maps ;;
    *) echo "Unknown SMF_CRON_SET='${SMF_CRON_SET}' (use all, short or long)" >&2; exit 1 ;;
esac
echo "cron set: ${SMF_CRON_SET:-all}: $(ls /etc/cron.d | tr '\n' ' ')"

# Cron-driven scripts call the API at http://localhost:8000, but the API lives
# in the `backend` container. Forward localhost:8000 there (restarted if it dies).
( while true; do /opt/venv/bin/python /app/scripts/localhost_forward.py; sleep 2; done ) \
    >> /app/logs/worker_forward.log 2>&1 &

# cron starts its jobs with an almost empty environment, so the variables from `env_file` (R2 keys, API tokens...) never
# reach them. Containers that bind-mount the repo get them from /app/.env (load_dotenv), but the pinned long image has no
# .env (.dockerignore), so a forecast could not upload to the CDN. Save the container environment where run_job.sh finds it.
export -p | grep -v -E '^export (PATH|HOME|PWD|OLDPWD|SHLVL|_|TZ|HOSTNAME|TERM|SHELL|LOGNAME|USER)=' > /etc/smf_container_env.sh || true
chmod 600 /etc/smf_container_env.sh || true

exec "$@"
