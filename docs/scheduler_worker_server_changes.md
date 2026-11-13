# Server changes for the heavy scheduler worker (copy-paste)

Written against the server's real `~/showmefire/docker-compose.yml` and `deploy.sh`. The repo (`~/showmefire/showmefire-api`) is bind-mounted
at `/app` in every container, and `./showmefire-api/data` is `/app/data`, so the worker shares the DB, the locks and the status files
with the API automatically. Nothing below touches the repo; these are edits to the two files outside it.

## 0. First deploy (no server edits)

`./deploy.sh` as it is now. With `SMF_SCHEDULER_GROUPS` unset the API still runs every job. Back up first:

```bash
cd ~/showmefire
docker exec show-me-fire-api sh -c 'ls -lh /app/data/showmefire.db* && df -h /app/data'
docker exec show-me-fire-api sqlite3 /app/data/showmefire.db ".backup '/app/data/showmefire.pre-smf-numbers.db'"
docker exec show-me-fire-api sqlite3 /app/data/showmefire.pre-smf-numbers.db "PRAGMA integrity_check;"   # prints: ok
```

(The backup lands in `~/showmefire/showmefire-api/data/`. Delete it after a few clean days: it holds everything in the database.)
No `requirements`/`Dockerfile` change, so `deploy.sh` does `docker compose restart backend worker`.

## 1. docker-compose.yml: add the scheduler service

Add this service next to `worker:` (same image, volumes and env_file; it starts after the API so database setup is done by the API first):

```yaml
  scheduler:
    # Runs only the "heavy" APScheduler jobs (forecast_v1, hourly HRRR, RTMA, graphics, archives, verification...). An API deploy
    # does not restart it. See showmefire-api/docs/scheduler_worker.md.
    image: showmefire-api:latest
    container_name: show-me-fire-scheduler
    depends_on:
      - backend
    command: ["python", "-m", "core.scheduler_worker", "--groups", "heavy"]
    mem_limit: 10g
    memswap_limit: 10g
    stop_grace_period: 30m
    environment:
      - SMF_SCHEDULER_GROUPS=heavy
      - SMF_START_CRON=0
      - SMF_DOWNLOAD_MODOT_ROADS=0
      - SMF_METRICS_ENABLED=0
      - SMF_SCHEDULER_STOP_WAIT_SECONDS=1500
      - SMF_GIS_PUBLISH_DIR=/app/gis
      - SMF_GIS_RESOLUTION_METERS=3000
      - SMF_GIS_RETENTION_DAYS=30
      - SMF_DAY_ARCHIVE_PRUNE_LOCAL=false
      - SMF_DAY_ARCHIVE_DELETE=false
    volumes:
      - /mnt/adrive/showmefire-data/gis:/app/gis
      - ./showmefire-api:/app
      - ./showmefire-api/data:/app/data
      - /mnt/adrive/showmefire-data/archive:/app/archive
      - /mnt/adrive/showmefire-data/cache:/app/cache
      - /mnt/adrive/showmefire-data/data_archive:/app/data_archive
      - /mnt/adrive/showmefire-data/files:/app/files
    env_file:
      - ./showmefire-api/.env
    healthcheck:
      test: ["CMD-SHELL", "test $$(( $$(date +%s) - $$(stat -c %Y /app/data/current_conditions/status/scheduler_heavy.json) )) -lt 300"]
      interval: 1m
      start_period: 5m
    restart: unless-stopped
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

Copy any `SMF_*` variables the `backend` service sets under `environment:` that heavy jobs also need (the day-archive flags above are copied
already). Everything in `.env` reaches it through `env_file`, including `SMF_FORECAST_V1_*` and the R2 keys.

Memory: `backend` is 14g and `worker` 8g. forecast_v1 alone is limited to 8 GB. Check `free -h` before choosing the scheduler's limit; once
the heavy jobs have moved out you can lower the backend's.

Then, **in the same edit**, give the `backend` service the groups setting (this is the cutover):

```yaml
    environment:
      - SMF_SCHEDULER_GROUPS=api       # <- add; the heavy jobs now run in `scheduler`
      - SMF_START_CRON=0
      ...
```

Do this in a quiet window. `./deploy.sh` sees the changed compose checksum and runs `docker compose up -d`, which recreates `backend` with
the new setting and **starts the new `scheduler` service**. The existing guard (forecast-v1 phase, cron jobs, daily-job lead time) runs
first, and at that point the heavy jobs are still in the backend, so it protects them. Check afterwards:

```bash
docker ps --format '{{.Names}}\t{{.Status}}' | grep show-me-fire
docker logs --tail 20 show-me-fire-scheduler      # "Scheduler worker running groups heavy"
jq . ~/showmefire/showmefire-api/data/current_conditions/status/scheduler_heavy.json | head
curl -s localhost:5001/health/ready | jq .         # scheduler check: "heavy scheduler worker alive"
```

Rollback: remove the `SMF_SCHEDULER_GROUPS` line from `backend`, run `./deploy.sh`, then `docker compose stop scheduler`.

## 2. deploy.sh edits

Goal: an API-only deploy restarts only `backend` and needs no wait; a deploy that restarts the workers keeps the full guard.

**a. Header comment** (add under `--force`):

```bash
#   ./deploy.sh --api-only pull, restart ONLY the backend (no guard: heavy jobs live in the scheduler container)
```

**b. Argument parsing** (add a case, and `SCOPE=all` next to `MODE=deploy`):

```bash
MODE=deploy
SCOPE=all
for arg in "$@"; do
    case "$arg" in
        --check) MODE=check ;;
        --force) MODE=force ;;
        --api-only) SCOPE=api ;;
        *) echo "Unknown option: $arg (use --check, --force or --api-only)"; exit 2 ;;
    esac
done
```

**c. Constants** (next to `IGNORED_JOBS`):

```bash
HEAVY_STATE=$REPO/data/current_conditions/status/scheduler_heavy.json
# Heavy jobs that run every few minutes and simply resume or run again, so a restart does not lose their work.
IGNORED_HEAVY="hrrr_hourly_forecast capture_latest_mrms end_of_day_archive rtma_spread_rate_pipeline forecast_v1_admin_requests"
```

**d. In `guard_collect`**, after the forecast-v1/staging checks (section 2) add:

```bash
    # 2b. Jobs running right now in the scheduler container (it writes this file every minute; ignore a stale one).
    if [ -f "$HEAVY_STATE" ]; then
        local beat_age=$(( now_epoch - $(stat -c %Y "$HEAVY_STATE") )) hjob hskip hi
        if [ "$beat_age" -lt 300 ]; then
            while read -r hjob; do
                [ -z "$hjob" ] && continue
                hskip=false
                for hi in $IGNORED_HEAVY; do [ "$hjob" = "$hi" ] && hskip=true; done
                $hskip || GUARD_REASONS+=("scheduler job '$hjob' is running")
            done < <(jq -r '(.running_jobs // [])[]' "$HEAVY_STATE" 2>/dev/null)
        fi
    fi
```

The forecast-v1 `admin-job.json` and "still staging" checks you already have stay as they are: they read files on the host, so they work
whichever container runs the job.

**e. Skip the guard for an API-only deploy** (replace the `if [ "$MODE" = force ]` line):

```bash
if [ "$MODE" = force ]; then
    echo "--force: skipping the safety guard."
elif [ "$SCOPE" = api ] && [ "$ACTION" = restart ]; then
    echo "--api-only: restarting only the backend; the scheduler and worker are untouched."
else
```

**f. Apply step** (the `restart)` case):

```bash
    restart)
        if [ "$SCOPE" = api ]; then
            echo "Code-only change, API only -- restarting backend."
            docker compose restart backend
        else
            echo "Code-only changes -- restarting backend, worker and scheduler without rebuild."
            docker compose restart backend worker scheduler
        fi
        ;;
```

(`rebuild` and `recreate` already use `docker compose up -d`, which handles the new service; `--api-only` does not apply to them because a
rebuilt image or a changed config affects every container, so they keep the full guard.)

**g. Health check** (add the scheduler to the loop):

```bash
for name in show-me-fire-api show-me-fire-worker show-me-fire-scheduler; do
```

## 3. Two things to know about the bind mount

- Because `./showmefire-api` is mounted at `/app` in every container, `git pull` changes the files under a **running** scheduler. A running Python
  process keeps the modules it already loaded, but anything it imports later (several jobs import inside the function) is read from the new
  files. So a scheduler that has not been restarted can run a mix of old and new code. Keep changes to heavy-job code additive, and restart the
  scheduler (a normal `./deploy.sh`, with the guard) when heavy-job code changes. `--api-only` is for changes that don't touch it.
- The cron `worker` has the same property, and cron scripts start fresh Python processes anyway, so they pick up new code on their next run.
