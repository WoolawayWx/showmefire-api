# Heavy scheduler worker (runbook)

The long APScheduler jobs (forecast_v1, the hourly HRRR build, the RTMA/spread-rate pipelines, precipitation and burn-ban graphics,
all the archives, verification, retraining, storage cleanup) run in their **own container**, so a deploy of the API (`show-me-fire-api`)
no longer kills a forecast that is in the middle of a run.

```
show-me-fire-api         uvicorn + the "api" jobs   (stations, alerts, pushes, WildCAD, fire detections, drift check, watchdog...)
show-me-fire-scheduler   python -m core.scheduler_worker   the "heavy" jobs   <- new
show-me-fire-worker      cron (forecasts_12z, ensemble_12z, fire_outlook, maps...)            unchanged
```

Which job runs where is `core/job_groups.py` (`JOB_GROUPS`). A test fails if a scheduled job has no group.
`SMF_SCHEDULER_GROUPS` selects a process's groups: unset or `api,heavy` = everything in one process (today's behaviour, also dev and
tests), `api` = the API container once the worker exists, `heavy` = the worker.

## What changed in the code (and why it is safe to split)

- The API keeps the station feeds in memory and serves them. `fetch_synoptic` and `fetch_raws_stations` therefore stay in the API, and each
  successful update is also written to `data/current_conditions/status/station_feed.json` / `raws_feed.json`
  (`services/station_feed_snapshot.py`). forecast_v1, the RTMA analysis and the spread-rate pipeline read the feed through
  `station_data_for_job()` / `raws_data_for_job()`: live data when there is some, otherwise that snapshot, **never one older than 30 minutes**
  (the job skips its tick instead of forecasting from stale observations).
- forecast_v1's single-run lock is now a file lock on the shared volume (`data/forecast-v1/.run.lock`, `fcntl.flock`), so it holds across
  containers and is released by the kernel if the holder dies. The admin "Run now" button only **queues** a request (`admin-job.json`,
  status `queued`); the heavy group starts it within 30 s (`forecast_v1_admin_requests`).
- On start the worker runs `reconcile_orphaned_run()`: a job file still saying `running` while nothing holds the lock belonged to a process
  that died, so it is marked failed (and its `staging` forecast run too) and the next 30-minute tick retries, instead of waiting out the
  4-hour guard.
- The worker writes `data/current_conditions/status/scheduler_heavy.json` every minute (groups, jobs with next run time, running jobs).
  `/health/ready`, the admin Logs page, the model schedule and the forecast-v1 admin page read it. The **API** watches it
  (`scheduler_worker_watch`, every 5 min) and alerts staff on Discord + the staff app if it goes quiet for 5 minutes, then reminds every
  6 hours and says when it recovered. It does nothing before a worker has ever reported, and nothing when `heavy` is in the API's own groups.
- Database setup (`init_database` and friends) runs under `core/startup_lock.py` in both processes, so they cannot race the table rebuild.

**The exact compose service and `deploy.sh` edits for the real server files are in `docs/scheduler_worker_server_changes.md`; use that. The generic snippet below is kept for reference.**

## Server setup (the server's `docker-compose.yml` and `.env` are not in this repo)

Add to the server compose file (same image, volumes and `env_file` as the API; copy them from the existing `show-me-fire-api` /
`show-me-fire-worker` services, and keep the paths identical so the locks and the snapshots are shared):

```yaml
  scheduler:
    image: showmefire-api:latest            # the same image the API runs
    container_name: show-me-fire-scheduler
    restart: unless-stopped
    command: ["python", "-m", "core.scheduler_worker", "--groups", "heavy"]
    env_file: .env                          # same file as the API
    environment:
      SMF_SCHEDULER_GROUPS: heavy
      SMF_START_CRON: "0"
      SMF_DOWNLOAD_MODOT_ROADS: "0"
      SMF_METRICS_ENABLED: "0"              # the API owns the metrics snapshots
      SMF_SCHEDULER_STOP_WAIT_SECONDS: "1500"
      # SMF_FORECAST_V1_* and the R2 credentials must be here too (they move with forecast_v1); put them in .env.
    volumes: []                             # <- copy the API service's volumes here (data, logs, .locks, cache, images, gis, archive, ...)
    stop_grace_period: 30m                  # longer than SMF_SCHEDULER_STOP_WAIT_SECONDS
    mem_limit: 12g                          # forecast_v1 alone is limited to 8 GB (SMF_FORECAST_V1_MEMORY_LIMIT_GB)
    healthcheck:
      test: ["CMD-SHELL", "test $$(( $$(date +%s) - $$(stat -c %Y /app/data/current_conditions/status/scheduler_heavy.json) )) -lt 300"]
      interval: 1m
      start_period: 5m
```

Then in the API service: `environment: { SMF_SCHEDULER_GROUPS: api }` (keep `run_sch: "true"`), and **remove** `SMF_FORECAST_V1_ENABLED` and the
other forecast_v1 variables from it if you want one place to configure them (the forecast-v1 admin page reads the worker's setting from its
heartbeat). The worker does not need the localhost:8000 port-forward: no scheduled job calls the API (only the cron scripts do).

## Cutover (do it in a quiet window: no forecast_v1 phase, no HRRR build, none of the daily cron jobs about to start)

1. Deploy this code first with the defaults (no `SMF_SCHEDULER_GROUPS`): nothing changes yet. `./deploy.sh --check` shows whether it is safe.
2. In one `docker compose up -d`: start `scheduler` and recreate `api` with `SMF_SCHEDULER_GROUPS=api`. Heavy jobs miss at most one tick.
3. Check: `docker logs show-me-fire-scheduler` ("Scheduler worker running groups heavy"), the heartbeat file, `/health/ready` (scheduler "heavy
   scheduler worker alive"), and the admin Logs page: every job should still run exactly once (heavy ones show "scheduler worker").
4. Rollback: set the API back to `api,heavy` (or unset), recreate it, stop `scheduler`.

## `deploy.sh` changes (the user edits this on the server)

- A deploy that only changes code the API runs: recreate **only** the API. The wait for forecast-v1 phases no longer applies to it.
- The guard now matters for the **scheduler** container (and the cron worker, as before). Read the running state from files, not the API:
  - forecast_v1: `data/forecast-v1/admin-job.json` `status` is `running` **and** `updated_at` is under ~15 min old (a crashed run leaves a stale
    file; the next worker start reconciles it), or `flock -n data/forecast-v1/.run.lock true` fails (a run holds the lock);
  - heavy jobs in general: `jq '.running_jobs' data/current_conditions/status/scheduler_heavy.json` is not empty;
  - cron jobs in the cron worker: unchanged (`pgrep -f run_job.sh`).
- Recreate the scheduler container only when its code matters (a heavy job changed) and the guard says nothing is running. A stop waits up to
  `SMF_SCHEDULER_STOP_WAIT_SECONDS` for running jobs; forecast_v1 itself cannot be paused mid-run, so a stop during one still loses that run
  (it restarts from the next 30-minute tick).
- The repo is bind-mounted into every container, so a `git pull` changes files under a running scheduler: it keeps the modules it already loaded but reads later (lazy) imports from the new files. Keep heavy-job changes additive and restart the scheduler (full guard) when they change.

## Manual actions that still run inside the API

An API deploy during one of these kills it (they are one-off and short): forecast-v1 `POST /retention`, spread-rate `generate`/`warmup`,
`archive_admin POST /sync`, `day_archive_admin POST /run`, rtma-peak `generate`, beta verification, burn-ban map regeneration. They are not
serialized against the worker's scheduled copy of the same job.

## Known gaps

- `retrain_detection_confidence` and `post_promotion_monitor` run in the worker, but `services/detection_confidence._booster` is loaded once per
  process, so a retrained model reaches the API at its next restart (as before). Whether `post_promotion_monitor`'s registry rollbacks reach the
  API's model loaders without a restart was not checked.
- `drift_check` stays in the API on purpose (it fills `drift_monitor._state`, which routers read).
