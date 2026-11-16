# Long jobs in their own containers

Some jobs run for hours (forecast-v1 about 6 h; the 12z forecast, ensemble fire danger and seven-day outlook for tens of
minutes to hours). A restart kills them, so deploys used to wait for them. They now run in two containers that a deploy never
restarts, built from a pinned image with the code baked in.

| container | runs | image |
|---|---|---|
| `show-me-fire-forecast` | scheduler group `forecast`: `run_forecast_v1_shadow`, `forecast_v1_admin_requests`, `prune_forecast_v1_hot_storage` | `showmefire-api:long` |
| `show-me-fire-worker-long` | cron `SMF_CRON_SET=long`: `forecasts_12z`, `forecasts_09z`, `ensemble_12z`, `fire_outlook`, `fire_outlook_verification` | `showmefire-api:long` |
| `show-me-fire-scheduler` | scheduler group `heavy`: HRRR hourly, RTMA, graphics, archives, verification (minutes each) | `showmefire-api:latest`, code bind-mounted |
| `show-me-fire-worker` | cron `SMF_CRON_SET=short`: `maps`, `verification`, `empty_outlook_maps` | `showmefire-api:latest`, code bind-mounted |
| `show-me-fire-api` | web API + quick jobs | `showmefire-api:latest`, code bind-mounted |

Without the two new compose services nothing changes: the defaults (`SMF_SCHEDULER_GROUPS` unset, `SMF_CRON_SET=all`) run
everything as before. To run both scheduler groups in one worker (the old layout): `--groups heavy,forecast`.

## How a deploy treats them
`./deploy.sh` never recreates or restarts them. At the end it compares the commit they run (image label `smf.git_sha`) with
the new one; if the diff touches code they import (anything but `routers/`, `main.py`, docs, tests) it moves them, but only
when idle (no forecast-v1 run, no long cron job running, no long daily job about to start). If busy it prints why, writes
`~/showmefire/.long-pending`, and leaves them on their current code. Later: `./deploy.sh --long` (also run by the next
`./deploy.sh`). `./deploy.sh --check` shows both states. They can therefore run older code than the API, for up to one run
(about 6 h). The database changes are additive, so that is safe; a change to something both sides write (a table the forecast
reads and the API migrates) needs a coordinated move: wait for idle, `./deploy.sh --long`.

Rollback: `docker tag showmefire-api:long-previous showmefire-api:long && docker compose up -d --no-deps forecast worker-long`.

## First-time cutover (one idle moment)
The old `scheduler` container still runs forecast-v1 until it is restarted onto the new code, so do these together:
1. `./deploy.sh --check` until it says SAFE (no forecast run, no cron job).
2. Edit `~/showmefire/docker-compose.yml` to match `docs/server/docker-compose.yml` (new `forecast` and `worker-long`
   services; `worker` gets `SMF_CRON_SET=short`; backend `build.args.GIT_SHA`). `mkdir -p showmefire-api/.locks showmefire-api/images`.
3. `./deploy.sh` (it sees the changed compose and recreates backend, scheduler and worker; the scheduler comes back on
   `--groups heavy` without the forecast jobs).
4. `./deploy.sh --long` (builds the pinned image and starts `forecast` and `worker-long`).
5. Verify:
   - `docker ps`: five healthy containers; `jq '.groups' data/current_conditions/status/scheduler_forecast.json` is `["forecast"]`.
   - Admin Logs page: forecast jobs show "forecast worker", cron jobs still list; `curl localhost:5001/health/ready` shows both workers alive.
   - Files the long containers wrote outside a mount (these would be lost on the next move). After a day:
     `docker exec show-me-fire-forecast sh -c 'find /app -type f -newer /proc/1/cmdline -not -path "*/__pycache__/*" -not -path "/app/data/*" | head -50'`
     and the same for `show-me-fire-worker-long`. Anything listed that matters needs a mount in the compose file.

## Watching run times
`services/job_runs.py` records every run (60 days) and the runs in progress. The admin Logs page job list now carries
`duration` (runs, median, p95, max, latest; runs under a minute are ignored so the seven-day outlook's instant "nothing to do"
exits do not distort it) and `running` (started, elapsed, overrunning). The `long_run_watch` job (API group, every ~10 min)
alerts staff once when a run exceeds 1.5 x its p95 (at least 15 min past it), after 5 runs of history. It lives in the API so it
still fires if a worker is the thing that is stuck.

Fire Weather Hybrid (`scripts/run_fire_weather_hybrid.py`) is not in the baked crontab or the APScheduler registry; if it is
started by hand or from the server's own crontab, run it through `scripts/run_job.sh <id> <script>` so its runs are tracked.
