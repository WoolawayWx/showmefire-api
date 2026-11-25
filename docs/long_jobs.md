# Long jobs in their own containers

Some jobs run for hours (forecast-v1 about 6 h; the 12z forecast, ensemble fire danger and seven-day outlook for tens of
minutes to hours). A restart kills them, so deploys used to wait for them. They now run in two containers with the code baked into the image. Their rollout waits for an idle window.

| container | runs | image |
|---|---|---|
| `show-me-fire-forecast` | scheduler group `forecast`: `run_forecast_v1_shadow`, `forecast_v1_admin_requests`, `prune_forecast_v1_hot_storage` | `showmefire-api:long` |
| `show-me-fire-worker-long` | cron `SMF_CRON_SET=long`: `forecasts_12z`, `forecasts_09z`, `ensemble_12z`, `extended_forecast`, `fire_outlook`, `fire_outlook_verification` | `showmefire-api:long` |
| `show-me-fire-scheduler` | scheduler group `heavy`: HRRR hourly, RTMA, graphics, archives, verification (minutes each) | `showmefire-api:latest`, code bind-mounted |
| `show-me-fire-worker` | cron `SMF_CRON_SET=short`: `maps`, `verification`, `empty_outlook_maps` | `showmefire-api:latest`, code bind-mounted |
| `show-me-fire-api` | web API + quick jobs | `showmefire-api:latest`, code bind-mounted |

Without the two new compose services nothing changes: the defaults (`SMF_SCHEDULER_GROUPS` unset, `SMF_CRON_SET=all`) run
everything as before. To run both scheduler groups in one worker (the old layout): `--groups heavy,forecast`.

## How a deploy treats them

`~/showmefire/deploy.sh` fetches the upstream commit and plans changes before advancing the live checkout. It tracks applied
revisions in `~/showmefire/.deploy-state`. An advisory-category map edit reloads the backend and reaches the short worker on its
next map run. `--api-only` leaves scheduler and long-worker work for a normal deploy. `--check` reports current restart blockers.

When a release needs an image, the ARM64 host builds a clean Git snapshot as `showmefire-api:<full-commit>`. A retry reuses
that tag. Standard services use `showmefire-api:latest`; the long containers use `showmefire-api:long`. For the long rollout,
the script takes an exclusive lock after the build and checks for running jobs and upcoming daily jobs. A busy rollout stays
in `~/showmefire/.long-pending`. Use `./deploy.sh --long` to retry the release already in the live checkout, or let the next
normal deploy pick it up. The script verifies the image IDs, cron process, and a new forecast heartbeat before clearing pending
state. Schedule any incompatible changes to shared data for a coordinated rollout.

## Install or update on the server

1. Once the release is on the tracked upstream branch, run the installer from `~/showmefire` before advancing the live
   checkout. It backs up both active files, updates the two long-worker environment lists, and validates Compose.

   ```bash
   cd ~/showmefire
   git -C showmefire-api fetch
   release=$(git -C showmefire-api rev-parse '@{upstream}')
   git -C showmefire-api show "$release:docs/server/install-deploy.sh" | bash -s -- "$release"
   ```

   The installer expects an existing split layout with the shared `./showmefire-api/.locks:/app/.locks` mount in both long
   services. For a first-time split, use the Compose template. The normal deploy applies the tracked Dockerfile.
2. Check free space on the Docker host with `df -h` and review cache usage with `docker buildx du`. Ensure
   `showmefire-api/.locks` and `showmefire-api/images` exist. Validate with `docker compose config --quiet` from
   `~/showmefire`. Confirm an idle window with `./deploy.sh --check` and verify the legacy workers have no active jobs before
   the first switch, since their jobs predate the shared lock.
3. Run `./deploy.sh`. If the long rollout is deferred, run `./deploy.sh --long` once idle. Confirm the API answers on
   `http://localhost:5001/health` and both long workers appear in `docker compose ps`. The admin Logs page should show
   forecast jobs on the forecast worker. For a first-time split, apply the Compose settings for all five services in the
   template so the heavy scheduler and short cron worker drop the jobs moving to the long containers.
4. To roll back the long containers, first confirm they are idle with `./deploy.sh --check`. From `~/showmefire` run:

   ```bash
   flock -n .deploy.lock flock -n showmefire-api/.locks/long-deploy.lock bash -ec '
       docker tag showmefire-api:long-previous showmefire-api:long
       docker compose up -d --no-build --no-deps forecast worker-long
   '
   ```

   Verify the restored image and heartbeat. Leave the pending marker in place so the intended rollout can be retried.

## Check build performance

Run two native ARM64 builds from clean snapshots of different commits whose differences are limited to application source.
Use separate local image tags and `docker build --progress=plain --build-arg GIT_SHA=<commit> -t <tag> <snapshot>`. Record the
first build time and the warm build time separately from the image-switch time. The warm build should show the OS package,
locked Python dependencies, Pyretechnics, and virtual environment copy steps as cached. The initial target is under two minutes.
When dependencies cache but export dominates, measure export and unpack separately. If dependency layers miss, check the builder
and its cache with `docker buildx ls` and `docker buildx du`; deploys rely on BuildKit disk-based cleanup.

Refresh the pinned Python and uv digests in `Dockerfile` deliberately as part of dependency maintenance. Resolve the new
digests with `docker buildx imagetools inspect python:3.11-slim` and `docker buildx imagetools inspect ghcr.io/astral-sh/uv:0.8`,
then verify a native ARM64 build before deploying that update.

## Watching run times
`services/job_runs.py` records every run (60 days) and the runs in progress. The admin Logs page job list now carries
`duration` (runs, median, p95, max, latest; runs under a minute are ignored so the seven-day outlook's instant "nothing to do"
exits do not distort it) and `running` (started, elapsed, overrunning). The `long_run_watch` job (API group, every ~10 min)
alerts staff once when a run exceeds 1.5 x its p95 (at least 15 min past it), after 5 runs of history. It lives in the API so it
still fires if a worker is the thing that is stuck.

Fire Weather Hybrid (`scripts/run_fire_weather_hybrid.py`) is not in the baked crontab or the APScheduler registry; if it is
started by hand or from the server's own crontab, run it through `scripts/run_job.sh <id> <script>` so its runs are tracked.
