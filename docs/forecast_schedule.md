# 12z forecast schedule (early start, prefetch, shared HRRR data)

The 12z HRRR hours are all on AWS by about 13:14Z (f04 ~12:56Z, **f15 ~13:10Z**, f18 ~13:14Z; measured on three consecutive
days), but the forecast used to start at a fixed 14:30Z. It now starts when its inputs are ready.

## Daily timeline (UTC, cron `TZ=UTC` in the worker container; see `api/Dockerfile`)

| Time | What | Where |
|---|---|---|
| 12:05 | 09z forecast starts polling, then runs (holds the forecast lock while it runs) | `scripts/forecasts_09z.sh` |
| 13:00 | **12z forecast** starts polling; runs once f04-f15 are on AWS (about 13:10Z) and it is at least 07:30 Central | `scripts/forecasts.sh` |
| 15:00 | **Ensemble (beta)**, needs REFS (~14:36Z) and HREF (~15:06Z) | `scripts/ensemble_12z.sh` |
| every 3 min, all day | **progressive hourly build**: add each newly published HRRR hour of the newest cycle to the product and publish it to the CDN (the peak fire danger product too) | APScheduler `hrrr_hourly_forecast` |

## Progressive build: each hour is shown as soon as it is processed

`services/hrrr_hourly/progressive.py`. Each tick (`scripts/hrrr_hourly_forecast.py --auto`) looks at the newest cycles (from 45 minutes
after the cycle, when f00 is about to post) and, for the newest one with work, processes every hour HRRR has published and that is not
done yet (fetch, derive, save a work file, as the one-shot build did). After **every** hour it:

* writes that hour's chunk into `<cycle>/forecast.zarr`, which was created at full length (f00-f18) with every array empty, so the
  metadata never changes and only new chunk files appear (`live_store.py`);
* rewrites `manifest.json`: `forecast_hours` (the hours in so far), `pending_hours`, `complete: false`, `zarr_leads` (the Zarr index
  of every hour), `revision`;
* updates the peak fire danger product for the local day the hour belongs to;
* uploads the new chunk(s) and the manifest to the CDN (`cdn_datasets.publish_update`). A chunk is cached for a year, a manifest
  for a run still filling in for a minute; the finished run's manifest is re-uploaded with the year-long cache. An upload that fails is
  queued in `<cycle>/live.json` and goes up with the next hour.

The cycle is only made **latest** (`latest.json` locally and on the CDN) once `SMF_HRRR_HOURLY_LATEST_AFTER_HOURS` (default 6) hours are
in. Until then it is listed in `runs.json` with `ready: false` and reachable by its id. When all 19 hours are in (or an hour has failed
3 times, or has not appeared 3 hours after the cycle, `SMF_HRRR_HOURLY_GIVE_UP_HOURS`) the cycle is finished: the NetCDF copy is written,
the manifest is marked `complete` and the work files are removed. State is all on disk, so a crashed tick carries on at the next one.

No longer waits for the daily forecast: it runs alongside it (it still waits for the 15-minute maps and for free memory and disk).
`SMF_HRRR_HOURLY_PROGRESSIVE=0` goes back to building whole cycles after f18 (`job.run_auto`); `--cycle` builds are always one-shot.

## Peak fire danger product (estimate)

`services/hrrr_hourly/peak.py`, dataset `hrrr-fire-danger-peak` on the CDN, `/hrrr-fire-danger-peak/...` on the API. For each local day an
HRRR run reaches at least 3 hours of the 10 AM - 9 PM Central window of: the peak fire danger (maximum over the window), the hour it first
happens, minimum RH and fuel moisture, maximum wind, gust, temperature and HDW. The same peak logic as the daily forecast, but an
**HRRR-only estimate**: the rule is applied (`danger.py`, parity-tested against `core.fire_danger.calculate_fire_danger`) to the hourly
product's estimated fuel moisture (3 + 0.25 x RH), not the RAWS-based fuel moisture the official forecast starts from, and without its snow
mask, smoothing or green-up adjustment. A day shows the peak so far until all its window hours are in (`volatile_steps` in the manifest).
The hourly product also has the per-hour `fire_danger_est` layer it is built from. Local files: `data/hrrr_fire_danger_peak/` (`SMF_HRRR_PEAK_DIR`).

## How the 12z forecast start works

* `scripts/forecasts.sh` exports the 12z settings, runs `scripts/wait_for_hrrr.py` **before** taking the forecast lock (so
  `maps.sh` keeps running while it only polls), then takes the lock and runs everything as before.
* `FORECAST_AVAILABILITY_LAG_HOURS=1` makes `DailyForecast.py`, `DailyForecast_ModelFD.py` (the beta run later in the
  script) and `services/hrrr_readiness.cycle_run_time` treat today's 12z as the current run from 13:00Z. The default stays
  2 (the old "before 14:00Z use yesterday's run" rule) for every other flow, e.g. 09z.
* `FORECAST_NOT_BEFORE_LOCAL=07:30` holds the start until 07:30 Central, because the forecast starts from RAWS fuel
  moisture observed around 7 AM (+-1 h). That is 12:30Z in daylight time (before the files exist, so no effect) and
  13:30Z in standard time (the real limit in winter).
* `HRRR_READINESS_TIMEOUT_SECONDS=5400` from the 13:00Z start is 14:30Z, the old fixed start. If HRRR is that late the
  forecast starts anyway with `REQUIRE_COMPLETE_HRRR=false`, as before.
* A running 09z forecast holds the lock; 12z waits for it (blocking `flock`), so a slow 09z delays 12z.
* The ensemble step moved to its own cron at 15:00Z (`scripts/ensemble_12z.sh`, same lock, always exits 0) so it still
  gets REFS/HREF and no longer delays publication or the notices.

## The shared cache

* The 12:45Z prefetch cron is gone: the progressive build (above) saves each hour's work file `.work/<cycle>/f##.npz` as it
  posts and, once f04-f15 are in, assembles the forecast's cache file. (`--prefetch` still works by hand, or from a leftover
  cron entry in an old image: it makes one quick progressive pass. With `SMF_HRRR_HOURLY_PROGRESSIVE=0` it is the old
  polling `job.prefetch_cycle`.)
* The work files also keep `raw_*` surface fields (2 m T in K, RH, 10 m wind, gust) and `grid_*` projection details; the
  loader drops them so they never reach the Zarr, series or manifest.
* When f04-f15 are saved the progressive build assembles the forecast's cache file (`services/hrrr_shared_cache.py`):
  the four fields above from the work files, precipitation from the identical `:APCP:` FastHerbie call (the forecast
  differences it as a cumulative record, so it is not rebuilt from the 1-hour windows).

### Rolling the shared cache out

1. **Comparison mode (default).** The file is written to `cache/hrrr_shared/` and `DailyForecast` downloads its own as
   always. `scripts/compare_hrrr_cache.py` (run from `forecasts.sh`, report only) compares the two and appends a line to
   `data/hrrr_shared_cache_compare.jsonl`; the log says MATCHES or DIFFERS with the reasons.
2. After a few days of MATCHES, set `FORECAST_USE_SHARED_HRRR=true` in the worker environment. The build then writes
   `/app/cache/hrrr/hrrr_<date>_12z_f04-15.nc` (atomically) and `DailyForecast` takes its existing cache-hit path.
   `DailyForecast` refuses a cache that lacks `t2m/r2/u10/v10` or any forecast hour and downloads instead.
3. Rollback: unset `FORECAST_USE_SHARED_HRRR`. Nothing else depends on it.

## Every hourly run

The hourly viewer product is built for **every** HRRR cycle (24 a day), not just 12z.

* `progressive.advance` runs every 3 minutes (APScheduler, all day; `job.run_auto` when `SMF_HRRR_HOURLY_PROGRESSIVE=0`). Each tick takes the newest cycle old enough to be ready
  (`fetch.cycle_candidates`: cycle + 1h05m or later; f18 is out about 1h14m after the cycle) plus
  `SMF_HRRR_HOURLY_BACKFILL_CYCLES` (default 3) earlier ones, and builds the newest that is published and not built yet.
  A builder that falls behind therefore skips old cycles instead of building them hours late.
* Only the **12z** cycle waits for the daily forecast to report `updated` (or 4 h after the cycle, `FORECAST_GRACE`). Every
  cycle waits while the forecast holds its lock, and builds at the lowest priority with the same memory and disk guards.
* `latest.json` (and so the viewer's "Latest") only ever moves forward, even if a late older cycle finishes after a newer one.
* Retention: 7 days locally (`SMF_HRRR_HOURLY_KEEP_DAYS`, was 14) and 7 days on the CDN, about 168 runs. The CDN run list is
  capped at 400. Abandoned `.work/<cycle>` folders are removed after 2 days.
* The shared HRRR cache for the daily forecast remains 12z only.
* The viewer groups its Run menu by local (Central) day.

Rough load at 24 runs a day (estimates; check `timings.json` and the server): about 32 MB of Zarr per run on the CDN
(about 5.4 GB for 7 days; progressive publishing makes about 28 chunk uploads per hour plus a few small files, so roughly 14,000 uploads a day, and the peak product adds a few thousand), the NetCDF copies grow `data/hrrr_hourly/nc` and the R2 archive
(`archive_store.py` source `hrrr_hourly`) by one file per run, and each run downloads roughly 19 hours of HRRR (the code
comments say about 60 MB per hour, so on the order of 1 GB per run, 25 GB a day). To back down, set `SMF_HRRR_HOURLY_CYCLE_HOURS`
(for example `0,6,12,18`, or `12` for the old one-a-day) on the backend and restart it; runs already built stay until they age out.

## Admin controls (website: Admin > System > HRRR Runs)

* **Which runs are built.** The page edits `data/hrrr_hourly/settings.json` (`services/hrrr_hourly/controls.py`), through
  `GET/PUT /admin/hrrr-hourly/settings`. The scheduler starts a fresh `--auto` process every 3 minutes and it reads the file,
  so a change applies within 10 minutes with no restart. Precedence: the saved hours, then `SMF_HRRR_HOURLY_CYCLE_HOURS`, then
  all 24. "Stop overriding" clears the saved hours.
* **Pause.** Stops new builds from starting (a build already running finishes); the status reads `paused` and the viewer
  says updates are paused. An explicit `--cycle` or `--force` run is not blocked. Unpausing clears the status on the next tick.
* **History.** Every build outcome (complete, partial, failed, with runtime and error) is appended to
  `data/hrrr_hourly/history.jsonl` (trimmed to the last 500), shown on the page next to a 3-day timeline of cycles
  (`GET /admin/hrrr-hourly/coverage`): built, partial, enabled but not built, not enabled.

## API metrics (website: Admin > System > API Metrics)

`core/metrics.py` times every HTTP request in a pure ASGI middleware (outermost, registered in `main.py`) and keeps
per-minute and per-route rows for 24 hours, saved to `data/api_metrics.json` every minute and reloaded at start-up.
`GET /admin/metrics/api?window=<minutes>` (admin only) returns the overview, a time series and a table of routes by
template (never the raw URL, so route count stays bounded at 400). Latency is the time until the response starts. Client
addresses are not kept. It counts this API process only, so cron jobs and the HRRR builder subprocesses are not included, and
the metrics page's own requests are skipped.

### Server metrics (the Server tab of the same page)

`core/server_metrics.py` samples the machine every 10 seconds with psutil (a pinned dependency) in a background thread started
by the API, averages it per minute for 24 hours, and saves it to `data/server_metrics.json` once a minute.
`GET /admin/metrics/server?window=<minutes>` returns the current reading, a time series, disk volumes, folder sizes, the
busiest processes and warnings. Shown: CPU (with time waiting on disk and stolen by the hypervisor), load against core
count, memory, swap, disk and network rates, the API process (CPU, memory, threads, open files), the container's memory and
CPU against its limit, disk space, the size of `data`, `data/hrrr_hourly`, `images`, `cache`, `archive` and `logs`, and
warnings (disk over 90% full, memory under 10% free, swap over half used, load over twice the cores, CPU over 90% for
minutes, the container near its memory limit).

On the Ubuntu server with the API in Docker: CPU, memory, swap, load and disk are the host's (the container shares its
`/proc`); network is the container's own interface; the container's memory and CPU come from its cgroup (v2 on current
Ubuntu, v1 also handled). Processes are those in the API's container, which includes the HRRR build while it runs; the cron
jobs in the worker container are not listed. Processes are shown by name and the basename of a `.py` script, never the full
command line. Folder sizes are counted every 10 minutes, each folder within 15 seconds, and marked partial if it takes longer.

If the worker container also starts this app on the shared data volume, set `SMF_METRICS_ENABLED=0` there (or give it its own
`SMF_API_METRICS_PATH` / `SMF_SERVER_METRICS_PATH`) so it does not overwrite the API's saved history.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `FORECAST_AVAILABILITY_LAG_HOURS` | 2 (1 in `forecasts.sh`) | hours after the cycle before it is "today's run" |
| `FORECAST_NOT_BEFORE_LOCAL` | unset (07:30 in `forecasts.sh`) | earliest start, Central `HH:MM` |
| `HRRR_READINESS_TIMEOUT_SECONDS` / `_POLL_SECONDS` | 7200 / 300 (5400 in `forecasts.sh`) | how long / how often `wait_for_hrrr.py` polls |
| `SMF_HRRR_HOURLY_PREFETCH_TIMEOUT_SECONDS` / `_POLL_SECONDS` | 7200 / 60 | prefetch wait for an unpublished hour |
| `SMF_HRRR_HOURLY_BACKFILL_CYCLES` | 3 | how many earlier cycles a tick may still build |
| `SMF_HRRR_HOURLY_CYCLE_HOURS` | all 24 | UTC hours whose run is built; `0,6,12,18` backs down to four a day, `12` to one |
| `SMF_HRRR_HOURLY_KEEP_DAYS` | 7 | days of runs kept on the server |
| `SMF_METRICS_ENABLED` | on | `0` stops the metrics saving and the server sampler in that container |
| `SMF_API_METRICS_PATH` / `SMF_SERVER_METRICS_PATH` | `data/api_metrics.json` / `data/server_metrics.json` | where the snapshots are saved |
| `SMF_METRICS_DIRS` | `data,data/hrrr_hourly,images,cache,archive,logs` | folders whose size is tracked (relative to the API folder) |
| `FORECAST_USE_SHARED_HRRR` | off | write the live cache path instead of the shadow copy |
| `SMF_HRRR_CACHE_DIR` / `SMF_HRRR_SHARED_CACHE_DIR` | `/app/cache/hrrr` / `cache/hrrr_shared` | live and shadow cache directories |

## Deploying

The cron entries are baked into the **worker** image, so the Dockerfile change needs a rebuild and redeploy of the worker.
The backend (APScheduler) needs the new `core/scheduler.py` window (hour 7-14 Central). Worker and backend must share
`data/` and `.locks/`. Roll back the schedule by restoring `30 14` for `forecasts.sh`, removing the `hrrrprefetch` and
`ensemble12z` cron files, and moving the ensemble step back into `forecasts.sh`.
