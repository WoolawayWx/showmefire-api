# showmefire-api
api and server files for the backend of showmefire.org

The API collects and archives operational data, imports verified model
releases, and serves predictions. Training and static-raster preprocessing
belong in `ShowMeFire-Models`. See its
[`docs/spatial_fuel_moisture_runbook.md`](https://github.com/Cade417/ShowMeFire-Models/blob/main/docs/spatial_fuel_moisture_runbook.md)
for the complete operator workflow, or its
[`docs/model_lifecycle.md`](https://github.com/Cade417/ShowMeFire-Models/blob/main/docs/model_lifecycle.md)
for the general publish/import/promote workflow shared by every model family
(not just the spatial one) — including the website's Import button
(`POST /api/admin/models/{family}/import`) as an alternative to the CLI
commands below.

## RTMA and historical fuel-moisture capture

Hourly RTMA capture is registered with APScheduler at minute 50 and targets
the previous complete UTC analysis hour. Historical maintenance commands are
explicit and resumable:

```bash
# Backfill the rolling one-year Synoptic entitlement in UTC daily chunks.
python scripts/backfill_synoptic.py --dry-run
python scripts/backfill_synoptic.py

# Sync versioned HRRR, observations, and forecasts to R2. RTMA is excluded.
python -m services.archive_bundler
```

Fuel moisture is sourced only from Synoptic station observations. RTMA is
stored as analyzed meteorological input and is never used as an FM label.
Hourly live RTMA is retained for seven days by default and is not permanently
archived. Spatial inference uses initialization minus 12 hours through
initialization (13 causal frames). It carries an earlier frame across at most
two missing hours and falls back to XGBoost when three are missing. The API
never loads future RTMA. Historical/realized RTMA is fetched in `ShowMeFire-Models` from local HRRR
initialization timestamps.

Spatial releases contain ONNX, checkpoint, static NetCDF, manifest,
evaluation, and smoke-test assets. Import verifies the whole contract before
registering beta:

```bash
python pipelines/import_model.py --model fuel_moisture_spatial --tag <release-tag> --repo Cade417/ShowMeFire-Models
```

Once explicitly promoted, forecast generation attempts spatial inference and
automatically retains XGBoost output on any missing input or contract/runtime
failure. Current status is exposed at `/api/model/spatial/diagnostics`.

## Verification against RTMA and API access limits

The nightly report (`forecast/endOfDayReport.py`) adds an `rtma_metrics` block next to the station metrics: forecast-vs-RTMA and station-vs-RTMA scores sampled at station locations from the cached hourly RTMA (`services/rtma_verification.py`), plus the statewide peak-class forecast raster against the RTMA peak raster. RTMA fuel moisture and fire danger are derived from RTMA RH, not observed. Bias is first-named minus RTMA. RTMA hours are only cached 7 days, so metrics are computed and stored at report time (backfill with `POST /api/admin/verification/rerun` inside that window).

Public read routes (`/verification/*`, `/forecast-09z/metrics/*`, `/routes`) are rate limited per client IP (`core/rate_limit.py`). The route list is at `/routes` (JSON) and `/routes/page` (HTML).

| Env var | Default | Effect |
| --- | --- | --- |
| `RATE_LIMIT_VERIFICATION_PER_MIN` | 60 | per-IP limit on `/verification/*` |
| `RATE_LIMIT_VERIFICATION_STATION_PER_MIN` | 20 | per-IP limit on station-level `/comparisons` |
| `RATE_LIMIT_FORECAST_09Z_PER_MIN` | 60 | per-IP limit on `/api/forecast-09z/metrics/*` |
| `API_READ_KEYS` | empty | comma-separated Bearer keys; 10x limit |
| `VERIFICATION_STATION_DATA_REQUIRES_KEY` | false | require key/admin for station-level rows |
| `REPORTS_DATA_PUBLIC` | false | allow public download of `/reports/*.json\|csv` |

Every non-admin `/api/<x>` route is also served at `/<x>` (`core/api_aliases.py`, applied at the end of `main.py`; set `API_UNPREFIXED_ALIASES=false` to disable). The `/api` paths are unchanged, and aliases use the same handlers, auth and rate limits. Not aliased: `/api/admin/*`, `/api/graphics/auth/*`, and paths that would collide with an existing route or mount (currently `/api/home-video` and `/api/images/timeline`; skips are logged at startup).

Limits are per worker process; behind a proxy set `TRUST_PROXY_HEADERS=true` so the real client IP is used.

## Mobile API and push notifications

`GET /api/mobile/content` provides the public forecast, outlook, SitRep,
fire-weather alert, and Missouri county contract used by the standalone mobile
app. Anonymous installations manage category and county preferences through
`PUT` and `DELETE /api/mobile/push-subscriptions/{installation_id}`.

Forecast completion, newly issued Red Flag Warnings and Fire Weather Watches,
and newly activated SitReps fan out through Expo Push Service. The first NWS
polling cycle seeds active products without notifying. APScheduler checks Expo
delivery receipts every 15 minutes and deletes subscriptions reported as
`DeviceNotRegistered`. `DELETE` is idempotent and removes the anonymous
subscription together with its tickets and receipts. Delivery records for
active subscriptions are purged after seven days.

Set `PUBLIC_API_BASE_URL` and `PUBLIC_CDN_BASE_URL` when the public hosts differ
from production defaults. Set `EXPO_ACCESS_TOKEN` only when enhanced Expo push
security is enabled; never expose it to the mobile client.

## Public fire reports and the fire-event store

`fire_events` is the unified store for fire observations from every source:
anonymous public reports, FIRMS/VIIRS and MODIS thermal anomalies, NGFS
events, and official confirmations. Each row carries a `source` discriminator
and a `verification_tier` (`unverified`, `admin_reviewed`,
`official_source_confirmed`). Only `official_source_confirmed` rows are
primary model labels; `admin_reviewed` rows form a down-weighted auxiliary
set, and `unverified` rows - including every satellite detection - are never
labels.

`POST /api/fires/reports` is the only unauthenticated write path in this
API. It requires a Cloudflare Turnstile token, enforces per-IP and global
rate limits, and stores every submission as `status='pending'`. Nothing is
publicly readable or label-eligible until an administrator approves it.
`GET /api/fires/events` and `GET /api/fires/events.geojson` return approved
events only; the GeoJSON properties intentionally match the uppercase shape
served by `/fires/satdet` so existing map clients need no changes. The
file-backed `/fires/satdet`, `/fires/detections/advanced`,
`/api/fires/missouri`, and `/api/fires/missouri/geojson` endpoints are
unchanged and remain the live feed.

Moderation runs through `/api/admin/fires/reports` (list, approve, reject)
and `/api/admin/fires/events/{id}` (edit, delete). Every state change appends
to `fire_event_moderation`, which is append-only. Training labels are
exported with `python scripts/export_fire_labels.py`, which writes a CSV and
a checksummed manifest to `data/fire_labels/`. Backfill and re-ingest of the
detection files is `python scripts/backfill_fire_detections.py --dry-run`.

Environment variables: `TURNSTILE_SECRET_KEY` (required in production; the
API refuses to start without it), `FIRE_REPORT_IP_SALT`,
`TRUST_PROXY_HEADERS`, `FIRE_REPORT_LIMIT_PER_HOUR`,
`FIRE_REPORT_LIMIT_PER_DAY`, `FIRE_REPORT_GLOBAL_LIMIT_PER_DAY`,
`FIRE_REPORT_MAX_AGE_DAYS`, `FIRE_LABEL_ADMIN_REVIEWED_WEIGHT`.

See [`docs/fire_report_moderation_runbook.md`](docs/fire_report_moderation_runbook.md)
for the operator playbook.
# GIS publication

The production forecast and observation jobs publish MapServer-ready data when
`SMF_GIS_PUBLISH_DIR` is set. Use a persistent host-mounted directory shared
with the `showmefire-mapserver` deployment; the API needs read/write access and
MapServer mounts the same directory read-only as `/data`.

Optional settings:

- `SMF_GIS_RESOLUTION_METERS` (default `3000`) controls the fixed Missouri
  EPSG:32615 grid resolution.
- `SMF_GIS_RETENTION_DAYS` (default `30`) controls time-stamped raster history.

The API also serves `catalog.json` and `burn_bans.geojson` beneath `/gis`.
Operational GeoTIFFs and GeoPackages use EPSG:32615; public GeoJSON uses
EPSG:4326.

## R2 archive and data operations

Versioned, checksummed data objects replace daily ZIP production. The admin
page at `/admin/archive` tracks source coverage, upload progress, and processing
status. See [the migration and operations runbook](docs/r2_archive_runbook.md)
for configuration, existing ZIP imports, restores, and training downloads.
Local pruning is opt-in; existing ZIPs remain readable.

## Day archive and route list

Each completed Central day is bundled into R2 (`day-archive/`) with everything needed to
reconstruct it, kept 30 days, with Discord reminders to download it; the server keeps source
files 7 days. Download with `scripts/pull_day_archives.py` or the `/admin/day-archive`
endpoints. Details: [`docs/r2_archive_runbook.md`](docs/r2_archive_runbook.md). Every route,
including admin routes, is listed in [`docs/ROUTES.md`](docs/ROUTES.md), generated by
`python scripts/generate_routes_doc.py` (`--check` to verify it is current).

## Hourly HRRR forecast (12z run)

`services/hrrr_hourly/` builds an hour-by-hour grid product from the HRRR 12z run, f00-f18: mixing height,
transport wind, ventilation, inversion, lapse rate, Haines, HDW, 2 m T/RH/dewpoint/VPD, 10 m wind and gust,
estimated fuel moisture, LCL, CAPE/CIN, PWAT, cloud cover, hourly precip and smoke. It is separate from the daily
forecast: its own directories (`data/hrrr_hourly/`, `images/hrrr_hourly/`), and it never writes to
`archive/forecasts`, `cache/hrrr`, `images/mo-*`, `gis/` or the station forecast table.

- Job: `scripts/hrrr_hourly_forecast.py --auto` (APScheduler `hrrr_hourly_forecast`, every 20 min 09:00-14:59 Central).
  The daily forecast always goes first: this waits until f18 of the 12z run is published, `status.json`
  `ForecastFireDanger` reports that same 12z run as `updated`, and the forecast lock is free (its script holds the
  lock through the Discord/mobile notices and the beta run). If the forecast never reports, it proceeds 6 h after
  the cycle. Nothing runs alongside the forecast.
  Priorities are forecast+notices, then the 15-minute maps, then this: the build runs at nice 19 / idle I/O with one
  numeric thread, pauses between hours while the forecast or a `maps.sh` cycle holds its lock (capped at 45 / 10
  min), waits for 3 GB of free memory, and refuses to start with under 3 GB of free disk
  (`SMF_HRRR_HOURLY_MIN_MEM_MB`, `SMF_HRRR_HOURLY_MIN_DISK_MB`). The status file shows `phase: paused` and `paused_for`.
- Front-end status: `data/hrrr_hourly/status.json`, also `GET /hrrr-hourly/status`. `state` is `waiting_for_hrrr`,
  `waiting_for_forecast`, `building` (`phase` fetching, `hours_done`/`hours_total`), `rendering`, `complete`,
  `partial` or `failed`; `updating` is true only while building/rendering, `pending` while waiting, and
  `serving_cycle` is the cycle still being served until the new one is swapped in.
  Manual: `--cycle YYYYMMDD [--hours 0-18] [--force]`. Status: `data/hrrr_hourly/status.json`.
- Storage per cycle (`<YYYYMMDD>_12z`): `forecast.zarr` (one-hour chunks, consolidated metadata, readable from JS
  and Python), `manifest.json` (run/issue time, variables with units, grid, statewide hourly series), and a flat
  NetCDF `nc/hrrr_hourly_<cycle>.nc` that the R2 archive and day bundles pick up. Zarr needs the `zarr` package;
  without it only the NetCDF is written.
- Routes (no `/api`): `GET /hrrr-hourly/runs`, `/runs/{cycle|latest}`, `/runs/{cycle}/zarr/...`, `/netcdf`,
  `/point?lat=&lon=`, `/image/{element}/{hour}`. Local retention `SMF_HRRR_HOURLY_KEEP_DAYS` (14).
- Python: `xr.open_zarr("https://api.showmefire.org/hrrr-hourly/runs/latest/zarr")`. JS: open the same URL with zarrita.

## Admin sessions

Admin sign-ins now last 30 days by default (`REFRESH_TOKEN_EXPIRE_DAYS`). The
shorter access cookie renews automatically on ordinary API requests while the
refresh cookie remains valid, including after reopening a tab. Renewal happens
before the endpoint runs, so saving a form does not require replaying its request.
The refresh expiry is fixed from sign-in; requests do not extend it indefinitely.
Explicit logout clears both HttpOnly cookies and never triggers renewal.

Deploy the API and website together. Existing refresh cookies retain their
original expiry; sign in again after deployment to receive the longer lifetime.
`ACCESS_TOKEN_EXPIRE_MINUTES` (or `ACCESS_TOKEN_EXPIRE_HOURS`) still controls the
short-lived access token. Keep the same `JWT_SECRET` across restarts and workers
so valid sessions survive deployments.
