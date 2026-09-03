# showmefire-api
api and server files for the backend of showmefire.org

The API collects and archives operational data, imports verified model
releases, and serves predictions. Training and static-raster preprocessing
belong in `ShowMeFire-Models`. See its
[`docs/spatial_fuel_moisture_runbook.md`](https://github.com/Cade417/ShowMeFire-Models/blob/main/docs/spatial_fuel_moisture_runbook.md)
for the complete operator workflow.

## RTMA and historical fuel-moisture capture

Hourly RTMA capture is registered with APScheduler at minute 50 and targets
the previous complete UTC analysis hour. Historical maintenance commands are
explicit and resumable:

```bash
# Backfill the rolling one-year Synoptic entitlement in UTC daily chunks.
python scripts/backfill_synoptic.py --dry-run
python scripts/backfill_synoptic.py

# Bundle HRRR, observations, and forecasts. RTMA is intentionally excluded.
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
