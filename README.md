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
delivery receipts every 15 minutes and disables tokens reported as
`DeviceNotRegistered`.

Set `PUBLIC_API_BASE_URL` and `PUBLIC_CDN_BASE_URL` when the public hosts differ
from production defaults. Set `EXPO_ACCESS_TOKEN` only when enhanced Expo push
security is enabled; never expose it to the mobile client.
