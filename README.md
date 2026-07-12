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
# Inspect or backfill RTMA. Run the archive bundler afterward to atomically
# merge new hours into existing daily archives.
python scripts/backfill_rtma.py --dry-run
python scripts/backfill_rtma.py
python -m services.archive_bundler

# Backfill the rolling one-year Synoptic entitlement in UTC daily chunks.
python scripts/backfill_synoptic.py --dry-run
python scripts/backfill_synoptic.py
```

Fuel moisture is sourced only from Synoptic station observations. RTMA is
stored as analyzed meteorological input and is never used as an FM label.

Spatial releases contain ONNX, checkpoint, static NetCDF, manifest,
evaluation, and smoke-test assets. Import verifies the whole contract before
registering beta:

```bash
python pipelines/import_model.py --model fuel_moisture_spatial --tag <release-tag> --repo Cade417/ShowMeFire-Models
```

Once explicitly promoted, forecast generation attempts spatial inference and
automatically retains XGBoost output on any missing input or contract/runtime
failure. Current status is exposed at `/api/model/spatial/diagnostics`.
