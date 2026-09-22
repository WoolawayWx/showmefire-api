# Forecast v1 operations

This package is the isolated 72-hour forecast path. It does not replace the
legacy forecast unless `SMF_FORECAST_V1_PUBLIC=true`; without that flag, runs
are complete shadow runs and remain available by immutable run ID.

## Automatic Herbie acquisition

The default scheduled source mode is `herbie`. Every 30 minutes the scheduler
checks for the newest 12Z cycle that is at least six hours old. Completed run
IDs are skipped, incomplete NOAA feeds are retried, and the existing public
pointer is not changed by a failed attempt.

The downloader requests indexed GRIB byte ranges rather than complete model
files, clips every source to the Missouri one-degree buffer, validates the
valid-time sequence, and archives the clipped native grid before reprojecting.
It acquires HRRR f00-f48, RRFS control f00-f72, seven REFS/RRFS ensemble
members f00-f72, and GEFS control plus all 30 perturbed members at three-hour
source intervals. GEFS precipitation is distributed over its source interval
and continuous fields are interpolated to the hourly uncertainty grid.

GEFS members are normalized and appended to one disk-backed NetCDF cube as
they download, so only one loaded member is retained in memory at a time. All
31 real members remain available for the immutable native archive and
station-member extract. Before public-grid reprojection, the existing summary
step reduces the ensemble to mean-minus-spread and mean-plus-spread
pseudo-members; this bounds reprojection memory without discarding the
member-level source archive.

```text
SMF_FORECAST_V1_ENABLED=true
SMF_FORECAST_V1_SOURCE_MODE=herbie
SMF_FORECAST_V1_PUBLIC=false
SMF_FORECAST_V1_MIN_CYCLE_AGE_HOURS=6
SMF_FORECAST_V1_POLL_MINUTES=30
SMF_FORECAST_V1_MEMORY_LIMIT_GB=8
SMF_HERBIE_THREADS=1
SMF_ENSEMBLE_MEMBER_THREADS=1
SMF_CPU_POOL_WORKERS=2
```

The single-thread defaults bound the number of full-domain GRIB fields open
during acquisition; increasing them trades additional peak memory for download
speed. Each query is clipped and eagerly loaded before its backing dataset is
closed. Downloaded subsets remain in the seven-day cache instead of being
removed while cfgrib arrays may still reference them. The forecast worker's default 8 GiB
address-space ceiling leaves headroom beneath a 12 GB container limit; an
over-budget run fails without replacing the previous public forecast. The
shared heavy-job pool defaults to at most two workers to prevent CPU-rich hosts
from accidentally running many memory-intensive grid jobs simultaneously.

`SMF_REFS_MEMBERS`, `SMF_GEFS_MEMBERS`, `SMF_RRFS_PRODUCT`, and
`SMF_RRFS_DOMAIN` can override feed details without changing code. The local
RRFS template targets the NOAA operational `noaa-rrfs-ops-pds` bucket; the
Docker build checks that the weather `herbie-data` distribution supports the
RRFS rotated grid. HRRR remains authoritative through hour 48. If RRFS is
unavailable, the GEFS ensemble mean supplies a coarse synoptic fallback for
hours 49-72. Those hours carry `coarse_synoptic_fallback`, are disclosed in
the run manifest, and have category/meteorological confidence capped at 49.
If neither RRFS nor GEFS is complete, publication fails closed.

Run-level `warnings` retain their string-array API contract and contain only
operational degradation: RRFS loss, GEFS fallback, or partial/unavailable
meteorological confidence. `sourceDiagnostics` in the manifest and admin
status provide source/role/severity, affected leads, missing fields, and the
sanitized full acquisition error. FV3-HIRES is a zero-weight shadow source;
its missing fields remain informational and do not raise the global degraded
banner.

Meteorological confidence uses available model agreement, REFS/GEFS ensemble
spread, previous-cycle consistency at matching valid times, 30-day station
verification, and lead time. It renormalizes the configured weights over
available components. A real score needs agreement or spread and at least
35% available weight; otherwise it remains NoData. The hourly Synoptic refresh
stores normalized sensor values and matches completed forecast hours to
QC-eligible observations within 30 minutes. Rolling verification activates
only after 30 pairs from at least three stations in a lead bucket.

The authenticated operations console is `/admin/forecast-v1`. Its status API
reports whether the application scheduler and forecast job are enabled, the
current/next eligible 12Z cycle, recent runs, warnings, source diagnostics,
confidence-component coverage, RRFS/fallback counts, verification samples,
storage totals, static
graphics, and station choices. Admins can queue a non-blocking run or invoke
the normal retention cleanup. A shared execution lock prevents a manual run
from overlapping the scheduled acquisition.

During a run, `GET /api/admin/forecast-v1/job` provides a lightweight progress
feed with the current phase, source/member or asset counts, percent complete,
elapsed time, memory ceiling, and the latest 30 activity events. The operations
console polls this endpoint every two seconds; the larger status response is
refreshed separately so live progress does not repeatedly query all products.

## Staged input contract

When `SMF_FORECAST_V1_SOURCE_MODE=staged`, the opt-in scheduler reads
`SMF_FORECAST_V1_SOURCE_DIR` (default
`data/forecast-v1-input`) at 09:15 America/Chicago. The directory contains:

- `cycle.json`: `{"cycleTime":"2026-09-06T12:00:00Z","fallbackFuelMoisture":12}`
- `stations.json`: station objects with `station_id`, `name`, `latitude`,
  `longitude`, `network_type`, `elevation_m`, and sensor capabilities.
- Any of `hrrr.nc`, `rrfs.nc`, `refs.nc`, and `gefs.nc`.

Input NetCDF files are already clipped to Missouri plus the one-degree buffer.
They retain their native grid, carry a `crs` attribute, and expose projected
`x`/`y`, `time`, and optional `member` coordinates. Source aliases and unit
normalization live in `adapters.py`. HRRR and RRFS are strict about core fields;
GEFS may carry unavailable fields because it is used for uncertainty rather
than as an equal deterministic grid.

Enable the staged shadow path with:

```text
SMF_FORECAST_V1_ENABLED=true
SMF_FORECAST_V1_SOURCE_MODE=staged
SMF_FORECAST_V1_PUBLIC=false
SMF_FORECAST_V1_ROOT=data/forecast-v1
```

Set the public flag only after the verification gates pass. R2 upload is
automatic when the standard R2 credentials are present. Objects use the
`forecast-v1/` namespace and are checked after upload. `latest.json` is updated
last.

For rollout, migrate SQLite first by running `forecast_v1.repository.ensure_schema`
or starting the new API, then deploy the worker. Leave
`SMF_FORECAST_V1_PUBLIC=false` for a two-hour RRFS canary and a complete
73-hour shadow run. Require two consecutive successful 12Z shadow cycles
with RRFS hours 49-72, no GEFS fallback, populated supported confidence,
and aligned public grids before changing the public flag.

## Persisted data

- Native source and derived cubes are NetCDF4, shuffled, compression level 6,
  and scale/offset packed according to `contracts.ARCHIVE_ENCODINGS`.
- Public rasters are band-interleaved 256×256 tiled COGs. Hourly products have
  73 bands; daily products have three. Fire categories use nearest overviews;
  continuous, confidence, and probability data use averaged overviews.
- Public and member-level station rows are Zstd-compressed, dictionary-encoded
  Parquet. PyArrow is intentionally a runtime dependency.
- SQLite stores current query rows and a rebuildable index. Physical values are
  `REAL`, timestamps are UTC RFC3339 `TEXT`, category/confidence values are
  constrained integers, and JSON columns are validated text.
- `255`, `65535`, and `-32768` are packed NoData sentinels. APIs convert missing
  values to JSON `null`.

The retention task keeps the download cache for seven days, local archived
artifacts for 30 days when R2 is configured, and station query rows for 90 days.
Metadata and R2 objects are not pruned.

## Manual run

```text
python -m forecast_v1.pipeline \
  --cycle 2026-09-06T12:00:00Z \
  --source hrrr=/data/staged/hrrr.nc \
  --source rrfs=/data/staged/rrfs.nc \
  --source refs=/data/staged/refs.nc \
  --stations /data/staged/stations.json
```

Add `--public` only for a promoted run. The command archives native sources,
regrids the derived product to the fixed 267×264 EPSG:32615 grid, produces all
artifacts, then commits the hot index.

Use the same command with `--archive-only` and no `--stations` argument for the
00Z, 06Z, and 18Z cycles retained for four-cycle consistency and trend inputs.

Legacy JSON archives can be converted with
`python scripts/migrate_forecast_v1_history.py <paths...>`. Add `--relational`
only for archives needed by historical verification; otherwise the migration
creates the immutable Parquet research object without synthetic SQLite runs.
