# Datasets on the CDN

Finished dataset runs are published to Cloudflare R2 (bucket `cdn-showmefire`, served as `https://cdn.showmefire.org`)
so browsers read them from the edge instead of the API. Code: `services/cdn_datasets.py` (generic) and
`services/hrrr_hourly/publish.py` (the HRRR hourly glue, called by `job.py` after a build).

```
datasets/index.json                         every published dataset: title, kind, schema version, pointers
datasets/<dataset>/latest.json              {run, manifest, time_utc, updated_at}      cache 60 s
datasets/<dataset>/runs.json                {latest, runs: [...]} newest first          cache 60 s
datasets/<dataset>/runs/<run>/manifest.json                                             cache 1 year, immutable
datasets/<dataset>/runs/<run>/...           the run's files (HRRR: forecast.zarr/)      cache 1 year, immutable
```

* A run's files are never rewritten, so they are cached for a year. Only the three small pointer files change.
* Upload order is files, manifest, `runs.json`, `latest.json`, `index.json`: a reader following the pointer never sees
  a half-uploaded run. Runs older than `SMF_DATASETS_KEEP_DAYS` (default 7) are deleted after the pointers move; the
  newest run is never deleted.
* Publishing is best effort. With no R2 credentials, or if R2 fails, the build still completes and the API still serves
  every run; the viewer falls back to the API for anything the CDN does not have.
* Two datasets use it: `hrrr-hourly` and `hrrr-fire-danger-peak` (the peak fire danger estimate). Both are published **while the run is
  being built** (`publish_update`): each new chunk once (cached for a year), a manifest that changes (cached for a minute until the run
  is `complete`), and a `runs.json` row with `ready: false` until enough of the run is in to be the dataset's latest.
* Not on the CDN: the NetCDF copy (large; the R2 archive already picks it up, `archive_store.py` source `hrrr_hourly`)
  and `/hrrr-hourly/point` and `/status`, which the API computes.

## Adding a dataset

Write the run to a folder with a `manifest.json`, then call `cdn_datasets.try_publish_run(dataset_id, run_id, run_dir,
summary, title=..., kind=..., schema_version=...)`. `summary` is the run's row in `runs.json` and needs `id` and `time_utc`.
`useDataViewer({ dataset: '<dataset_id>' })` on the site reads `datasets/<dataset_id>/` and the `/<dataset_id>` API routes.

## Operations

* Publish a run that is already built: `python scripts/hrrr_hourly_forecast.py --publish YYYYMMDD`.
* The bucket needs a CORS rule allowing `https://showmefire.org` (GET) for the Zarr reads.
* The site reads from `NUXT_PUBLIC_DATASETS_URL` (default `https://cdn.showmefire.org/datasets`; empty in dev).
* Check Cloudflare is caching: `cf-cache-status: HIT` on a chunk URL.
