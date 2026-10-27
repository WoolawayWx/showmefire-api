# Detection history: FIRMS backfill and the R2 training export

## Why
`fire_events` only holds detections from when ingest was switched on, and the live feeds keep days, not months.
Training the detection-confidence dataset needs detections over the same dates as its label sources (NERIS,
feedback, reports). Nothing in the API deletes old detections ("not a fire" incidents are soft-deleted and the export
includes them), so history only grows from here; this fills in what came before and ships it to the training machine.

## 1. Backfill VIIRS/MODIS from the FIRMS archive
Needs `MODIS_KEY` (the FIRMS MAP_KEY the live poller already uses). NGFS has no archive and cannot be backfilled.

    python scripts/backfill_firms_history.py --start 2025-12-01 --end 2026-08-08 --dry-run   # count only
    python scripts/backfill_firms_history.py --start 2025-12-01 --end 2026-08-08

- Uses the Standard Processing products (`VIIRS_SNPP_SP`, `VIIRS_NOAA20_SP`, `VIIRS_NOAA21_SP`, `MODIS_SP`) plus the NRT
  ones for recent days, 5 days per request. FIRMS allows 5000 requests per 10 minutes per key.
- An archive row gets the id its NRT twin would have, so a detection in both lands on one row (no duplicates);
  re-running is safe.
- Raw fetched features are kept in `data/firms_backfill/firms_history_<start>_<end>.geojson`.
- Ingest does not score; the normal scoring pass picks the new rows up later, newest first.
- The live ingest now also stores brightness (4 um / 11 um channels), pixel area and day/night for VIIRS/MODIS. Rows
  already in the DB are filled in the next time the live feed (or a backfill) sees them.

## 2. Publish the training export to R2
Needs `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`. Bucket is `R2_TRAINING_BUCKET`, else
`R2_ARCHIVE_BUCKET`, else `R2_BUCKET`, else `cdn-showmefire`; prefix is `R2_TRAINING_PREFIX` (default
`training/detections`). **Keep that prefix off any public/CDN path**: the zip has no contact details or IP hashes, but
it does include moderation outcomes.

    python scripts/publish_detection_training_data.py [--since ISO] [--keep 14]

Layout: `<prefix>/latest.json` (written last) and `<prefix>/snapshots/<UTC stamp>/{detection_training_export.zip,manifest.json}`.
Old snapshots beyond `--keep` are deleted; the one `latest.json` points to never is.

Scheduled: set `SMF_DETECTION_R2_PUBLISH=true` (off by default) to publish daily at `SMF_DETECTION_R2_PUBLISH_HOUR`
(America/Chicago, default 03:20); `SMF_DETECTION_R2_KEEP` sets retention.

## 3. Training side
    python -m detection_dataset.build --pull      # newest R2 snapshot (checksummed)

With R2 credentials in `model-training/.env`, a first dashboard run also pulls from R2 automatically.
