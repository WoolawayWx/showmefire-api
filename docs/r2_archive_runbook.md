# Versioned R2 data archive

The API keeps live working files on disk and archives individual objects to R2
under `data-v2/objects/YYYY-MM-DD/source/sha256/filename`. A changed file creates
a new version. After reading back and verifying the uploaded SHA-256, the service
publishes its JSON manifest under the corresponding `data-v2/manifests/` key.
Daily ZIP uploads are replaced by a scan every 15 minutes. RTMA remains a
seven-day operational cache and is intentionally excluded from permanent storage.

Included sources: station forecasts, Synoptic observations, HRRR and RRFS grids,
and the isolated Testbed forecast and HRRR directories. This does not move the
application database, models, images, or GIS serving assets into object storage.
Forecast/observation producers continue using their existing working paths.

## Configuration

This deployment uses the existing `cdn-showmefire` bucket with the new `data-v2`
prefix. Set `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, and `R2_SECRET_ACCESS_KEY`.
`R2_ARCHIVE_BUCKET` can override the destination; otherwise it uses existing
`R2_BUCKET` or `cdn-showmefire`. `R2_DATA_PREFIX` defaults
to `data-v2`. Credentials need object read/write/list privileges for the archive
bucket. The existing Cloudflare boto3 S3 endpoint and `auto` region are used:
https://developers.cloudflare.com/r2/examples/aws/boto3/

The SQLite index lives in `data/archive_inventory_<namespace>.sqlite3`, scoped
by bucket and prefix, and must use the persistent data mount. It stores revisions,
errors, and run counters. R2 manifests can rebuild the inventory; run history is
local. The process lock uses POSIX flock (Linux/macOS) on that same data mount.
Multiple hosts must not independently prune the same working directory.

`SMF_ARCHIVE_ROOT` overrides the API root. Sources currently use the standard
Testbed directory beneath that root; deployments overriding `TESTBED_PRODUCTS_DIR`
need to map their archived products into this layout before enabling pruning.
`SMF_ARCHIVE_SETTLE_SECONDS` defaults to 300: recently modified files wait for the
next scan. A temporary snapshot needs space for one source file. Each new upload
is read back in full to verify it. Unchanged verified files are skipped using
local size/mtime; external modifications must preserve neither value.

## Migration and rollback

Run from `api/` with its normal Python environment:

```sh
# No uploads or deletions: index the available working data first.
python scripts/archive_storage.py inventory
python scripts/archive_storage.py status

# Upload and verify working data. This retains all source files.
python scripts/archive_storage.py sync

# Import local ZIPs or every legacy ZIP in data-archive/ in the configured bucket.
python scripts/archive_storage.py import-zip --zip data_archive_day/20260712.zip
python scripts/archive_storage.py import-r2-zips

# Rebuild metadata on another server or after losing the local inventory.
python scripts/archive_storage.py rebuild
python scripts/archive_storage.py restore --date 2026-07-12
```

Legacy imports are resumable and never delete original ZIPs. Unknown members
(including historical RTMA) remain in their original ZIPs; they are not migrated
into v2. If legacy ZIPs live in a different bucket, download them from that bucket
and use `import-zip` with the new destination configured. ZIP timestamps lack a
timezone, so migrated member timestamps use UTC as an ordering approximation.
Verify current working files after importing legacy history if timestamps overlap.
The verification restore wrapper reads v2 first, then fills missing local files
from the old ZIP. Authentication/download failures propagate instead of being
misread as an absent archive. Restores refuse to overwrite different local v2 data.

Do not enable cleanup until migration and restores have been checked. Explicit
`sync --prune` or `SMF_ARCHIVE_PRUNE=true` removes only verified sources whose
filename date AND modification time are older than `SMF_ARCHIVE_LOCAL_DAYS`
(default seven, minimum one). No R2 object deletion or version expiration is
performed. Rollback: disable the scheduler change, restore required files, and
resume the legacy workflow; the original ZIP history is still available.

## Operator dashboard

`/admin/archive` shows daily counts by source, forecast records, observation
samples, bytes, upload failures, persisted run progress, and current forecast /
observation processing states. It refreshes every ten seconds. Use Sync now to
retry failures. Scheduled scans retry automatically. A restarted interrupted run
is marked interrupted when the next sync acquires the lock.

A filename date is not an observation validity date. Manifests separately record
UTC start/end timestamps and station/record counts. Counts refer to the latest
revision of each filename; observation samples can overlap between files and
are not deduplicated training labels. Empty records and missing days do not
establish source completeness. Legacy coverage remains unknown until imported.
The dashboard uses existing job state files; primary and realtime producers
currently expose completion status, not percentage progress. Backfill day states
come from `synoptic_backfill_manifest.json`.

The existing historical observation list/load endpoints also consult the R2
inventory when local files have been removed. Rebuild the inventory before
serving historical data from a replacement server.

## Training machine

Install `ShowMeFire-Models/requirements-archive.txt`, export the same R2 settings
and `SMF_DATA_ROOT`, and run:

```sh
python scripts/pull_r2_archives.py --start 2026-07-01 --end 2026-07-31
# Or limit bandwidth to one source:
python scripts/pull_r2_archives.py --start 2026-07-01 --end 2026-07-31 --source observations
```

The downloader selects the newest revision, verifies every downloaded checksum,
resumes by skipping matching files, and refuses to overwrite different local data.
The existing SSH/ZIP pull script is still available for legacy archives.

## Day bundles (everything needed to reconstruct a day)

`services/day_archive.py` builds one bundle per completed Central day under
`day-archive/YYYY-MM-DD/` in R2 (nightly at 03:30 Central, after the 02:45 map zips and the
01:00 full-observation capture). R2 is the long-term store; the server only uses a
transient work directory (`cache/day-archive/`, cleaned after each upload).

Components (zip each, verified by reading the upload back, then `manifest.json` is
published last): `forecasts`, `observations` (RAWS verification pull), `observations_full`
(all networks 1/2/156/65, whole Central day, `services/full_obs_archive.py`), `hrrr`, `rrfs`,
`ensemble`, `forecasts_beta`, `hrrr_beta`, `rtma` (the day's 24 hourly files, captured
before the 7-day cache rolls over), `derived` (peak rasters, validation report, 09z/RTMA
metrics) and `maps` (server-side copies of the `map-archive/` zips for the two UTC days a
Central day spans; size-checked, the CLI re-checks size on download). A component with no
data on the server is recorded as `missing`; a bundle is `complete` only when forecasts,
observations, derived and maps are present and nothing failed or is partial. Incomplete days
are retried for `SMF_DAY_ARCHIVE_RETRY_DAYS` (4). First run backfills the last
`SMF_DAY_ARCHIVE_BACKFILL_DAYS` (7) days; older days have no local source left.

### Window, reminders, deletion
The ledger (`day-archive/ledger.json`, cached in `data/day_archive/`) gives each bundle
`expires = created + SMF_DAY_ARCHIVE_KEEP_DAYS` (30). Discord reminders go out 7, 3 and 1
days before expiry for bundles not yet downloaded, then a daily "overdue - NOT deleted"
notice. A bundle is deleted from R2 only if it is past its window AND has a verified
download receipt AND `SMF_DAY_ARCHIVE_DELETE=true` (default false). Nothing is ever deleted
undownloaded.

### Server retention (7 days)
With `SMF_DAY_ARCHIVE_PRUNE_LOCAL=true` the nightly pass deletes server source files older
than `SMF_ARCHIVE_LOCAL_DAYS` (7), only for components read back from R2 and only when
each file's sha256 still equals the manifest. `reports/` and the `gis/*/archive` rasters are
never pruned (the public `/verification` routes serve them). Leave the flag off
(it then logs what it would delete) until the first bundles have been checked. The older
`SMF_ARCHIVE_PRUNE` for `data-v2/` is separate and still off; see the cleanup note above.

### Downloading (reads R2, not the server)
- CLI (SSH or your own machine; needs boto3 and the R2 env vars):
  `python scripts/pull_day_archives.py --list`, `--dest DIR --all-pending`,
  `--dest DIR --date 2026-10-16`, `--since DATE`, `--no-mark`. It verifies every sha256
  against the manifest and only then writes `day-archive/receipts/<date>.json`.
- Admin endpoints (JSON only, no web page, no `/api` prefix): `GET /admin/day-archive`,
  `GET /admin/day-archive/{date}`, `GET /admin/day-archive/{date}/links` (presigned R2 URLs,
  default 1 h), `POST /admin/day-archive/{date}/mark-downloaded`, `POST /admin/day-archive/run`.
  Links point at R2 directly. Mark a day downloaded only after verifying the files.

The full route list, including admin routes, is in `docs/ROUTES.md`
(`python scripts/generate_routes_doc.py`).
