# Capture-only model data (RRFS, GEFS, NBM, MRMS)

The server keeps a Missouri-cropped copy of every weather model we may train on, whether or not a product reads it yet.
Code: `services/data_capture/` (job `capture_models`, every 20 minutes at :07/:27/:47). Plan: `_plans/PIPELINE_RESET.md`.

## What it keeps
| Model | Source | Cycles / leads | Folder (server and training PC) |
|---|---|---|---|
| RRFS deterministic 13 km | `noaa-rrfs-ops-pds` | 00/06/12/18z, f01-f48, hourly | `cache/rrfs/rrfs_YYYYMMDD_HHz_f01-48.nc` |
| GEFS mean + spread 0.25° | `noaa-gefs-pds` | 00/12z, f06-f168 every 6 h | `cache/gefs/gefs_mean_…`, `gefs_spread_…` |
| NBM core (CONUS) | `noaa-nbm-grib2-pds` | 13z, f03-f72 every 3 h | `cache/nbm/nbm_…` (grid row order corrected, see below) |
| MRMS hourly QPE | `mrms.ncep.noaa.gov` | hourly | `cache/mrms/mrms_YYYYMMDD_HHz.nc` |

Files use the training PC's folder layout and names, so the nightly day bundle restores them with no renaming. Each is
written atomically (`.part` then rename) with lossless zlib compression. A run is captured only when its **last lead** is
published, never twice, and a missing optional field is stored as NaN and listed in the file's `missing_fields` attribute.
HRRR, RTMA and the REFS/RRFS ensemble members are still captured by the existing jobs.

NBM files store alternate rows right-to-left (`alternativeRowScanning`). The shared GRIB reader doesn't undo that, so the
capture does it (`alternate_row_scan: true` in `capture_config.json`). Found by checking Springfield IL (-2 °C on a 17 °C
day) and fixed; any new model should be spot-checked the same way before it is trusted.

## Settings (all optional; none are needed for a normal start)
| Variable | Default | Meaning |
|---|---|---|
| `SMF_CAPTURE_MODELS_ENABLED` | `true` | `false` stops the job. |
| `SMF_CAPTURE_MIN_FREE_GB` | `5` | Nothing is captured while free disk under the data root is below this. The server has ~17 GB free (2026-11-18). |
| `SMF_CAPTURE_MAX_RUNS` | `3` | Model runs per tick, so a catch-up can never hog the server (it runs at nice 19, ionice idle, 1 thread). |
| `keep_days` in `capture_config.json` | `14` | Local retention; the bundle has uploaded each day long before then. |

Turn the individual models off with `"enabled": false` on that model in `capture_config.json`.

## Rollout and checks (server changes happen only when you deploy)
1. Deploy, then watch `logs/api.log` for `data capture:` lines (captured / failed / skipped).
2. After one cycle each folder under `/app/cache/<model>/` should hold files. Expected size/day, from real test runs:
   RRFS ≈ 0.4 MB per 2 leads (≈ 9 MB for 48), GEFS ≈ 80 KB per 2 leads, NBM ≈ 2.8 MB per 2 leads. Total well under 350 MB/day.
3. The next nightly bundle (03:30 CT) should list `rrfs`, `gefs`, `nbm` and `mrms` as `present` in its manifest.
4. On the training PC: download the bundle (Data Manager), run **Restore nightly bundles**; the Data health tab shows each model
   current instead of "Not collected yet".

## Disk and cost controls (separate switches, currently OFF on the server)
- `SMF_DAY_ARCHIVE_PRUNE_LOCAL=true` and `SMF_ARCHIVE_PRUNE=true`: delete local files after they are verified in R2.
  Without them `cache/hrrr`, `archive/forecasts` and `archive/raw_data` grow forever (the server is 83% full).
  Turn on only after a day with a verified bundle; see `docs/r2_archive_runbook.md`.
- `SMF_DAY_ARCHIVE_DELETE=true`: delete a bundle from R2 after 30 days **and** a download receipt from the Data Manager.

## Old RRFS map script
`forecast/DailyForecast_RRFS.py` / `scripts/forecast_RRFS.sh` have not run since February (nothing schedules them) and their
cache lookup searches for the wrong file name, so they can never have worked as an archive. Leave them in place until the
capture has run a week and the RRFS bundle component is confirmed; then they can be moved to an attic folder.
