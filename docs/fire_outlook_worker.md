# Seven-day fire outlook worker (runbook)

What it is: 28 consecutive six-hour windows through 168 h, from all 31 GEFS members, shown as "scenario agreement" (the share of weather
scenarios reaching Elevated-or-higher / Critical-or-higher Show Me Fire conditions). It forecasts conditions favourable for fire *given an
ignition*; it does not predict ignitions or perimeters. Design and status: `_plans/PLAN.md`, `_plans/CLAUDE_PLAN.md`.

**It is OFF by default.** Nothing runs until `SMF_FIRE_OUTLOOK_ENABLED=true` is set in the worker container. The plan requires a staff-only
30-day shadow pilot before public use; do not turn on public display before that review.

## Staff-only pilot (what staff can see before it is public)

Three independent switches, all default **off**, so enabling the worker alone exposes nothing:

| Switch | Where | When on |
|---|---|---|
| `SMF_FIRE_OUTLOOK_ENABLED` | worker | the cron ticks build runs (local files only) |
| `SMF_FIRE_OUTLOOK_PUBLISH` | worker | runs are also uploaded to the **public** CDN dataset `fire-outlook` |
| `SMF_FIRE_OUTLOOK_PUBLIC` | API | the public `/fire-outlook/...` routes answer instead of 404 |
| `NUXT_PUBLIC_FIRE_OUTLOOK_ENABLED` | website | the "7-day outlook" source appears in the public data viewer |

Pilot setup: set only `SMF_FIRE_OUTLOOK_ENABLED=true`. Staff then open **Admin > Forecasting > Seven-Day Outlook (pilot)** (`/admin/fire-outlook`): the worker's
state and settings, errors with their full text, what the latest run is made of (conditioning level, precipitation audit, sources, rule/seasonal state), the verification
export, and the same map the public will get. It reads `/api/admin/fire-outlook/...` (admin session, private cache headers); nothing is uploaded to the CDN and the public
routes return 404. To go public: pilot review, then `SMF_FIRE_OUTLOOK_PUBLISH=true` and `SMF_FIRE_OUTLOOK_PUBLIC=true` on the API/worker and the website flag; to retract,
turn `SMF_FIRE_OUTLOOK_PUBLIC` off (closes the public routes at once).

## What runs and when

| Job id | Cron (UTC, worker container) | Script | Does |
|---|---|---|---|
| `fire_outlook` | `7,37 * * * *` | `scripts/fire_outlook.sh` -> `scripts/run_fire_outlook.py` | Finds the latest 00Z/12Z cycle; waits until all 31 members' f168 are published (about 5 h after the cycle: ~05:00Z and ~17:00Z); then builds, verifies and publishes the run. Almost every tick exits in seconds. |
| `fire_outlook_verification` | `40 06 * * *` | `scripts/fire_outlook_verification.sh` | Joins forecasts with later observations into the training records (a window is verified 6 h after it ends). |

The ticks sit at :07/:37 so they never land on the `*/15` map refresh. Each tick takes its own lock (no two at once) and the shared forecast lock
(`scripts/forecast_lock.sh`) with a 20 s wait: while the daily forecast/ensemble holds it, the tick is skipped and retried 30 minutes later; while
the outlook computes it holds the lock so `maps.sh` yields CPU. One processing thread (`OMP/OPENBLAS/MKL_NUM_THREADS`), `nice -n 10`, hard 4-hour limit
(exit 124). The worker is **not** registered in the API scheduler (`core/scheduler.py`) and runs in its own process.

## Environment (all optional except the enable switch)

| Variable | Default | Meaning |
|---|---|---|
| `SMF_FIRE_OUTLOOK_ENABLED` | `false` | `true` to run at all (both scripts) |
| `SMF_FIRE_OUTLOOK_PUBLISH` | `false` | `true` to upload runs to the public CDN (needed for public delivery; leave off in the staff pilot) |
| `SMF_FIRE_OUTLOOK_PUBLIC` | `false` | `true` to open the public `/fire-outlook` API routes (the staff routes under `/api/admin/fire-outlook` always work for admins) |
| `SMF_FIRE_OUTLOOK_DIR` | `data/fire-outlook` | worker root (under the mounted `/app/data` volume): `runs/`, `cache/`, `verification/`, `status.json`, `latest.json`, `published.json` |
| `SMF_FIRE_OUTLOOK_MEMORY_LIMIT_GB` | `8` | address-space ceiling (Linux `RLIMIT_AS`) |
| `SMF_FIRE_OUTLOOK_THREADS` | `1` | processing threads |
| `SMF_FIRE_OUTLOOK_PUBLIC_KEEP_DAYS` | `7` | public runs kept locally and on the CDN |
| `SMF_FIRE_OUTLOOK_CACHE_KEEP_DAYS` | `30` | worker cache (GEFS/NBM/MRMS) kept |
| `SMF_FIRE_OUTLOOK_TIMEOUT_SECONDS` | `14400` | hard limit per tick |
| `SMF_FIRE_OUTLOOK_VERIFY_LATENCY_HOURS` | `6` | wait after a window ends before verifying it |
| `R2_*` (existing) | | CDN uploads need R2 credentials AND `SMF_FIRE_OUTLOOK_PUBLISH=true`; otherwise the run stays local |

## Compute and storage (measured 2026-10-29 on a Windows dev box with real data; not yet on the production worker)

~6.4 GB downloaded per cycle (byte ranges, 0.25 deg GEFS), limited by request latency (~10.6k requests), not CPU. One cycle end to end:
**~10 minutes, 2.4 GB peak** with an EMC seed; add ~3 min for the NBM baseline (88 leads) and the RTMA replay once an RTMA cache exists.
Output 43 MB per run (14 runs ~ 0.6 GB); cache ~24 MB GEFS per cycle plus NBM/MRMS. **Benchmark on the chosen worker before enabling** (plan gate).

## Where each input comes from

* **GEFS** (`services/fire_outlook/acquisition.py`): 31 members, 0.25 deg `pgrb2sp25`, 3-hourly to 168 h, cropped to the domain per member and cached (resumable).
* **Initial fuel moisture** (`history.py`, `conditioning.py`): causal replay of the RTMA cache ending at the cycle hour, anchored to RAWS 10-h observations that
  existed at or before the cycle (taken from the `observations` table, newest reading per station within 3 h, every in-domain station; no RAWS data means "conditioned", not
  "anchored", and the manifest says which). RTMA hours whose precipitation was flagged missing (the cache stores zeros with `missing="true"`) are filled from
  **MRMS** (Pass2, else Pass1), else skipped; they are never replayed as dry. More than 12 unfilled hours, or under 120 hours of history, refuses the run
  (it does not publish an unconditioned number). The manifest records the conditioning level and a precipitation audit.
* **NBM** (`nbm.py`): station values only, a *comparison baseline* for the weather verification. Never mixed into the agreement. Hourly to f048, then 3-hourly
  (interpolated like GEFS); unavailable means no `*_nbm` columns, never a partial baseline.
* **Observations** (`verification.py`): the `observations` table (RAWS reporting 10-h fuel moisture in the last 30 days).

## Checking on it

* Staff: **Admin > Seven-Day Outlook (pilot)**, or `GET /api/admin/fire-outlook/status` (admin session; settings, full worker status with error text, latest run details, verification counts).
* Public (only once `SMF_FIRE_OUTLOOK_PUBLIC` is on): `GET /api/fire-outlook/status` (stage, cycle age, runtime, memory, source coverage, last complete run; no error text).
* Admin Logs page: jobs `fire_outlook` and `fire_outlook_verification`; logs `fire_outlook_YYYYMMDD.log`, `fire_outlook_verification.log`, `cron_fire_outlook.log`.
* Runs: `GET /api/fire-outlook/runs`; manifest `/runs/{run_id}/manifest`; point series `/runs/{run_id}/point?lat=&lon=`.

## Failure behaviour

A failed cycle never replaces the previous complete run: the local `latest.json` only moves after a run is written and checksum-verified, and the CDN `latest` only
after a successful upload. An upload failure is recorded as a `publishing` failure and retried on the next tick without recomputing. A cycle with an incomplete member
set, missing windows, or insufficient conditioning history is not published. `status.json` keeps the stage and error. To rerun a cycle:
`python scripts/run_fire_outlook.py --cycle 2026-10-29T12 --no-publish` (a completed, verified cycle is skipped).

## Turning it off / rolling back

Set `SMF_FIRE_OUTLOOK_ENABLED=false` (ticks then exit immediately). Public runs age out after 7 days; to retract one immediately delete its folder under
`<root>/runs/` and remove it from the CDN dataset `fire-outlook`. The existing daily, hourly HRRR and 72-hour products are untouched by any of this.

## Training / verification loop

Copy `<root>/verification/records/` to the training machine's `$SMF_DATA_ROOT/fire_outlook/verification/` and run `python -m fire_outlook.retrain` in `model-training`
(see `model-training/fire_outlook/`). Calibration stays a candidate; the public label remains "scenario agreement" until a calibration version is reviewed.

## Known gaps (as of 2026-10-29)

Not yet exercised on the production worker or against real weeks of data; the RTMA replay has only been run against synthetic cache files (the dev box has 13 RTMA hours);
the observations table's real timestamp format/cadence is unverified (the code parses mixed formats); no nonburnable mask by decision; the map view is not built.
