# Fire danger model tracking

How each fire-danger model's county-day predictions line up with confirmed fires. Informational only: nothing here changes
a model, a score, or anything public. Admin page: `/admin/fire-danger-tracking`. Code: `services/fire_danger_tracking.py`,
`routers/fire_danger_tracking.py`, table `fire_danger_model_scores`.

## What is tracked
| Model | Where its county result comes from | Valid date |
|---|---|---|
| `fire_weather_index` (shadow) | `model-shadow/fire-weather-index/*.fire_weather_index_score.json` (score + category) | `valid_local_date` |
| `fire_weather_index:blend` (shadow) | `model-shadow/fire-weather-index/*.fire_weather_index_blend_score.json` (HRRR+RRFS+FV3 blend, below) | `valid_local_date` |
| `risk_fusion_glm` (shadow) | `model-shadow/risk-fusion-glm/*.glm_score.json` (P(>=1 fire)) | `valid_local_date` |
| `ensemble:<track>` (shadow) | `model-shadow/ensemble-fire-danger/*.json` (county categorical + P(>=Elevated)) | America/Chicago date of the anchor cycle; only 09Z-15Z daytime cycles |
| `public_rule` | `county_forecast_days` (the public rule-based category) | `forecast_date` |

Not tracked: `fire_weather_ml` (it stores map-level summaries only, no county result).

**There is no production blend of these models.** Each scores independently. The HRRR + RRFS + FV3-HIRES blend exists only in
the offline training panel (`model-training/fire_weather_index/build_county_days.py`); live scoring is HRRR-only. The
"consensus" block is an analysis (mean category across model families on the same county-days) to show whether combining them
would help; nothing serves it.

## How it runs
- Nightly job `fire_danger_tracking` (05:10 America/Chicago) copies new evidence into `fire_danger_model_scores`. Append-only and
  idempotent: a re-ingest never changes a stored row. The admin page has a "Copy new model scores now" button.
- Metrics are computed on request from the table. Nothing runs a model.
- The public rule's county table is overwritten by each run, so it only has history from the first night this job ran.

## Outcome (what counts as a fire day)
An APPROVED fire event at tier >= `admin_reviewed` (or `official_source_confirmed` only) on the America/Chicago date, excluding
`prescribed`, `agricultural` and `not_a_fire` causes (the same convention as the risk-fusion training labels). A day with no event is
NOT proof of no fire, so:
- only dates at least `lag_days` (default 3) old are scored, so reports and reviews can catch up;
- every result carries its counts, and a window with fewer than 10 fire county-days says "too few to conclude" instead of an AUC.

## One run per model per day
Only runs produced before 14:00 local on the valid date count as forecasts (`FIRE_DANGER_TRACKING_CUTOFF_HOUR`); later reruns have
seen part of the day. The number excluded per model is shown, never silently dropped. Of the runs that remain, the latest wins.

## What is reported
Per model, for the recent window and all time: county-days, fire days, AUC with a 95% interval resampled by date (county-days on
one date are not independent), average precision, how many fire days a model flagged at Elevated or higher and what share of all
days that was (hit rate, alert share, lift over the base rate), fire rate by category, whether the rate rises with category
(a drop within sampling error is allowed), and Brier vs a prevalence-only baseline for the model whose probability means
P(fire) (`risk_fusion_glm`). Models have different histories, so a **same-rows** block compares everyone on only the county-days
they all scored.

## Caveats
- Fire days are sparse and depend on what gets reported and reviewed; early numbers will be noisy.
- Approved satellite detections are fire evidence only after review. An unreviewed backlog lowers the fire-day count.
- The ensemble's valid date is assumed to be the Chicago date of a 09Z-15Z anchor cycle.
- Counties with a NULL `county_fips` on an event cannot be attributed and are left out.

## The blended Fire Weather Severity Index (beta/preview)
`services/fire_weather_index_blend.py` scores the same index on the HRRR + RRFS + HiResW FV3 average and draws it in the house
style, so the blend can be compared with the HRRR-only shadow. It runs inside the 12Z ensemble run (cron 15:00Z), which has already
fetched and regridded all three onto the forecast grid for the same valid hours; nothing new is downloaded.
- Each source's hourly fields are reduced with exactly the HRRR-only hook's four reductions (afternoon-min RH, max wind, max VPD,
  24 h precip), then averaged with equal weight over the sources that have a value in each cell (the offline blend's rule). The
  same bundle and scoring are used, so the two products differ only in their weather input. A parity test checks the reductions
  against the HRRR-only hook.
- One member per source: `hrrr_0`, `rrfs_0` (else `rrfs_m6`), `fv3_0`. Needs at least two sources.
- Outputs: image `testbed/images/fire_weather_index_blend_shadow_latest.png`, manifest key `fire_weather_index_blend_shadow`
  (testbed page card), evidence `<run_id>.fire_weather_index_blend_score.json` (tracked above as `fire_weather_index:blend`).
- Gated by the same enabled switch as the fire_weather_index shadow, best-effort, and it never reads or writes that shadow's health
  state, so a blend failure cannot disable it.
- Not included (same as the live HRRR-only shadow): antecedent rain, the fuel-type precip threshold and score smoothing that the
  offline diagnostic adds. FV3-HIRES temperature is a (max+min)/2 proxy, which biases VPD; this is stated on the map.
- RRFS and FV3-HIRES are only blended in when the ensemble run fetched them. FV3 is optional and only available for the 00Z/12Z
  cycles on NOMADS.
