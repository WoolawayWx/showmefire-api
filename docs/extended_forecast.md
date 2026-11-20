# Day 2 / Day 3 forecast (beta)

`forecast/ExtendedForecast.py` runs Day 1's forecast logic on the next two days, as a fast alternative to the
72-hour `forecast_v1` pipeline's Day 2/3 products. Built 2026-11-19; multi-run schedule, HRRR/RRFS blend and station
scoring added 2026-11-20. Beta, Testbed only. Scheduled in the Dockerfile (needs an image rebuild on the server).

| Day | Source | Local window (standard time) |
|-----|--------|------------------------------|
| 2   | HRRR 3 km blended with RRFS 13 km (fixed 70 / 30) | 10:00-21:00 CT the next day |
| 3   | RRFS 13 km | 10:00-21:00 CT two days out |

(Day 1 is HRRR f04-f15 of the 12z run. Each later day is the same afternoon window, 24 h further on. Under daylight
time the window is 11:00-22:00.)

## Which model runs feed it

The window is the afternoon (16z-03z) of *the run's UTC date + day - 1*, so the forecast hours shift with the run:

| Run | Day 2 hours | Day 3 hours |
|-----|-------------|-------------|
| 00z | not possible (f40-51, past HRRR's f48) | not used |
| 06z | f34-45 | not used (f58-69 is past the RRFS limit below) |
| 12z | f28-39 | f52-63 |
| 18z | f22-33 (warm-up is cut at f00, so it is flagged `short_warmup`) | f46-57 |

`SMF_EXTENDED_RRFS_MAX_LEAD` (default 63) is deliberately conservative: only RRFS f63 of the 12z run has been confirmed.
Raise it once longer leads are confirmed and 00z/06z can feed Day 3 too. Changing it changes which runs are used, not
the code.

Cron (Dockerfile) ticks at 01:30, 10:30, 16:30, 19:30 and 22:30 UTC: HRRR's last needed hour is out ~4 h after the
cycle, RRFS's ~7 h. Each tick builds only a day whose newest eligible run is not already built (the day's JSON records the
run), so the five ticks never repeat work. A day that fell back to HRRR alone (`rrfs_blend_missing`) is rebuilt on the
next tick. `--force` rebuilds anyway.

## Peak conditions: a 2-hour block that moves

The maps show conditions in the **2 hours where Missouri's danger is highest**, not the extremes of the whole 12-hour
window, and which 2 hours it is changes with every run. The whole window is still modelled (the fuel-moisture model
needs that history, and the rain map is the 24 hours ending at the window's end).

* **How it is picked:** every hourly step is scored by the state-wide mean danger category (0-4) over cells inside
  Missouri, with the state-wide mean RH as the tie-break (so an all-Low day still gets its driest stretch). The best
  2 consecutive steps win (`choose_peak_block`). `SMF_EXTENDED_PEAK_HOURS` sets the length.
* **What it drives:** the peak danger, minimum fuel moisture, minimum RH, maximum wind and maximum temperature maps, the
  percent-of-state numbers, and the labels ("Peak 3:00 PM-4:00 PM CT" in the subtitle and descriptions; the two hours
  are the local valid times of the two hourly steps).
* **In the JSON:** `peak_block: {first_lead, last_lead, hours, local}`; `window_local` is still the wide window.
* **Scoring:** snapshots store the whole window plus the peak block's leads, and each run is scored on both (`scope`
  column: `window`, `peak`). The shadow blend weights use only the `window` scope: two hours per station is too few to rank
  the models steadily. The peak scope is reported in `summary.json` metrics, and needs both hours observed.

## What is shared with Day 1

The stable fuel-moisture model and its features (`DailyForecast.predict_fm_grid`), the fire-wind reduction factor,
the canonical fire-danger rule with its seasonal softening, the SNODAS snow mask, hourly risk smoothing, the summary
smoothing, and the map house style (`forecast/map_style.py`, same palettes and legends). The six statewide maps are
the same six as Day 1: peak fire danger, minimum fuel moisture, minimum RH, maximum wind, maximum temperature, rain.

## Why it is fast

`forecast_v1` pulls HRRR, RRFS and 31 GEFS members with ~15 fields each, blends them, and scores 51 scenarios per
cell. This job asks `forecast_v1.acquisition.acquire_fields` for 5 fields (T, RH, wind, gust, rain) per model, only for
the hours it needs, clipped to Missouri; the clipped result is cached under `cache/extended/`, so re-runs are instant
and the RRFS download is shared between Day 2's blend and Day 3. The fire-danger rule is only evaluated on cells drier
than the Low threshold (Day 1 loops over every cell in Python).

## The Day 2 blend

RRFS is interpolated (bilinear) onto the HRRR grid and mixed with HRRR **before** the fuel-moisture model and the rule
run, once: temperature, RH and run-total rain by weight, wind as *speed* (averaging u and v would shrink it wherever
the models disagree on direction). Where RRFS has no value the HRRR value is kept. The HRRR share is
`SMF_EXTENDED_DAY2_HRRR_WEIGHT` (default 0.7). If RRFS is unavailable after the wait, Day 2 is HRRR alone and the JSON
says so (`flags: rrfs_blend_missing`); `--no-blend` forces that.

Blending flattens extremes (a mean of two forecasts), so the peak category can read slightly lower than either model.
The weights are a judgment call: no verification yet shows RRFS improves on HRRR. That is what the scoring below is for.

## Station snapshots and scoring (what an adaptive blend would need)

Every run saves what each model said at the RAWS/observation stations for the mapped hours (HRRR, RRFS and the blend),
and later runs score matured snapshots against the station observations (`forecast/extended_verify.py`). Nothing here
changes a forecast.

* **Window statistics** (the whole window and the peak block, scored separately), same definition for forecast and observation: max temperature, min RH, max 10 m wind, total
  rain. An observed value needs 2/3 of the window's hours (rain needs all of them); nothing is zero-filled. Scoring
  waits 6 h after the window for late observations.
* **Storage uses the project's compression process.** One file per run, `(model, lead_hour, station)`, written by
  `hrrr_hourly.store.write_netcdf`: values rounded to each element's step (same names and steps as the hourly HRRR
  product), packed as int16, zlib-compressed. All models of a run share a file because the per-file overhead dominates
  at this size: about 120 stations x 12 h x 5 fields measured 47 KB for three models in one file against 114 KB as three
  files (plain float32 would be ~48 KB *per model*). Scores go through `fire_outlook.verification.write_table` (Parquet,
  CSV without pyarrow). Roughly 0.3 MB/day in total; snapshots and scores are pruned after 60 days
  (`SMF_EXTENDED_VERIFY_KEEP_DAYS`).
* **Where:** `<TESTBED_PRODUCTS_DIR>/verification/extended/` (`forecast/`, `scores/`, `summary.json`;
  `SMF_EXTENDED_VERIFY_DIR` overrides).
* **`summary.json`** holds MAE / RMSE / bias per day, model and variable over the last 21 days, and `weights_day2`: the
  HRRR share an adaptive blend *would* pick (inverse squared error on station-values both models forecast, per variable,
  clamped to 30-70 %, prior 70 % until there are 150 paired values over 7 run days). It is `status: "shadow"`,
  `applied: false`. Rain is never tuned. Turning it on is a separate, deliberate change once a few weeks of record show
  the weights are stable and beat 70/30.
* Stations are the RAWS stations that reported fuel moisture in the last 30 days (`core.database.get_db_path`), and one
  farther than 20 km from every model cell is skipped.

## Differences from Day 1 (all deliberate)

* **Not started from observations.** Day 1's spatial model starts from the morning RAWS fuel moisture. At 24-48 h
  that start is irrelevant, so Day 2/3 run the XGBoost model from the forecast's own T/RH/wind/rain history, beginning
  24 h before the window so its 24-hour rain features are real (Day 1's window starts with an empty history, so its
  `precip_24h` is always 0).
* **Day 3 is regridded.** RRFS is 13 km. The window summaries are interpolated (linear) to 3 km, then smoothed
  exactly as Day 1 smooths, so contours are not blocky. No hourly risk smoothing at 13 km. The coarse grid is first
  extended by 2 cells of repeated edge values: interpolation alone only covers the area between the outermost cell
  centres, which stopped short of Missouri's border (a ragged rim the width of a coarse cell).
* **Map layout.** Titles longer than 44 characters (the RH and fuel-moisture maps) break after the day number so they do
  not run over the state's northeast corner, and the peak hours sit on their own subtitle line (not on the rain map).
  The rain legend labels are two lines (`0.50"` over `12.7 mm`) in every map script so the legend never reaches the state.
* **Rain map** is the 24 hours ending at the end of the window (Day 1 shows start-of-run to f15). Every precipitation map
  (Day 1, 2, 3 and the 72-hour graphics) uses one color table, the AccuWeather QPF scale in `assets/AccuWeather_QPF_in.pal`,
  loaded by `core/precip_palette.py`: 41 discrete steps from 0.025" to 40" (white below the first step, the top color above
  the last). To change it, replace the .pal file; the legend ticks and contour lines are `TICK_VALUES` / `LINE_VALUES` there.
* **No side effects:** no station_forecasts rows, v5 / risk-fusion / fire-weather-index hooks, ensemble hand-off, GIS
  publication, `forecast_peak` archive, or R2 upload. Beta-vs-stable shadow scoring is switched off in this process.
* **No GEFS / REFS fallback.** If RRFS is missing, Day 3 fails and Day 2 is still produced (from HRRR alone).

## Output

Maps `mo-forecast-day{2,3}-{firedanger,fuelmoisture,minrh,maxwind,maxtemp,rainfall}.png` and
`extended-forecast-day{2,3}.json` go to `BETA_IMAGES_DIR` (`TESTBED_IMAGES_DIR`, default `data/testbed/images`), and the
Testbed manifest gets an `extended_forecast` product entry listing them. They are overwritten by each newer run. The
JSON has the run, window, the blend weights, percent of the state at Elevated+ / Critical+, minimum fuel moisture,
runtime, and any flags (`precipitation_missing`, `snow_cover_missing`, `short_warmup`, `rrfs_blend_missing`,
`blend_failed`).

## Running

```
python forecast/ExtendedForecast.py                 # both days, newest eligible run old enough, skips what is built
python forecast/ExtendedForecast.py --day 2 --cycle 2026-11-19T18 --no-render
python forecast/ExtendedForecast.py --wait-minutes 150   # retry every 5 min while a run is still publishing
python forecast/ExtendedForecast.py --force --no-blend --no-verify
```

`scripts/extended_forecast.sh` wraps it for cron (forecast lock, log file, honest exit code). Minimum run age:
HRRR 4 h (`SMF_EXTENDED_HRRR_MIN_AGE_HOURS`), RRFS 7 h (`SMF_EXTENDED_RRFS_MIN_AGE_HOURS`).

Tests: `tests/test_extended_forecast.py` (windows per run, run selection, lead sets, rain differencing, rule parity,
regridding, blend) and `tests/test_extended_verify.py` (snapshots, packing round trip, scoring, shadow weights, pruning).
The real fuel-moisture model path needs the registered stable model, and the RRFS path needs the Docker-only patch, so
those are exercised only where they exist (Docker/prod).
