# Day 2 / Day 3 forecast (beta)

`forecast/ExtendedForecast.py` runs Day 1's forecast logic on the next two days, as a fast alternative to the
72-hour `forecast_v1` pipeline's Day 2/3 products. Built 2026-11-19. Beta, Testbed only, not scheduled.

| Day | Model run | Hours | Local window (standard time) |
|-----|-----------|-------|------------------------------|
| 2   | HRRR 12z, 3 km  | f28-f39 | 10:00-21:00 CT the next day |
| 3   | RRFS 12z, 13 km | f52-f63 | 10:00-21:00 CT two days out |

(Day 1 is HRRR f04-f15. Each later day is the same afternoon window, 24 h further on. Under daylight time the window
is 11:00-22:00.)

## What is shared with Day 1

The stable fuel-moisture model and its features (`DailyForecast.predict_fm_grid`), the fire-wind reduction factor,
the canonical fire-danger rule with its seasonal softening, the SNODAS snow mask, hourly risk smoothing, the summary
smoothing, and the map house style (`forecast/map_style.py`, same palettes and legends). The six statewide maps are
the same six as Day 1: peak fire danger, minimum fuel moisture, minimum RH, maximum wind, maximum temperature, rain.

## Why it is fast

`forecast_v1` pulls HRRR, RRFS and 31 GEFS members with ~15 fields each, blends them, and scores 51 scenarios per
cell. This job asks `forecast_v1.acquisition.acquire_fields` for 5 fields (T, RH, wind, gust, rain) from one
deterministic model, only for the 36 hours it needs, clipped to Missouri; the clipped result is cached under
`cache/extended/` so re-runs and the second day are instant. The fire-danger rule is only evaluated on cells drier than
the Low threshold (Day 1 loops over every cell in Python).

## Differences from Day 1 (all deliberate)

* **Not started from observations.** Day 1's spatial model starts from the morning RAWS fuel moisture. At 24-48 h
  that start is irrelevant, so Day 2/3 run the XGBoost model from the forecast's own T/RH/wind/rain history, beginning
  24 h before the window so its 24-hour rain features are real (Day 1's window starts with an empty history, so its
  `precip_24h` is always 0).
* **Day 3 is regridded.** RRFS is 13 km. The window summaries are interpolated (linear) to 3 km, then smoothed
  exactly as Day 1 smooths, so contours are not blocky. No hourly risk smoothing at 13 km.
* **Rain map** is the 24 hours ending at the end of the window (Day 1 shows start-of-run to f15).
* **No side effects:** no station_forecasts rows, v5 / risk-fusion / fire-weather-index hooks, ensemble hand-off, GIS
  publication, `forecast_peak` archive, or R2 upload. Beta-vs-stable shadow scoring is switched off in this process.
* **No GEFS / REFS fallback.** If RRFS is missing, Day 3 fails and Day 2 is still produced.

## Output

Maps `mo-forecast-day{2,3}-{firedanger,fuelmoisture,minrh,maxwind,maxtemp,rainfall}.png` and
`extended-forecast-day{2,3}.json` go to `BETA_IMAGES_DIR` (`TESTBED_IMAGES_DIR`, default `data/testbed/images`), and the
Testbed manifest gets an `extended_forecast` product entry listing them. The JSON has the cycle, window, percent of the
state at Elevated+ / Critical+, minimum fuel moisture, runtime, and any flags (`precipitation_missing`,
`snow_cover_missing`).

## Running

```
python forecast/ExtendedForecast.py                 # both days, newest 12z old enough
python forecast/ExtendedForecast.py --day 2 --cycle 2026-11-19T12 --no-render
python forecast/ExtendedForecast.py --wait-minutes 150   # retry every 5 min while the run is still publishing
```

`scripts/extended_forecast.sh` wraps it for cron (forecast lock, log file, honest exit code). The Dockerfile cron line
to use is in the script's header; it is not added yet. The newest 12z is chosen by age: HRRR 4 h
(`SMF_EXTENDED_HRRR_MIN_AGE_HOURS`), RRFS 7 h (`SMF_EXTENDED_RRFS_MIN_AGE_HOURS`).

Tests: `tests/test_extended_forecast.py` (windows, lead sets, rain differencing, rule parity, regridding). The real
fuel-moisture model path needs the registered stable model, so it is exercised only where that exists (Docker/prod).
