"""
Ensemble fire danger (Day 1, BETA): chance of >= Moderate / Elevated
(High) / Critical (Very High) / Extreme fire danger, plus a categorical
ensemble forecast, from short-range convection-allowing guidance.

Two tracks, run side by side (see model-training/docs/ensemble_fire_danger.md
for the full design and why there are two):

- Track M "members": an HREF-style ensemble of REAL members - HRRR
  and RRFS, each also as time-lagged runs - each
  pushed through the operational fuel-moisture model and fire danger rule.
- Track E "ensprod": synthetic members drawn from the REFS published
  ensemble mean + spread (REFS publishes no per-member surface grids -
  confirmed live 2026-09-22). HREF was a second input until NOAA retired it.

Modules marked "contract mirror" in their docstring (core, grib_idx,
members, regrid, tracks, fm_features, member_config.json) are byte-
identical to model-training/ensemble_fire_danger/ and listed in
core/contract_mirrors.json. The rest (runtime, render, forecast_state,
bundle) are api-only.

Kept import-light on purpose: DailyForecast imports forecast_state from
this package on every run.
"""
