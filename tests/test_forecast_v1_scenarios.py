from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import xarray as xr

from core.fire_danger import MPS_TO_KNOTS, calculate_fire_danger
from forecast_v1.engine import _classify_values, _emc_rh_sensitivity, add_risk_and_confidence, daily_aggregates

CYCLE = datetime(2026, 9, 6, 12, tzinfo=timezone.utc)
TIMES = np.array([np.datetime64(CYCLE.replace(tzinfo=None) + timedelta(hours=lead), "ns") for lead in range(73)])


def _forecast(cells: list[tuple[float, float, float]]) -> xr.Dataset:
    """Constant hourly fm/RH/wind (m/s) per cell, one row of cells."""
    fm, rh, wind = (np.tile(np.array(column, dtype=np.float32), (73, 1, 1)) for column in zip(*cells))
    dims = ("time", "y", "x")
    coords = {"time": TIMES, "y": [0.0], "x": np.arange(len(cells), dtype=float)}
    return xr.Dataset(
        {
            "fuel_moisture_p10": (dims, fm), "fuel_moisture_p50": (dims, fm), "fuel_moisture_p90": (dims, fm),
            "relative_humidity_2m": (dims, rh), "wind_speed_10m": (dims, wind), "wind_gust_10m": (dims, wind * 1.5),
            "temperature_2m": (dims, np.full_like(fm, 25.0)), "quality_mask": (dims, np.zeros(fm.shape, dtype=np.uint16)),
            "precipitation_increment": (dims, np.zeros_like(fm)), "snow_water_equivalent": (dims, np.zeros_like(fm)),
        },
        coords=coords,
    )


BENIGN = (25.0, 80.0, 1.0)  # wet fuels: always Low
# Dry fuels, 21 kt (well clear of the 12 kt Elevated wind but under every
# Critical/Extreme combination at this RH), RH exactly at the Elevated limit.
AT_ELEVATED_RH = (7.0, 35.0, 11.0)


def test_vectorized_rule_matches_the_canonical_scalar_rule():
    rng = np.random.default_rng(0)
    fm, rh, wind = rng.uniform(0, 30, 3000), rng.uniform(0, 100, 3000), rng.uniform(0, 20, 3000)
    expected = [calculate_fire_danger(f, r, w * MPS_TO_KNOTS) for f, r, w in zip(fm, rh, wind)]
    assert _classify_values(fm, rh, wind).tolist() == expected
    assert _classify_values(np.array([np.nan]), np.array([30.0]), np.array([5.0])).tolist() == [255]


def test_chance_of_elevated_is_zero_when_benign_and_split_at_the_threshold():
    result = add_risk_and_confidence(_forecast([BENIGN, AT_ELEVATED_RH]), scenario_count=201)
    hourly = result.probability_elevated_plus.isel(y=0).values
    assert (hourly[:, 0] == 0).all()
    assert 35 <= hourly[30, 1] <= 65
    critical = result.probability_critical_plus.values
    assert (critical <= result.probability_elevated_plus.values).all()
    assert int(result.category_confidence.isel(time=30, y=0, x=0)) == 100  # the old metric: "certain" when benign


def test_daily_chance_holds_errors_for_the_day_instead_of_compounding_hours():
    # Independent hourly noise would make "any of 24 hours" nearly certain at a
    # 50/50 cell; a day-long error keeps the daily chance near the hourly one.
    result = add_risk_and_confidence(_forecast([AT_ELEVATED_RH]), scenario_count=201)
    daily = daily_aggregates(result)
    day_2 = int(daily.probability_elevated_plus.isel(day=1, y=0, x=0))
    assert abs(day_2 - int(result.probability_elevated_plus.isel(time=30, y=0, x=0))) <= 10
    assert daily.probability_critical_plus.dtype == np.uint8
    assert (daily.probability_critical_plus <= daily.probability_elevated_plus).all()


def test_measured_error_spread_widens_the_scenarios():
    # 3 points under the RH limit: barely uncertain with a small spread.
    near = (7.0, 32.0, 11.0)
    narrow = add_risk_and_confidence(
        _forecast([near]), scenario_count=201,
        error_sigma={"relative_humidity_2m": xr.DataArray(np.full(73, 1.0), dims="time", coords={"time": TIMES})},
    )
    wide = add_risk_and_confidence(
        _forecast([near]), scenario_count=201,
        error_sigma={"relative_humidity_2m": xr.DataArray(np.full(73, 12.0), dims="time", coords={"time": TIMES})},
    )
    narrow_chance = int(narrow.probability_elevated_plus.isel(time=30, y=0, x=0))
    wide_chance = int(wide.probability_elevated_plus.isel(time=30, y=0, x=0))
    # Fuel (7% vs the 9% limit) and wind spreads keep even the narrow case short of certain.
    assert narrow_chance >= 85 and wide_chance <= 70


def test_fuel_moisture_follows_humidity_errors():
    temperature = xr.DataArray([25.0, 25.0])
    sensitivity = _emc_rh_sensitivity(temperature, xr.DataArray([20.0, 60.0]))
    assert (sensitivity > 0.05).all() and (sensitivity < 0.5).all()
