from __future__ import annotations

import numpy as np

from services.verification_rainfall import (
    CONTRACT_VERSION,
    adjust_category,
    adjust_grid,
    category_reduction,
    combine_category_grids,
    default_nlcd_raster_path,
    provider_precedence,
    regime_for_nlcd,
)


def test_nlcd_classes_map_to_documented_regimes():
    assert regime_for_nlcd(71) == "grass_pasture"
    assert regime_for_nlcd(82) == "agriculture"
    assert regime_for_nlcd(52) == "shrubland"
    assert regime_for_nlcd(42) == "dense_forest"
    assert regime_for_nlcd(999) is None


def test_default_nlcd_path_matches_acquisition_location():
    assert default_nlcd_raster_path().name == "nlcd_class.tif"
    assert default_nlcd_raster_path().parent.name == "static"


def test_threshold_event_reduces_danger_by_at_most_two_levels():
    suppression = category_reduction(
        10.0,
        "grass_pasture",
        relative_humidity=60,
        wind_kts=2,
    )
    assert suppression["contract_version"] == CONTRACT_VERSION
    assert suppression["reduction"] == 2
    assert adjust_category(4, suppression) == 2


def test_relief_decays_and_weather_can_remove_reduction():
    recent = category_reduction(2.5, "grass_pasture", relative_humidity=70, wind_kts=0)
    stale = category_reduction(
        2.5,
        "grass_pasture",
        hours_since_rain=72,
        relative_humidity=10,
        wind_kts=25,
    )
    assert recent["reduction"] == 2
    assert stale["reduction"] == 0


def test_missing_inputs_fail_closed():
    missing_land_use = category_reduction(20, None)
    missing_rain = category_reduction(None, "shrubland")
    assert missing_land_use["reason"] == "land_use_unavailable"
    assert missing_rain["reason"] == "rainfall_unavailable"


def test_provider_precedence_prefers_mrms_then_rtma_then_station():
    assert provider_precedence(mrms_mm=1, rtma_mm=3, station_mm=4)["provider"] == "mrms"
    assert provider_precedence(rtma_mm=3, station_mm=4)["provider"] == "rtma"
    assert provider_precedence(station_mm=4)["provider"] == "station"
    assert provider_precedence()["provider"] is None


def test_grid_adjustment_and_combination_preserve_nodata():
    adjusted, reductions = adjust_grid(
        np.array([[4, 2], [np.nan, 1]], dtype=float),
        np.array([[5, 0], [5, 5]], dtype=float),
        np.array([[71, 71], [71, 71]], dtype=float),
        relative_humidity=60,
        wind_kts=2,
    )
    assert adjusted[0, 0] == 2
    assert reductions[0, 0] == 2
    assert np.isnan(adjusted[1, 0])
    combined = combine_category_grids(adjusted, np.array([[1, 1], [1, 1]], dtype=float))
    assert combined[0, 0] == 2
    assert np.isfinite(combined).all()
