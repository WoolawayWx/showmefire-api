"""services/raws_calibration.py: RAWS residuals spread across a grid, fading out away from stations."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from services import raws_calibration as rc

# A 0.05-degree grid around central Missouri, about 330 x 440 km.
LAT, LON = np.meshgrid(np.arange(37.0, 40.0, 0.05), np.arange(-94.5, -90.5, 0.05), indexing="ij")


def _km_east(km):
    return km / (rc.KM_PER_DEG_LON_EQUATOR * np.cos(np.deg2rad(LAT.mean())))


def _cell(lat, lon):
    return np.unravel_index(rc.nearest_cell(LON, LAT, lon, lat), LAT.shape)


def test_one_station_full_at_the_station_half_at_the_fade_distance_little_far_away():
    correction = rc.correction_field(LON, LAT, [-92.5], [38.5], [-6.0], fade=75)
    assert correction[_cell(38.5, -92.5)] == pytest.approx(-6.0, abs=0.1)
    assert correction[_cell(38.5, -92.5 + _km_east(75))] == pytest.approx(-3.0, abs=0.2)
    assert abs(correction[_cell(38.5, -92.5 + _km_east(190))]) < 1.0


def test_stations_close_together_keep_their_correction_between_them():
    lons = [-92.5, -92.5 + _km_east(60)]
    correction = rc.correction_field(LON, LAT, lons, [38.5, 38.5], [-6.0, -6.0], fade=75)
    assert correction[_cell(38.5, -92.5 + _km_east(30))] < -4.5


def test_calibrate_applies_the_mean_offset_everywhere_and_local_departures_near_their_station():
    estimate = np.full(LAT.shape, 13.0)
    stations = [{"longitude": -91.0, "latitude": 39.5, "fuel_moisture": 22.0},     # +9, the odd one out
                {"longitude": -94.0, "latitude": 37.3, "fuel_moisture": 16.0},     # +3
                {"longitude": -94.2, "latitude": 37.5, "fuel_moisture": 16.0}]     # +3
    calibrated, meta = rc.calibrate(estimate, LON, LAT, stations, fade=75)
    assert meta == {"station_count": 3, "applied": True, "fade_km": 75, "mean_residual": 5.0}
    assert calibrated[_cell(39.5, -91.0)] == pytest.approx(22.0, abs=0.3)
    assert calibrated[_cell(37.4, -94.1)] == pytest.approx(16.0, abs=0.3)
    # Between them, ~150 km from either, it is close to the mean offset (13 + 5), not pulled to 22 or 16.
    assert calibrated[_cell(38.5, -92.6)] == pytest.approx(18.0, abs=1.0)


def test_calibrate_needs_three_usable_stations():
    estimate = np.full(LAT.shape, 13.0)
    estimate[_cell(38.5, -92.0)] = np.nan                  # a station over a missing cell is skipped
    stations = [{"longitude": -92.0, "latitude": 38.5, "fuel_moisture": 18.0},
                {"longitude": -91.0, "latitude": 38.0, "fuel_moisture": 18.0},
                {"longitude": -93.0, "latitude": 39.0, "fuel_moisture": 18.0}]
    calibrated, meta = rc.calibrate(estimate, LON, LAT, stations)
    assert meta["applied"] is False and meta["station_count"] == 2
    assert calibrated is estimate


def test_fresh_station_fuel_moisture_ignores_stale_readings():
    now = datetime(2026, 10, 22, 15, tzinfo=timezone.utc)
    station = {"observations": {"fuel_moisture": {"value": 11.3, "time": "2026-10-22T14:53:00Z"}}}
    assert rc.fresh_station_fuel_moisture(station, now=now) == 11.3
    stale = {"observations": {"fuel_moisture": {"value": 11.3, "time": "2026-10-22T12:53:00Z"}}}
    assert rc.fresh_station_fuel_moisture(stale, now=now) is None
    assert rc.fresh_station_fuel_moisture({"observations": {}}, now=now) is None


def test_synoptic_fuel_observations_takes_the_closest_reading_in_the_window():
    payload = {"STATION": [{"STID": "BEVM7", "LATITUDE": "38.0", "LONGITUDE": "-92.0", "OBSERVATIONS": {
        "date_time": ["2026-10-21T14:53:00Z", "2026-10-21T15:53:00Z"], "fuel_moisture_set_1": [11.3, 11.8]}}]}
    hour = datetime(2026, 10, 21, 16, tzinfo=timezone.utc)
    [reading] = rc.synoptic_fuel_observations(payload, hour)
    assert reading["fuel_moisture"] == 11.8 and reading["station"] == "BEVM7"
    assert rc.synoptic_fuel_observations(payload, hour + timedelta(hours=3)) == []
