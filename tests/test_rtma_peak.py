from datetime import date, datetime, timezone

import numpy as np
import xarray as xr

from services.rtma_peak import (
    PEAK_WINDOW_HOURS,
    _calibrate_fuel_moisture,
    _classify_grid,
    _fuel_moisture_observations,
    _hours_for_local_date,
    _lon180,
    _lon_lat_meshes,
)


def test_rtma_peak_hours_match_verification_window():
    hours = _hours_for_local_date(date(2026, 8, 29))
    assert len(hours) == PEAK_WINDOW_HOURS
    chicago = hours[0].astimezone(__import__("zoneinfo").ZoneInfo("America/Chicago"))
    assert chicago.hour == 10
    last = hours[-1].astimezone(__import__("zoneinfo").ZoneInfo("America/Chicago"))
    assert last.hour == 21
    assert last.date() == date(2026, 8, 29)


def test_rtma_longitude_is_converted_to_western_hemisphere():
    lon, lat = _lon_lat_meshes(np.array([265.5, 270.2]), np.array([36.0, 38.0, 40.0]))
    assert lon.min() < 0
    assert lon.max() < 0
    assert np.isclose(lon[0, 0], 265.5 - 360)
    assert lat.shape == lon.shape == (3, 2)
    assert np.all(_lon180(np.array([267.0, -92.0])) == np.array([267.0 - 360, -92.0]))


def test_archive_observations_select_closest_measured_fuel_moisture():
    payload = {
        "STATION": [{
            "STID": "TEST",
            "LATITUDE": "38.0",
            "LONGITUDE": "-92.0",
            "OBSERVATIONS": {
                "date_time": ["2026-09-04T15:53:00Z", "2026-09-04T16:53:00Z"],
                "fuel_moisture_set_1": [12.0, 8.0],
            },
        }],
    }
    observations = _fuel_moisture_observations(
        payload, datetime(2026, 9, 4, 17, tzinfo=timezone.utc)
    )
    assert len(observations) == 1
    assert observations[0]["fuel_moisture"] == 8.0


def test_measured_fuel_moisture_calibrates_rh_estimate():
    lon, lat = np.meshgrid(np.array([-93.0, -92.0, -91.0]), np.array([37.0, 38.0, 39.0]))
    estimate = np.full(lon.shape, 15.0)
    observations = [
        {"longitude": -93.0, "latitude": 37.0, "fuel_moisture": 7.0},
        {"longitude": -92.0, "latitude": 38.0, "fuel_moisture": 7.0},
        {"longitude": -91.0, "latitude": 39.0, "fuel_moisture": 7.0},
    ]
    calibrated = _calibrate_fuel_moisture(estimate, lon, lat, observations)
    assert np.all(calibrated < estimate)
    assert np.isclose(calibrated[1, 1], 7.0, atol=0.1)


def test_rtma_classification_falls_back_to_rh_estimate_without_measurements():
    ds = xr.Dataset({
        "longitude": (("y", "x"), np.array([[268.0, 269.0], [268.0, 269.0]])),
        "latitude": (("y", "x"), np.array([[37.0, 37.0], [38.0, 38.0]])),
        "r2": (("y", "x"), np.full((2, 2), 20.0)),
        "u10": (("y", "x"), np.full((2, 2), 10.0)),
        "v10": (("y", "x"), np.zeros((2, 2))),
    })
    result, _, _ = _classify_grid(ds)
    assert result.shape == (2, 2)
    assert np.isfinite(result).all()
