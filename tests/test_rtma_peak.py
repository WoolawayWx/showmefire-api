from datetime import date

import numpy as np

from services.rtma_peak import PEAK_WINDOW_HOURS, _hours_for_local_date, _lon180, _lon_lat_meshes


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
