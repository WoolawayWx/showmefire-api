import unittest
from datetime import datetime, timedelta, timezone

from services.station_et import derive_all, derive_station_et, hourly_eto_mm, solar_geometry

NOW = datetime(2026, 10, 15, 2, 0, tzinfo=timezone.utc)


def _raws(skip=(), stale_hours=0, drop=(), elevation_ft=1100.0, hours=26):
    """26 hourly RAWS observations (at :53) of a clear mid-October day in Missouri.

    ``skip`` = indexes (0 = newest) with no observation; ``drop`` = variables absent entirely.
    """
    lat, lon = 38.0, -92.0
    newest = NOW - timedelta(minutes=7 + 60 * stale_hours)
    whens = [newest - timedelta(hours=k) for k in range(hours) if k not in skip][::-1]
    columns = {"air_temp": [], "relative_humidity": [], "wind_speed": [], "solar_radiation": []}
    for when in whens:
        _, sun = solar_geometry(lat, lon, when)
        sun = max(0.0, sun)
        columns["air_temp"].append(12.0 + 9.0 * sun)
        columns["relative_humidity"].append(85.0 - 40.0 * sun)
        columns["wind_speed"].append(2.5)
        columns["solar_radiation"].append(700.0 * sun)
    observations = {"date_time": [t.isoformat() for t in whens]}
    observations.update({f"{name}_set_1": values for name, values in columns.items() if name not in drop})
    return {"STID": "TEST", "NAME": "Test RAWS", "STATE": "MO", "LATITUDE": str(lat), "LONGITUDE": str(lon),
            "ELEVATION": elevation_ft, "OBSERVATIONS": observations}


class HourlyEtoTests(unittest.TestCase):
    def test_matches_fao56_hourly_example(self):
        # ASCE-EWRI short crop (Cd 0.24 by day) runs ~4 % above the FAO-56 published value
        # (0.63 mm/h, Cd 0.34) for the same inputs, so the expectation is ~0.65.
        # FAO-56 Example 19: 1 Oct, 14:00-15:00 local standard time (Lz = 15 E = UTC+1) at
        # 16 13'N 16 15'E, 8 m; T 38 C, RH 52 %, u2 3.3 m/s, Rs 2.450 MJ/m2/h; inputs from the published example.
        when = datetime(2026, 10, 1, 13, 30, tzinfo=timezone.utc)
        lat, lon = 16 + 13 / 60, 16 + 15 / 60
        eto, ratio = hourly_eto_mm(38, 52, 3.3, 2.450 / 0.0036, lat, lon, 8, when, wind_height_m=2.0)
        self.assertAlmostEqual(eto, 0.65, delta=0.02)
        self.assertIsNotNone(ratio)  # a daytime hour reports its measured Rs/Rso
        ra, _ = solar_geometry(lat, lon, when)
        self.assertAlmostEqual(ra, 3.54, delta=0.2)  # published 3.543; ours is ~4 % low

    def test_night_is_small_and_never_negative(self):
        when = datetime(2026, 10, 15, 6, 0, tzinfo=timezone.utc)  # 1 am local
        eto, ratio = hourly_eto_mm(14, 90, 1.0, 0.0, 38.0, -92.0, 300, when)
        self.assertGreaterEqual(eto, 0.0)
        self.assertLess(eto, 0.05)
        self.assertIsNone(ratio)  # no measurable Rs/Rso at night

    def test_drier_windier_air_evaporates_more(self):
        when = datetime(2026, 7, 15, 18, 0, tzinfo=timezone.utc)
        humid, _ = hourly_eto_mm(28, 85, 1.0, 600, 38.0, -92.0, 300, when)
        dry, _ = hourly_eto_mm(28, 30, 5.0, 600, 38.0, -92.0, 300, when)
        self.assertGreater(dry, humid)


class StationEtTests(unittest.TestCase):
    def test_full_day_is_plausible_and_consistent(self):
        record = derive_station_et(_raws(), NOW)
        self.assertEqual(record["hours_used"], 24)
        self.assertGreater(record["et_24h_in"], 0.04)   # ~1 mm
        self.assertLess(record["et_24h_in"], 0.18)      # ~4.5 mm; mid-October Missouri
        self.assertGreater(record["et_max_in_hr"], record["et_min_in_hr"])
        self.assertLessEqual(record["et_min_in_hr"], 0.001)  # overnight
        self.assertLessEqual(record["et_24h_in"], 24 * record["et_max_in_hr"])
        self.assertTrue(record["in_state"])
        self.assertAlmostEqual(record["elevation_m"], 335.3, places=1)  # ELEVATION is in feet

    def test_one_missing_hour_is_interpolated(self):
        full = derive_station_et(_raws(), NOW)["et_24h_in"]
        gap = derive_station_et(_raws(skip={6}), NOW)
        self.assertEqual(gap["hours_used"], 23)
        self.assertAlmostEqual(gap["et_24h_in"], full, delta=full * 0.05)

    def test_too_many_missing_hours_withholds_total_but_keeps_rates(self):
        record = derive_station_et(_raws(skip={5, 6, 7}), NOW)
        self.assertIsNone(record["et_24h_in"])
        self.assertEqual(record["hours_used"], 21)
        self.assertIsNotNone(record["et_max_in_hr"])
        self.assertIsNotNone(record["et_current_in_hr"])

    def test_stale_station_has_no_record(self):
        self.assertIsNone(derive_station_et(_raws(stale_hours=5), NOW))

    def test_missing_input_or_elevation_has_no_record(self):
        self.assertIsNone(derive_station_et(_raws(drop={"solar_radiation"}), NOW))
        station = _raws()
        station["ELEVATION"] = None
        self.assertIsNone(derive_station_et(station, NOW))

    def test_implausible_hour_is_dropped_not_zeroed(self):
        station = _raws()
        station["OBSERVATIONS"]["relative_humidity_set_1"][10] = 400.0
        record = derive_station_et(station, NOW)
        self.assertEqual(record["hours_used"], 23)

    def test_ignored_stations_excluded(self):
        self.assertEqual(derive_all([_raws()], {"TEST"}, NOW), [])
        self.assertEqual(len(derive_all([_raws()], frozenset(), NOW)), 1)


if __name__ == "__main__":
    unittest.main()
