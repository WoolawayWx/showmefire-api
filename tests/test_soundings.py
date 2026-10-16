"""Radiosonde parsing and fire-weather metrics (no network)."""
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import soundings as snd

VALID = datetime(2026, 10, 16, 12, tzinfo=timezone.utc)


def synthetic_rows(surface_temp=20.0, inversion=False):
    """A well-mixed column (potential temperature falling very slightly with
    height, so a surface parcel never stops) from 970 to 300 hPa, westerly winds."""
    rows = [{"pres": 1000.0, "hght": 100.0, "tmpc": None, "dwpc": None, "drct": None, "sknt": None}]  # below ground
    theta_surface = (surface_temp + 273.15) * (1000.0 / 970.0) ** 0.2857
    for index, pres in enumerate(range(970, 299, -10)):
        height = 400.0 + index * 100.0
        temp = (theta_surface - 0.01 * index) * (pres / 1000.0) ** 0.2857 - 273.15
        if inversion and height <= 600:
            temp = surface_temp + 0.02 * (height - 400.0)  # warms 4 C over the lowest 200 m
        rows.append({"pres": float(pres), "hght": height, "tmpc": temp, "dwpc": temp - 15,
                     "drct": 270.0, "sknt": 20.0})
    return rows


class ParseTests(unittest.TestCase):
    def test_surface_is_first_level_with_temperature(self):
        sounding = snd.parse_profile("KSGF", VALID, synthetic_rows())
        self.assertEqual(sounding.pressure_hpa[0], 970.0)
        self.assertEqual(sounding.height_agl_m[0], 0.0)
        self.assertTrue(np.all(np.diff(sounding.pressure_hpa) < 0))

    def test_winds_are_earth_relative_components(self):
        sounding = snd.parse_profile("KSGF", VALID, synthetic_rows())
        # 20 kt from the west blows toward the east: u > 0, v ~ 0.
        self.assertAlmostEqual(sounding.u_ms[0], 20 / 1.943844, places=3)
        self.assertAlmostEqual(sounding.v_ms[0], 0.0, places=6)

    def test_incomplete_profile_is_rejected(self):
        rows = [row for row in synthetic_rows() if row["pres"] >= 700]
        self.assertIsNone(snd.parse_profile("KSGF", VALID, rows))

    def test_special_launch_detection(self):
        self.assertFalse(snd.is_special(datetime(2026, 10, 16, 0, tzinfo=timezone.utc)))
        self.assertFalse(snd.is_special(VALID))
        self.assertTrue(snd.is_special(datetime(2026, 10, 16, 18, tzinfo=timezone.utc)))


class MetricTests(unittest.TestCase):
    def test_well_mixed_afternoon_profile(self):
        metrics = snd.fire_weather_metrics(snd.parse_profile("KSGF", VALID, synthetic_rows()))
        self.assertGreater(metrics["mixing_height_ft"], 10000)  # never meets the dry adiabat in the column
        self.assertEqual(metrics["inversion_strength_f"], 0.0)
        self.assertAlmostEqual(metrics["transport_speed_mph"], 23.0, delta=0.1)
        self.assertEqual(metrics["transport_dir_deg"], 270)
        self.assertEqual(metrics["ventilation_category"], "Excellent")

    def test_morning_inversion_traps_mixing(self):
        metrics = snd.fire_weather_metrics(snd.parse_profile("KSGF", VALID, synthetic_rows(inversion=True)))
        self.assertEqual(metrics["mixing_height_ft"], 0)
        self.assertAlmostEqual(metrics["inversion_strength_f"], 7.2, places=1)  # 4 C
        self.assertEqual(metrics["inversion_top_ft"], round(200 * 3.28084))
        self.assertEqual(metrics["ventilation_category"], "Poor")


class ListTests(unittest.TestCase):
    def test_list_launches_returns_unique_times_including_specials(self):
        csv_text = ("station,validUTC,levelcode,pressure_mb\n"
                    "KOAX,2026-10-16 00:00:00,4,1000\nKOAX,2026-10-16 00:00:00,4,925\n"
                    "KOAX,2026-10-16 18:00:00,4,1000\n")
        response = mock.Mock(text=csv_text)
        response.raise_for_status = mock.Mock()
        with mock.patch.object(snd.requests, "get", return_value=response):
            launches = snd.list_launches("KOAX", VALID, VALID)
        self.assertEqual(launches, [datetime(2026, 10, 16, 0, tzinfo=timezone.utc),
                                    datetime(2026, 10, 16, 18, tzinfo=timezone.utc)])


class RetentionTests(unittest.TestCase):
    def test_keeps_two_latest_plus_recent_specials(self):
        times = [datetime(2026, 10, day, hour, tzinfo=timezone.utc) for day, hour in
                 [(1, 18), (14, 0), (15, 12), (15, 18), (16, 0), (16, 12)]]
        now = datetime(2026, 10, 16, 14, tzinfo=timezone.utc)
        kept = snd.kept_launches(times, now)
        # Newest two, plus the 15th's 18Z special; the Oct 1 special is older than a week.
        self.assertEqual(kept, [times[5], times[4], times[3]])


class ManifestTests(unittest.TestCase):
    def setUp(self):
        from maps import sounding_maps as sm

        self.sm = sm
        base = snd.parse_profile("KSGF", VALID, synthetic_rows())
        later = snd.parse_profile("KSGF", datetime(2026, 10, 16, 18, tzinfo=timezone.utc),
                                  synthetic_rows(inversion=True))

        def record(sounding):
            return {"station": "KSGF", "valid_time": sounding.valid.isoformat(), "special": snd.is_special(sounding.valid),
                    "fire_weather": snd.fire_weather_metrics(sounding), "thermodynamics": {"pwat_in": 0.5},
                    "omega": {"source": "HRRR analysis", "pressure_hpa": [1000.0], "omega_pa_s": [0.1]}}

        self.previous, self.record = record(base), record(later)

    def test_changes_since_previous(self):
        delta = self.sm.changes(self.record, self.previous)
        self.assertEqual(delta["since"], VALID.isoformat())
        self.assertLess(delta["mixing_height_ft"], 0)       # inversion trapped mixing
        self.assertAlmostEqual(delta["inversion_strength_f"], 7.2, places=1)
        self.assertEqual(delta["pwat_in"], 0.0)

    def test_manifest_entry_names_and_labels(self):
        entry = self.sm.manifest_entry("KSGF", self.record, self.previous)
        self.assertEqual(entry["image"], "sounding-sgf-20261016-1800z.png")
        self.assertEqual(entry["thumbnail"], "sounding-sgf-20261016-1800z-sm.webp")
        self.assertTrue(entry["special"])
        self.assertIn("(Special)", entry["label"])
        self.assertEqual(entry["omega_source"], "HRRR analysis")

    def test_change_line_skips_zero_changes(self):
        line = self.sm._change_line(self.record, self.previous, self.sm.changes(self.record, self.previous))
        self.assertIn("Change since 12Z Oct 16", line)
        self.assertNotIn("PW", line)

    def test_sounding_round_trips_through_archive_record(self):
        sounding = snd.parse_profile("KSGF", VALID, synthetic_rows())
        record = {"station": "KSGF", "valid_time": VALID.isoformat(),
                  "profile": {key: [float(v) for v in getattr(sounding, attr)] for key, attr in
                              (("pressure_hpa", "pressure_hpa"), ("height_m", "height_m"), ("temp_c", "temp_c"),
                               ("dewpoint_c", "dewpoint_c"), ("u_ms", "u_ms"), ("v_ms", "v_ms"))}}
        rebuilt = snd.sounding_from_record(record)
        np.testing.assert_allclose(rebuilt.temp_c, sounding.temp_c)
        self.assertEqual(rebuilt.valid, VALID)


class Cw3eTests(unittest.TestCase):
    LISTING = (
        '<a href="USCOU_20260314_1200.txt">x</a> <a href="USCOU_20260315_18.txt">x</a> '
        '<a href="USCOU_USCOU_20260122_21Z.txt">x</a> <a href="USCOU_20260122_2100.txt">x</a> '
        '<a href="ReadMe.txt">x</a>'
    )
    FLIGHT = ("Height (m), Pressure (hPa), Temperature (C), Dew Point (C), Wind Direction, Wind Speed (m/s)\n"
              + "\n".join(f"{253 + 100 * i},{964.1 - 10 * i},{17.2 - 0.9 * i},{12.3 - 1.5 * i},200,{6 + 0.2 * i}"
                          for i in range(70)))

    def _response(self, text):
        response = mock.Mock(text=text)
        response.raise_for_status = mock.Mock()
        return response

    def test_listing_handles_every_file_name_style(self):
        with mock.patch.object(snd.requests, "get", return_value=self._response(self.LISTING)):
            files = snd._cw3e_files("USCOU")
        self.assertEqual(sorted(files), [datetime(2026, 1, 22, 21, tzinfo=timezone.utc),
                                         datetime(2026, 3, 14, 12, tzinfo=timezone.utc),
                                         datetime(2026, 3, 15, 18, tzinfo=timezone.utc)])
        self.assertEqual(files[datetime(2026, 3, 15, 18, tzinfo=timezone.utc)], "USCOU_20260315_18.txt")

    def test_flight_file_parses_with_winds_in_knots(self):
        sounding = snd.parse_cw3e("USCOU", VALID, self.FLIGHT)
        self.assertEqual(sounding.pressure_hpa[0], 964.1)
        self.assertEqual(sounding.height_m[0], 253.0)
        # 6 m/s from 200 degrees: blowing toward the north-northeast.
        self.assertAlmostEqual(float(np.hypot(sounding.u_ms[0], sounding.v_ms[0])), 6.0, places=3)
        self.assertGreater(sounding.v_ms[0], 0)

    def test_university_flights_are_always_special_and_named_cou(self):
        self.assertTrue(snd.launch_is_special("USCOU", VALID))     # even at 12Z
        self.assertFalse(snd.launch_is_special("KSGF", VALID))
        self.assertEqual(snd.site_slug("USCOU"), "cou")
        self.assertEqual(snd.site_slug("KSGF"), "sgf")

    def test_parcel_excess_ignores_near_surface_sensor_noise(self):
        rows = synthetic_rows()
        surface = rows[1]
        # A 0.1 C warm wiggle 10 m above the surface, like a settling sensor; a real
        # inversion would be larger or deeper than the 0.5 K parcel excess.
        rows.insert(2, {**surface, "pres": surface["pres"] - 1.1, "hght": surface["hght"] + 10,
                        "tmpc": surface["tmpc"] + 0.1})
        metrics = snd.fire_weather_metrics(snd.parse_profile("KSGF", VALID, rows))
        self.assertGreater(metrics["mixing_height_ft"], 1000)


if __name__ == "__main__":
    unittest.main()
