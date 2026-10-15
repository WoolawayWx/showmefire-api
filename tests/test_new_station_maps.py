import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

API_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(API_DIR))

from services import derived_met


class WindHeightTests(unittest.TestCase):
    def test_same_height_is_unchanged(self):
        self.assertAlmostEqual(derived_met.wind_at_height(10.0, 2.0, 2.0), 10.0)

    def test_lower_height_is_slower_and_higher_is_faster(self):
        self.assertLess(derived_met.wind_at_height(10.0, 10.0, 2.0), 10.0)
        self.assertGreater(derived_met.wind_at_height(10.0, 2.0, 10.0), 10.0)

    def test_very_low_sensor_height_is_left_alone(self):
        self.assertEqual(derived_met.wind_at_height(10.0, 0.1, 2.0), 10.0)


class SolarElevationTests(unittest.TestCase):
    def test_midday_is_high_and_midnight_is_below_horizon(self):
        noon = datetime(2026, 6, 21, 18, 0, tzinfo=timezone.utc)  # ~1 pm CDT
        midnight = datetime(2026, 6, 22, 5, 0, tzinfo=timezone.utc)
        self.assertGreater(float(derived_met.solar_elevation_deg(38.5, -92.5, noon)), 65)
        self.assertLess(float(derived_met.solar_elevation_deg(38.5, -92.5, midnight)), 0)


class PasquillTests(unittest.TestCase):
    def test_calm_sunny_is_unstable_and_calm_clear_night_is_stable(self):
        self.assertLessEqual(derived_met.pasquill_stability_index(1.0, 70), 1.5)
        self.assertGreaterEqual(derived_met.pasquill_stability_index(1.0, -20), 6.0)

    def test_strong_wind_is_neutral_day_or_night(self):
        self.assertEqual(derived_met.pasquill_stability_index(8.0, -20), 4.0)
        self.assertEqual(derived_met.pasquill_stability_index(8.0, 20), 4.0)


class PressureUnitTests(unittest.TestCase):
    def test_standard_pressure_converts_to_29_92_inhg(self):
        self.assertAlmostEqual(float(derived_met.hpa_to_inhg(1013.25)), 29.92, places=2)


class DryStreakTests(unittest.TestCase):
    def setUp(self):
        from maps import rainfall_maps

        self.streak = rainfall_maps.dry_streak

    def test_counts_back_to_first_wet_day(self):
        totals = {"2026-10-13": 0.0, "2026-10-12": 0.02, "2026-10-11": 0.0, "2026-10-10": 0.4, "2026-10-09": 0.0}
        self.assertEqual(self.streak(totals, "2026-10-13", 0.10), 3)

    def test_rain_today_resets_to_zero(self):
        self.assertEqual(self.streak({"2026-10-13": 0.30, "2026-10-12": 0.0}, "2026-10-13", 0.25), 0)

    def test_missing_day_stops_the_count_and_missing_today_is_unknown(self):
        self.assertEqual(self.streak({"2026-10-13": 0.0, "2026-10-11": 0.0}, "2026-10-13", 0.10), 1)
        self.assertIsNone(self.streak({"2026-10-12": 0.0}, "2026-10-13", 0.10))


if __name__ == "__main__":
    unittest.main()
