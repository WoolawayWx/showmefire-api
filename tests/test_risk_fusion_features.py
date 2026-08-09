import unittest

import numpy as np
import pandas as pd

from core import risk_fusion_features as rff


class CalendarFeaturesTests(unittest.TestCase):
    def test_new_years_day_is_a_holiday(self):
        dates = pd.DatetimeIndex(["2026-01-01"])
        result = rff.calendar_features(dates)
        self.assertTrue(bool(result["is_holiday_us"].iloc[0]))

    def test_ordinary_tuesday_is_neither_weekend_nor_holiday(self):
        dates = pd.DatetimeIndex(["2026-03-10"])  # a Tuesday
        result = rff.calendar_features(dates)
        self.assertFalse(bool(result["is_weekend"].iloc[0]))
        self.assertFalse(bool(result["is_holiday_us"].iloc[0]))

    def test_saturday_is_a_weekend(self):
        dates = pd.DatetimeIndex(["2026-03-14"])  # a Saturday
        result = rff.calendar_features(dates)
        self.assertTrue(bool(result["is_weekend"].iloc[0]))

    def test_doy_sin_cos_are_bounded(self):
        dates = pd.date_range("2026-01-01", "2026-12-31", freq="17D")
        result = rff.calendar_features(dates)
        self.assertTrue((result["valid_doy_sin"].abs() <= 1.0001).all())
        self.assertTrue((result["valid_doy_cos"].abs() <= 1.0001).all())


class KeetchByramDroughtIndexTests(unittest.TestCase):
    def test_sustained_dry_heat_increases_kbdi_monotonically(self):
        n = 60
        temps = np.full(n, 32.0)  # 32C ~ 90F
        precip = np.zeros(n)
        result = rff.keetch_byram_drought_index(temps, precip, mean_annual_precip_mm=1000.0)
        kbdi = result["kbdi"]
        diffs = np.diff(kbdi)
        self.assertTrue(np.all(diffs >= -1e-9), "KBDI must not decrease with zero rain")
        self.assertGreater(kbdi[-1], kbdi[0])

    def test_kbdi_stays_within_the_standard_bounds(self):
        n = 120
        temps = np.full(n, 35.0)
        precip = np.zeros(n)
        result = rff.keetch_byram_drought_index(temps, precip, mean_annual_precip_mm=800.0)
        self.assertTrue(np.all(result["kbdi"] >= 0.0))
        self.assertTrue(np.all(result["kbdi"] <= 203.2 + 1e-6))

    def test_heavy_rain_drops_kbdi_substantially(self):
        temps = np.full(20, 30.0)
        precip = np.zeros(20)
        precip[15] = 60.0  # a 60mm rain event after two dry weeks
        result = rff.keetch_byram_drought_index(temps, precip, mean_annual_precip_mm=1000.0)
        kbdi = result["kbdi"]
        self.assertLess(kbdi[15], kbdi[14])

    def test_cold_days_do_not_increase_kbdi(self):
        temps = np.full(10, 5.0)  # well under the 50F/10C ET threshold
        precip = np.zeros(10)
        result = rff.keetch_byram_drought_index(temps, precip, mean_annual_precip_mm=1000.0)
        self.assertTrue(np.allclose(result["kbdi"], 0.0))

    def test_first_spinup_days_are_marked_invalid(self):
        n = 100
        result = rff.keetch_byram_drought_index(np.full(n, 25.0), np.zeros(n), mean_annual_precip_mm=1000.0, spinup_days=90)
        self.assertFalse(result["valid"][0])
        self.assertFalse(result["valid"][89])
        self.assertTrue(result["valid"][90])

    def test_rejects_non_positive_mean_annual_precip(self):
        with self.assertRaises(ValueError):
            rff.keetch_byram_drought_index(np.array([20.0]), np.array([0.0]), mean_annual_precip_mm=0.0)

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            rff.keetch_byram_drought_index(np.array([20.0, 21.0]), np.array([0.0]), mean_annual_precip_mm=1000.0)


class GrowingDegreeDaysTests(unittest.TestCase):
    def test_accumulates_only_positive_departures_from_base_temp(self):
        dates = pd.date_range("2026-03-01", periods=5, freq="D")
        temps = np.array([5.0, 15.0, 20.0, 8.0, 25.0])  # base 10C
        gdd = rff.growing_degree_days(temps, dates, base_temp_c=10.0)
        expected_daily = np.maximum(0.0, temps - 10.0)
        np.testing.assert_allclose(gdd, np.cumsum(expected_daily))

    def test_resets_at_the_next_march_first(self):
        # Two days in the season that started 2025-03-01, then the new
        # season starting 2026-03-01 must restart from zero rather than
        # continuing the prior season's accumulation.
        dates = pd.DatetimeIndex(["2026-02-27", "2026-02-28", "2026-03-01"])
        temps = np.array([20.0, 20.0, 20.0])
        gdd = rff.growing_degree_days(temps, dates, base_temp_c=10.0)
        self.assertAlmostEqual(gdd[0], 10.0)
        self.assertAlmostEqual(gdd[1], 20.0)  # accumulated across the two prior-season days
        self.assertAlmostEqual(gdd[2], 10.0)  # reset: new season, one day only

    def test_out_of_order_dates_still_accumulate_correctly(self):
        dates = pd.DatetimeIndex(["2026-03-02", "2026-03-01"])
        temps = np.array([20.0, 20.0])
        gdd = rff.growing_degree_days(temps, dates, base_temp_c=10.0)
        self.assertAlmostEqual(gdd[1], 10.0)
        self.assertAlmostEqual(gdd[0], 20.0)


class ReduceCellsToCountyTests(unittest.TestCase):
    def test_reduces_mapped_cells_with_the_given_reducer(self):
        grid = np.zeros((3, 3))
        grid[0, 0] = 10.0
        grid[0, 1] = 20.0
        grid[1, 1] = 5.0
        cell_to_fips = {"0,0": "29019", "0,1": "29019", "1,1": "29027"}
        result = rff.reduce_cells_to_county(grid, cell_to_fips, reducer=np.nanmean)
        self.assertAlmostEqual(result["29019"], 15.0)
        self.assertAlmostEqual(result["29027"], 5.0)

    def test_supports_max_reducer(self):
        grid = np.array([[1.0, 9.0], [3.0, 4.0]])
        cell_to_fips = {"0,0": "29019", "0,1": "29019"}
        result = rff.reduce_cells_to_county(grid, cell_to_fips, reducer=np.nanmax)
        self.assertEqual(result["29019"], 9.0)


class VaporPressureDeficitTests(unittest.TestCase):
    def test_zero_at_saturation(self):
        vpd = rff.vapor_pressure_deficit_kpa(np.array([25.0]), np.array([100.0]))
        self.assertAlmostEqual(vpd[0], 0.0, places=4)

    def test_increases_as_humidity_drops(self):
        vpd_dry = rff.vapor_pressure_deficit_kpa(np.array([30.0]), np.array([10.0]))
        vpd_humid = rff.vapor_pressure_deficit_kpa(np.array([30.0]), np.array([90.0]))
        self.assertGreater(vpd_dry[0], vpd_humid[0])

    def test_never_negative(self):
        vpd = rff.vapor_pressure_deficit_kpa(np.array([-10.0, 40.0]), np.array([100.0, 0.0]))
        self.assertTrue(np.all(vpd >= 0.0))


if __name__ == "__main__":
    unittest.main()
