"""Fire-weather index math on synthetic columns with hand-checkable answers."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.fire_weather import indices as ix


def column(*values):
    """A (level, 1, 1) profile from bottom-up values."""
    return np.array(values, dtype=float).reshape(-1, 1, 1)


class HolzworthTests(unittest.TestCase):
    def test_well_mixed_column_mixes_to_the_top(self):
        heights = column(500, 1000, 1500)
        theta = column(300, 300, 300)
        result = ix.mixing_height_holzworth(heights, theta, np.full((1, 1), 300.0))
        self.assertEqual(result[0, 0], 1500)

    def test_crossing_is_interpolated(self):
        heights = column(500, 1500)
        theta = column(299, 303)  # parcel (300 K) meets the profile 1/4 of the way up
        result = ix.mixing_height_holzworth(heights, theta, np.full((1, 1), 300.0))
        self.assertAlmostEqual(result[0, 0], 750.0)

    def test_surface_inversion_gives_zero(self):
        result = ix.mixing_height_holzworth(column(300), column(305), np.full((1, 1), 300.0))
        self.assertEqual(result[0, 0], 0.0)


class TransportWindTests(unittest.TestCase):
    def test_constant_wind_averages_to_itself(self):
        heights = column(10, 80, 500, 1500)
        u = column(5, 5, 5, 5)
        v = column(0, 0, 0, 0)
        mean_u, mean_v = ix.transport_wind(heights, u, v, np.full((1, 1), 1000.0))
        self.assertAlmostEqual(mean_u[0, 0], 5.0)
        self.assertAlmostEqual(mean_v[0, 0], 0.0)

    def test_layer_above_mixing_height_is_ignored(self):
        heights = column(10, 1000, 2000)
        u = column(2, 2, 50)
        mean_u, _ = ix.transport_wind(heights, u, np.zeros_like(u), np.full((1, 1), 1000.0))
        self.assertAlmostEqual(mean_u[0, 0], 2.0)

    def test_shallow_mixed_layer_uses_lowest_wind(self):
        heights = column(10, 80)
        mean_u, _ = ix.transport_wind(heights, column(3, 9), column(0, 0), np.full((1, 1), 5.0))
        self.assertAlmostEqual(mean_u[0, 0], 3.0)

    def test_below_ground_levels_are_skipped(self):
        heights = column(np.nan, 10, 1000)
        u = column(99, 4, 4)
        mean_u, _ = ix.transport_wind(heights, u, np.zeros_like(u), np.full((1, 1), 1000.0))
        self.assertAlmostEqual(mean_u[0, 0], 4.0)

    def test_direction_is_where_wind_comes_from(self):
        self.assertAlmostEqual(float(ix.wind_direction_from(0.0, -5.0)), 0.0)    # from the north
        self.assertAlmostEqual(float(ix.wind_direction_from(-5.0, 0.0)), 90.0)   # from the east
        self.assertAlmostEqual(float(ix.wind_direction_from(5.0, 0.0)), 270.0)   # from the west


class VentilationTests(unittest.TestCase):
    def test_rate_and_category(self):
        rate = ix.ventilation_rate_kt_ft(10.0, 3000.0)
        self.assertEqual(rate, 30000.0)
        self.assertEqual(int(ix.ventilation_category(rate)), 0)  # Poor (<40k)

    def test_springfield_category_edges(self):
        rates = np.array([39999, 40000, 59999, 60000, 99999, 100000, 149999, 150000, np.nan])
        self.assertEqual(ix.ventilation_category(rates).tolist(), [0, 1, 1, 2, 2, 3, 3, 4, -1])


class InversionAndLapseTests(unittest.TestCase):
    def test_surface_inversion_strength_and_top(self):
        heights = column(2, 100, 300, 600)
        temp = column(5, 8, 9, 7)
        strength, top = ix.surface_inversion(heights, temp)
        self.assertEqual(strength[0, 0], 4.0)
        self.assertEqual(top[0, 0], 300.0)

    def test_no_inversion_when_cooling_with_height(self):
        strength, top = ix.surface_inversion(column(2, 500), column(20, 15))
        self.assertEqual((strength[0, 0], top[0, 0]), (0.0, 0.0))

    def test_missing_level_does_not_end_the_inversion(self):
        strength, top = ix.surface_inversion(column(2, np.nan, 300), column(5, np.nan, 7))
        self.assertEqual((strength[0, 0], top[0, 0]), (2.0, 300.0))

    def test_dry_adiabatic_lapse_rate(self):
        heights = column(2, 1002)
        temp = column(20.0, 10.2)
        rate = ix.lapse_rate_c_per_km(heights, temp, depth_m=1002.0)
        self.assertAlmostEqual(rate[0, 0], 9.8, places=6)


class HainesTests(unittest.TestCase):
    """Edges from the low-elevation table in NWS WR Technical Attachment 00-08."""

    def test_stability_term_edges(self):
        t850, td850 = 10.0, 9.0  # moisture term 1
        self.assertEqual(ix.haines_low(13.0, t850, td850), 2)  # 3 C -> 1
        self.assertEqual(ix.haines_low(14.0, t850, td850), 3)  # 4 C -> 2
        self.assertEqual(ix.haines_low(17.0, t850, td850), 3)  # 7 C -> 2
        self.assertEqual(ix.haines_low(18.0, t850, td850), 4)  # 8 C -> 3

    def test_moisture_term_edges(self):
        t950, t850 = 11.0, 10.0  # stability term 1
        self.assertEqual(ix.haines_low(t950, t850, 5.0), 2)   # 5 C -> 1
        self.assertEqual(ix.haines_low(t950, t850, 4.0), 3)   # 6 C -> 2
        self.assertEqual(ix.haines_low(t950, t850, 1.0), 3)   # 9 C -> 2
        self.assertEqual(ix.haines_low(t950, t850, 0.0), 4)   # 10 C -> 3

    def test_range_is_two_to_six(self):
        self.assertEqual(ix.haines_low(30.0, 10.0, -30.0), 6)


class HotDryWindyTests(unittest.TestCase):
    def test_product_of_layer_maxima(self):
        vpd = column(ix.vapor_pressure_deficit_hpa(30.0, 10.0), ix.vapor_pressure_deficit_hpa(25.0, 15.0))
        wind = column(5.0, 12.0, 8.0)
        expected = float(ix.vapor_pressure_deficit_hpa(30.0, 10.0)) * 12.0
        self.assertAlmostEqual(ix.hot_dry_windy(vpd, wind)[0, 0], expected)

    def test_bolton_saturation_vapor_pressure(self):
        self.assertAlmostEqual(float(ix.saturation_vapor_pressure_hpa(0.0)), 6.112)
        self.assertAlmostEqual(float(ix.saturation_vapor_pressure_hpa(20.0)), 23.37, places=1)


class WindRotationTests(unittest.TestCase):
    def test_no_rotation_on_central_meridian(self):
        u, v = ix.earth_relative_winds(np.array(1.0), np.array(2.0), -97.5, -97.5, 38.5)
        self.assertAlmostEqual(float(u), 1.0)
        self.assertAlmostEqual(float(v), 2.0)

    def test_grid_north_points_east_of_true_north_east_of_meridian(self):
        u, v = ix.earth_relative_winds(np.array(0.0), np.array(1.0), -90.0, -97.5, 38.5)
        self.assertGreater(float(u), 0.0)
        self.assertAlmostEqual(float(np.hypot(u, v)), 1.0)


if __name__ == "__main__":
    unittest.main()
