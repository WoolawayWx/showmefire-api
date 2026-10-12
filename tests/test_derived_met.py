import sys
import unittest
from pathlib import Path

import numpy as np

API_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(API_DIR))

from services import derived_met


class HeatIndexTests(unittest.TestCase):
    def test_nws_chart_spot_check_90f_70rh(self):
        # NWS heat index chart: 90F/70%RH -> 105F "feels like".
        self.assertAlmostEqual(derived_met.heat_index_f(90.0, 70.0), 105.0, delta=2.0)

    def test_near_no_adjustment_boundary_80f_40rh(self):
        # Below the (temp+rh)<160 simple-formula cutover, heat index should
        # stay close to actual air temperature.
        self.assertAlmostEqual(derived_met.heat_index_f(80.0, 40.0), 80.0, delta=2.0)

    def test_array_input_preserves_shape(self):
        result = derived_met.heat_index_f(np.array([90.0, 95.0]), np.array([70.0, 50.0]))
        self.assertEqual(result.shape, (2,))


class WindChillTests(unittest.TestCase):
    def test_nws_chart_spot_check_30f_15mph(self):
        # NWS wind chill chart: 30F/15mph -> 19F.
        self.assertAlmostEqual(derived_met.wind_chill_f(30.0, 15.0), 19.0, delta=1.0)


class FeelsLikeTests(unittest.TestCase):
    def test_hot_branch_uses_heat_index(self):
        temp_f, rh_pct, wind_mph = 90.0, 70.0, 5.0
        self.assertAlmostEqual(
            derived_met.feels_like_f(temp_f, rh_pct, wind_mph),
            derived_met.heat_index_f(temp_f, rh_pct),
        )

    def test_cold_windy_branch_uses_wind_chill(self):
        temp_f, rh_pct, wind_mph = 30.0, 50.0, 15.0
        self.assertAlmostEqual(
            derived_met.feels_like_f(temp_f, rh_pct, wind_mph),
            derived_met.wind_chill_f(temp_f, wind_mph),
        )

    def test_cold_but_calm_falls_through_to_plain_temp(self):
        # Below 50F but wind under 3mph - NWS doesn't apply wind chill here.
        temp_f, rh_pct, wind_mph = 30.0, 50.0, 1.0
        self.assertEqual(derived_met.feels_like_f(temp_f, rh_pct, wind_mph), temp_f)

    def test_mild_temperature_falls_through_to_plain_temp(self):
        temp_f, rh_pct, wind_mph = 65.0, 50.0, 10.0
        self.assertEqual(derived_met.feels_like_f(temp_f, rh_pct, wind_mph), temp_f)


class DewpointDepressionTests(unittest.TestCase):
    def test_simple_subtraction(self):
        self.assertEqual(derived_met.dewpoint_depression_f(80.0, 65.0), 15.0)


class WetBulbTests(unittest.TestCase):
    def test_stull_reference_example(self):
        # Stull (2011): 20C/50%RH -> ~13.7C wet-bulb.
        temp_f = derived_met.c_to_f(20.0)
        wet_bulb_c = derived_met.f_to_c(derived_met.wet_bulb_f(temp_f, 50.0))
        self.assertAlmostEqual(wet_bulb_c, 13.7, delta=0.5)

    def test_wet_bulb_never_exceeds_air_temp(self):
        temp_f = 95.0
        self.assertLessEqual(derived_met.wet_bulb_f(temp_f, 60.0), temp_f)


class WbgtTests(unittest.TestCase):
    def test_estimated_wbgt_is_between_wetbulb_and_air_temp_ish(self):
        # Not a precise physical bound, but a sanity check: on a hot humid
        # day, estimated WBGT should land well above typical cool readings.
        hot_humid = derived_met.estimate_wbgt_f(95.0, 70.0, 5.0)
        mild_dry = derived_met.estimate_wbgt_f(70.0, 20.0, 5.0)
        self.assertGreater(hot_humid, mild_dry)


class ThetaETests(unittest.TestCase):
    def test_theta_e_exceeds_dry_potential_temperature(self):
        # Equivalent potential temperature must be >= the dry potential
        # temperature at any positive moisture content.
        temp_c, dewpoint_c, pressure_hpa = 20.0, 15.0, 950.0
        theta_e = derived_met.theta_e_k(temp_c, dewpoint_c, pressure_hpa)
        dry_theta = (temp_c + 273.15) * (1000.0 / pressure_hpa) ** 0.2854
        self.assertGreater(theta_e, dry_theta)

    def test_standard_atmosphere_pressure_decreases_with_elevation(self):
        sea_level = derived_met.standard_atmosphere_pressure_hpa(0.0)
        self.assertAlmostEqual(sea_level, derived_met.SEA_LEVEL_PRESSURE_HPA, delta=0.5)
        higher = derived_met.standard_atmosphere_pressure_hpa(500.0)
        self.assertLess(higher, sea_level)


class SpecificHumidityTests(unittest.TestCase):
    def test_higher_dewpoint_means_more_moisture(self):
        pressure_hpa = 1000.0
        dry = derived_met.specific_humidity_g_per_kg(40.0, pressure_hpa)
        humid = derived_met.specific_humidity_g_per_kg(70.0, pressure_hpa)
        self.assertGreater(humid, dry)

    def test_reasonable_magnitude_for_warm_humid_dewpoint(self):
        # A 70F dewpoint at ~1000 hPa should be on the order of 10-20 g/kg,
        # not an order of magnitude off (kg/kg vs g/kg mixups are the classic
        # bug here).
        q = derived_met.specific_humidity_g_per_kg(70.0, 1000.0)
        self.assertGreater(q, 5.0)
        self.assertLess(q, 25.0)


class ConversionTests(unittest.TestCase):
    def test_c_to_f_and_back(self):
        self.assertAlmostEqual(derived_met.f_to_c(derived_met.c_to_f(25.0)), 25.0, delta=1e-9)

    def test_mph_ms_round_trip(self):
        self.assertAlmostEqual(derived_met.ms_to_mph(derived_met.mph_to_ms(10.0)), 10.0, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
