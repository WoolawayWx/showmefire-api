"""Physical Fire Danger Index: each link of the chain against hand-checkable values, then the index's behaviour."""
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.fire_weather import physical_danger as pd
from services.fire_weather_hybrid import behavior
from services.hrrr_hourly import BY_NAME, derive

MPH = 1.0 / pd.MS_TO_MPH
NOON = datetime(2026, 3, 15, 18, tzinfo=timezone.utc)        # about solar noon in Missouri


def danger(temp_c=24.0, rh=20.0, wind_mph=20.0, solar=800.0, **kwargs):
    return pd.physical_fire_danger(temp_c, rh, np.asarray(wind_mph) * MPH, solar, **kwargs)


class SunTests(unittest.TestCase):
    def test_haurwitz_overhead_and_below_horizon(self):
        self.assertAlmostEqual(float(pd.clear_sky_irradiance_wm2(90.0)), 1098.0 * np.exp(-0.057), places=6)
        self.assertEqual(float(pd.clear_sky_irradiance_wm2(0.0)), 0.0)
        self.assertEqual(float(pd.clear_sky_irradiance_wm2(-20.0)), 0.0)
        self.assertTrue(np.isnan(pd.clear_sky_irradiance_wm2(np.nan)))

    def test_cloud_transmission_ends(self):
        self.assertEqual(float(pd.cloud_transmission(0.0)), 1.0)
        self.assertAlmostEqual(float(pd.cloud_transmission(100.0)), 0.25)
        self.assertAlmostEqual(float(pd.cloud_transmission(50.0)), 1.0 - 0.75 * 0.5 ** 3.4)

    def test_missouri_noon_and_midnight(self):
        noon = float(pd.solar_irradiance_wm2(38.5, -92.5, NOON, 0.0))
        self.assertTrue(700 < noon < 850, noon)
        self.assertEqual(float(pd.solar_irradiance_wm2(38.5, -92.5, NOON.replace(hour=6), 0.0)), 0.0)


class FuelMoistureTests(unittest.TestCase):
    def test_simard_branches(self):
        # 68 F: 0.03229 + 0.281073*5 - 0.000578*5*68;  2.22749 + 0.160107*30 - 0.01478*68;  the >= 50 branch at 80 %
        self.assertAlmostEqual(float(pd.equilibrium_moisture_pct(20.0, 5.0)), 1.241135, places=5)
        self.assertAlmostEqual(float(pd.equilibrium_moisture_pct(20.0, 30.0)), 6.02566, places=5)
        self.assertAlmostEqual(float(pd.equilibrium_moisture_pct(20.0, 80.0)), 16.11668, places=5)

    def test_emc_rises_with_humidity_and_falls_with_temperature(self):
        rh = np.arange(1.0, 100.0)
        self.assertTrue(np.all(np.diff(pd.equilibrium_moisture_pct(20.0, rh)) > -0.35))   # small steps at branch joins only
        self.assertTrue(np.all(pd.equilibrium_moisture_pct(35.0, rh) <= pd.equilibrium_moisture_pct(5.0, rh)))
        # The sign-reversed fit elsewhere in the repo pins this range at 1 %.
        self.assertGreater(float(pd.equilibrium_moisture_pct(20.0, 40.0)), 7.0)

    def test_fuel_level_air_without_sun_is_the_shelter_air(self):
        temp, rh = pd.fuel_level_conditions(20.0, 40.0, 0.0, 3.0)
        self.assertEqual((float(temp), float(rh)), (20.0, 40.0))

    def test_fuel_level_warming_by_hand(self):
        # 1 langley/min, calm: 1 / 0.026 = 38.46 F warmer; RH x exp(-0.033 * 38.46)
        temp, rh = pd.fuel_level_conditions(20.0, 40.0, pd.WM2_PER_LANGLEY_MIN, 0.0)
        self.assertAlmostEqual(float(temp), 20.0 + 38.4615 / 1.8, places=3)
        self.assertAlmostEqual(float(rh), 40.0 * np.exp(-0.033 * 38.4615), places=3)

    def test_wind_ventilates_the_warming_away(self):
        calm = pd.dead_fuel_moisture(20.0, 40.0, 800.0, 0.0)[0]
        windy = pd.dead_fuel_moisture(20.0, 40.0, 800.0, 5.0)[0]
        shade = pd.dead_fuel_moisture(20.0, 40.0, 0.0, 5.0)[0]
        self.assertTrue(float(calm) < float(windy) < float(shade))

    def test_fosberg_ratios_and_rain(self):
        fm1, fm10 = pd.dead_fuel_moisture(20.0, 30.0, 0.0, 2.0)
        self.assertAlmostEqual(float(fm1), 1.03 * 6.02566, places=4)
        self.assertAlmostEqual(float(fm10), 1.28 * 6.02566, places=4)
        rain = np.array([0.0, 0.2, 0.25, 5.0, np.nan])
        fm1, fm10 = pd.dead_fuel_moisture(20.0, 30.0, 0.0, 2.0, rain)
        np.testing.assert_allclose(fm1[:2], 1.03 * 6.02566, rtol=1e-5)
        np.testing.assert_array_equal(fm1[2:4], [35.0, 35.0])
        np.testing.assert_array_equal(fm10[2:4], [35.0, 35.0])
        self.assertTrue(np.isnan(fm1[4]) and np.isnan(fm10[4]))     # unknown rain is not "no rain"


class WindTests(unittest.TestCase):
    def test_log_profile_and_similarity_constants(self):
        self.assertAlmostEqual(pd.WIND_10M_TO_20FT, np.log(6.096 / 0.03) / np.log(10 / 0.03))
        self.assertAlmostEqual(pd.NEUTRAL_TURBULENCE_INTENSITY, 1.0 / np.log(10 / 0.03))
        self.assertAlmostEqual(pd.unsheltered_midflame_factor(102), 1.83 / np.log(20.36 / 0.13))   # GR2 is 1 ft deep

    def test_design_wind_from_gust(self):
        # U = 10, gust = 16: sigma = 2, design = 10 + 1.2816 * 2
        self.assertAlmostEqual(float(pd.design_wind_20ft_ms(10.0, 16.0)), 12.5632 * pd.WIND_10M_TO_20FT, places=6)

    def test_gust_below_the_mean_wind_adds_nothing(self):
        self.assertAlmostEqual(float(pd.design_wind_20ft_ms(10.0, 8.0)), 10.0 * pd.WIND_10M_TO_20FT)

    def test_missing_gust_uses_neutral_turbulence(self):
        expected = 10.0 * (1 + 1.2816 * pd.NEUTRAL_TURBULENCE_INTENSITY) * pd.WIND_10M_TO_20FT
        self.assertAlmostEqual(float(pd.design_wind_20ft_ms(10.0)), expected)
        self.assertAlmostEqual(float(pd.design_wind_20ft_ms(10.0, np.nan)), expected)


class LiveFuelTests(unittest.TestCase):
    def test_mid_month_values_are_the_table(self):
        self.assertEqual(pd.live_moisture_pct(datetime(2026, 1, 15, 12)), (30.0, 60.0))
        herb, _woody = pd.live_moisture_pct(datetime(2026, 6, 15))
        self.assertAlmostEqual(herb, 120.0, delta=0.5)

    def test_no_jump_across_a_month_or_the_new_year(self):
        for before, after in ((datetime(2026, 4, 30), datetime(2026, 5, 1)), (datetime(2026, 12, 31), datetime(2027, 1, 1))):
            self.assertLess(abs(pd.live_moisture_pct(before)[0] - pd.live_moisture_pct(after)[0]), 3.0)


class IndexTests(unittest.TestCase):
    def test_index_anchors(self):
        feet = np.array([0.0, 1.0, 4.0, 8.0, 11.0, 15.0, 40.0, np.nan]) / pd.M_TO_FT
        index = pd.index_from_flame_length(feet)
        np.testing.assert_allclose(index[:7], [0, 20, 40, 60, 80, 100, 100], atol=1e-6)
        self.assertTrue(np.isnan(index[7]))

    def test_category_is_the_index_band(self):
        rh, wind = np.meshgrid(np.linspace(5, 95, 19), np.linspace(0, 40, 21))
        out = danger(rh=rh, wind_mph=wind)
        expected = np.minimum(np.floor(out["index"] / 20.0 + 1e-9), 4)
        np.testing.assert_array_equal(out["category"], expected)
        self.assertEqual(set(np.unique(out["category"])), {0.0, 1.0, 2.0, 3.0, 4.0})   # the whole scale is reachable

    def test_head_fire_is_the_rothermel_port_below_its_wind_limit(self):
        wind = np.linspace(0.0, 12.0, 13)                       # 20 ft m/s; the port's limit is near 13 here
        for herb in (30.0, 75.0):
            ros, intensity, flame = pd.head_fire(102, 5.0, 6.0, 7.0, herb, 60.0, wind)
            port = behavior.surface_fire(np.full(13, 102.0), 5.0, 6.0, 7.0, herb, 60.0, wind, 0.0, 0.0, 0.0, 0.0, 0.0)
            np.testing.assert_allclose(ros, port["ros_m_min"], rtol=1e-9)
            np.testing.assert_allclose(intensity, port["intensity_kw_m"], rtol=1e-9)
            np.testing.assert_allclose(flame, port["flame_length_m"], rtol=1e-9)

    def test_head_fire_keeps_rising_past_the_wind_limit(self):
        wind = np.array([16.0, 20.0, 24.0])
        ros, _intensity, flame = pd.head_fire(102, 5.0, 6.0, 7.0, 30.0, 60.0, wind)
        port = behavior.surface_fire(np.full(3, 102.0), 5.0, 6.0, 7.0, 30.0, 60.0, wind, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.assertEqual(len(set(port["ros_m_min"].round(6))), 1)        # the port is capped: one value
        self.assertTrue(np.all(np.diff(ros) > 0) and np.all(ros > port["ros_m_min"]))
        self.assertTrue(np.all(np.diff(flame) > 0))

    def test_head_fire_too_wet_is_zero_and_missing_is_nan(self):
        ros, intensity, flame = pd.head_fire(102, np.array([20.0, np.nan, 5.0]), 20.0, 20.0, 30.0, 60.0,
                                             np.array([10.0, 10.0, np.nan]))
        self.assertEqual((ros[0], intensity[0], flame[0]), (0.0, 0.0, 0.0))
        self.assertTrue(np.isnan(flame[1]) and np.isnan(flame[2]))

    def test_byram_flame_length(self):
        out = danger(fm1_pct=5.0, fm10_pct=6.0, gust_10m_ms=30 * MPH)
        self.assertAlmostEqual(float(out["flame_length_ft"]) / pd.M_TO_FT,
                               0.0775 * float(out["intensity_kw_m"]) ** 0.46, places=2)

    def test_never_lower_when_hotter_drier_windier_or_sunnier(self):
        def non_decreasing(values):
            return bool(np.all(np.diff(values) >= -1e-9))
        self.assertTrue(non_decreasing(danger(temp_c=np.linspace(-5, 40, 46))["index"]))
        self.assertTrue(non_decreasing(danger(rh=np.linspace(100, 3, 98))["index"]))
        self.assertTrue(non_decreasing(danger(solar=np.linspace(0, 1000, 41))["index"]))
        # Wind: always with fuel moisture held fixed; with the estimate, on a dry day (wind also cools the fuel).
        self.assertTrue(non_decreasing(danger(wind_mph=np.linspace(0, 60, 61), fm1_pct=12.0, fm10_pct=13.0)["index"]))
        self.assertTrue(non_decreasing(danger(wind_mph=np.linspace(0, 60, 61))["index"]))
        self.assertTrue(non_decreasing(danger(gust_10m_ms=np.linspace(20, 60, 41) * MPH)["index"]))
        self.assertTrue(non_decreasing(danger(live_herb_pct=np.linspace(120, 30, 19))["index"]))   # greener never burns hotter

    def test_wet_or_green_fuel_does_not_burn(self):
        for out in (danger(rh=95.0, solar=0.0), danger(precip_mm_1h=2.0), danger(live_herb_pct=120.0, wind_mph=5.0)):
            self.assertLess(float(out["index"]), 20.0)
            self.assertEqual(float(out["category"]), 0.0)
        self.assertEqual(float(danger(precip_mm_1h=2.0)["index"]), 0.0)       # above extinction: a real zero

    def test_red_flag_day_is_critical_and_a_humid_calm_day_is_not(self):
        self.assertGreaterEqual(float(danger(temp_c=24, rh=20, wind_mph=20, gust_10m_ms=35 * MPH)["category"]), 3.0)
        self.assertLessEqual(float(danger(temp_c=15, rh=65, wind_mph=5, solar=300.0)["category"]), 1.0)

    def test_missing_inputs_give_nan(self):
        for kwargs in ({"temp_c": np.nan}, {"rh": np.nan}, {"wind_mph": np.nan}, {"solar": np.nan},
                       {"precip_mm_1h": np.nan}):
            out = danger(**kwargs)
            self.assertTrue(np.isnan(out["index"]) and np.isnan(out["category"]), kwargs)

    def test_supplied_fuel_moisture_replaces_the_estimate(self):
        dry, wet = danger(fm1_pct=3.0, fm10_pct=4.0), danger(fm1_pct=13.0, fm10_pct=14.0)
        self.assertEqual(float(dry["fm1_pct"]), 3.0)
        self.assertGreater(float(dry["index"]), float(wet["index"]))

    def test_grid_shape_is_kept(self):
        out = pd.physical_fire_danger_at(np.full((3, 4), 22.0), np.full((3, 4), 30.0), np.full((3, 4), 7.0),
                                         np.zeros((3, 4)), np.full((3, 4), 38.5), np.full((3, 4), -92.5), NOON)
        self.assertTrue(all(values.shape == (3, 4) for values in out.values()))
        self.assertTrue(np.all(out["solar_wm2"] > 700))


class HourlyProductTests(unittest.TestCase):
    def profiles(self, **extra):
        shape = (2, 3)
        lon = np.full(shape, -92.5)
        return SimpleNamespace(t2_c=np.full(shape, 24.0), td2_c=np.full(shape, 0.0), u10=np.full(shape, 6.0),
                               v10=np.full(shape, 6.0), lon=lon, central_longitude=-92.5, standard_parallel=38.5, **extra)

    def derive(self, profiles, extras):
        names = ("mixing_height_ft", "holzworth_ft", "transport_speed_mph", "transport_dir_deg", "ventilation_kt_ft",
                 "ventilation_category", "inversion_strength_f", "inversion_top_ft", "lapse_rate_c_km", "haines", "hdw")
        original = derive.hp.derive_fields
        derive.hp.derive_fields = lambda p: {name: np.zeros(p.t2_c.shape) for name in names}
        try:
            return derive.derive_hour(profiles, extras)
        finally:
            derive.hp.derive_fields = original

    def test_elements_are_registered_and_derived(self):
        profiles = self.profiles(valid_time=NOON, lat=np.full((2, 3), 38.5))
        out = self.derive(profiles, {"cloud": np.zeros((2, 3)), "gust": np.full((2, 3), 14.0)})
        for name in ("pfdi", "pfdi_category", "pfdi_flame_length_ft", "pfdi_fm1_pct"):
            self.assertIn(name, BY_NAME)
            self.assertEqual(out[name].shape, (2, 3))
        self.assertGreaterEqual(float(out["pfdi_category"][0, 0]), 2.0)        # 24 C, about 20 % RH, 19 mph, full sun

    def test_omitted_without_cloud_cover_or_valid_time(self):
        self.assertNotIn("pfdi", self.derive(self.profiles(valid_time=NOON, lat=np.full((2, 3), 38.5)), {}))
        self.assertNotIn("pfdi", self.derive(self.profiles(), {"cloud": np.zeros((2, 3))}))


if __name__ == "__main__":
    unittest.main()
