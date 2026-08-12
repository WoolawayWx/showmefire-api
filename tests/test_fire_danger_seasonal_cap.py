import unittest

from core.fire_danger import (
    CURED_SEASON_GDD_FLOOR_C,
    DROUGHT_OVERRIDE_KBDI_MM,
    GREEN_SEASON_GDD_CEILING_C,
    RULE_SPEC,
    FireDangerCategory,
    calculate_fire_danger,
    seasonal_dampening_adjustment,
)

THRESHOLDS = RULE_SPEC["thresholds"]


class SeasonalDampeningAdjustmentTests(unittest.TestCase):
    def test_low_and_moderate_are_never_touched(self):
        # Deepest green-up, but category is below Elevated: must pass through unchanged.
        self.assertEqual(seasonal_dampening_adjustment(int(FireDangerCategory.LOW), 20.0, 80.0, 2.0, 0.0),
                          int(FireDangerCategory.LOW))
        self.assertEqual(seasonal_dampening_adjustment(int(FireDangerCategory.MODERATE), 12.0, 50.0, 12.0, 0.0),
                          int(FireDangerCategory.MODERATE))

    def test_none_category_passes_through(self):
        self.assertIsNone(seasonal_dampening_adjustment(None, 5.0, 20.0, 20.0, 0.0))

    def test_no_gdd_signal_leaves_category_unchanged(self):
        category = calculate_fire_danger(5.0, 20.0, 20.0)
        self.assertEqual(seasonal_dampening_adjustment(category, 5.0, 20.0, 20.0, None), category)

    def test_fully_cured_season_leaves_category_unchanged(self):
        # A marginal Elevated cell (just crosses the very-dry OR-clause).
        fm, rh, wind = THRESHOLDS["elevated_fm"] - 0.1, THRESHOLDS["elevated_very_dry_rh"] - 0.1, THRESHOLDS["elevated_very_dry_wind"] + 0.1
        category = calculate_fire_danger(fm, rh, wind)
        self.assertEqual(category, int(FireDangerCategory.ELEVATED))
        self.assertEqual(seasonal_dampening_adjustment(category, fm, rh, wind, CURED_SEASON_GDD_FLOOR_C), category)

    def test_marginal_elevated_cell_demotes_during_deep_greenup(self):
        # Just barely crosses the loose "very dry" OR-clause - far from Critical's thresholds.
        fm = THRESHOLDS["elevated_fm"] - 0.1
        rh = THRESHOLDS["elevated_very_dry_rh"] - 0.1
        wind = THRESHOLDS["elevated_very_dry_wind"] + 0.1
        category = calculate_fire_danger(fm, rh, wind)
        self.assertEqual(category, int(FireDangerCategory.ELEVATED))
        adjusted = seasonal_dampening_adjustment(category, fm, rh, wind, GREEN_SEASON_GDD_CEILING_C)
        self.assertEqual(adjusted, int(FireDangerCategory.MODERATE))

    def test_deep_elevated_cell_survives_deep_greenup(self):
        # Right at Critical's own thresholds - deep within the Elevated band, not marginal.
        fm = THRESHOLDS["elevated_fm"] - 0.1
        rh = THRESHOLDS["critical_rh"] + 0.1
        wind = THRESHOLDS["critical_wind"] - 0.1
        category = calculate_fire_danger(fm, rh, wind)
        self.assertEqual(category, int(FireDangerCategory.ELEVATED))
        adjusted = seasonal_dampening_adjustment(category, fm, rh, wind, GREEN_SEASON_GDD_CEILING_C)
        self.assertEqual(adjusted, int(FireDangerCategory.ELEVATED))

    def test_marginal_critical_cell_demotes_during_deep_greenup(self):
        # Just barely crosses into Critical - far from Extreme's thresholds.
        fm = THRESHOLDS["elevated_fm"] - 0.1
        rh = THRESHOLDS["critical_rh"] - 0.1
        wind = THRESHOLDS["critical_wind"] + 0.1
        category = calculate_fire_danger(fm, rh, wind)
        self.assertEqual(category, int(FireDangerCategory.CRITICAL))
        adjusted = seasonal_dampening_adjustment(category, fm, rh, wind, GREEN_SEASON_GDD_CEILING_C)
        self.assertEqual(adjusted, int(FireDangerCategory.ELEVATED))

    def test_deep_critical_cell_survives_deep_greenup(self):
        # Right at Extreme's own thresholds - deep within Critical, genuinely severe weather.
        fm = THRESHOLDS["extreme_fm"] + 0.1
        rh = THRESHOLDS["extreme_rh"] + 0.1
        wind = THRESHOLDS["extreme_wind"] - 0.1
        category = calculate_fire_danger(fm, rh, wind)
        self.assertEqual(category, int(FireDangerCategory.CRITICAL))
        adjusted = seasonal_dampening_adjustment(category, fm, rh, wind, GREEN_SEASON_GDD_CEILING_C)
        self.assertEqual(adjusted, int(FireDangerCategory.CRITICAL))

    def test_marginal_extreme_cell_demotes_during_deep_greenup(self):
        fm = THRESHOLDS["extreme_fm"] - 0.1
        rh = THRESHOLDS["extreme_rh"] - 0.1
        wind = THRESHOLDS["extreme_wind"] + 0.1
        category = calculate_fire_danger(fm, rh, wind)
        self.assertEqual(category, int(FireDangerCategory.EXTREME))
        adjusted = seasonal_dampening_adjustment(category, fm, rh, wind, GREEN_SEASON_GDD_CEILING_C)
        self.assertEqual(adjusted, int(FireDangerCategory.CRITICAL))

    def test_deeply_extreme_cell_survives_deep_greenup(self):
        fm, rh, wind = 0.0, 0.0, 60.0
        category = calculate_fire_danger(fm, rh, wind)
        self.assertEqual(category, int(FireDangerCategory.EXTREME))
        adjusted = seasonal_dampening_adjustment(category, fm, rh, wind, GREEN_SEASON_GDD_CEILING_C)
        self.assertEqual(adjusted, int(FireDangerCategory.EXTREME))

    def test_drought_override_leaves_category_unchanged_even_in_green_season(self):
        fm = THRESHOLDS["elevated_fm"] - 0.1
        rh = THRESHOLDS["critical_rh"] - 0.1
        wind = THRESHOLDS["critical_wind"] + 0.1
        category = calculate_fire_danger(fm, rh, wind)
        self.assertEqual(category, int(FireDangerCategory.CRITICAL))
        adjusted = seasonal_dampening_adjustment(category, fm, rh, wind, GREEN_SEASON_GDD_CEILING_C,
                                                 kbdi_mm=DROUGHT_OVERRIDE_KBDI_MM)
        self.assertEqual(adjusted, category)

    def test_never_demotes_more_than_one_tier(self):
        fm = THRESHOLDS["extreme_fm"] - 0.1
        rh = THRESHOLDS["extreme_rh"] - 0.1
        wind = THRESHOLDS["extreme_wind"] + 0.1
        category = calculate_fire_danger(fm, rh, wind)
        self.assertEqual(category, int(FireDangerCategory.EXTREME))
        adjusted = seasonal_dampening_adjustment(category, fm, rh, wind, GREEN_SEASON_GDD_CEILING_C)
        self.assertGreaterEqual(adjusted, int(FireDangerCategory.CRITICAL))

    def test_transition_band_partially_reduces_demotion_reach(self):
        # Midway through the transition band, green_factor is ~0.5, so the
        # demotion threshold (green * MAX_DEMOTION_FRACTION) is smaller than
        # at deep green-up - a cell that demotes at full green-up may not
        # demote midway through the transition.
        midpoint_gdd = (GREEN_SEASON_GDD_CEILING_C + CURED_SEASON_GDD_FLOOR_C) / 2.0
        fm = THRESHOLDS["elevated_fm"] - 0.1
        rh = THRESHOLDS["elevated_very_dry_rh"] - 0.1
        wind = THRESHOLDS["elevated_very_dry_wind"] + 0.1
        category = calculate_fire_danger(fm, rh, wind)
        self.assertEqual(category, int(FireDangerCategory.ELEVATED))
        at_full_green = seasonal_dampening_adjustment(category, fm, rh, wind, GREEN_SEASON_GDD_CEILING_C)
        at_midpoint = seasonal_dampening_adjustment(category, fm, rh, wind, midpoint_gdd)
        self.assertEqual(at_full_green, int(FireDangerCategory.MODERATE))
        self.assertEqual(at_midpoint, int(FireDangerCategory.ELEVATED))


if __name__ == "__main__":
    unittest.main()
