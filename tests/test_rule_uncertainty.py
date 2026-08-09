import unittest

import numpy as np

from core.fire_danger import RULE_SPEC, calculate_fire_danger
from services import rule_uncertainty as ru


class CategoryVectorizedParityTests(unittest.TestCase):
    def setUp(self):
        self.thresholds = RULE_SPEC["thresholds"]

    def test_matches_scalar_reference_over_random_and_boundary_points(self):
        # This is the exact guard the module docstring calls mandatory:
        # a second implementation of the rule must never silently drift
        # from core/fire_danger.py::calculate_fire_danger.
        ru.check_parity(calculate_fire_danger, self.thresholds, n_samples=1000, seed=42)

    def test_low_fm_overrides_every_other_condition(self):
        # fm >= low_fm must be LOW even with extreme rh/wind.
        category = ru.category_vectorized(np.array([20.0]), np.array([5.0]), np.array([50.0]), self.thresholds)
        self.assertEqual(category[0], ru.CATEGORY_LOW)

    def test_missing_input_returns_missing_sentinel(self):
        category = ru.category_vectorized(np.array([float("nan")]), np.array([50.0]), np.array([10.0]), self.thresholds)
        self.assertEqual(category[0], ru.MISSING_CATEGORY)

    def test_broadcasts_across_grid_shape(self):
        fm = np.full((3, 4), 5.0)
        rh = np.full((3, 4), 15.0)
        wind = np.full((3, 4), 30.0)
        category = ru.category_vectorized(fm, rh, wind, self.thresholds)
        self.assertEqual(category.shape, (3, 4))
        self.assertTrue(np.all(category == ru.CATEGORY_EXTREME))

    def test_parity_check_raises_on_injected_mismatch(self):
        def broken_reference(fm, rh, wind, missing_category=None):
            return ru.CATEGORY_LOW  # always wrong except where the real rule also says low
        with self.assertRaises(AssertionError):
            ru.check_parity(broken_reference, self.thresholds, n_samples=50, seed=1)


class SampleCategoryProbabilitiesTests(unittest.TestCase):
    def setUp(self):
        self.thresholds = RULE_SPEC["thresholds"]

    def test_probabilities_sum_to_one_per_point(self):
        result = ru.sample_category_probabilities(
            fm=np.array([5.0, 20.0]), rh=np.array([15.0, 50.0]), wind_kts=np.array([30.0, 5.0]),
            fm_sigma=np.array([2.0, 2.0]), rh_sigma=np.array([5.0, 5.0]), wind_sigma_log=np.array([0.3, 0.3]),
            thresholds=self.thresholds, n_draws=500, seed=1,
        )
        totals = sum(result[f"category_probability_{label}"] for label in ru.CATEGORY_LABELS)
        np.testing.assert_allclose(totals, 1.0, atol=1e-9)

    def test_zero_uncertainty_collapses_to_deterministic_with_full_stability(self):
        result = ru.sample_category_probabilities(
            fm=np.array([5.0]), rh=np.array([15.0]), wind_kts=np.array([30.0]),
            fm_sigma=np.array([1e-9]), rh_sigma=np.array([1e-9]), wind_sigma_log=np.array([1e-9]),
            thresholds=self.thresholds, n_draws=200, seed=2,
        )
        self.assertAlmostEqual(float(result["stability"][0]), 1.0, places=3)
        self.assertFalse(bool(result["modal_disagrees"][0]))

    def test_deterministic_category_matches_category_vectorized(self):
        fm, rh, wind = np.array([5.0, 20.0]), np.array([15.0, 50.0]), np.array([30.0, 5.0])
        result = ru.sample_category_probabilities(
            fm=fm, rh=rh, wind_kts=wind,
            fm_sigma=np.array([2.0, 2.0]), rh_sigma=np.array([5.0, 5.0]), wind_sigma_log=np.array([0.3, 0.3]),
            thresholds=self.thresholds, n_draws=100, seed=3,
        )
        expected = ru.category_vectorized(fm, rh, wind, self.thresholds)
        np.testing.assert_array_equal(result["deterministic_category"], expected)

    def test_is_deterministic_given_the_same_seed(self):
        kwargs = dict(
            fm=np.array([5.0]), rh=np.array([15.0]), wind_kts=np.array([30.0]),
            fm_sigma=np.array([2.0]), rh_sigma=np.array([5.0]), wind_sigma_log=np.array([0.3]),
            thresholds=self.thresholds, n_draws=300, seed=811,
        )
        first = ru.sample_category_probabilities(**kwargs)
        second = ru.sample_category_probabilities(**kwargs)
        np.testing.assert_array_equal(first["stability"], second["stability"])

    def test_county_index_produces_independent_but_reproducible_streams(self):
        kwargs = dict(
            fm=np.array([5.0]), rh=np.array([15.0]), wind_kts=np.array([30.0]),
            fm_sigma=np.array([2.0]), rh_sigma=np.array([5.0]), wind_sigma_log=np.array([0.3]),
            thresholds=self.thresholds, n_draws=300,
        )
        county_1 = ru.sample_category_probabilities(**kwargs, county_index=1)
        county_2 = ru.sample_category_probabilities(**kwargs, county_index=2)
        county_1_again = ru.sample_category_probabilities(**kwargs, county_index=1)
        np.testing.assert_array_equal(county_1["stability"], county_1_again["stability"])
        # Different counties draw independent streams - not asserting
        # inequality (could coincidentally match), just that both run cleanly
        # and are internally consistent.
        self.assertEqual(county_2["stability"].shape, county_1["stability"].shape)

    def test_probability_at_or_above_elevated_is_monotone_superset_of_critical(self):
        result = ru.sample_category_probabilities(
            fm=np.array([5.0]), rh=np.array([15.0]), wind_kts=np.array([16.0]),
            fm_sigma=np.array([3.0]), rh_sigma=np.array([8.0]), wind_sigma_log=np.array([0.4]),
            thresholds=self.thresholds, n_draws=1000, seed=7,
        )
        self.assertGreaterEqual(
            float(result["probability_at_or_above_elevated"][0]),
            float(result["probability_at_or_above_critical"][0]),
        )
        self.assertGreaterEqual(
            float(result["probability_at_or_above_critical"][0]),
            float(result["probability_at_or_above_extreme"][0]),
        )


if __name__ == "__main__":
    unittest.main()
