import sys
import unittest
from pathlib import Path

import numpy as np

API_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(API_DIR))

from models.fm_uncertainty import fit_uncertainty, intervals


class FmUncertaintyTests(unittest.TestCase):
    def test_quantile_widths_are_non_negative(self):
        rng = np.random.default_rng(0)
        actual = rng.uniform(3, 30, size=1000)
        prediction = actual + rng.normal(0, 2, size=1000)
        regime = rng.integers(1, 13, size=1000)
        uncertainty = fit_uncertainty(actual, prediction, regime, target=0.8, minimum_rows=50)
        self.assertGreaterEqual(uncertainty["global"], 0.0)
        self.assertGreater(len(uncertainty["regimes"]), 0)
        for entry in uncertainty["regimes"].values():
            self.assertGreaterEqual(entry["half_width"], 0.0)

    def test_coverage_is_close_to_target_on_synthetic_data(self):
        rng = np.random.default_rng(1)
        n = 5000
        actual = rng.uniform(3, 30, size=n)
        errors = rng.normal(0, 2, size=n)
        prediction = actual + errors
        regime = np.full(n, 1)

        uncertainty = fit_uncertainty(actual, prediction, regime, target=0.8, minimum_rows=300)
        bounds = intervals(prediction, regime, uncertainty)
        lo, hi = bounds[:, 0], bounds[:, 2]
        covered = np.mean((actual >= lo) & (actual <= hi))
        # Interval is symmetric (+/- the target-quantile of |error|), so true
        # coverage on a symmetric error distribution should land noticeably
        # above the nominal one-sided target - just checking it's in a sane
        # neighborhood, not exact, since this is empirical/finite-sample.
        self.assertGreater(covered, 0.8)
        self.assertLess(covered, 1.0)

    def test_missing_regime_falls_back_to_global(self):
        rng = np.random.default_rng(2)
        actual = rng.uniform(3, 30, size=500)
        prediction = actual + rng.normal(0, 2, size=500)
        regime = np.full(500, "month_1")
        uncertainty = fit_uncertainty(actual, prediction, regime, minimum_rows=50)

        bounds = intervals([15.0], ["never_seen_regime"], uncertainty)
        expected_width = uncertainty["global"]
        self.assertAlmostEqual(bounds[0, 2] - bounds[0, 0], 2 * expected_width, places=6)

    def test_sparse_regime_below_minimum_rows_is_excluded(self):
        rng = np.random.default_rng(3)
        actual = rng.uniform(3, 30, size=400)
        prediction = actual + rng.normal(0, 2, size=400)
        regime = np.array(["common"] * 380 + ["rare"] * 20)
        uncertainty = fit_uncertainty(actual, prediction, regime, minimum_rows=300)
        self.assertIn("common", uncertainty["regimes"])
        self.assertNotIn("rare", uncertainty["regimes"])


if __name__ == "__main__":
    unittest.main()
