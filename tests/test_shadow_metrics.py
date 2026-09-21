import unittest

from services import shadow_metrics


class CategoriesForTests(unittest.TestCase):
    def test_matches_calculate_fire_danger_directly(self):
        from core.fire_danger import calculate_fire_danger
        fm, rh, wind = [8.0, 15.0], [30.0, 60.0], [20.0, 5.0]
        expected = [calculate_fire_danger(f, r, w) for f, r, w in zip(fm, rh, wind)]
        self.assertEqual(shadow_metrics.categories_for(fm, rh, wind), expected)


class CategoryDisagreementSummaryTests(unittest.TestCase):
    def test_counts_disagreements_and_unavailable_rows(self):
        stable = [0, 1, 2, None, 3]
        beta = [0, 2, 2, 1, None]
        summary = shadow_metrics.category_disagreement_summary(stable, beta)
        # index 1 (1!=2), index 3 (None!=1), index 4 (3!=None) all disagree -
        # a None on either side always counts as a disagreement too, matching
        # the original inlined `a != b` behavior this replaces verbatim.
        self.assertEqual(summary["category_disagreements"], 3)
        self.assertEqual(summary["unavailable"], 2)  # index 3 and index 4 have a None on one side
        self.assertEqual(summary["comparable_rows"], 3)
        # Among the 3 comparable rows (index 0,1,2): only index 1 (1!=2)
        # actually disagrees - disagreement_rate = 1/3, not
        # category_disagreements(3)/comparable_rows(3) which would
        # wrongly include the two None-caused mismatches.
        self.assertAlmostEqual(summary["disagreement_rate"], 1 / 3, places=4)

    def test_all_unavailable_gives_none_disagreement_rate(self):
        summary = shadow_metrics.category_disagreement_summary([None], [None])
        self.assertEqual(summary["comparable_rows"], 0)
        self.assertIsNone(summary["disagreement_rate"])

    def test_perfect_agreement(self):
        summary = shadow_metrics.category_disagreement_summary([0, 1, 2], [0, 1, 2])
        self.assertEqual(summary["category_disagreements"], 0)
        self.assertEqual(summary["disagreement_rate"], 0.0)


class ContinuousErrorSummaryTests(unittest.TestCase):
    def test_computes_mae_and_bias(self):
        stable = [10.0, 20.0, 30.0]
        beta = [11.0, 19.0, 33.0]
        summary = shadow_metrics.continuous_error_summary(stable, beta)
        self.assertEqual(summary["n"], 3)
        self.assertAlmostEqual(summary["mae"], (1 + 1 + 3) / 3, places=4)
        self.assertAlmostEqual(summary["bias"], (1 - 1 + 3) / 3, places=4)

    def test_ignores_non_finite_rows(self):
        stable = [10.0, float("nan"), 30.0]
        beta = [11.0, 5.0, float("nan")]
        summary = shadow_metrics.continuous_error_summary(stable, beta)
        self.assertEqual(summary["n"], 1)
        self.assertAlmostEqual(summary["mae"], 1.0, places=4)

    def test_empty_input_returns_none(self):
        summary = shadow_metrics.continuous_error_summary([], [])
        self.assertEqual(summary, {"mae": None, "bias": None, "n": 0})


if __name__ == "__main__":
    unittest.main()
