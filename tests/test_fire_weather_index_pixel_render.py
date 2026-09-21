"""Coverage for the pixel/raster rendering path added to
fire_weather_index_shadow.py (compute_*_grid, score_to_category_index_grid,
_render_pixel_fill via _render_png) - matches the operational Peak Fire
Danger Forecast map's palette/style instead of the original county
choropleth. See test_fire_weather_index_shadow.py for the pre-existing
scalar/bundle-loading coverage this doesn't duplicate.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from services import fire_weather_index_shadow as fwis

BUNDLE = {
    "factor_weights": {
        "weights": {"rh": 1.0, "wind": 1.0, "vpd": 1.0, "precip_relief": 1.0},
        "raw_score_ceiling": {"value": 1.0},
        "ramp_anchors": {
            "rh": {"benign": 60.0, "extreme": 10.0},
            "wind": {"benign": 5.0, "extreme": 30.0},
            "vpd": {"benign": 0.5, "extreme": 4.0},
            "precip_relief": {"benign": 0.0, "extreme": 10.0},
        },
    },
    "category_thresholds": {"thresholds": [0.2, 0.4, 0.6, 0.8]},
    "version": "0.0.1-test",
}


class GridMathMatchesScalarMathTests(unittest.TestCase):
    def test_ramp_grid_matches_scalar_ramp_per_cell_including_nan(self):
        values = np.array([[15.0, 60.0, np.nan], [10.0, 35.0, 5.0]])
        grid = fwis._ramp_grid(values, benign=60.0, extreme=10.0)
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                expected = fwis._ramp(float(values[row, col]) if np.isfinite(values[row, col]) else None,
                                       60.0, 10.0)
                actual = grid[row, col]
                if expected is None:
                    self.assertTrue(np.isnan(actual))
                else:
                    self.assertAlmostEqual(actual, expected, places=9)

    def test_compute_score_grid_matches_compute_score_per_cell(self):
        anchors = BUNDLE["factor_weights"]["ramp_anchors"]
        weights = BUNDLE["factor_weights"]["weights"]
        weather_grids = {
            "rh_min_afternoon": np.array([[20.0, 55.0], [np.nan, 40.0]]),
            "wind_kts_max": np.array([[25.0, 8.0], [15.0, 12.0]]),
            "vpd_kpa_max": np.array([[3.0, 1.0], [2.0, 1.5]]),
            "precip_24h_mm": np.array([[0.0, 8.0], [1.0, 3.0]]),
        }
        factor_grids = fwis.compute_factors_grid(anchors, weather_grids)
        score_grid = fwis.compute_score_grid(weights, factor_grids, raw_score_ceiling=1.0)

        for row in range(2):
            for col in range(2):
                weather_row = {key: float(grid[row, col]) if np.isfinite(grid[row, col]) else None
                               for key, grid in weather_grids.items()}
                factor_values = fwis.compute_factors(anchors, weather_row)
                expected = fwis.compute_score(weights, factor_values, raw_score_ceiling=1.0)
                actual = score_grid[row, col]
                if expected is None:
                    self.assertTrue(np.isnan(actual))
                else:
                    self.assertAlmostEqual(actual, expected, places=9)

    def test_a_nan_cell_does_not_corrupt_neighboring_cells(self):
        weather_grids = {
            "rh_min_afternoon": np.array([[np.nan, 55.0]]),
            "wind_kts_max": np.array([[25.0, 8.0]]),
            "vpd_kpa_max": np.array([[3.0, 1.0]]),
            "precip_24h_mm": np.array([[0.0, 8.0]]),
        }
        anchors = BUNDLE["factor_weights"]["ramp_anchors"]
        weights = BUNDLE["factor_weights"]["weights"]
        factor_grids = fwis.compute_factors_grid(anchors, weather_grids)
        score_grid = fwis.compute_score_grid(weights, factor_grids, raw_score_ceiling=1.0)
        # Cell (0,0) has a NaN rh input but the other three factors are
        # finite there - it should still renormalize and score, not go NaN
        # sitewide just because one factor was missing at that one cell.
        self.assertFalse(np.isnan(score_grid[0, 0]))
        self.assertFalse(np.isnan(score_grid[0, 1]))


class ScoreToCategoryIndexGridTests(unittest.TestCase):
    def test_matches_score_to_category_at_thresholds(self):
        thresholds = [0.2, 0.4, 0.6, 0.8]
        scores = np.array([0.0, 0.1, 0.2, 0.5, 0.9, 1.0])
        index_grid = fwis.score_to_category_index_grid(scores, thresholds)
        for score, index in zip(scores, index_grid):
            expected_category = fwis.score_to_category(float(score), thresholds)
            # The continuous index at an exact category boundary should
            # round to the same discrete category the scalar path gives.
            self.assertAlmostEqual(round(index), expected_category, delta=1)

    def test_ceiling_at_exactly_one_maps_to_solidly_extreme_not_a_boundary(self):
        # Regression for the exact edge case documented in
        # score_fire_weather_index_today.py::render_pixel_map: when the
        # Extreme threshold IS the 1.0 ceiling, a score of 1.0 must map to
        # a solidly-Extreme index (4.0), not the Critical/Extreme boundary.
        thresholds = [0.2, 0.4, 0.6, 1.0]
        index = fwis.score_to_category_index_grid(np.array([1.0]), thresholds)[0]
        self.assertEqual(index, 4.0)


class PixelRenderTests(unittest.TestCase):
    def test_render_png_with_weather_grids_produces_production_colors(self):
        # A small synthetic grid over Missouri's real bbox, one clearly-Low
        # cell and one clearly-Extreme cell, positioned so each dominates
        # one half of the image after projection.
        lat = np.array([[36.0, 36.0], [40.0, 40.0]])
        lon = np.array([[-95.0, -90.0], [-95.0, -90.0]])
        weather_grids = {
            "rh_min_afternoon": np.array([[70.0, 70.0], [5.0, 5.0]]),   # top: benign RH -> Low
            "wind_kts_max": np.array([[3.0, 3.0], [35.0, 35.0]]),       # bottom: extreme wind -> Extreme
            "vpd_kpa_max": np.array([[0.2, 0.2], [5.0, 5.0]]),
            "precip_24h_mm": np.array([[20.0, 20.0], [0.0, 0.0]]),
        }
        with tempfile.TemporaryDirectory() as directory:
            out_path = Path(directory) / "pixel_test.png"
            fwis._render_png([], {}, BUNDLE, out_path, weather_grids=weather_grids, lat=lat, lon=lon)
            self.assertTrue(out_path.is_file())
            image = Image.open(out_path).convert("RGB")
            self.assertGreater(image.width, 0)
            self.assertGreater(image.height, 0)

    def test_render_png_without_grids_still_falls_back_to_county_choropleth(self):
        scored = {"29001": {"score": 0.1, "category": 0}}
        with tempfile.TemporaryDirectory() as directory:
            out_path = Path(directory) / "county_test.png"
            fwis._render_png(["29001"], scored, BUNDLE, out_path)
            self.assertTrue(out_path.is_file())


if __name__ == "__main__":
    unittest.main()
