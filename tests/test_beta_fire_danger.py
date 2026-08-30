import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from core.beta_fire_danger import score_fire_danger
from core.fire_danger import calculate_fire_danger
from services import beta_products
try:
    from routers import forecast_admin
except ModuleNotFoundError:
    # The minimal local test environment may not include API auth extras.
    forecast_admin = None


class BetaFireDangerTests(unittest.TestCase):
    def test_official_category_matches_canonical_rules(self):
        samples = [
            (20, 70, 5),
            (14, 44, 5),
            (8, 34, 12),
            (8, 24, 5),
            (8, 24, 15),
            (6, 19, 25),
        ]
        for fm, rh, wind in samples:
            result = score_fire_danger(fm, rh, wind)
            self.assertEqual(result["official_category"], calculate_fire_danger(fm, rh, wind))

    def test_score_increases_beyond_an_or_branch_threshold(self):
        near = score_fire_danger(8, 34.9, 12.1)
        deeper = score_fire_danger(8, 25, 20)
        self.assertGreater(deeper["score"], near["score"])
        self.assertGreater(deeper["criteria"]["elevated_branches"]["rh35_wind12"], 0)

    def test_higher_categories_start_at_their_scale_boundary(self):
        self.assertEqual(score_fire_danger(8.9, 34.9, 12)["score"], 1)
        self.assertEqual(score_fire_danger(8.9, 24.9, 15)["score"], 2)
        self.assertEqual(score_fire_danger(6.9, 19.9, 25)["score"], 3)

    def test_score_is_zero_for_low_conditions(self):
        result = score_fire_danger(20, 70, 5)
        self.assertEqual(result["official_label"], "Low")
        self.assertEqual(result["score"], 0)

    def test_invalid_values_are_rejected(self):
        with self.assertRaises(ValueError):
            score_fire_danger(float("nan"), 40, 10)

    def test_observation_products_are_written_to_beta_paths(self):
        station = {
            "stid": "TEST",
            "name": "Test Station",
            "state": "MO",
            "longitude": -92,
            "latitude": 38,
            "observations": {
                "relative_humidity": {"value": 30},
                "wind_speed": {"value": 15},
            },
        }
        raw_station = {
            "stid": "TEST",
            "observations": {"fuel_moisture": {"value": 8}},
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(beta_products, "BETA_GIS_DIR", root / "gis"), \
                    patch.object(beta_products, "BETA_MANIFEST_PATH", root / "manifest.json"), \
                    patch.object(beta_products, "BETA_OBSERVATION_STATE_PATH", root / "state.json"):
                result = beta_products.refresh_observation_products(
                    {"stations": [station]},
                    {"stations": [raw_station]},
                )
                self.assertEqual(result["products"]["realtime_current"]["features"][0]["properties"]["stid"], "TEST")
                self.assertTrue((root / "gis" / "realtime_current.geojson").exists())
                self.assertEqual(result["manifest"]["scorer_version"], "1.0.0")

    @unittest.skipIf(forecast_admin is None, "API authentication dependencies are not installed")
    def test_forecast_trigger_requires_admin(self):
        with patch.object(forecast_admin, "verify_token", return_value=None):
            with self.assertRaises(Exception) as context:
                forecast_admin._require_admin()
        self.assertEqual(getattr(context.exception, "status_code", None), 401)

    @unittest.skipIf(forecast_admin is None, "API authentication dependencies are not installed")
    def test_forecast_trigger_accepts_admin(self):
        with patch.object(forecast_admin, "verify_token", return_value="admin@example.org"):
            self.assertEqual(forecast_admin._require_admin(), "admin@example.org")


if __name__ == "__main__":
    unittest.main()
