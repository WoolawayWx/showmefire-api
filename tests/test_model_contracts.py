import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

API_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(API_DIR))

from core.fire_danger import (FireDangerCategory, calculate_fire_danger,
                              meters_per_second_to_knots, miles_per_hour_to_knots)
from models.features import build_causal_features, validate_feature_contract
from models import versioning
from pipelines.generate_training_set import align_observations_to_weather


class FireDangerContractTests(unittest.TestCase):
    def test_category_boundaries_preserve_public_ids(self):
        cases = [
            ((15, 10, 40), FireDangerCategory.LOW),
            ((14.9, 44.9, 0), FireDangerCategory.MODERATE),
            ((8.9, 34.9, 12), FireDangerCategory.ELEVATED),
            ((8.9, 24.9, 15), FireDangerCategory.CRITICAL),
            ((6.9, 19.9, 25), FireDangerCategory.EXTREME),
            ((7.0, 19.9, 25), FireDangerCategory.CRITICAL),
        ]
        for inputs, expected in cases:
            with self.subTest(inputs=inputs):
                self.assertEqual(calculate_fire_danger(*inputs), int(expected))

    def test_wind_conversions(self):
        self.assertAlmostEqual(meters_per_second_to_knots(1), 1.9438444924406)
        self.assertAlmostEqual(miles_per_hour_to_knots(1), 0.86897624190065)


class CausalFeatureTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "station_id": ["A", "A", "A"],
            "obs_time": ["2026-01-01T00:00Z", "2026-01-01T02:00Z", "2026-01-01T08:00Z"],
            "temp_c": [0.0, 2.0, 8.0], "rel_humidity": [50.0, 40.0, 20.0],
            "wind_speed_ms": [1.0, 2.0, 3.0], "precip_mm": [0.0, 1.0, 0.0],
        })

    def test_windows_are_time_based_and_causal(self):
        result = build_causal_features(self.frame)
        self.assertEqual(result.loc[2, "temp_mean_3h"], 8.0)
        self.assertEqual(result.loc[2, "hours_since_rain"], 6.0)
        changed = self.frame.copy(); changed.loc[2, "temp_c"] = 50
        changed_result = build_causal_features(changed)
        self.assertEqual(result.loc[1, "temp_mean_3h"], changed_result.loc[1, "temp_mean_3h"])

    def test_contract_rejects_missing_features(self):
        with self.assertRaisesRegex(ValueError, "Missing required"):
            validate_feature_contract(self.frame, {"feature_columns": ["unknown"]})


class TemporalAlignmentTests(unittest.TestCase):
    def test_nearest_prior_only_and_tolerance(self):
        observations = pd.DataFrame({"station_id": ["A", "A"],
                                     "obs_time": ["2026-01-01T01:00Z", "2026-01-01T04:00Z"],
                                     "target_fm": [8, 9]})
        weather = pd.DataFrame({"station_id": ["A", "A", "A"],
                                "weather_time": ["2026-01-01T00:30Z", "2026-01-01T01:05Z", "2026-01-01T02:00Z"],
                                "temp_c": [1, 2, 3]})
        result = align_observations_to_weather(observations, weather, 60)
        self.assertEqual(result.loc[0, "temp_c"], 1)
        self.assertEqual(result.loc[0, "match_age_minutes"], 30)
        self.assertTrue(pd.isna(result.loc[1, "weather_time"]))

    def test_invalid_timestamp_fails(self):
        obs = pd.DataFrame({"station_id": ["A"], "obs_time": ["bad"]})
        wx = pd.DataFrame({"station_id": ["A"], "weather_time": ["2026-01-01Z"]})
        with self.assertRaisesRegex(ValueError, "Invalid timestamps"):
            align_observations_to_weather(obs, wx)


class RegistrySafetyTests(unittest.TestCase):
    def _metadata(self, shadow_passed=True):
        return {
            "feature_schema_version": "2.0.0", "rule_spec_version": "1.0.0",
            "training_window": {}, "data_match_policy": {}, "validation_folds": [],
            "class_support": {}, "feature_columns": [], "promotion_gates": {"offline": True},
            "shadow_required": True, "shadow": {"passed": shadow_passed},
        }

    def test_promotion_gate_and_rollback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); models = root / "models"; versions = models / "versions"
            models.mkdir(); source1 = root / "one.json"; source1.write_text("one")
            source2 = root / "two.json"; source2.write_text("two")
            with patch.multiple(versioning, API_DIR=root, MODELS_DIR=models,
                                CONFIG_PATH=models / "config.json", VERSIONS_DIR=versions):
                first_beta = versioning.register_trained_model("fuel_moisture", source1,
                    channel="beta", metadata=self._metadata(False))
                with self.assertRaisesRegex(ValueError, "shadow validation"):
                    versioning.promote("fuel_moisture", first_beta)
                versioning.update_beta_metadata("fuel_moisture", {"shadow": {"passed": True}})
                self.assertEqual(versioning.promote("fuel_moisture", first_beta), "0.0.1")
                second_beta = versioning.register_trained_model("fuel_moisture", source2,
                    channel="beta", metadata=self._metadata(True))
                self.assertEqual(versioning.promote("fuel_moisture", second_beta), "0.0.2")
                active_path = root / versioning.get_model_entry("fuel_moisture")["stable"]["file"]
                active_path.unlink()
                self.assertEqual(versioning.load_active_model_path("fuel_moisture", auto_rollback=True),
                                 versions / "fuel_moisture_0.0.1.json")
                self.assertEqual((models / "fuel_moisture_model.json").read_text(), "one")


if __name__ == "__main__":
    unittest.main()
