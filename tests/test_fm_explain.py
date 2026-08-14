import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xgboost as xgb

API_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(API_DIR))

from models import versioning

FEATURES = ["temp_c", "rel_humidity", "wind_speed_ms"]


def _train_tiny_booster(path):
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(200, len(FEATURES))) * [40, 100, 20]
    y = 20 - 0.1 * X[:, 0] + 0.15 * X[:, 1] - 0.5 * X[:, 2]
    dmat = xgb.DMatrix(X, label=y, feature_names=FEATURES)
    booster = xgb.train({"objective": "reg:squarederror", "max_depth": 3}, dmat, num_boost_round=20)
    booster.save_model(str(path))


class FmExplainTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        models_dir = root / "models"
        models_dir.mkdir()
        model_path = root / "fuel_moisture.json"
        _train_tiny_booster(model_path)

        patcher = patch.multiple(
            versioning, API_DIR=root, MODELS_DIR=models_dir,
            CONFIG_PATH=models_dir / "config.json", VERSIONS_DIR=models_dir / "versions",
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        metadata = {
            "feature_schema_version": "2.0.0", "rule_spec_version": "1.0.0",
            "training_window": {}, "data_match_policy": {}, "validation_folds": [],
            "class_support": {}, "feature_columns": FEATURES,
            "shadow_required": False,
        }
        version = versioning.register_trained_model(
            "fuel_moisture", model_path, channel="beta", metadata=metadata)
        versioning.promote("fuel_moisture", version)

        # fm_explain resolves the active model at IMPORT time - reload it now,
        # inside the patched registry, so it picks up the fixture model
        # instead of whatever is (or isn't) registered on this machine.
        global fm_explain
        import services.fm_explain as fm_explain  # noqa: PLC0415 - deliberate late import, see above
        importlib.reload(fm_explain)
        self.fm_explain = fm_explain

    def test_global_importance_covers_every_feature_non_negative(self):
        importance = self.fm_explain.global_importance()
        self.assertEqual(len(importance), len(FEATURES))
        for value in importance.values():
            self.assertGreaterEqual(value, 0.0)

    def test_explain_prediction_contributions_sum_to_prediction(self):
        row = {"temp_c": 20.0, "rel_humidity": 30.0, "wind_speed_ms": 8.0}
        result = self.fm_explain.explain_prediction(row)
        total = result["base_value"] + sum(result["contributions"].values())
        self.assertAlmostEqual(total, result["prediction"], places=4)

    def test_explain_prediction_rejects_missing_feature(self):
        with self.assertRaises(ValueError):
            self.fm_explain.explain_prediction({"temp_c": 20.0})


if __name__ == "__main__":
    unittest.main()
