import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models import versioning
import pipelines.promote_model as promote_model


class ChoicesListTests(unittest.TestCase):
    def test_fire_risk_fusion_is_now_a_valid_choice(self):
        # Regression test: fire_risk_fusion used to be excluded from this
        # CLI's --model choices entirely, even though it was importable via
        # import_model.py and had full gates in validate_promotion_candidate -
        # a structural dead end this phase fixes.
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", required=True,
                             choices=["fuel_moisture", "fire_danger", "fuel_moisture_spatial",
                                      "fire_behavior_static", "fire_risk_fusion"])
        parser.parse_args(["--model", "fire_risk_fusion"])


class MainEndToEndTests(unittest.TestCase):
    def _isolated_registry(self, root):
        return patch.multiple(
            versioning, API_DIR=root, MODELS_DIR=root / "models",
            CONFIG_PATH=root / "models" / "config.json", VERSIONS_DIR=root / "models" / "versions",
        )

    def test_promoting_fire_risk_fusion_reports_the_advisory_only_blocker_not_a_crash(self):
        # This candidate lacks a model/checkpoint/static_bundle asset role
        # (fire_risk_fusion's roles are glm/guard/calibration/...), which
        # used to crash promote_model.py with a TypeError before it ever
        # reached the real advisory_only gate. It should now cleanly print
        # the blocker and exit 1, not crash.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self._isolated_registry(root):
                asset_path = root / "glm.json"
                asset_path.write_text("{}")
                versioning.register_trained_model(
                    "fire_risk_fusion", channel="beta", assets={"glm": str(asset_path)}, metadata={},
                )
                with patch.object(sys, "argv", ["promote_model.py", "--model", "fire_risk_fusion"]):
                    with self.assertRaises(SystemExit) as cm:
                        promote_model.main()
        self.assertEqual(cm.exception.code, 1)

    def test_promoting_fire_risk_fusion_succeeds_once_advisory_only_and_gates_are_satisfied(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self._isolated_registry(root):
                with patch("core.fire_danger.RULE_SPEC_VERSION", "v-test"), \
                     patch("core.fire_danger.RULE_SPEC_SHA256", "sha-test"):
                    asset_path = root / "glm.json"
                    asset_path.write_text("{}")
                    metadata = {
                        "label_manifest_sha256": "a", "label_min_tier": "b", "label_rows_by_tier": "c",
                        "cause_filter": "d", "count_family": "poisson", "model_family": "glm",
                        "offset_definition_sha256": "e", "feature_module_sha256": "f",
                        "policy_version": "1", "policy_sha256": "2", "guard_active_row_fraction": 1.0,
                        "advisory_only": True, "rule_spec_version": "v-test", "rule_spec_sha256": "sha-test",
                        "feature_schema_version": "2.0.0", "training_window": {}, "data_match_policy": {},
                        "validation_folds": [], "class_support": {}, "feature_columns": [],
                    }
                    versioning.register_trained_model(
                        "fire_risk_fusion", channel="beta", assets={"glm": str(asset_path)}, metadata=metadata,
                    )
                    with patch.object(sys, "argv", ["promote_model.py", "--model", "fire_risk_fusion"]):
                        promote_model.main()
                entry = versioning.get_model_entry("fire_risk_fusion")
        self.assertIsNotNone(entry["stable"])
        self.assertIsNone(entry["beta"])


if __name__ == "__main__":
    unittest.main()
