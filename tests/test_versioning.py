import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models import versioning


class MultiAssetPromotionCrashRegressionTests(unittest.TestCase):
    """register_trained_model() only sets a non-None top-level `file` when an
    asset role is named model/checkpoint/static_bundle (see the `primary`
    lookup in register_trained_model). fire_risk_fusion (glm/guard/
    calibration/...), fire_weather_index (factor_weights/category_thresholds),
    and fire_weather_ml all lack such a role, so their beta["file"] is None.
    validate_promotion_candidate() and promote() both used to dereference
    that None (`API_DIR / None` -> TypeError) before ever reaching their real
    gate logic. These tests pin the fix: such candidates should be
    evaluated on their real gates, not crash first."""

    def _isolated_registry(self, root):
        return patch.multiple(
            versioning, API_DIR=root, MODELS_DIR=root / "models",
            CONFIG_PATH=root / "models" / "config.json", VERSIONS_DIR=root / "models" / "versions",
        )

    def _register_bundle_without_model_role(self, root, model_type, metadata=None):
        asset_path = root / "asset.json"
        asset_path.write_text("{}")
        return versioning.register_trained_model(
            model_type, channel="beta", assets={"factor_weights": str(asset_path)},
            metadata=metadata or {},
        )

    def test_validate_promotion_candidate_does_not_crash_on_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self._isolated_registry(root):
                self._register_bundle_without_model_role(root, "fire_weather_index",
                    metadata={"model_family": "x", "advisory_only": True})
                beta = versioning.get_model_entry("fire_weather_index")["beta"]
                self.assertIsNone(beta["file"])
                # Must not raise TypeError - real gate logic decides the outcome.
                blockers = versioning.validate_promotion_candidate("fire_weather_index", beta)
        self.assertEqual(blockers, [])

    def test_promote_does_not_crash_on_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self._isolated_registry(root):
                self._register_bundle_without_model_role(root, "fire_weather_index",
                    metadata={"model_family": "x", "advisory_only": True})
                version = versioning.promote("fire_weather_index")
                entry = versioning.get_model_entry("fire_weather_index")
        self.assertEqual(entry["stable"]["version"], version)
        self.assertIsNone(entry["beta"])

    def test_promote_still_blocks_fire_risk_fusion_missing_advisory_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self._isolated_registry(root):
                self._register_bundle_without_model_role(root, "fire_risk_fusion", metadata={})
                with self.assertRaises(ValueError):
                    versioning.promote("fire_risk_fusion")


class PromoteFilenameTests(unittest.TestCase):
    """promote() no longer renames the on-disk file to drop the -beta.N
    suffix - it's already immutable/content-addressed under
    models/versions/, and the rename was the same root cause as the crash
    above for multi-asset bundles. Confirms the version string is still
    cleaned even though the filename is untouched."""

    def test_registry_version_is_cleaned_but_filename_is_not_renamed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.multiple(versioning, API_DIR=root, MODELS_DIR=root / "models",
                                CONFIG_PATH=root / "models" / "config.json", VERSIONS_DIR=root / "models" / "versions"):
                source = root / "one.json"
                source.write_text("one")
                versioning.register_trained_model("fake_single_file", source, channel="beta")
                version = versioning.promote("fake_single_file")
                entry = versioning.get_model_entry("fake_single_file")

        self.assertEqual(version, "0.0.1")
        self.assertEqual(entry["stable"]["version"], "0.0.1")
        self.assertTrue(entry["stable"]["file"].endswith("fake_single_file_0.0.1-beta.1.json"))


class SaveConfigWindowsFallbackTests(unittest.TestCase):
    def test_falls_back_to_copy_when_replace_is_denied(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.multiple(versioning, API_DIR=root, MODELS_DIR=root / "models",
                                CONFIG_PATH=root / "models" / "config.json", VERSIONS_DIR=root / "models" / "versions"):
                real_replace = Path.replace

                def denied_replace(self, target):
                    if self.suffix == ".tmp":
                        raise PermissionError("simulated locked destination")
                    return real_replace(self, target)

                with patch.object(Path, "replace", denied_replace):
                    versioning._save_config({"fake_model": {"stable": None, "beta": None, "history": []}})

                self.assertTrue(versioning.CONFIG_PATH.exists())
                self.assertIn("fake_model", versioning._load_config())


if __name__ == "__main__":
    unittest.main()
