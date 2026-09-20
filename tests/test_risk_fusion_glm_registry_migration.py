"""Covers Phase 2's risk_fusion_glm pilot migration: install_bundle() now
dual-writes into the unified registry (models/versioning.py), and
load_bundle() can resolve a bundle from that registry's `stable` channel
when no explicit directory/BUNDLE_ENV override is set. Uses
tests/test_risk_fusion_glm_shadow.py's own _write_bundle() fixture shape."""
import json
import tempfile
import zipfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models import versioning
from models import shadow_bundles
from services import risk_fusion_glm_shadow as rfgs
from tests.test_risk_fusion_glm_shadow import _write_bundle, _real_feature_module_sha256


class InstallBundleDualWriteTests(unittest.TestCase):
    def _isolated(self, root):
        return (
            patch.multiple(versioning, API_DIR=root, MODELS_DIR=root / "reg-models",
                           CONFIG_PATH=root / "reg-models" / "config.json", VERSIONS_DIR=root / "reg-models" / "versions"),
            patch.object(shadow_bundles, "BUNDLES_ROOT", root / "model-bundles"),
        )

    def _zip_bundle(self, root) -> Path:
        bundle_dir = root / "candidate"
        bundle_dir.mkdir()
        _write_bundle(bundle_dir)
        (bundle_dir / "registered_version.json").write_text(
            json.dumps({"model_type": "fire_risk_fusion", "version": "0.0.1-beta.3"}))
        archive_path = root / "bundle.zip"
        with zipfile.ZipFile(archive_path, "w") as zf:
            for file in bundle_dir.iterdir():
                zf.write(file, file.name)
        return archive_path

    def test_install_bundle_registers_a_beta_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry_patch, bundles_patch = self._isolated(root)
            with registry_patch, bundles_patch:
                archive_path = self._zip_bundle(root)
                result = shadow_bundles.install_bundle("risk_fusion_glm", archive_path, uploaded_by="tester")

                self.assertIsNotNone(result["registry_version"])
                entry = versioning.get_model_entry("risk_fusion_glm")
                beta = entry["beta"]
                self.assertEqual(beta["version"], result["registry_version"])
                self.assertEqual(beta["metadata"]["shadow_bundle_version"], "0.0.1-beta.3")
                self.assertTrue(beta["metadata"]["advisory_only"])
                self.assertEqual(beta["metadata"]["model_family"], "glm")
                self.assertIn("climatology", beta["assets"])
                self.assertIn("contract", beta["assets"])

    def test_other_families_are_unaffected_by_dual_write(self):
        # All 5 real guarded-shadow families (v4, v5, risk_fusion_glm,
        # fire_weather_ml, fire_weather_index) are now migrated - this test
        # instead confirms install_bundle() only dual-writes families
        # actually listed in _REGISTRY_MIGRATED_FAMILIES, using a
        # hypothetical not-yet-migrated family so the guard itself (not
        # just "every real family happens to be migrated") stays covered.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry_patch, bundles_patch = self._isolated(root)
            with registry_patch, bundles_patch:
                with patch.object(shadow_bundles, "SHADOW_FAMILIES", ["not_yet_migrated"]), \
                     patch.object(shadow_bundles, "_validators", return_value={
                         "not_yet_migrated": (rfgs.BUNDLE_ENV, lambda d: {"version": "x"})
                     }):
                    bundle_dir = root / "candidate"
                    bundle_dir.mkdir()
                    (bundle_dir / "factor_weights.json").write_text("{}")
                    archive_path = root / "bundle.zip"
                    with zipfile.ZipFile(archive_path, "w") as zf:
                        zf.write(bundle_dir / "factor_weights.json", "factor_weights.json")
                    result = shadow_bundles.install_bundle("not_yet_migrated", archive_path, uploaded_by="tester")
                self.assertIsNone(result["registry_version"])
                self.assertEqual(versioning.get_model_entry("not_yet_migrated"), {})


class LoadBundleRegistryFallbackTests(unittest.TestCase):
    def _isolated(self, root):
        return patch.multiple(versioning, API_DIR=root, MODELS_DIR=root / "models",
                              CONFIG_PATH=root / "models" / "config.json", VERSIONS_DIR=root / "models" / "versions")

    def test_falls_back_to_registry_stable_when_env_and_directory_are_unset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_dir = root / "source"
            bundle_dir.mkdir()
            _write_bundle(bundle_dir)

            with self._isolated(root), patch.dict("os.environ", {rfgs.BUNDLE_ENV: ""}, clear=False):
                assets = {name: str(bundle_dir / filename) for name, filename in rfgs.BUNDLE_ASSET_FILENAMES.items()}
                versioning.register_trained_model(
                    "risk_fusion_glm", channel="beta", assets=assets,
                    metadata={"advisory_only": True, "model_family": "glm",
                             "feature_module_sha256": _real_feature_module_sha256(),
                             "shadow_bundle_version": "0.0.1-beta.7"},
                )
                versioning.promote("risk_fusion_glm")

                bundle = rfgs.load_bundle(directory=None)

        self.assertEqual(bundle["contract"]["model_family"], "glm")
        self.assertEqual(bundle["version"], "0.0.1-beta.7")

    def test_explicit_directory_still_wins_over_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry_bundle_dir = root / "registry_source"
            registry_bundle_dir.mkdir()
            _write_bundle(registry_bundle_dir, county_fips=("29001",))

            override_dir = root / "operator_override"
            override_dir.mkdir()
            _write_bundle(override_dir, county_fips=("29510",))

            with self._isolated(root), patch.dict("os.environ", {rfgs.BUNDLE_ENV: ""}, clear=False):
                assets = {name: str(registry_bundle_dir / filename) for name, filename in rfgs.BUNDLE_ASSET_FILENAMES.items()}
                versioning.register_trained_model(
                    "risk_fusion_glm", channel="beta", assets=assets,
                    metadata={"advisory_only": True, "model_family": "glm",
                             "feature_module_sha256": _real_feature_module_sha256()},
                )
                versioning.promote("risk_fusion_glm")

                bundle = rfgs.load_bundle(directory=override_dir)

        self.assertIn("29510", bundle["county_reference"])
        self.assertNotIn("29001", bundle["county_reference"])


if __name__ == "__main__":
    unittest.main()
