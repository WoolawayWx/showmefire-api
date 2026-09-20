"""Covers Phase 2's fire_weather_index pilot migration (third family):
install_bundle() dual-writes into the unified registry, and load_bundle()
can resolve from the registry's `stable` channel. Uses
tests/test_fire_weather_index_shadow.py's own _write_bundle() fixture."""
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from models import versioning
from models import shadow_bundles
from services import fire_weather_index_shadow as fwis
from tests.test_fire_weather_index_shadow import _write_bundle


class InstallBundleDualWriteTests(unittest.TestCase):
    def _isolated(self, root):
        return (
            patch.multiple(versioning, API_DIR=root, MODELS_DIR=root / "reg-models",
                           CONFIG_PATH=root / "reg-models" / "config.json", VERSIONS_DIR=root / "reg-models" / "versions"),
            patch.object(shadow_bundles, "BUNDLES_ROOT", root / "model-bundles"),
        )

    def _zip_bundle(self, root, version="0.0.1-beta.4") -> Path:
        bundle_dir = root / f"candidate-{version.replace('.', '_')}"
        bundle_dir.mkdir()
        _write_bundle(bundle_dir, version=version)
        archive_path = root / f"bundle-{version.replace('.', '_')}.zip"
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
                result = shadow_bundles.install_bundle("fire_weather_index", archive_path, uploaded_by="tester")

                self.assertIsNotNone(result["registry_version"])
                entry = versioning.get_model_entry("fire_weather_index")
                beta = entry["beta"]
                self.assertEqual(beta["version"], result["registry_version"])
                self.assertEqual(beta["metadata"]["shadow_bundle_version"], "0.0.1-beta.4")
                self.assertTrue(beta["metadata"]["advisory_only"])
                self.assertEqual(beta["metadata"]["model_family"], "fire_weather_index")
                self.assertIn("factor_weights", beta["assets"])
                self.assertIn("category_thresholds", beta["assets"])


class LoadBundleRegistryFallbackTests(unittest.TestCase):
    def _isolated(self, root):
        return patch.multiple(versioning, API_DIR=root, MODELS_DIR=root / "models",
                              CONFIG_PATH=root / "models" / "config.json", VERSIONS_DIR=root / "models" / "versions")

    def test_falls_back_to_registry_stable_when_env_and_directory_are_unset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_dir = root / "source"
            bundle_dir.mkdir()
            _write_bundle(bundle_dir, version="0.0.1-beta.7")

            with self._isolated(root), patch.dict("os.environ", {fwis.BUNDLE_ENV: ""}, clear=False):
                assets = {name: str(bundle_dir / filename) for name, filename in fwis.BUNDLE_ASSET_FILENAMES.items()}
                versioning.register_trained_model(
                    "fire_weather_index", channel="beta", assets=assets,
                    metadata={"advisory_only": True, "model_family": "fire_weather_index",
                             "shadow_bundle_version": "0.0.1-beta.7"},
                )
                versioning.promote("fire_weather_index")

                bundle = fwis.load_bundle(directory=None)

        self.assertEqual(bundle["version"], "0.0.1-beta.7")
        self.assertIn("weights", bundle["factor_weights"])

    def test_explicit_directory_still_wins_over_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry_bundle_dir = root / "registry_source"
            registry_bundle_dir.mkdir()
            _write_bundle(registry_bundle_dir, version="0.0.1-beta.1")

            override_dir = root / "operator_override"
            override_dir.mkdir()
            _write_bundle(override_dir, version="0.0.1-beta.9")

            with self._isolated(root), patch.dict("os.environ", {fwis.BUNDLE_ENV: ""}, clear=False):
                assets = {name: str(registry_bundle_dir / filename) for name, filename in fwis.BUNDLE_ASSET_FILENAMES.items()}
                versioning.register_trained_model(
                    "fire_weather_index", channel="beta", assets=assets,
                    metadata={"advisory_only": True, "model_family": "fire_weather_index"},
                )
                versioning.promote("fire_weather_index")

                bundle = fwis.load_bundle(directory=override_dir)

        self.assertEqual(bundle["version"], "0.0.1-beta.9")


if __name__ == "__main__":
    unittest.main()
