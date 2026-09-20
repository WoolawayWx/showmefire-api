"""Covers Phase 2's v4/v5 pilot migration (fourth and fifth families,
completing the guarded-shadow -> unified registry migration): install_bundle()
dual-writes into the unified registry, and validate_bundle() can resolve
from the registry's `stable` channel. Uses tests/test_v5_safety.py's own
_bundle() fixture shape (mirrored here for v4 with its own asset names)."""
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from core.fire_danger import RULE_SPEC_SHA256
from core.precipitation import PRECIPITATION_CONTRACT_SHA256, PRECIPITATION_CONTRACT_VERSION
from models import versioning
from models import shadow_bundles
from services import v4_shadow, v5_shadow


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_v4_bundle(directory: Path, version="0.0.1-beta.1") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    assets = {}
    for name, value in (("base_xgboost.json", "base"), ("guarded_gru.pt", "residual"),
                        ("lead_guard.json", "lead_guard")):
        path = directory / name
        path.write_text(value)
        assets[name] = _sha(path)
    contract = {"rule_spec_sha256": RULE_SPEC_SHA256, "manifest_sha256": "manifest",
                "precipitation_contract_version": PRECIPITATION_CONTRACT_VERSION,
                "precipitation_contract_sha256": PRECIPITATION_CONTRACT_SHA256,
                "base_model_sha256": assets["base_xgboost.json"],
                "residual_model_sha256": assets["guarded_gru.pt"],
                "lead_guard_sha256": assets["lead_guard.json"]}
    (directory / "contract.json").write_text(json.dumps(contract))
    names = (*assets.keys(), "contract.json")
    shadow = {"status": "experimental_shadow_only", "registry_channel": None,
              "rule_spec_sha256": RULE_SPEC_SHA256,
              "precipitation_contract_sha256": PRECIPITATION_CONTRACT_SHA256,
              "assets": {name: _sha(directory / name) for name in names}}
    (directory / "shadow_bundle_manifest.json").write_text(json.dumps(shadow))
    if version is not None:
        (directory / "registered_version.json").write_text(json.dumps({"model_type": "v4", "version": version}))


def _write_v5_bundle(directory: Path, version="0.0.1-beta.1") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    assets = {}
    for name, value in (("base_xgboost.json", "base"), ("specialist_xgboost.json", "specialist"),
                        ("guard.json", "guard"), ("uncertainty.json", "uncertainty")):
        path = directory / name
        path.write_text(value)
        assets[name] = _sha(path)
    contract = {"rule_spec_sha256": RULE_SPEC_SHA256, "manifest_sha256": "manifest",
                "precipitation_contract_version": PRECIPITATION_CONTRACT_VERSION,
                "precipitation_contract_sha256": PRECIPITATION_CONTRACT_SHA256,
                "base_model_sha256": assets["base_xgboost.json"],
                "specialist_model_sha256": assets["specialist_xgboost.json"],
                "guard_sha256": assets["guard.json"], "uncertainty_sha256": assets["uncertainty.json"]}
    (directory / "contract.json").write_text(json.dumps(contract))
    names = (*assets.keys(), "contract.json")
    shadow = {"status": "experimental_shadow_only", "registry_channel": None,
              "rule_spec_sha256": RULE_SPEC_SHA256,
              "precipitation_contract_version": PRECIPITATION_CONTRACT_VERSION,
              "precipitation_contract_sha256": PRECIPITATION_CONTRACT_SHA256,
              "assets": {name: _sha(directory / name) for name in names}}
    (directory / "shadow_bundle_manifest.json").write_text(json.dumps(shadow))
    if version is not None:
        (directory / "registered_version.json").write_text(json.dumps({"model_type": "v5", "version": version}))


class InstallBundleDualWriteTests(unittest.TestCase):
    def _isolated(self, root):
        return (
            patch.multiple(versioning, API_DIR=root, MODELS_DIR=root / "reg-models",
                           CONFIG_PATH=root / "reg-models" / "config.json", VERSIONS_DIR=root / "reg-models" / "versions"),
            patch.object(shadow_bundles, "BUNDLES_ROOT", root / "model-bundles"),
        )

    def _zip(self, root, family, writer, version):
        bundle_dir = root / f"{family}-candidate-{version.replace('.', '_')}"
        writer(bundle_dir, version=version)
        archive_path = root / f"{family}-{version.replace('.', '_')}.zip"
        with zipfile.ZipFile(archive_path, "w") as zf:
            for file in bundle_dir.iterdir():
                zf.write(file, file.name)
        return archive_path

    def test_install_v4_bundle_registers_a_beta_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry_patch, bundles_patch = self._isolated(root)
            with registry_patch, bundles_patch:
                archive_path = self._zip(root, "v4", _write_v4_bundle, "0.0.1-beta.3")
                result = shadow_bundles.install_bundle("v4", archive_path, uploaded_by="tester")

                self.assertIsNotNone(result["registry_version"])
                entry = versioning.get_model_entry("v4")
                beta = entry["beta"]
                self.assertEqual(beta["metadata"]["shadow_bundle_version"], "0.0.1-beta.3")
                self.assertTrue(beta["metadata"]["advisory_only"])
                self.assertEqual(beta["metadata"]["rule_spec_sha256"], RULE_SPEC_SHA256)
                self.assertIn("model", beta["assets"])
                self.assertIn("shadow_manifest", beta["assets"])

    def test_install_v5_bundle_registers_a_beta_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry_patch, bundles_patch = self._isolated(root)
            with registry_patch, bundles_patch:
                archive_path = self._zip(root, "v5", _write_v5_bundle, "0.0.1-beta.4")
                result = shadow_bundles.install_bundle("v5", archive_path, uploaded_by="tester")

                self.assertIsNotNone(result["registry_version"])
                entry = versioning.get_model_entry("v5")
                beta = entry["beta"]
                self.assertEqual(beta["metadata"]["shadow_bundle_version"], "0.0.1-beta.4")
                self.assertTrue(beta["metadata"]["advisory_only"])
                self.assertIn("guard", beta["assets"])
                self.assertIn("uncertainty", beta["assets"])


class ValidateBundleRegistryFallbackTests(unittest.TestCase):
    def _isolated(self, root):
        return patch.multiple(versioning, API_DIR=root, MODELS_DIR=root / "models",
                              CONFIG_PATH=root / "models" / "config.json", VERSIONS_DIR=root / "models" / "versions")

    def test_v4_falls_back_to_registry_stable_when_env_and_directory_are_unset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_dir = root / "source"
            _write_v4_bundle(bundle_dir, version="0.0.1-beta.7")

            with self._isolated(root), patch.dict("os.environ", {v4_shadow.BUNDLE_ENV: ""}, clear=False):
                assets = {name: str(bundle_dir / filename) for name, filename in v4_shadow.BUNDLE_ASSET_FILENAMES.items()}
                versioning.register_trained_model(
                    "v4", channel="beta", assets=assets,
                    metadata={"advisory_only": True, "rule_spec_sha256": RULE_SPEC_SHA256,
                             "precipitation_contract_version": PRECIPITATION_CONTRACT_VERSION,
                             "precipitation_contract_sha256": PRECIPITATION_CONTRACT_SHA256,
                             "shadow_bundle_version": "0.0.1-beta.7"},
                )
                versioning.promote("v4")

                contract = v4_shadow.validate_bundle(directory=None)

        self.assertEqual(contract["registered_version"], "0.0.1-beta.7")

    def test_v5_falls_back_to_registry_stable_when_env_and_directory_are_unset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_dir = root / "source"
            _write_v5_bundle(bundle_dir, version="0.0.1-beta.8")

            with self._isolated(root), patch.dict("os.environ", {v5_shadow.BUNDLE_ENV: ""}, clear=False):
                assets = {name: str(bundle_dir / filename) for name, filename in v5_shadow.BUNDLE_ASSET_FILENAMES.items()}
                versioning.register_trained_model(
                    "v5", channel="beta", assets=assets,
                    metadata={"advisory_only": True, "rule_spec_sha256": RULE_SPEC_SHA256,
                             "precipitation_contract_version": PRECIPITATION_CONTRACT_VERSION,
                             "precipitation_contract_sha256": PRECIPITATION_CONTRACT_SHA256,
                             "shadow_bundle_version": "0.0.1-beta.8"},
                )
                versioning.promote("v5")

                contract = v5_shadow.validate_bundle(directory=None)

        self.assertEqual(contract["registered_version"], "0.0.1-beta.8")

    def test_explicit_directory_still_wins_over_registry_for_v5(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry_bundle_dir = root / "registry_source"
            _write_v5_bundle(registry_bundle_dir, version="0.0.1-beta.1")

            override_dir = root / "operator_override"
            _write_v5_bundle(override_dir, version="0.0.1-beta.9")

            with self._isolated(root), patch.dict("os.environ", {v5_shadow.BUNDLE_ENV: ""}, clear=False):
                assets = {name: str(registry_bundle_dir / filename) for name, filename in v5_shadow.BUNDLE_ASSET_FILENAMES.items()}
                versioning.register_trained_model(
                    "v5", channel="beta", assets=assets,
                    metadata={"advisory_only": True, "rule_spec_sha256": RULE_SPEC_SHA256,
                             "precipitation_contract_version": PRECIPITATION_CONTRACT_VERSION,
                             "precipitation_contract_sha256": PRECIPITATION_CONTRACT_SHA256},
                )
                versioning.promote("v5")

                contract = v5_shadow.validate_bundle(directory=override_dir)

        self.assertEqual(contract["registered_version"], "0.0.1-beta.9")


if __name__ == "__main__":
    unittest.main()
