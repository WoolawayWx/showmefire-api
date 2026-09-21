"""Covers Phase 2's "Activate = promote" cutover for risk_fusion_glm, the
first family migrated off shadow_bundles.py's ungated activate onto the
gated models.versioning.promote(). Calls the router's async endpoint
functions directly (no HTTP layer) with _require_admin patched out, since
this only needs the routing/promotion logic, not auth."""
import asyncio
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from models import versioning
from models import shadow_bundles
import routers.model_admin as model_admin
from tests.test_risk_fusion_glm_shadow import _write_bundle, _real_feature_module_sha256


def _run(coro):
    return asyncio.run(coro)


class ActivateRiskFusionGlmTests(unittest.TestCase):
    def _isolated(self, root):
        return (
            patch.multiple(versioning, API_DIR=root, MODELS_DIR=root / "reg-models",
                           CONFIG_PATH=root / "reg-models" / "config.json", VERSIONS_DIR=root / "reg-models" / "versions"),
            patch.object(shadow_bundles, "BUNDLES_ROOT", root / "model-bundles"),
            patch.object(model_admin, "_require_admin", return_value="tester@example.com"),
            patch.object(model_admin, "_require_confirmation", return_value=None),
        )

    def _install_zip(self, root, version="0.0.1-beta.9"):
        bundle_dir = root / f"candidate-{version.replace('.', '_')}"
        bundle_dir.mkdir()
        _write_bundle(bundle_dir)
        (bundle_dir / "registered_version.json").write_text(
            json.dumps({"model_type": "fire_risk_fusion", "version": version}))
        archive_path = root / f"bundle-{version.replace('.', '_')}.zip"
        with zipfile.ZipFile(archive_path, "w") as zf:
            for file in bundle_dir.iterdir():
                zf.write(file, file.name)
        return shadow_bundles.install_bundle("risk_fusion_glm", archive_path, uploaded_by="tester")

    def test_activate_promotes_the_matching_registry_beta(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reg_patch, bundles_patch, admin_patch, confirm_patch = self._isolated(root)
            with reg_patch, bundles_patch, admin_patch, confirm_patch:
                installed = self._install_zip(root)

                response = _run(model_admin.activate_family_version(
                    "risk_fusion_glm", model_admin.ActivateRequest(version=installed["version"])))

                self.assertTrue(response["success"])
                entry = versioning.get_model_entry("risk_fusion_glm")
                self.assertIsNotNone(entry["stable"])
                self.assertIsNone(entry["beta"])
                self.assertEqual(entry["stable"]["metadata"]["shadow_bundle_version"], installed["version"])

                active = shadow_bundles.get_active("risk_fusion_glm")
                self.assertEqual(active["version"], installed["version"])

    def test_activate_unknown_version_returns_404_not_a_crash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reg_patch, bundles_patch, admin_patch, confirm_patch = self._isolated(root)
            with reg_patch, bundles_patch, admin_patch, confirm_patch:
                self._install_zip(root)
                from fastapi import HTTPException
                with self.assertRaises(HTTPException) as cm:
                    _run(model_admin.activate_family_version(
                        "risk_fusion_glm", model_admin.ActivateRequest(version="does-not-exist")))
                self.assertEqual(cm.exception.status_code, 404)

    def test_reactivating_an_already_stable_version_is_a_no_op_not_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reg_patch, bundles_patch, admin_patch, confirm_patch = self._isolated(root)
            with reg_patch, bundles_patch, admin_patch, confirm_patch:
                installed = self._install_zip(root)
                _run(model_admin.activate_family_version(
                    "risk_fusion_glm", model_admin.ActivateRequest(version=installed["version"])))

                response = _run(model_admin.activate_family_version(
                    "risk_fusion_glm", model_admin.ActivateRequest(version=installed["version"])))
                self.assertTrue(response["success"])

    def test_reactivating_an_older_promoted_version_rolls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reg_patch, bundles_patch, admin_patch, confirm_patch = self._isolated(root)
            with reg_patch, bundles_patch, admin_patch, confirm_patch:
                first = self._install_zip(root, version="0.0.1-beta.1")
                _run(model_admin.activate_family_version(
                    "risk_fusion_glm", model_admin.ActivateRequest(version=first["version"])))

                second = self._install_zip(root, version="0.0.1-beta.2")
                _run(model_admin.activate_family_version(
                    "risk_fusion_glm", model_admin.ActivateRequest(version=second["version"])))

                response = _run(model_admin.activate_family_version(
                    "risk_fusion_glm", model_admin.ActivateRequest(version=first["version"])))
                self.assertTrue(response["success"])
                entry = versioning.get_model_entry("risk_fusion_glm")
                self.assertEqual(entry["stable"]["metadata"]["shadow_bundle_version"], first["version"])


if __name__ == "__main__":
    unittest.main()
