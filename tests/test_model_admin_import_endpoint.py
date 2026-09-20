"""Covers Phase 4's website Import feature: POST /api/admin/models/{family}/import,
the server-side equivalent of pipelines/import_model.py's CLI. Calls the
router's async endpoint function directly (no HTTP layer) with
_require_admin patched out, same pattern as test_model_admin_activate.py."""
import asyncio
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models import versioning
import routers.model_admin as model_admin
from pipelines.import_model import ImportValidationError


def _run(coro):
    return asyncio.run(coro)


class ImportEndpointTests(unittest.TestCase):
    def _isolated(self, root):
        return (
            patch.multiple(versioning, API_DIR=root, MODELS_DIR=root / "models",
                           CONFIG_PATH=root / "models" / "config.json", VERSIONS_DIR=root / "models" / "versions"),
            patch.object(model_admin, "_require_admin", return_value="tester@example.com"),
            patch.dict("os.environ", {"SMF_GITHUB_REPO": "owner/ShowMeFire-Models"}, clear=False),
        )

    def test_happy_path_registers_a_beta_and_returns_the_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reg_patch, admin_patch, env_patch = self._isolated(root)
            with reg_patch, admin_patch, env_patch:
                with patch.object(model_admin.import_model, "import_release", return_value="0.0.1-beta.1") as fake_import:
                    response = _run(model_admin.import_model_release(
                        "fuel_moisture", model_admin.ImportRequest(tag="fuel_moisture-v0.0.1-beta.1")))
                self.assertTrue(response["success"])
                self.assertEqual(response["version"], "0.0.1-beta.1")
                fake_import.assert_called_once()
                args, kwargs = fake_import.call_args
                self.assertEqual(args[0], "fuel_moisture")
                self.assertEqual(args[1], "fuel_moisture-v0.0.1-beta.1")
                self.assertEqual(args[2], "owner/ShowMeFire-Models")
                self.assertEqual(kwargs["timeout"], model_admin.IMPORT_TIMEOUT_SECONDS)

    def test_fire_weather_ml_and_fire_weather_index_are_importable_despite_being_guarded_shadow_families(self):
        # Regression test: these two are in GUARDED_SHADOW_TYPES (shown as
        # "Guarded Shadow" in the website dropdown) but DO have a
        # training-side GitHub-release path and are accepted by
        # pipelines/import_model.py - the endpoint must gate on
        # import_model.IMPORTABLE_MODEL_TYPES, not REGISTRY_MODEL_TYPES,
        # or these two become importable via CLI but not via the website.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reg_patch, admin_patch, env_patch = self._isolated(root)
            with reg_patch, admin_patch, env_patch:
                for family in ("fire_weather_ml", "fire_weather_index"):
                    with patch.object(model_admin.import_model, "import_release", return_value="0.0.1-beta.1"):
                        response = _run(model_admin.import_model_release(
                            family, model_admin.ImportRequest(tag=f"{family}-v0.0.1-beta.1")))
                    self.assertTrue(response["success"], msg=family)

    def test_unknown_family_returns_400(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reg_patch, admin_patch, env_patch = self._isolated(root)
            with reg_patch, admin_patch, env_patch:
                from fastapi import HTTPException
                with self.assertRaises(HTTPException) as cm:
                    _run(model_admin.import_model_release(
                        "not_a_real_family", model_admin.ImportRequest(tag="x-v1")))
                self.assertEqual(cm.exception.status_code, 400)

    def test_repo_not_in_allowlist_returns_403(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reg_patch, admin_patch, env_patch = self._isolated(root)
            with reg_patch, admin_patch, env_patch:
                from fastapi import HTTPException
                with self.assertRaises(HTTPException) as cm:
                    _run(model_admin.import_model_release(
                        "fuel_moisture", model_admin.ImportRequest(tag="x-v1", repo="attacker/evil-repo")))
                self.assertEqual(cm.exception.status_code, 403)

    def test_missing_tag_returns_400(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reg_patch, admin_patch, env_patch = self._isolated(root)
            with reg_patch, admin_patch, env_patch:
                from fastapi import HTTPException
                with self.assertRaises(HTTPException) as cm:
                    _run(model_admin.import_model_release(
                        "fuel_moisture", model_admin.ImportRequest(tag="   ")))
                self.assertEqual(cm.exception.status_code, 400)

    def test_timeout_returns_504(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reg_patch, admin_patch, env_patch = self._isolated(root)
            with reg_patch, admin_patch, env_patch:
                with patch.object(model_admin.import_model, "import_release",
                                  side_effect=subprocess.TimeoutExpired(cmd="gh", timeout=300)):
                    from fastapi import HTTPException
                    with self.assertRaises(HTTPException) as cm:
                        _run(model_admin.import_model_release(
                            "fuel_moisture", model_admin.ImportRequest(tag="x-v1")))
                    self.assertEqual(cm.exception.status_code, 504)

    def test_validation_failure_returns_422(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reg_patch, admin_patch, env_patch = self._isolated(root)
            with reg_patch, admin_patch, env_patch:
                with patch.object(model_admin.import_model, "import_release",
                                  side_effect=ImportValidationError("checksum mismatch")):
                    from fastapi import HTTPException
                    with self.assertRaises(HTTPException) as cm:
                        _run(model_admin.import_model_release(
                            "fuel_moisture", model_admin.ImportRequest(tag="x-v1")))
                    self.assertEqual(cm.exception.status_code, 422)

    def test_invalid_bump_returns_400(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reg_patch, admin_patch, env_patch = self._isolated(root)
            with reg_patch, admin_patch, env_patch:
                from fastapi import HTTPException
                with self.assertRaises(HTTPException) as cm:
                    _run(model_admin.import_model_release(
                        "fuel_moisture", model_admin.ImportRequest(tag="x-v1", bump="huge")))
                self.assertEqual(cm.exception.status_code, 400)

    def test_explicit_repo_override_must_still_be_allowlisted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reg_patch, admin_patch, env_patch = self._isolated(root)
            with reg_patch, admin_patch, env_patch:
                with patch.object(model_admin.import_model, "import_release", return_value="0.0.1-beta.1") as fake_import:
                    response = _run(model_admin.import_model_release(
                        "fuel_moisture", model_admin.ImportRequest(tag="x-v1", repo="owner/ShowMeFire-Models")))
                self.assertTrue(response["success"])
                fake_import.assert_called_once()


if __name__ == "__main__":
    unittest.main()
