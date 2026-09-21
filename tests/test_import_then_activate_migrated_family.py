"""Regression test for a real bug found in manual testing: importing a
migrated shadow family (fire_weather_index/fire_weather_ml) via
POST /{family}/import reported success, but the version never showed up in
GET /{family}/versions, and clicking Activate on it would have 404'd.

Root cause: GET /versions returned shadow_bundles.list_versions() alone for
every guarded-shadow family, but import_model_release() never touches
shadow_bundles - it registers straight into the unified registry, which
carries no shadow_bundle_version metadata for an imported candidate (that
field only exists for zip-uploaded bundles). The activate dispatch
(_registry_action_for_shadow_bundle) also only matched on
metadata["shadow_bundle_version"], so an imported version had no way to
match at all.
"""
import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models import versioning
from models import shadow_bundles
import routers.model_admin as model_admin


def _run(coro):
    return asyncio.run(coro)


class ImportThenActivateTests(unittest.TestCase):
    def _isolated(self, root):
        return (
            patch.multiple(versioning, API_DIR=root, MODELS_DIR=root / "models",
                           CONFIG_PATH=root / "models" / "config.json", VERSIONS_DIR=root / "models" / "versions"),
            patch.object(shadow_bundles, "BUNDLES_ROOT", root / "model-bundles"),
            patch.object(model_admin, "_require_admin", return_value="tester@example.com"),
            patch.object(model_admin, "_require_confirmation", return_value=None),
            patch.dict("os.environ", {"SMF_GITHUB_REPO": "owner/ShowMeFire-Models"}, clear=False),
        )

    def test_imported_version_shows_up_and_can_be_activated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reg_patch, bundles_patch, admin_patch, confirm_patch, env_patch = self._isolated(root)
            with reg_patch, bundles_patch, admin_patch, confirm_patch, env_patch:
                asset_path = root / "factor_weights.json"
                asset_path.write_text("{}")

                def fake_import_release(model_type, tag, repo, bump="patch", timeout=None):
                    return versioning.register_trained_model(
                        model_type, channel="beta", assets={"factor_weights": str(asset_path)},
                        metadata={"advisory_only": True, "model_family": "fire_weather_index"},
                        performance={"source_release_tag": tag},
                    )

                with patch.object(model_admin.import_model, "import_release", side_effect=fake_import_release):
                    response = _run(model_admin.import_model_release(
                        "fire_weather_index",
                        model_admin.ImportRequest(tag="fire_weather_index-v0.0.1-beta.7")))
                self.assertTrue(response["success"])
                imported_version = response["version"]

                # Bug 1: the imported version must show up in /versions.
                listing = _run(model_admin.list_family_versions("fire_weather_index"))
                versions_seen = {v["version"] for v in listing["versions"]}
                self.assertIn(imported_version, versions_seen,
                              msg="imported beta is invisible in the version list - this was the reported bug")

                # Bug 2: Activate must work on that same version, not 404.
                activate_response = _run(model_admin.activate_family_version(
                    "fire_weather_index", model_admin.ActivateRequest(version=imported_version)))
                self.assertTrue(activate_response["success"])

                entry = versioning.get_model_entry("fire_weather_index")
                self.assertIsNotNone(entry["stable"])
                # promote() strips the "-beta.N" suffix for the clean stable
                # version - "0.0.1-beta.1" (beta) legitimately becomes "0.0.1"
                # (stable), same as every other registry family.
                self.assertEqual(entry["stable"]["version"], "0.0.1")
                self.assertIsNone(entry["beta"])

                # After activation, /versions should report it as active/stable.
                listing_after = _run(model_admin.list_family_versions("fire_weather_index"))
                self.assertEqual(listing_after["active"]["version"], "0.0.1")


if __name__ == "__main__":
    unittest.main()
