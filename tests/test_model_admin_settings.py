"""Covers Phase 3's website enable/disable toggle: GET/POST
/api/admin/models/{family}/settings, and their merge into GET /status's
guarded_shadows response. Calls the router's async endpoint functions
directly (no HTTP layer), same pattern as test_model_admin_activate.py."""
import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import core.database as database
import routers.model_admin as model_admin


def _run(coro):
    return asyncio.run(coro)


class ModelAdminSettingsTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = Path(self._tmpdir.name) / "test.db"
        self._db_patcher = patch.object(database, "get_db_path", return_value=self._db_path)
        self._db_patcher.start()
        database.init_database()
        self._admin_patcher = patch.object(model_admin, "_require_admin", return_value="tester@example.com")
        self._admin_patcher.start()

    def tearDown(self):
        self._admin_patcher.stop()
        self._db_patcher.stop()
        self._tmpdir.cleanup()

    def test_get_settings_returns_the_seeded_row(self):
        result = _run(model_admin.get_family_settings("v5"))
        self.assertEqual(result["family"], "v5")
        self.assertIn("enabled", result)

    def test_get_settings_404s_for_a_non_shadow_family(self):
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as cm:
            _run(model_admin.get_family_settings("fuel_moisture"))
        self.assertEqual(cm.exception.status_code, 404)

    def test_post_settings_updates_and_records_the_admin_email(self):
        result = _run(model_admin.set_family_settings("v5", model_admin.ShadowSettingsRequest(enabled=True)))
        self.assertTrue(result["success"])
        self.assertTrue(result["setting"]["enabled"])
        self.assertEqual(result["setting"]["updated_by"], "tester@example.com")

        # Confirm it actually persisted, not just echoed back.
        refetched = _run(model_admin.get_family_settings("v5"))
        self.assertTrue(refetched["enabled"])

    def test_post_settings_404s_for_a_non_shadow_family(self):
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as cm:
            _run(model_admin.set_family_settings("fire_behavior_static", model_admin.ShadowSettingsRequest(enabled=True)))
        self.assertEqual(cm.exception.status_code, 404)

    def test_status_endpoint_merges_requested_enabled_into_guarded_shadows(self):
        _run(model_admin.set_family_settings("fire_weather_index", model_admin.ShadowSettingsRequest(enabled=True)))
        status = _run(model_admin.get_model_status())
        fwi = status["guarded_shadows"]["fire_weather_index"]
        self.assertTrue(fwi["requested_enabled"])
        self.assertEqual(fwi["settings_updated_by"], "tester@example.com")

    def test_status_endpoint_does_not_crash_for_risk_fusion_which_has_no_settings_row(self):
        # risk_fusion (Phase A) is in guarded_shadows but not in
        # shadow_bundles.SHADOW_FAMILIES - must be skipped gracefully, not
        # raise a KeyError when merging settings.
        status = _run(model_admin.get_model_status())
        self.assertIn("risk_fusion", status["guarded_shadows"])
        self.assertNotIn("requested_enabled", status["guarded_shadows"]["risk_fusion"])


if __name__ == "__main__":
    unittest.main()
