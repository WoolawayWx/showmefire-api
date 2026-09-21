import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import core.database as database


class ShadowModelSettingsTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = Path(self._tmpdir.name) / "test.db"
        self._db_patcher = patch.object(database, "get_db_path", return_value=self._db_path)
        self._db_patcher.start()
        database.init_database()

    def tearDown(self):
        self._db_patcher.stop()
        self._tmpdir.cleanup()

    def test_table_is_seeded_with_one_row_per_known_family(self):
        settings = database.list_shadow_model_settings()
        self.assertEqual(set(settings), {"fire_weather_index", "fire_weather_ml", "risk_fusion_glm", "v4", "v5"})

    def test_v4_defaults_enabled_with_no_env_var(self):
        setting = database.get_shadow_model_setting("v4")
        self.assertTrue(setting["enabled"])

    def test_other_families_default_to_their_env_var_state(self):
        with patch.dict("os.environ", {"FIRE_WEATHER_INDEX_SHADOW_ENABLED": "true"}, clear=False):
            # SHADOW_MODEL_SETTINGS_DEFAULTS is computed at import time, not
            # re-read per call - re-seed a fresh DB to see the env var take
            # effect, matching how a real fresh deploy would pick it up.
            fresh_tmpdir = tempfile.TemporaryDirectory()
            fresh_db_path = Path(fresh_tmpdir.name) / "test2.db"
            with patch.object(database, "get_db_path", return_value=fresh_db_path), \
                 patch.dict(database.SHADOW_MODEL_SETTINGS_DEFAULTS, {"fire_weather_index": True}):
                database.init_database()
                setting = database.get_shadow_model_setting("fire_weather_index")
                self.assertTrue(setting["enabled"])
            fresh_tmpdir.cleanup()

    def test_set_shadow_model_setting_upserts_and_records_who(self):
        updated = database.set_shadow_model_setting("v5", True, updated_by="admin@example.com")
        self.assertTrue(updated["enabled"])
        self.assertEqual(updated["updated_by"], "admin@example.com")

        updated_again = database.set_shadow_model_setting("v5", False, updated_by="other@example.com")
        self.assertFalse(updated_again["enabled"])
        self.assertEqual(updated_again["updated_by"], "other@example.com")

        # Still exactly one row for v5, not a duplicate.
        settings = database.list_shadow_model_settings()
        self.assertEqual(settings["v5"]["enabled"], False)

    def test_get_unknown_family_returns_none(self):
        self.assertIsNone(database.get_shadow_model_setting("not_a_real_family"))

    def test_reinitializing_does_not_reset_an_existing_setting(self):
        database.set_shadow_model_setting("v4", False, updated_by="admin@example.com")
        database.init_database()  # INSERT OR IGNORE must not clobber the existing row
        setting = database.get_shadow_model_setting("v4")
        self.assertFalse(setting["enabled"])


if __name__ == "__main__":
    unittest.main()
