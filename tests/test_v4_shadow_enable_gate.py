"""v4 previously had no _configured()/_requested()/ENABLED_ENV at all - a
bare always-True _state["enabled"] flipped off only by consecutive
failures. This pins the new gate added to bring it in line with the other
four shadow modules, without changing record_predictions()'s own guard
(still the raw _state["enabled"] flag - v4 has no live production caller
today, confirmed via grep, so nothing calls diagnostics() before
record_predictions the way v5's score_and_record does)."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import core.database as database
from services import v4_shadow


class V4ShadowEnableGateTests(unittest.TestCase):
    def setUp(self):
        v4_shadow._state.update(consecutive_failures=0, last_error=None, runs=0, unavailable=0)

    def _isolated_db(self):
        tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(tmpdir.name) / "test.db"
        patcher = patch.object(database, "get_db_path", return_value=db_path)
        patcher.start()
        database.init_database()
        return tmpdir, patcher

    def test_default_is_enabled_with_no_env_var_and_no_db_row(self):
        with patch.dict("os.environ", {}, clear=False):
            # No DB patched - get_shadow_model_setting will fail to find the
            # real table/row and fall back to the env var default ("true").
            self.assertTrue(v4_shadow._requested())

    def test_env_var_can_disable_when_no_db_row_exists(self):
        with patch.dict("os.environ", {v4_shadow.ENABLED_ENV: "false"}, clear=False):
            self.assertFalse(v4_shadow._requested())

    def test_db_row_overrides_env_var(self):
        tmpdir, patcher = self._isolated_db()
        try:
            database.set_shadow_model_setting("v4", False, updated_by="tester")
            with patch.dict("os.environ", {v4_shadow.ENABLED_ENV: "true"}, clear=False):
                self.assertFalse(v4_shadow._requested())  # DB row (False) wins over env var (true)

            database.set_shadow_model_setting("v4", True, updated_by="tester")
            self.assertTrue(v4_shadow._requested())
        finally:
            patcher.stop()
            tmpdir.cleanup()

    def test_diagnostics_reflects_configured_requested_and_auto_disabled(self):
        with patch.object(v4_shadow, "_configured", return_value=True), \
             patch.object(v4_shadow, "_requested", return_value=True):
            diagnostics = v4_shadow.diagnostics()
            self.assertTrue(diagnostics["configured"])
            self.assertTrue(diagnostics["requested"])
            self.assertFalse(diagnostics["auto_disabled"])
            self.assertTrue(diagnostics["enabled"])

    def test_diagnostics_reports_disabled_when_not_configured(self):
        with patch.object(v4_shadow, "_configured", return_value=False), \
             patch.object(v4_shadow, "_requested", return_value=True):
            self.assertFalse(v4_shadow.diagnostics()["enabled"])

    def test_max_failures_auto_disable_still_works(self):
        v4_shadow._state.update(consecutive_failures=v4_shadow.MAX_FAILURES, enabled=False)
        with patch.object(v4_shadow, "_configured", return_value=True), \
             patch.object(v4_shadow, "_requested", return_value=True):
            diagnostics = v4_shadow.diagnostics()
            self.assertTrue(diagnostics["auto_disabled"])
            self.assertFalse(diagnostics["enabled"])

    def test_record_predictions_guard_is_unaffected_by_the_new_gate(self):
        # record_predictions still checks the raw _state["enabled"] flag
        # directly, not diagnostics()["enabled"] - manually setting it True
        # (as existing tests already do) must still let a call through even
        # when _configured()/_requested() would say otherwise.
        v4_shadow._state.update(enabled=True)
        with patch.object(v4_shadow, "_configured", return_value=False):
            with tempfile.TemporaryDirectory() as directory:
                # A bad bundle_dir still fails for its own reason (missing
                # bundle), but it must fail INSIDE the try block (consuming
                # a failure count), not be short-circuited by the new gate.
                result = v4_shadow.record_predictions(
                    "run", ["row"], [9], [[5, 6, 7, 8, 9, 10, 11]], [20], [15], [.8], [1],
                    Path(directory) / "missing", evidence_root=Path(directory) / "evidence")
                self.assertFalse(result)
                self.assertEqual(v4_shadow._state["consecutive_failures"], 1)


if __name__ == "__main__":
    unittest.main()
