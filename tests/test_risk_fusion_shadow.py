import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core.fire_danger import RULE_SPEC
from services import risk_fusion_shadow as rfs


class RiskFusionShadowTests(unittest.TestCase):
    def setUp(self):
        # diagnostics() recomputes _state["enabled"] from _requested() (the
        # env var) on every call - matching v5_shadow's cron-process reread
        # pattern - so tests must patch _requested rather than only setting
        # _state["enabled"] directly, or diagnostics() silently overwrites it.
        self.requested_patch = patch.object(rfs, "_requested", return_value=True)
        self.requested_patch.start()

        # STATE_PATH/EVIDENCE_ROOT are process-global and diagnostics()
        # persists to *and* rereads from that real path on every call - left
        # unpatched, one test's persisted shadow-state.json leaks into the
        # next test (and into the repo's real api/data/ directory). Isolate
        # every test behind its own temp directory.
        self.state_dir = tempfile.TemporaryDirectory()
        self.state_root_patch = patch.object(rfs, "EVIDENCE_ROOT", Path(self.state_dir.name))
        self.state_path_patch = patch.object(rfs, "STATE_PATH", Path(self.state_dir.name) / "shadow-state.json")
        self.state_root_patch.start()
        self.state_path_patch.start()

        rfs._state.update(
            enabled=True, consecutive_failures=0, last_error=None, runs=0, healthy=True,
            auto_disabled=False, counties_scored=0, county_days_recorded=0,
            rule_mc_parity_status="ok",
        )
        self.thresholds = RULE_SPEC["thresholds"]

    def tearDown(self):
        self.state_path_patch.stop()
        self.state_root_patch.stop()
        self.state_dir.cleanup()
        self.requested_patch.stop()

    def _record(self, evidence_root, run_id="run1", n_counties=2):
        return rfs.record_rule_mc(
            run_id=run_id,
            county_fips=[f"2901{i}" for i in range(n_counties)],
            valid_local_date="2026-07-01",
            fm=np.full(n_counties, 6.0),
            rh=np.full(n_counties, 20.0),
            wind_kts=np.full(n_counties, 15.0),
            fm_sigma=np.full(n_counties, 2.0),
            rh_sigma=np.full(n_counties, 5.0),
            wind_sigma_log=np.full(n_counties, 0.3),
            thresholds=self.thresholds,
            n_draws=200,
            evidence_root=evidence_root,
        )

    def test_disabled_by_default_without_env_flag(self):
        with patch.object(rfs, "_requested", return_value=False):
            self.assertFalse(rfs.diagnostics()["enabled"])

    def test_writes_immutable_evidence_and_second_write_fails_without_corrupting_first(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory)
            self.assertTrue(self._record(evidence))
            written = (evidence / "run1.rule_mc.json").read_text()
            record = json.loads(written)
            self.assertEqual(record["county_fips"], ["29010", "29011"])
            self.assertEqual(len(record["rule_category_stability"]), 2)

            # Re-recording the same run_id must fail (immutable open("x"))
            # without corrupting the original evidence file.
            self.assertFalse(self._record(evidence))
            self.assertEqual((evidence / "run1.rule_mc.json").read_text(), written)

    def test_row_alignment_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            result = rfs.record_rule_mc(
                run_id="run2", county_fips=["29010", "29011"], valid_local_date="2026-07-01",
                fm=np.array([6.0, 6.0]), rh=np.array([20.0]),  # mismatched length
                wind_kts=np.array([15.0, 15.0]), fm_sigma=np.array([2.0, 2.0]),
                rh_sigma=np.array([5.0, 5.0]), wind_sigma_log=np.array([0.3, 0.3]),
                thresholds=self.thresholds, n_draws=100, evidence_root=Path(directory),
            )
            self.assertFalse(result)
            self.assertFalse((Path(directory) / "run2.rule_mc.json").exists())

    def test_parity_failure_blocks_recording(self):
        rfs._state["rule_mc_parity_status"] = "not_yet_checked"
        with patch.object(rfs, "check_rule_mc_parity", return_value=False):
            with tempfile.TemporaryDirectory() as directory:
                result = self._record(Path(directory))
        self.assertFalse(result)
        self.assertIn("parity", rfs._state["last_error"])

    def test_repeated_failure_disables_only_risk_fusion_shadow(self):
        for index in range(rfs.MAX_FAILURES):
            result = rfs.record_rule_mc(
                run_id=f"fail{index}", county_fips=["29010"], valid_local_date="2026-07-01",
                fm=np.array([6.0]), rh=np.array([]),  # forces the alignment ValueError every time
                wind_kts=np.array([15.0]), fm_sigma=np.array([2.0]),
                rh_sigma=np.array([5.0]), wind_sigma_log=np.array([0.3]),
                thresholds=self.thresholds, evidence_root=Path(tempfile.mkdtemp()),
            )
            self.assertFalse(result)
        self.assertFalse(rfs.diagnostics()["enabled"])
        self.assertTrue(rfs._state["auto_disabled"])

    def test_disabled_shadow_never_writes_evidence(self):
        rfs._state["enabled"] = False
        with patch.object(rfs, "_requested", return_value=False):
            with tempfile.TemporaryDirectory() as directory:
                result = self._record(Path(directory))
                self.assertFalse(result)
                self.assertEqual(list(Path(directory).glob("*")), [])

    def test_record_skipped_run_marks_unhealthy_without_writing_files(self):
        result = rfs.record_skipped_run("no fresh forecast available")
        self.assertTrue(result)
        self.assertFalse(rfs._state["healthy"])
        self.assertEqual(rfs._state["last_error"], "no fresh forecast available")

    def test_check_rule_mc_parity_passes_against_real_rule(self):
        self.assertTrue(rfs.check_rule_mc_parity())
        self.assertEqual(rfs._state["rule_mc_parity_status"], "ok")

    def test_probability_fields_present_and_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory)
            self._record(evidence)
            record = json.loads((evidence / "run1.rule_mc.json").read_text())
            for field in ("rule_probability_at_or_above_elevated", "rule_probability_at_or_above_critical",
                         "rule_probability_at_or_above_extreme"):
                for value in record[field]:
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
