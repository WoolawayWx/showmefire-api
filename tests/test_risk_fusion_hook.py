import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from services import risk_fusion_glm_shadow as rfgs
from services import risk_fusion_hook as hook
from services import risk_fusion_shadow as rfs


FAKE_CELLS = {
    "grid_shape": [2, 2],
    "cell_to_fips": {"0,0": "29019", "0,1": "29019", "1,0": "29027", "1,1": "29027"},
}


class RiskFusionHookTests(unittest.TestCase):
    def setUp(self):
        self.requested_patch = patch.object(rfs, "_requested", return_value=True)
        self.requested_patch.start()
        self.state_dir = tempfile.TemporaryDirectory()
        self.state_root_patch = patch.object(rfs, "EVIDENCE_ROOT", Path(self.state_dir.name))
        self.state_path_patch = patch.object(rfs, "STATE_PATH", Path(self.state_dir.name) / "shadow-state.json")
        self.state_root_patch.start()
        self.state_path_patch.start()
        rfs._state.update(enabled=True, consecutive_failures=0, last_error=None, runs=0,
                          healthy=True, auto_disabled=False, rule_mc_parity_status="ok")

        self.cells_patch = patch.object(hook, "county_cells", return_value=FAKE_CELLS)
        self.cells_patch.start()

    def tearDown(self):
        self.cells_patch.stop()
        self.state_path_patch.stop()
        self.state_root_patch.stop()
        self.state_dir.cleanup()
        self.requested_patch.stop()

    def _grids(self, n_hours=12, fm=6.0, rh=20.0, ws_kts=15.0):
        shape = (2, 2)
        return (
            [np.full(shape, fm) for _ in range(n_hours)],
            [np.full(shape, rh) for _ in range(n_hours)],
            [np.full(shape, ws_kts) for _ in range(n_hours)],
        )

    def test_records_shadow_evidence_for_matching_grid_shape(self):
        fm, rh, ws = self._grids()
        result = hook.run_shadow_for_forecast(fm, rh, ws, run_id="run1", valid_local_date="2026-07-01")
        self.assertTrue(result)
        evidence_path = Path(self.state_dir.name) / "run1.rule_mc.json"
        self.assertTrue(evidence_path.exists())

    def test_grid_shape_mismatch_records_skip_not_garbage(self):
        fm = [np.full((5, 5), 6.0) for _ in range(12)]  # wrong shape vs FAKE_CELLS' (2,2)
        rh = [np.full((5, 5), 20.0) for _ in range(12)]
        ws = [np.full((5, 5), 15.0) for _ in range(12)]
        result = hook.run_shadow_for_forecast(fm, rh, ws, run_id="run2", valid_local_date="2026-07-01")
        self.assertFalse(result)
        self.assertIn("grid shape mismatch", rfs._state["last_error"])
        self.assertFalse((Path(self.state_dir.name) / "run2.rule_mc.json").exists())

    def test_disabled_shadow_does_nothing(self):
        with patch.object(rfs, "_requested", return_value=False):
            fm, rh, ws = self._grids()
            result = hook.run_shadow_for_forecast(fm, rh, ws, run_id="run3", valid_local_date="2026-07-01")
            self.assertFalse(result)

    def test_empty_hourly_arrays_is_handled_gracefully(self):
        result = hook.run_shadow_for_forecast([], [], [], run_id="run4", valid_local_date="2026-07-01")
        self.assertFalse(result)

    def test_scores_every_county_in_the_cell_index(self):
        fm, rh, ws = self._grids()
        hook.run_shadow_for_forecast(fm, rh, ws, run_id="run5", valid_local_date="2026-07-01")
        import json
        record = json.loads((Path(self.state_dir.name) / "run5.rule_mc.json").read_text())
        self.assertEqual(sorted(record["county_fips"]), ["29019", "29027"])

    def test_extreme_conditions_produce_high_probability_at_or_above_extreme(self):
        # fm=3, rh=8, wind=40kts is deep in EXTREME territory (thresholds
        # fm<7, rh<20, wind>=25) with enough margin that the hook's fallback
        # sigmas (rh_sigma=7, wind lognormal sigma=0.3) don't wash out an
        # unambiguous signal - verified numerically at ~0.82 for this input.
        fm, rh, ws = self._grids(fm=3.0, rh=8.0, ws_kts=40.0)
        hook.run_shadow_for_forecast(fm, rh, ws, run_id="run6", valid_local_date="2026-07-01")
        import json
        record = json.loads((Path(self.state_dir.name) / "run6.rule_mc.json").read_text())
        for prob in record["rule_probability_at_or_above_extreme"]:
            self.assertGreater(prob, 0.5)

    def test_internal_exception_never_propagates(self):
        with patch.object(hook, "_reduce", side_effect=RuntimeError("boom")):
            fm, rh, ws = self._grids()
            result = hook.run_shadow_for_forecast(fm, rh, ws, run_id="run7", valid_local_date="2026-07-01")
        self.assertFalse(result)


class RunGlmShadowForForecastTests(unittest.TestCase):
    def setUp(self):
        from tests.test_risk_fusion_glm_shadow import _write_bundle
        self.bundle_dir = tempfile.TemporaryDirectory()
        _write_bundle(Path(self.bundle_dir.name), county_fips=("29019", "29027"))
        self.bundle_env_patch = patch.dict(os.environ, {rfgs.BUNDLE_ENV: self.bundle_dir.name})
        self.bundle_env_patch.start()

        self.requested_patch = patch.object(rfgs, "_requested", return_value=True)
        self.requested_patch.start()

        self.state_dir = tempfile.TemporaryDirectory()
        self.evidence_root_patch = patch.object(rfgs, "EVIDENCE_ROOT", Path(self.state_dir.name))
        self.state_path_patch = patch.object(rfgs, "STATE_PATH", Path(self.state_dir.name) / "shadow-state.json")
        self.evidence_root_patch.start()
        self.state_path_patch.start()
        rfgs._state.update(enabled=True, consecutive_failures=0, last_error=None, runs=0, successful_runs=0,
                           healthy=True, auto_disabled=False, counties_scored=0, county_days_recorded=0)

        self.cells_patch = patch.object(hook, "county_cells", return_value=FAKE_CELLS)
        self.cells_patch.start()

    def tearDown(self):
        self.cells_patch.stop()
        self.state_path_patch.stop()
        self.evidence_root_patch.stop()
        self.requested_patch.stop()
        self.bundle_env_patch.stop()
        self.bundle_dir.cleanup()
        self.state_dir.cleanup()

    def _all_grids(self, n_hours=12, fm=6.0, rh=40.0, ws_kts=15.0, temp_c=30.0, precip_mm=0.0):
        shape = (2, 2)
        return (
            [np.full(shape, fm) for _ in range(n_hours)],
            [np.full(shape, rh) for _ in range(n_hours)],
            [np.full(shape, ws_kts) for _ in range(n_hours)],
            [np.full(shape, temp_c) for _ in range(n_hours)],
            [np.full(shape, precip_mm) for _ in range(n_hours)],
        )

    def test_records_glm_evidence_for_matching_grid_shape(self):
        fm, rh, ws, temp, precip = self._all_grids()
        result = hook.run_glm_shadow_for_forecast(
            fm, rh, ws, temp, precip, run_id="grun1", valid_local_date="2026-07-01")
        self.assertTrue(result)
        evidence_path = Path(self.state_dir.name) / "grun1.glm_score.json"
        self.assertTrue(evidence_path.exists())
        record = json.loads(evidence_path.read_text())
        self.assertEqual(sorted(record["county_fips"]), ["29019", "29027"])
        self.assertEqual(len(record["lam"]), 2)

    def test_grid_shape_mismatch_records_skip_not_garbage(self):
        fm = [np.full((5, 5), 6.0) for _ in range(12)]
        rh = [np.full((5, 5), 40.0) for _ in range(12)]
        ws = [np.full((5, 5), 15.0) for _ in range(12)]
        temp = [np.full((5, 5), 30.0) for _ in range(12)]
        precip = [np.full((5, 5), 0.0) for _ in range(12)]
        result = hook.run_glm_shadow_for_forecast(
            fm, rh, ws, temp, precip, run_id="grun2", valid_local_date="2026-07-01")
        self.assertFalse(result)
        self.assertIn("grid shape mismatch", rfgs._state["last_error"])

    def test_disabled_shadow_does_nothing(self):
        with patch.object(rfgs, "_requested", return_value=False):
            fm, rh, ws, temp, precip = self._all_grids()
            result = hook.run_glm_shadow_for_forecast(
                fm, rh, ws, temp, precip, run_id="grun3", valid_local_date="2026-07-01")
            self.assertFalse(result)

    def test_empty_hourly_arrays_is_handled_gracefully(self):
        result = hook.run_glm_shadow_for_forecast(
            [], [], [], [], [], run_id="grun4", valid_local_date="2026-07-01")
        self.assertFalse(result)

    def test_internal_exception_never_propagates(self):
        with patch.object(hook, "_reduce", side_effect=RuntimeError("boom")):
            fm, rh, ws, temp, precip = self._all_grids()
            result = hook.run_glm_shadow_for_forecast(
                fm, rh, ws, temp, precip, run_id="grun5", valid_local_date="2026-07-01")
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
