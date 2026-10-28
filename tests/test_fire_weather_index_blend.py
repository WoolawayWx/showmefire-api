import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.risk_fusion_county_reference import county_cells
from services import beta_products
from services import fire_weather_index_blend as blend
from services import fire_weather_index_hook as hook
from services import fire_weather_index_shadow as fwis

CELLS = county_cells()
SHAPE = tuple(CELLS["grid_shape"])
MPS_TO_KNOTS = blend.MPS_TO_KNOTS
BUNDLE = {
    "factor_weights": {
        "weights": {"rh": 1.0, "wind": 1.0, "vpd": 1.0, "precip_relief": 1.0},
        "raw_score_ceiling": {"value": 1.0},
        "ramp_anchors": {"rh": {"benign": 60.0, "extreme": 10.0}, "wind": {"benign": 5.0, "extreme": 30.0},
                         "vpd": {"benign": 0.5, "extreme": 4.0}, "precip_relief": {"benign": 0.0, "extreme": 10.0}},
    },
    "category_thresholds": {"thresholds": [0.2, 0.4, 0.6, 0.8]},
    "version": "0.0.1-test", "bundle_checksum": "abc",
}


def _fields(seed: int, hours: int = 12, shift: float = 0.0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "temp_c": 18 + 6 * rng.random((hours, *SHAPE)) + shift,
        "rh": np.clip(55 - 25 * rng.random((hours, *SHAPE)) - shift, 5, 100),
        "wind_ms": 2 + 6 * rng.random((hours, *SHAPE)),
        "precip_mm": np.where(rng.random((hours, *SHAPE)) > 0.97, rng.random((hours, *SHAPE)), 0.0),
    }


def _latlon():
    lat = np.linspace(35.8, 40.7, SHAPE[0])[:, None] * np.ones((1, SHAPE[1]))
    lon = np.linspace(-95.8, -89.1, SHAPE[1])[None, :] * np.ones((SHAPE[0], 1))
    return lat, lon


class ReductionParityTests(unittest.TestCase):
    """The blend must reduce exactly like the HRRR-only hook, or the two maps would differ for the wrong reason."""

    def test_single_source_matches_the_hrrr_only_hook_grids_and_county_rows(self):
        f = _fields(0)
        captured = {}
        lat, lon = _latlon()
        with patch.object(fwis, "diagnostics", return_value={"enabled": True}), \
                patch.object(fwis, "score_for_forecast", side_effect=lambda **kw: captured.update(kw) or True):
            hook.run_fire_weather_index_shadow_for_forecast(
                list(f["rh"]), list(f["wind_ms"] * 0.8 * MPS_TO_KNOTS), list(f["temp_c"]), list(f["precip_mm"]),
                run_id="x", valid_local_date="2026-10-28", lat=lat, lon=lon)
        mine = blend.reduce_member(f, 0.8)
        for key in blend.GRID_KEYS:
            np.testing.assert_allclose(mine[key], captured["weather_grids"][key], rtol=1e-9, atol=1e-9, err_msg=key)
        rows = blend.county_rows(mine, CELLS["cell_to_fips"])
        self.assertEqual(set(rows), set(captured["weather_rows"]))
        for fips, row in rows.items():
            for key, value in row.items():
                self.assertAlmostEqual(value, captured["weather_rows"][fips][key], places=6, msg=f"{fips} {key}")

    def test_lead_offsets_use_the_same_indexing_as_the_hook(self):
        afternoon, full = blend.lead_offsets(12, 4)
        self.assertEqual((afternoon, full), (list(range(0, 6)), list(range(0, 12))))
        self.assertEqual(blend.lead_offsets(3, 4), ([0, 1, 2], [0, 1, 2]))


class BlendMathTests(unittest.TestCase):
    def test_equal_weight_mean_over_finite_sources_and_coverage(self):
        a = {k: np.full((2, 2), 10.0) for k in blend.GRID_KEYS}
        b = {k: np.full((2, 2), 20.0) for k in blend.GRID_KEYS}
        b["rh_min_afternoon"][0, 0] = np.nan  # this source has no value here
        out, cov = blend.blend_grids({"HRRR": a, "RRFS": b})
        self.assertEqual(out["rh_min_afternoon"][0, 0], 10.0)   # only HRRR counts in that cell
        self.assertEqual(out["rh_min_afternoon"][1, 1], 15.0)   # equal-weight mean elsewhere
        self.assertEqual(cov, {"HRRR": 1.0, "RRFS": 0.75})
        both_nan = {k: np.full((1, 1), np.nan) for k in blend.GRID_KEYS}
        out2, _ = blend.blend_grids({"A": both_nan, "B": both_nan})
        self.assertTrue(np.isnan(out2["wind_kts_max"]).all())

    def test_member_choice_prefers_current_rrfs_cycle_and_skips_missing_sources(self):
        plan = [SimpleNamespace(member_id="rrfs_m6", cycle=datetime(2026, 10, 28, 6)),
                SimpleNamespace(member_id="rrfs_0", cycle=datetime(2026, 10, 28, 12))]
        chosen = blend.pick_members({"hrrr_0": {}, "rrfs_m6": {}, "rrfs_0": {}}, plan)
        self.assertEqual(chosen["RRFS"]["member_id"], "rrfs_0")
        self.assertEqual(chosen["RRFS"]["cycle"], "2026-10-28T12:00Z")
        self.assertEqual(blend.pick_members({"hrrr_0": {}, "rrfs_m6": {}})["RRFS"]["member_id"], "rrfs_m6")
        self.assertNotIn("FV3-HIRES", chosen)


class RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.anchor = datetime(2026, 10, 28, 12)
        self.lat, self.lon = _latlon()
        self.members = {"hrrr_0": _fields(1), "rrfs_0": _fields(2, shift=1.5), "fv3_0": _fields(3, shift=-1.0)}
        self.plan = [SimpleNamespace(member_id=m, cycle=self.anchor) for m in self.members]
        self.patches = [patch.object(fwis, "diagnostics", return_value={"enabled": True}),
                        patch.object(fwis, "load_bundle", return_value=BUNDLE),
                        patch.object(beta_products, "BETA_ROOT", self.root / "beta")]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def _run(self, members=None, **kw):
        return blend.run_for_ensemble(self.plan, members or self.members, {m: 0.8 for m in self.members}, self.lat, self.lon,
                                      self.anchor, evidence_root=self.root / "evidence", **kw)

    def test_writes_evidence_image_and_manifest_and_never_overwrites(self):
        out = self._run()
        self.assertTrue(out["ok"], out)
        evidence = self.root / "evidence" / "20261028_12.fire_weather_index_blend_score.json"
        rec = json.loads(evidence.read_text())
        self.assertEqual(rec["valid_local_date"], "2026-10-28")
        self.assertEqual(rec["model"], "fire_weather_index:blend")
        self.assertEqual(rec["sources"]["used_short"], ["HRRR", "RRFS", "FV3"])
        self.assertTrue(rec["sources"]["fv3_t2m_is_tmax_tmin_proxy"])
        self.assertEqual(len(rec["county_fips"]), len(rec["score"]))
        image = Path(out["image"])
        self.assertTrue(image.exists() and image.stat().st_size > 10_000)
        manifest = json.loads((self.root / "beta" / "manifest.json").read_text()) if (self.root / "beta" / "manifest.json").exists() else None
        if manifest is not None:
            self.assertIn("fire_weather_index_blend_shadow", manifest["products"])
        again = self._run()
        self.assertFalse(again["ok"])
        self.assertIn("already recorded", again["reason"])

    def test_blend_scores_differ_from_hrrr_only_because_the_weather_does(self):
        out = self._run()
        hrrr_only = blend.county_rows(blend.reduce_member(self.members["hrrr_0"]), CELLS["cell_to_fips"])
        mixed = blend.county_rows(blend.blend_grids({f: blend.reduce_member(self.members[m]) for f, m in
                                                     (("HRRR", "hrrr_0"), ("RRFS", "rrfs_0"), ("FV3", "fv3_0"))})[0], CELLS["cell_to_fips"])
        self.assertTrue(out["ok"])
        fips = sorted(hrrr_only)[0]
        self.assertNotEqual(round(hrrr_only[fips]["rh_min_afternoon"], 3), round(mixed[fips]["rh_min_afternoon"], 3))

    def test_needs_two_sources_and_respects_the_shadow_switch_and_grid_shape(self):
        only_hrrr = self._run(members={"hrrr_0": self.members["hrrr_0"]})
        self.assertFalse(only_hrrr["ok"])
        self.assertIn("at least two", only_hrrr["reason"])
        self.assertFalse((self.root / "evidence").exists())
        with patch.object(fwis, "diagnostics", return_value={"enabled": False}):
            self.assertIn("not enabled", self._run()["reason"])
        bad = blend.run_for_ensemble(self.plan, self.members, {}, self.lat[:10, :10], self.lon[:10, :10], self.anchor,
                                     evidence_root=self.root / "evidence")
        self.assertIn("grid shape mismatch", bad["reason"])

    def test_neither_success_nor_failure_touches_the_hrrr_only_shadow_health_state(self):
        import copy
        before = copy.deepcopy(fwis._state)
        self.assertTrue(self._run()["ok"])
        with patch.object(fwis, "load_bundle", side_effect=RuntimeError("boom")):
            out = self._run()
        self.assertFalse(out["ok"])
        self.assertIn("boom", out["reason"])
        self.assertEqual(before, fwis._state)  # no run counts, failures or auto-disable were recorded


class TrackerIngestsTheBlendTests(unittest.TestCase):
    def test_blend_evidence_is_tracked_as_its_own_model_in_the_same_family(self):
        from core import database
        from services import fire_danger_tracking as ft
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "t.db"
            with patch.object(database, "get_db_path", return_value=db_path):
                database.init_database()
                d = root / "fwi"
                d.mkdir()
                (d / "20261028_12.fire_weather_index_blend_score.json").write_text(json.dumps({
                    "run_id": "20261028_12", "recorded_at": "2026-10-28T15:05:00Z", "valid_local_date": "2026-10-28",
                    "county_fips": ["29001", "29003"], "score": [0.4, 0.7], "category": [2, 3]}))
                out = ft.ingest_scores(db_path, {"fire_weather_index": d, "risk_fusion_glm": root / "a", "ensemble": root / "b"})
                self.assertEqual(out["added"], {"fire_weather_index:blend": 2})
                self.assertEqual(ft.ingest_scores(db_path, {"fire_weather_index": d, "risk_fusion_glm": root / "a", "ensemble": root / "b"})["added"], {})
                self.assertIn("blend", ft.model_label("fire_weather_index:blend"))
                self.assertEqual("fire_weather_index:blend".split(":", 1)[0], "fire_weather_index")


if __name__ == "__main__":
    unittest.main()
