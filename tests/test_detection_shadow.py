import json
import math
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import database
from services import detection_shadow as shadow


def _bundle(path: Path, supported=("viirs", "modis")) -> Path:
    features = ["frp_log", "n_1km_24h"]
    bundle = {"schema": shadow.BUNDLE_SCHEMA, "model_name": "veg_conf_logreg", "version": "test-1",
              "feature_schema": shadow.features_module().FEATURE_SCHEMA, "features": features,
              "missing_indicators": ["frp_log"], "impute": {"frp_log": 1.0, "n_1km_24h": 0.0},
              "scaler": {"mean": [1.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]}, "coef": [1.0, 0.5, -1.0], "intercept": 0.0,
              "calibrator": {"a": 1.0, "b": 0.0}, "supported_sources": list(supported)}
    path.write_text(json.dumps(bundle), encoding="utf-8")
    return path


class ShadowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "showmefire.db"
        self.patch = patch.object(database, "get_db_path", return_value=self.db_path)
        self.patch.start()
        database.init_database()
        self.now = datetime.now(timezone.utc)

    def tearDown(self):
        self.patch.stop()
        shadow._bundle_cache.clear()
        self.tmp.cleanup()

    def _event(self, ext, source="viirs", minutes_ago=30, lat=38.0, lon=-92.0, pct=None):
        when = (self.now - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")
        result = database.upsert_detection_event(
            source=source, external_id=ext, latitude=lat, longitude=lon, occurred_at=when, frp=8.0, confidence="n",
            bright_t7=330.0, bright_t13=295.0, pixel_area=0.14, daynight="D", land_cover="Cropland:50,Trees:50",
            cluster_incidents=False)
        conn = sqlite3.connect(self.db_path)
        conn.execute("UPDATE fire_events SET first_seen_at = ?, detection_confidence_pct = ? WHERE id = ?", (when, pct, result["event_id"]))
        conn.commit()
        conn.close()
        return result["event_id"]

    def _rows(self, sql, *args):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in conn.execute(sql, args).fetchall()]
        finally:
            conn.close()

    def test_v1_recorded_and_live_score_untouched(self):
        scored = self._event("a", pct=61.0)
        unscored = self._event("b")  # e.g. a masked detection the live pass never scores
        with patch.object(shadow, "bundle_path", return_value=Path(self.tmp.name) / "none.json"):
            out = shadow.run_shadow_pass(now=self.now, db_path=self.db_path)
        self.assertEqual(out["initial"]["snapshots"], 2)
        hist = {r["event_id"]: r for r in self._rows("SELECT * FROM detection_score_history")}
        self.assertAlmostEqual(hist[scored]["score"], 0.61)
        self.assertIn(unscored, hist)  # v1 computed for the shadow record only
        self.assertEqual(hist[scored]["model_name"], shadow.V1_NAME)
        live = {r["id"]: r["detection_confidence_pct"] for r in self._rows("SELECT id, detection_confidence_pct FROM fire_events")}
        self.assertEqual(live[scored], 61.0)
        self.assertIsNone(live[unscored])  # shadow never writes the live score

    def test_v2_only_for_supported_sensors_and_degrades_without_bundle(self):
        v = self._event("v", "viirs", pct=50.0)
        n = self._event("n", "ngfs", pct=50.0)
        model = _bundle(Path(self.tmp.name) / "m.json")
        with patch.object(shadow, "bundle_path", return_value=model):
            shadow.run_shadow_pass(now=self.now, db_path=self.db_path)
        v2 = {r["event_id"] for r in self._rows("SELECT event_id FROM detection_score_history WHERE model_name = 'veg_conf_logreg'")}
        self.assertEqual(v2, {v})
        self.assertNotIn(n, v2)

    def test_second_pass_adds_nothing_and_updated_pass_appends(self):
        eid = self._event("a", pct=50.0, minutes_ago=30)
        model = _bundle(Path(self.tmp.name) / "m.json")
        with patch.object(shadow, "bundle_path", return_value=model):
            shadow.run_shadow_pass(now=self.now, db_path=self.db_path)
            again = shadow.run_shadow_pass(now=self.now, db_path=self.db_path)
            self.assertEqual(again["initial"]["snapshots"], 0)
            self.assertEqual(again["updated"]["snapshots"], 0)  # not old enough yet
            later = self.now + timedelta(hours=7)
            out = shadow.run_shadow_pass(now=later, db_path=self.db_path)
        self.assertEqual(out["updated"]["snapshots"], 1)
        modes = [r["mode"] for r in self._rows("SELECT mode FROM detection_feature_snapshot WHERE event_id = ? ORDER BY id", eid)]
        self.assertEqual(modes, ["initial", "updated"])
        v2 = self._rows("SELECT mode FROM detection_score_history WHERE event_id = ? AND model_name = 'veg_conf_logreg' ORDER BY id", eid)
        self.assertEqual([r["mode"] for r in v2], ["initial", "updated"])  # initial row kept, never overwritten

    def test_initial_snapshot_ignores_detections_that_arrive_later(self):
        a = self._event("a", minutes_ago=60, pct=50.0)
        self._event("late", minutes_ago=0, lat=38.0001)
        conn = sqlite3.connect(self.db_path)
        conn.execute("UPDATE fire_events SET occurred_at = ?, first_seen_at = ? WHERE external_id = 'late'",
                     ((self.now - timedelta(minutes=70)).strftime("%Y-%m-%dT%H:%M:%SZ"), self.now.strftime("%Y-%m-%dT%H:%M:%SZ")))
        conn.commit()
        conn.close()
        shadow.run_shadow_pass(now=self.now, db_path=self.db_path)
        snap = self._rows("SELECT features_json FROM detection_feature_snapshot WHERE event_id = ? AND mode = 'initial'", a)[0]
        self.assertEqual(json.loads(snap["features_json"])["n_1km_24h"], 0)  # acquired earlier but arrived after scoring

    def test_score_bundle_matches_hand_calculation_and_marks_missing(self):
        bundle = json.loads(_bundle(Path(self.tmp.name) / "m.json").read_text())
        p, calibrated = shadow.score_bundle(bundle, {"frp_log": 2.0, "n_1km_24h": 2.0})
        self.assertTrue(calibrated)
        self.assertAlmostEqual(p, 1 / (1 + math.exp(-(1.0 * (2.0 - 1.0) + 0.5 * 2.0 + -1.0 * 0.0))))
        p_missing, _ = shadow.score_bundle(bundle, {"frp_log": None, "n_1km_24h": 2.0})
        self.assertAlmostEqual(p_missing, 1 / (1 + math.exp(-(0.0 + 1.0 - 1.0))))  # imputed to the median, indicator set

    def test_bundle_with_wrong_feature_schema_is_ignored(self):
        path = _bundle(Path(self.tmp.name) / "m.json")
        data = json.loads(path.read_text())
        data["feature_schema"] = "old"
        path.write_text(json.dumps(data))
        self.assertIsNone(shadow.load_bundle(path))


if __name__ == "__main__":
    unittest.main()
