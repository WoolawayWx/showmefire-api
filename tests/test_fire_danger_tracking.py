import json
import sqlite3
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import database
from services import fire_danger_tracking as ft

COUNTIES = [f"29{n:03d}" for n in range(1, 21)]
TODAY = "2026-10-31"


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db_path = self.root / "showmefire.db"
        self.patch = patch.object(database, "get_db_path", return_value=self.db_path)
        self.patch.start()
        database.init_database()
        self.dirs = {k: self.root / k for k in ("fire_weather_index", "risk_fusion_glm", "ensemble")}
        for d in self.dirs.values():
            d.mkdir()
        self.dates = [(date(2026, 9, 1) + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(40)]
        self.rng = np.random.default_rng(0)

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    # ---- builders
    def _write_world(self):
        """A latent 'danger' per county-day drives fires; fwi sees it, the glm sees it noisily, the public rule is noise."""
        self.latent, self.fires = {}, set()
        for d in self.dates:
            lat = self.rng.uniform(0, 1, len(COUNTIES))
            self.latent[d] = lat
            for f, v in zip(COUNTIES, lat):
                if self.rng.uniform() < v ** 3 * 0.5:
                    self.fires.add((f, d))
            run = d.replace("-", "") + "_12"
            (self.dirs["fire_weather_index"] / f"{run}.fire_weather_index_score.json").write_text(json.dumps({
                "run_id": run, "recorded_at": f"{d}T12:30:00Z", "valid_local_date": d, "bundle_checksum": "x",
                "county_fips": COUNTIES, "score": [float(v) for v in lat],
                "category": [int(min(4, v * 5)) for v in lat]}))
            noisy = np.clip(lat + self.rng.normal(0, 0.15, len(lat)), 0, 1)
            (self.dirs["risk_fusion_glm"] / f"{run}.glm_score.json").write_text(json.dumps({
                "run_id": run, "recorded_at": f"{d}T12:30:00Z", "valid_local_date": d, "bundle_checksum": "x",
                "county_fips": COUNTIES, "lam": [float(v) for v in noisy], "p_ge1_fire": [float(v * 0.5) for v in noisy]}))
            (self.dirs["ensemble"] / f"{run}.ensemble_fire_danger.json").write_text(json.dumps({
                "run_id": run, "anchor_cycle_utc": f"{d}T12:00Z", "recorded_at": f"{d}T13:00:00Z",
                "tracks": {"members": {"county_max_categorical": {f: int(min(4, v * 5)) for f, v in zip(COUNTIES, noisy)},
                                       "county_max_neighborhood_probability": {"elevated": {f: float(v) for f, v in zip(COUNTIES, noisy)}}}}}))
            conn = sqlite3.connect(self.db_path)
            for f in COUNTIES:
                conn.execute("INSERT OR REPLACE INTO county_forecast_days (forecast_date, county_fips, danger_level, forecast_run_id, published_at) VALUES (?, ?, ?, 'r', ?)",
                             (d, f, int(self.rng.integers(0, 5)), f"{d} 13:00:00"))
            conn.commit()
            conn.close()

    def _fire_event(self, fips, d, tier="admin_reviewed", status="approved", cause=None, hour="18:00:00"):
        eid = database.upsert_detection_event(source="viirs", external_id=f"{fips}-{d}-{tier}-{cause}", latitude=38.0, longitude=-92.0,
                                              occurred_at=f"{d}T{hour}Z", county_fips=fips, verification_tier=tier,
                                              cause_category=cause or "unknown", cluster_incidents=False)["event_id"]
        conn = sqlite3.connect(self.db_path)
        conn.execute("UPDATE fire_events SET status = ?, county_fips = ?, verification_tier = ?, cause_category = ? WHERE id = ?",
                     (status, fips, tier, cause or "unknown", eid))
        conn.commit()
        conn.close()

    def _all_fires(self):
        for f, d in sorted(self.fires):
            self._fire_event(f, d)

    # ---- ingest
    def test_ingest_copies_every_model_and_is_idempotent(self):
        self._write_world()
        out = ft.ingest_scores(self.db_path, self.dirs)
        n = len(COUNTIES) * len(self.dates)
        self.assertEqual(out["added"], {"fire_weather_index": n, "risk_fusion_glm": n, "ensemble:members": n, "public_rule": n})
        again = ft.ingest_scores(self.db_path, self.dirs)
        self.assertEqual(again["added"], {})
        self.assertEqual(again["skipped_files"], 2 * len(self.dates))

    def test_ensemble_night_cycles_are_skipped_and_a_bad_file_does_not_stop_the_rest(self):
        (self.dirs["ensemble"] / "n.ensemble_fire_danger.json").write_text(json.dumps({
            "run_id": "n", "anchor_cycle_utc": "2026-09-01T00:00Z", "recorded_at": "x",
            "tracks": {"members": {"county_max_categorical": {"29001": 1}}}}))
        (self.dirs["fire_weather_index"] / "bad.fire_weather_index_score.json").write_text("{not json")
        self._write_world()
        out = ft.ingest_scores(self.db_path, self.dirs)
        self.assertEqual(out["unreadable"], 1)
        conn = sqlite3.connect(self.db_path)
        try:
            stored = {r[0] for r in conn.execute("SELECT valid_date FROM fire_danger_model_scores")}
        finally:
            conn.close()
        self.assertNotIn("2026-08-31", stored)
        self.assertGreater(out["added"]["fire_weather_index"], 0)

    # ---- outcomes
    def test_outcome_filters_tier_status_cause_and_uses_chicago_date(self):
        self._fire_event("29001", "2026-09-10")                                       # counts
        self._fire_event("29002", "2026-09-10", tier="unverified")                    # below tier
        self._fire_event("29003", "2026-09-10", status="pending")                     # not approved
        self._fire_event("29004", "2026-09-10", cause="prescribed")                   # controlled burn
        self._fire_event("29005", "2026-09-11", hour="03:00:00")                      # 03Z = 22:00 on the 10th in Chicago
        conn = ft._connect(self.db_path)
        try:
            got = {(r.county_fips, r.date) for r in ft.fire_county_days(conn).itertuples()}
            official = ft.fire_county_days(conn, "official_source_confirmed")
        finally:
            conn.close()
        self.assertEqual(got, {("29001", "2026-09-10"), ("29005", "2026-09-10")})
        self.assertTrue(official.empty)

    # ---- metrics
    def test_informative_model_beats_noise_and_consensus_is_reported(self):
        self._write_world()
        self._all_fires()
        ft.ingest_scores(self.db_path, self.dirs)
        res = ft.evaluate(window_days=90, lag_days=3, today=TODAY, db_path=self.db_path)
        self.assertEqual(res["status"], "ok")
        allm = res["windows"]["all_time"]["models"]
        self.assertGreater(allm["fire_weather_index"]["auc"], 0.75)
        self.assertGreater(allm["fire_weather_index"]["auc"], allm["public_rule"]["auc"] + 0.15)
        self.assertLess(abs(allm["public_rule"]["auc"] - 0.5), 0.12)  # random categories carry no signal
        self.assertIn("brier", allm["risk_fusion_glm"])
        self.assertNotIn("brier", allm["fire_weather_index"])  # a category score is not a probability of fire
        rises = allm["fire_weather_index"]["rate_rises_with_category"]
        self.assertTrue(rises)
        rows = lambda *t: [{"county_days": n, "fire_days": f, "fire_rate": f / n} for n, f in t]
        self.assertTrue(ft._rises(rows((100, 0), (100, 2), (100, 9), (100, 30))))
        self.assertFalse(ft._rises(rows((200, 40), (200, 10), (200, 60))))  # a real, well-supported drop
        self.assertTrue(ft._rises(rows((160, 1), (160, 0), (160, 12))))      # one fire's difference is noise
        self.assertIsNone(ft._rises(rows((100, 1), (100, 2))))
        same = res["windows"]["all_time"]["same_rows"]
        self.assertEqual(same["county_days"], len(COUNTIES) * len(self.dates))
        cons = res["windows"]["all_time"]["consensus"]
        self.assertEqual(set(cons["families"]), {"fire_weather_index", "public_rule", "ensemble"})
        self.assertIn("auc", cons["consensus"])
        self.assertIsNotNone(allm["fire_weather_index"]["auc_ci95_by_date"])

    def test_recent_window_limits_dates_and_lag_excludes_the_latest_days(self):
        self._write_world()
        self._all_fires()
        ft.ingest_scores(self.db_path, self.dirs)
        res = ft.evaluate(window_days=10, lag_days=3, today="2026-10-09", db_path=self.db_path)  # last date is 2026-10-09
        self.assertEqual(res["outcome"]["latest_scored_date"], "2026-10-06")
        recent = res["windows"]["recent"]
        self.assertEqual(recent["models"]["fire_weather_index"]["dates"], 10)
        self.assertEqual(recent["to"], "2026-10-06")
        self.assertGreater(res["windows"]["all_time"]["models"]["fire_weather_index"]["dates"], 10)

    def test_too_few_fires_says_so_instead_of_concluding(self):
        self._write_world()
        for f, d in sorted(self.fires)[:3]:
            self._fire_event(f, d)
        ft.ingest_scores(self.db_path, self.dirs)
        res = ft.evaluate(today=TODAY, db_path=self.db_path)
        m = res["windows"]["all_time"]["models"]["fire_weather_index"]
        self.assertIn("too few to conclude", m["note"])
        self.assertNotIn("auc", m)

    def test_runs_after_the_local_cutoff_are_not_forecasts(self):
        self._write_world()
        late = self.dates[5]
        run = late.replace("-", "") + "_18"
        (self.dirs["fire_weather_index"] / f"{run}.fire_weather_index_score.json").write_text(json.dumps({
            "run_id": run, "recorded_at": f"{late}T23:00:00Z", "valid_local_date": late, "bundle_checksum": "x",
            "county_fips": COUNTIES, "score": [1.0] * len(COUNTIES), "category": [4] * len(COUNTIES)}))
        ft.ingest_scores(self.db_path, self.dirs)
        res = ft.evaluate(today=TODAY, db_path=self.db_path)
        self.assertEqual(res["cutoff"]["runs_after_cutoff_excluded"]["fire_weather_index"], 1)
        conn = ft._connect(self.db_path)
        try:
            runs, _ = ft._latest_runs(conn, None, TODAY)
        finally:
            conn.close()
        chosen = runs[(runs["model"] == "fire_weather_index") & (runs["valid_date"] == late)]["run_id"].unique()
        self.assertEqual(list(chosen), [late.replace("-", "") + "_12"])  # the morning run, not the evening rerun

    def test_no_scores_yet(self):
        res = ft.evaluate(today=TODAY, db_path=self.db_path)
        self.assertEqual(res["status"], "no_scores")

    def test_models_with_different_histories_are_compared_on_shared_rows_only(self):
        self._write_world()
        self._all_fires()
        for p in list(self.dirs["risk_fusion_glm"].glob("*.json"))[:20]:  # the GLM only has the last 20 days
            p.unlink()
        ft.ingest_scores(self.db_path, self.dirs)
        res = ft.evaluate(today=TODAY, db_path=self.db_path)
        block = res["windows"]["all_time"]
        self.assertEqual(block["models"]["risk_fusion_glm"]["dates"], 20)
        self.assertEqual(block["same_rows"]["county_days"], 20 * len(COUNTIES))
        self.assertEqual({m["n_county_days"] for m in block["same_rows"]["models"].values()}, {20 * len(COUNTIES)})

    def test_endpoint_returns_the_report(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from core.security import require_admin
        from routers import fire_danger_tracking as router_module
        self._write_world()
        self._all_fires()
        app = FastAPI()
        app.include_router(router_module.router)
        app.dependency_overrides[require_admin] = lambda: "admin"
        client = TestClient(app)
        with patch.object(ft, "EVIDENCE_DIRS", self.dirs):
            self.assertTrue(client.post("/api/admin/models/fire-danger-tracking/refresh").json()["added"])
        body = client.get("/api/admin/models/fire-danger-tracking?days=30").json()
        self.assertTrue(body["success"] and body["status"] == "ok")
        self.assertEqual(client.get("/api/admin/models/fire-danger-tracking?min_tier=bogus").status_code, 422)


if __name__ == "__main__":
    unittest.main()
