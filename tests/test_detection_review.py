import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import database
from services import detection_review as review


class DetectionReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "showmefire.db"
        self.patch = patch.object(database, "get_db_path", return_value=self.db_path)
        self.patch.start()
        database.init_database()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def _event(self, ext, source="viirs", lat=38.0, lon=-92.0, **extra):
        result = database.upsert_detection_event(
            source=source, external_id=ext, latitude=lat, longitude=lon, occurred_at="2026-09-01T18:00:00Z", frp=8.0,
            confidence="n", bright_t7=330.0, bright_t13=295.0, pixel_area=0.14, daynight="D", cluster_incidents=False)
        if extra:
            conn = sqlite3.connect(self.db_path)
            sets = ", ".join(f"{k} = ?" for k in extra)
            conn.execute(f"UPDATE fire_events SET {sets} WHERE id = ?", (*extra.values(), result["event_id"]))
            conn.commit()
            conn.close()
        return result["event_id"]

    def _sql(self, sql, *args):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in conn.execute(sql, args).fetchall()]
        finally:
            conn.close()

    def test_next_hides_scores_and_records_the_selection(self):
        self._event("a", detection_confidence_pct=77.0)
        item = review.next_detection("random", "me", self.db_path)
        text = json.dumps(item)
        self.assertNotIn("detection_confidence_pct", text)
        self.assertNotIn("score", text)
        self.assertIn("google_satellite", item["imagery_links"])
        sel = self._sql("SELECT * FROM detection_review_selection")
        self.assertEqual(len(sel), 1)
        self.assertEqual((sel[0]["stream"], sel[0]["sampling_version"], sel[0]["population_n"]), ("random", review.SAMPLING_VERSION, 1))

    def test_random_stream_includes_masked_detections(self):
        self._event("m", exclusion_zone_id=1)
        self.assertIsNotNone(review.next_detection("random", "me", self.db_path))

    def test_enriched_stream_records_bucket_and_does_not_flag_protected(self):
        self._event("m", recurring_source_id=1)
        item = review.next_detection("enriched", "me", self.db_path)
        self.assertIsNotNone(item)
        sel = self._sql("SELECT bucket, protected FROM detection_review_selection")[0]
        self.assertEqual((sel["bucket"], sel["protected"]), ("masked", 0))

    def test_enriched_returns_none_when_nothing_matches(self):
        self._event("plain")
        self.assertIsNone(review.next_detection("enriched", "me", self.db_path))

    def test_protected_flag_is_deterministic_and_about_one_in_five(self):
        flags = [review.is_protected(i) for i in range(2000)]
        self.assertEqual(flags, [review.is_protected(i) for i in range(2000)])
        self.assertTrue(300 < sum(flags) < 500)

    def test_reviewed_detection_is_not_offered_again(self):
        eid = self._event("a")
        review.submit_outcome(eid, "vegetation_fire", "me", db_path=self.db_path)
        self.assertIsNone(review.next_detection("random", "me", self.db_path))

    def test_agreement_conflict_and_resolution(self):
        eid = self._event("a")
        self.assertEqual(review.submit_outcome(eid, "vegetation_fire", "r1", db_path=self.db_path)["adjudication_status"], "single")
        self.assertEqual(review.submit_outcome(eid, "vegetation_fire", "r2", db_path=self.db_path)["adjudication_status"], "agree")
        other = self._event("b")
        review.submit_outcome(other, "vegetation_fire", "r1", db_path=self.db_path)
        self.assertEqual(review.submit_outcome(other, "not_vegetation_fire", "r2", db_path=self.db_path)["adjudication_status"], "conflict")
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        self.assertNotIn(other, review.final_outcomes(conn))  # an open conflict is not a label
        conn.close()
        self.assertEqual(review.submit_outcome(other, "not_vegetation_fire", "r3", db_path=self.db_path)["adjudication_status"], "resolved")
        self.assertEqual(len(self._sql("SELECT 1 FROM detection_review_outcome WHERE event_id = ?", other)), 3)  # append-only

    def test_validation(self):
        eid = self._event("a")
        with self.assertRaises(review.ReviewError):
            review.submit_outcome(eid, "maybe", "me", db_path=self.db_path)
        with self.assertRaises(review.ReviewError):
            review.submit_outcome(eid, "vegetation_fire", "me", reason="bogus", db_path=self.db_path)
        with self.assertRaises(review.ReviewError):
            review.submit_outcome(9999, "vegetation_fire", "me", db_path=self.db_path)

    def test_second_review_offers_only_other_reviewers_single_outcomes(self):
        eid = self._event("a")
        review.submit_outcome(eid, "vegetation_fire", "r1", db_path=self.db_path)
        self.assertIsNone(review.second_review("r1", self.db_path))
        self.assertEqual(review.second_review("r2", self.db_path)["detection"]["id"], eid)

    def test_stats_report_cannot_determine_and_conflicts(self):
        a, b = self._event("a"), self._event("b")
        review.submit_outcome(a, "cannot_determine", "r1", reason="insufficient_evidence", db_path=self.db_path)
        review.submit_outcome(b, "vegetation_fire", "r1", db_path=self.db_path)
        s = review.stats(self.db_path)
        self.assertEqual(s["reviewed"], 2)
        self.assertEqual(s["cannot_determine_rate"], 0.5)
        self.assertEqual(s["by_outcome"], {"cannot_determine": 1, "vegetation_fire": 1})

    def test_shadow_summary_says_too_few_and_then_computes(self):
        ids = [self._event(f"e{i}", lat=38.0 + i * 0.01) for i in range(40)]
        conn = sqlite3.connect(self.db_path)
        for i, eid in enumerate(ids):
            conn.execute("INSERT INTO detection_review_selection (event_id, stream, sampling_version, protected) VALUES (?, 'random', 'random-v1', ?)",
                         (eid, int(review.is_protected(eid))))
            conn.execute("INSERT INTO detection_score_history (event_id, model_name, model_version, mode, score) VALUES (?, 'v1_heuristic', 'formula', 'initial', ?)",
                         (eid, 0.9 if i % 2 == 0 else 0.1))
            conn.execute("INSERT INTO detection_score_history (event_id, model_name, model_version, mode, score) VALUES (?, 'veg_conf', '1', 'initial', 0.5)", (eid,))
        conn.commit()
        conn.close()
        for i, eid in enumerate(ids[:10]):
            review.submit_outcome(eid, "vegetation_fire" if i % 2 == 0 else "not_vegetation_fire", "r", db_path=self.db_path)
        early = review.shadow_summary(self.db_path)
        self.assertIn("note", early["models"]["v1_heuristic@formula"]["random_stream"])  # 10 reviews: no conclusions
        for i, eid in enumerate(ids[10:], start=10):
            review.submit_outcome(eid, "vegetation_fire" if i % 2 == 0 else "not_vegetation_fire", "r", db_path=self.db_path)
        full = review.shadow_summary(self.db_path)
        v1 = full["models"]["v1_heuristic@formula"]["random_stream"]
        self.assertEqual(v1["auc"], 1.0)  # v1 perfectly separates this synthetic case
        self.assertEqual(full["models"]["veg_conf@1"]["random_stream"]["auc"], 0.5)  # constant score = chance
        self.assertEqual(full["reviewed_random_stream"], 40)


if __name__ == "__main__":
    unittest.main()


class DetectionReviewRouterTests(DetectionReviewTests):
    def _client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from core.security import require_admin
        from routers import detection_review as router_module
        app = FastAPI()
        app.include_router(router_module.router)
        app.dependency_overrides[require_admin] = lambda: "admin@example.org"
        return TestClient(app)

    def test_endpoints_round_trip(self):
        eid = self._event("a", detection_confidence_pct=70.0)
        client = self._client()
        got = client.get("/api/admin/fires/detection-review/next").json()
        self.assertEqual(got["item"]["detection"]["id"], eid)
        self.assertNotIn("confidence_pct", json.dumps(got))
        ok = client.post(f"/api/admin/fires/detection-review/{eid}", json={"outcome": "not_vegetation_fire", "reason": "routine_industrial_heat"})
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(client.post(f"/api/admin/fires/detection-review/{eid}", json={"outcome": "bogus"}).status_code, 422)
        self.assertEqual(client.post("/api/admin/fires/detection-review/9999", json={"outcome": "vegetation_fire"}).status_code, 404)
        self.assertEqual(client.get("/api/admin/fires/detection-review/stats").json()["reviewed"], 1)
        self.assertIn("models", client.get("/api/admin/fires/detection-review/shadow-summary").json())
