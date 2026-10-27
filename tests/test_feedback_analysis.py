import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai import feedback_analysis as fa
from core import database


class FakeClient:
    model = "fake-model"
    configured = True

    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def generate_text(self, prompt, system=None, *, model=None, max_tokens=0):
        self.prompts.append(prompt)
        return self.reply


GOOD = json.dumps({"summary": "Most say it is a landowner burn.", "themes": ["brush pile", 7],
                   "location_hints": ["Hwy 5"], "red_flags": ["one joke note"], "trust": "HIGH", "approve": True})


class FeedbackAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "smf.db"
        for target in (database, fa):
            patcher = patch.object(target, "get_db_path", return_value=self.db)
            patcher.start()
            self.addCleanup(patcher.stop)
        # list_fire_incident_feedback lives in database and reads its own get_db_path (patched above)
        database.init_database()
        with sqlite3.connect(self.db) as conn:
            self.incident = conn.execute(
                "INSERT INTO fire_incidents (centroid_latitude, centroid_longitude, first_detected_at, last_detected_at, "
                "detection_count) VALUES (37, -92, 'a', 'b', 1)").lastrowid
        self._add("controlled_burn", "Farmer burning a brush pile off Hwy 5, has a permit", "pending")
        self._add("controlled_burn", "Same, saw the fire truck waiting nearby", "approved")
        self._add("not_a_fire", "lol", "rejected")
        self._add("confirmed_fire", "", "pending")

    def tearDown(self):
        try:
            self.tmp.cleanup()
        except PermissionError:
            pass

    def _add(self, classification, note, status, contact="me@example.com"):
        with sqlite3.connect(self.db) as conn:
            conn.execute("INSERT INTO fire_incident_feedback (incident_id, classification, note, contact, status, "
                         "submitter_ip_hash) VALUES (?, ?, ?, ?, ?, 'iphash')",
                         (self.incident, classification, note, contact, status))

    def test_sanitize_keeps_only_the_schema(self):
        a = fa.sanitize(json.loads(GOOD))
        self.assertEqual((a["trust"], a["themes"]), ("high", ["brush pile"]))
        self.assertNotIn("approve", a)
        self.assertEqual(fa.sanitize("junk")["summary"], None)

    def test_counts_come_from_the_rows_not_the_model_and_exclude_rejected(self):
        client = FakeClient(GOOD)
        self.assertEqual(fa.analyze(self.incident, client=client), "stored")
        stored = fa.get_analysis(self.incident, database.list_fire_incident_feedback(self.incident))
        self.assertEqual(stored["counts"], {"confirmed_fire": 1, "controlled_burn": 2, "not_a_fire": 0, "unsure": 0})
        self.assertEqual(stored["items_read"], 3)
        self.assertFalse(stored["stale"])

    def test_only_notes_and_choices_are_sent_never_contact_or_network_details(self):
        client = FakeClient(GOOD)
        fa.analyze(self.incident, client=client)
        prompt = client.prompts[0]
        self.assertIn("brush pile off Hwy 5", prompt)
        self.assertNotIn("me@example.com", prompt)
        self.assertNotIn("iphash", prompt)
        self.assertNotIn("lol", prompt)  # rejected feedback is not read

    def test_new_feedback_marks_the_analysis_stale_and_rerun_clears_it(self):
        fa.analyze(self.incident, client=FakeClient(GOOD))
        self._add("controlled_burn", "Burn is out now", "pending")
        feedback = database.list_fire_incident_feedback(self.incident)
        self.assertTrue(fa.get_analysis(self.incident, feedback)["stale"])
        fa.analyze(self.incident, client=FakeClient(GOOD))
        self.assertFalse(fa.get_analysis(self.incident, feedback)["stale"])

    def test_skips_without_notes_or_ai_and_fails_quietly_on_a_bad_reply(self):
        with sqlite3.connect(self.db) as conn:
            conn.execute("UPDATE fire_incident_feedback SET note = ''")
        self.assertEqual(fa.analyze(self.incident, client=FakeClient(GOOD)), "skipped")
        with sqlite3.connect(self.db) as conn:
            conn.execute("UPDATE fire_incident_feedback SET note = 'a long enough note here'")
        unconfigured = FakeClient(GOOD)
        unconfigured.configured = False
        self.assertEqual(fa.analyze(self.incident, client=unconfigured), "skipped")
        self.assertEqual(fa.analyze(self.incident, client=FakeClient("sorry")), "failed")
        self.assertIsNone(fa.get_analysis(self.incident, []))


class AnalyzeEndpointTests(FeedbackAnalysisTests):
    def setUp(self):
        super().setUp()
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from core.security import require_admin
        from routers import fires

        app = FastAPI()
        app.include_router(fires.router)
        app.dependency_overrides[require_admin] = lambda: "admin"
        self.http = TestClient(app)
        patcher = patch.object(fa, "CloudflareAIClient", lambda: FakeClient(GOOD))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_endpoint_returns_the_analysis_and_incident_detail_carries_it(self):
        url = f"/api/admin/fires/incidents/{self.incident}/feedback/analyze"
        body = self.http.post(url).json()
        self.assertEqual(body["analysis"]["trust"], "high")
        detail = self.http.get(f"/api/admin/fires/incidents/{self.incident}").json()["incident"]
        self.assertEqual(detail["feedback_analysis"]["summary"], "Most say it is a landowner burn.")
        self.assertEqual(self.http.post("/api/admin/fires/incidents/999/feedback/analyze").status_code, 404)

    def test_endpoint_explains_a_nothing_to_read_case(self):
        with sqlite3.connect(self.db) as conn:
            conn.execute("UPDATE fire_incident_feedback SET note = ''")
        self.assertEqual(self.http.post(f"/api/admin/fires/incidents/{self.incident}/feedback/analyze").status_code, 409)


if __name__ == "__main__":
    unittest.main()
