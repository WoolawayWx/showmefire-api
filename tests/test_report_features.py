import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai import models
from ai import report_features as rf
from core import database


class FakeClient:
    model = "fake-model"
    configured = True

    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def generate_text(self, prompt, system=None, *, model=None, max_tokens=0):
        self.prompts.append((prompt, system))
        return self.reply


GOOD = json.dumps({
    "event_type": "Grass Fire", "controlled_burn_indicators": False, "structures_threatened": True,
    "injuries_or_evacuations": None, "responders_on_scene": "yes", "spread": "rapid", "smoke": "thick black, drifting east",
    "fuels": ["grass", "Timber", "lava"], "location_hints": ["County Rd 12", " ", 5, "behind the co-op"],
    "urgency": "HIGH", "key_points": ["propane tank nearby"], "approve": True,
    "status": "Contained", "timing": "past", "reporter_role": "Fire Department", "acres_low": 50, "acres_high": 40,
})


class SanitizeTests(unittest.TestCase):
    def test_coerces_to_the_fixed_vocabulary(self):
        f = rf.sanitize(json.loads(GOOD))
        self.assertEqual(f["event_type"], "grass_fire")
        self.assertEqual(f["urgency"], "high")
        self.assertEqual(f["fuels"], ["grass", "timber"])           # "lava" dropped
        self.assertIsNone(f["responders_on_scene"])                 # "yes" is not a bool
        self.assertEqual(f["location_hints"], ["County Rd 12", "behind the co-op"])
        self.assertNotIn("approve", f)                              # unknown keys never pass through
        self.assertEqual((f["status"], f["timing"], f["reporter_role"]), ("contained", "past", "fire_department"))
        self.assertEqual((f["acres_low"], f["acres_high"]), (40, 50))   # swapped range is put right

    def test_single_acre_figure_fills_both_ends_and_junk_is_dropped(self):
        self.assertEqual(rf.sanitize({"acres_low": 12})["acres_high"], 12)
        f = rf.sanitize({"acres_low": "lots", "acres_high": -3})
        self.assertEqual((f["acres_low"], f["acres_high"]), (None, None))

    def test_garbage_becomes_nulls_not_errors(self):
        f = rf.sanitize("not a dict")
        self.assertEqual((f["event_type"], f["spread"], f["urgency"], f["fuels"]), ("unclear", "unknown", None, []))

    def test_parse_reply_tolerates_fences_and_chatter(self):
        self.assertEqual(rf.parse_reply('Sure!\n```json\n{"a": 1}\n```'), {"a": 1})
        with self.assertRaises(ValueError):
            rf.parse_reply("no json here")

    def test_report_text_drops_the_osint_source_line(self):
        text = rf.report_text({"description": "Barn fire on Hwy 5\n\nSource: https://x.example/p", "out_of_ordinary": "tank"})
        self.assertEqual(text, "Barn fire on Hwy 5\ntank")


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "smf.db"
        for target in (database, rf, models):
            patcher = patch.object(target, "get_db_path", return_value=self.db)
            patcher.start()
            self.addCleanup(patcher.stop)
        database.init_database()
        with sqlite3.connect(self.db) as conn:
            conn.execute("INSERT INTO fire_events (source, status, latitude, longitude, occurred_at, description) "
                         "VALUES ('user_submission', 'pending', 37, -92, '2026-10-01T12:00:00Z', "
                         "'Grass fire spreading fast toward a barn off County Rd 12')")
        self.event_id = 1

    def tearDown(self):
        try:
            self.tmp.cleanup()
        except PermissionError:
            pass

    def test_stores_features_and_keeps_the_raw_text(self):
        client = FakeClient(GOOD)
        self.assertEqual(rf.extract_and_store(self.event_id, client=client), "stored")
        stored = rf.get_features(self.event_id)
        self.assertEqual(stored["features"]["event_type"], "grass_fire")
        self.assertEqual(stored["model"], models.DEFAULT_TEXT_MODEL)
        with sqlite3.connect(self.db) as conn:
            self.assertIn("barn off County Rd 12", conn.execute("SELECT description FROM fire_events").fetchone()[0])
        self.assertIn("Grass fire spreading", client.prompts[0][0])
        self.assertIn("untrusted", client.prompts[0][1])

    def test_unchanged_text_is_not_sent_again_but_force_redoes_it(self):
        client = FakeClient(GOOD)
        rf.extract_and_store(self.event_id, client=client)
        self.assertEqual(rf.extract_and_store(self.event_id, client=client), "unchanged")
        self.assertEqual(rf.extract_and_store(self.event_id, client=client, force=True), "stored")
        self.assertEqual(len(client.prompts), 2)

    def test_a_bad_model_reply_fails_quietly_and_stores_nothing(self):
        self.assertEqual(rf.extract_and_store(self.event_id, client=FakeClient("sorry, I can't")), "failed")
        self.assertIsNone(rf.get_features(self.event_id))

    def test_unconfigured_ai_and_short_text_are_skipped(self):
        client = FakeClient(GOOD)
        client.configured = False
        self.assertEqual(rf.extract_and_store(self.event_id, client=client), "skipped")
        with sqlite3.connect(self.db) as conn:
            conn.execute("UPDATE fire_events SET description = 'fire'")
        self.assertEqual(rf.extract_and_store(self.event_id, client=FakeClient(GOOD)), "skipped")

    def test_admin_event_carries_the_features(self):
        rf.extract_and_store(self.event_id, client=FakeClient(GOOD))
        self.assertEqual(database.get_fire_event(self.event_id, admin=True)["extracted_features"]["features"]["urgency"], "high")

    def test_conflicts_between_the_form_and_the_text(self):
        features = rf.sanitize({"status": "out", "acres_low": 40, "acres_high": 50, "structures_threatened": False})
        event = {"acres": 200, "reported_fire_status": "active", "reported_structures_threatened": 1,
                 "reported_responders_on_scene": None}
        conflicts = rf.find_conflicts(event, features)
        self.assertEqual(len(conflicts), 3)
        self.assertTrue(any("200 acres, text says 40-50" in c for c in conflicts))
        self.assertEqual(rf.find_conflicts({"acres": 50, "reported_fire_status": "out"}, features), [])  # all agree
        self.assertEqual(rf.find_conflicts({"acres": 50}, rf.sanitize({})), [])  # nothing to compare

    def test_backfill_only_does_reports_without_features(self):
        with patch.object(rf, "CloudflareAIClient", lambda: FakeClient(GOOD)):
            self.assertEqual(rf.backfill(), {"stored": 1})
            self.assertEqual(rf.backfill(), {})


if __name__ == "__main__":
    unittest.main()


class RescanEndpointTests(StoreTests):
    """POST .../features/rescan, through the real router with admin auth overridden."""

    def setUp(self):
        super().setUp()
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from core.security import require_admin
        from routers import fires

        app = FastAPI()
        app.include_router(fires.router)
        app.dependency_overrides[require_admin] = lambda: "admin"
        self.client = TestClient(app)
        patcher = patch.object(rf, "CloudflareAIClient", lambda: FakeClient(GOOD))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_rescan_one_report_replaces_features_even_when_the_text_is_unchanged(self):
        rf.extract_and_store(self.event_id, client=FakeClient(json.dumps({"urgency": "low"})))
        self.assertEqual(rf.get_features(self.event_id)["features"]["urgency"], "low")
        response = self.client.post(f"/api/admin/fires/reports/{self.event_id}/features/rescan")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["report"]["extracted_features"]["features"]["urgency"], "high")

    def test_rescan_errors_are_explained(self):
        self.assertEqual(self.client.post("/api/admin/fires/reports/999/features/rescan").status_code, 404)
        with patch.object(rf, "CloudflareAIClient", lambda: FakeClient("not json")):
            self.assertEqual(self.client.post(f"/api/admin/fires/reports/{self.event_id}/features/rescan").status_code, 502)

    def test_rescan_incident_covers_reports_but_not_satellite_pixels(self):
        with sqlite3.connect(self.db) as conn:
            incident = conn.execute("INSERT INTO fire_incidents (centroid_latitude, centroid_longitude, first_detected_at, "
                                    "last_detected_at, detection_count) VALUES (37, -92, 'a', 'b', 2)").lastrowid
            conn.execute("UPDATE fire_events SET incident_id = ?", (incident,))
            conn.execute("INSERT INTO fire_events (source, status, latitude, longitude, occurred_at, description, incident_id) "
                         "VALUES ('viirs', 'pending', 37, -92, 'a', 'satellite pixel with some descriptive text', ?)", (incident,))
        body = self.client.post(f"/api/admin/fires/incidents/{incident}/features/rescan").json()
        self.assertEqual((body["scanned"], body["results"]), (1, {"stored": 1}))
        self.assertEqual(self.client.post("/api/admin/fires/incidents/999/features/rescan").status_code, 404)
