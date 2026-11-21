import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from core import database
from routers import fires
from services import cropland_burn, incident_history

# 20:00 UTC on 2026-11-20 = 2 pm Central, in the fall field-burning season.
START = datetime(2026, 11, 20, 20, 0, tzinfo=timezone.utc)
FIELDS = {"nlcd_groups": [{"group": "Cultivated crops", "pct": 90}, {"group": "Forest", "pct": 10}]}
WOODS = {"nlcd_groups": [{"group": "Forest", "pct": 85}, {"group": "Cultivated crops", "pct": 15}]}


def _members(hours=3.0, frp=20.0, start=START):
    return [{"occurred_at": (start + timedelta(hours=hours * i / 2)).isoformat(), "frp": frp} for i in range(3)]


class CroplandBurnModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        patcher = patch.object(database, "get_db_path", return_value=Path(self.tmp.name) / "smf.db")
        patcher.start()
        self.addCleanup(patcher.stop)
        database.init_database()
        self.incident = {"id": 1}

    def tearDown(self):
        try:
            self.tmp.cleanup()
        except PermissionError:
            pass

    def _likelihood(self, cover=FIELDS, members=None):
        features = cropland_burn.extract_features(self.incident, members or _members(), cover)
        return cropland_burn.predict(features)[0]

    def test_prior_separates_fields_from_woods(self):
        self.assertGreater(self._likelihood(FIELDS), cropland_burn.PROMPT_THRESHOLD)
        self.assertLess(self._likelihood(WOODS), 0.2)

    def test_multi_day_or_off_season_burn_is_less_likely(self):
        long_burn = self._likelihood(FIELDS, _members(hours=200))
        summer = self._likelihood(FIELDS, _members(start=datetime(2026, 8, 5, 20, 0, tzinfo=timezone.utc)))
        base = self._likelihood(FIELDS)
        self.assertLess(long_burn, base)
        self.assertLess(summer, base)

    def test_missing_ground_cover_never_raises_and_is_not_a_field(self):
        features = cropland_burn.extract_features(self.incident, [], {})
        self.assertEqual(features["crop"], 0.0)
        self.assertLess(cropland_burn.predict(features)[0], 0.1)

    def test_coarse_fuel_bundle_fallback_uses_the_cropland_class(self):
        features = cropland_burn.extract_features(self.incident, _members(), {"groups": [{"group": "Cropland", "pct": 80}]})
        self.assertEqual(features["crop"], 0.8)

    def test_answers_move_the_model_toward_the_answer(self):
        features = cropland_burn.extract_features(self.incident, _members(), FIELDS)
        prior = cropland_burn.predict(features)[0]
        delta: dict = {}
        for _ in range(8):
            delta = cropland_burn._learn(delta, features, 0, 1.0)
        self.assertLess(cropland_burn._predict(features, delta), prior)
        delta = {}
        weak = {**features, "crop": 0.45}
        for _ in range(8):
            delta = cropland_burn._learn(delta, weak, 1, 1.0)
        self.assertGreater(cropland_burn._predict(weak, delta), cropland_burn._predict(weak, {}))

    def test_one_answer_per_visitor_per_incident(self):
        first = cropland_burn.record_answer({"id": 1}, _members(), True, "public", "same")
        again = cropland_burn.record_answer({"id": 1}, _members(), False, "public", "same")
        other = cropland_burn.record_answer({"id": 1}, _members(), True, "public", "different")
        self.assertTrue(first["recorded"])
        self.assertFalse(again["recorded"])
        self.assertTrue(other["recorded"])
        self.assertEqual(cropland_burn.status()["answers"], 2)

    def test_staff_answers_count_for_more(self):
        cropland_burn.record_answer({"id": 1}, _members(), True, "staff", "admin-1")
        with closing(sqlite3.connect(database.get_db_path())) as conn:
            self.assertEqual(conn.execute("SELECT weight FROM cropland_burn_labels").fetchone()[0], cropland_burn.STAFF_WEIGHT)

    def test_replay_matches_live_learning_and_rollback_restores_a_snapshot(self):
        for index in range(12):
            cropland_burn.record_answer({"id": index + 1}, _members(), index % 3 != 0, "public", f"v{index}")
        live = cropland_burn.status()["learned_shift"]
        self.assertEqual(cropland_burn.rebuild_from_labels()["replayed"], 12)
        replayed = cropland_burn.status()["learned_shift"]
        for key, value in live.items():
            self.assertAlmostEqual(value, replayed[key], places=3)
        self.assertEqual(cropland_burn.rollback(), 10)          # the snapshot taken at 10 updates
        self.assertIsNone(cropland_burn.rollback(steps=5))

    def test_poor_precision_falls_back_to_the_rules_prior(self):
        for index in range(cropland_burn.PRECISION_MIN_CALLS + 2):
            cropland_burn.record_answer({"id": index + 1}, _members(), False, "public", f"v{index}")
        # Every confident call was answered No, so the learned model is not trusted.
        with closing(sqlite3.connect(database.get_db_path())) as conn:
            conn.execute("UPDATE cropland_burn_labels SET likelihood_before = 0.9")
            conn.commit()
        self.assertEqual(cropland_burn.status()["serving"], "rules")
        features = cropland_burn.extract_features(self.incident, _members(), FIELDS)
        self.assertEqual(cropland_burn.predict(features)[1], "rules")

    def test_assess_hides_the_prompt_once_answered_or_disabled(self):
        with patch.object(cropland_burn, "extract_features", return_value=cropland_burn.extract_features(self.incident, _members(), FIELDS)):
            self.assertTrue(cropland_burn.assess({"id": 7}, [])["show_prompt"])
            with patch.dict("os.environ", {"CROPLAND_BURN_PROMPT": "false"}):
                self.assertFalse(cropland_burn.assess({"id": 7}, [])["show_prompt"])
            cropland_burn.record_answer({"id": 7}, [], True, "public", "x")
            result = cropland_burn.assess({"id": 7}, [])
            self.assertFalse(result["show_prompt"])
            self.assertEqual(result["answered"], "yes")


class CroplandAnswerEndpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "smf.db"
        for module in (database, incident_history):
            patcher = patch.object(module, "get_db_path", return_value=self.db)
            patcher.start()
            self.addCleanup(patcher.stop)
        database.init_database()
        alerts = patch.object(fires, "notify_staff_alert")
        self.alerts = alerts.start()
        self.addCleanup(alerts.stop)
        for index in range(2):
            database.upsert_detection_event(
                source="viirs", external_id=f"d{index}", latitude=36.5, longitude=-89.7,
                occurred_at=(START + timedelta(minutes=10 * index)).isoformat().replace("+00:00", "Z"),
                county_fips="29143", county_name="New Madrid", frp=12.0, confidence="n", satellite="N20")
        with closing(sqlite3.connect(self.db)) as conn:
            self.slug = conn.execute("SELECT public_slug FROM fire_incidents LIMIT 1").fetchone()[0]
        app = FastAPI()
        app.include_router(fires.router)
        self.http = TestClient(app)

    def tearDown(self):
        try:
            self.tmp.cleanup()
        except PermissionError:
            pass

    def _feedback_rows(self):
        with closing(sqlite3.connect(self.db)) as conn:
            return conn.execute("SELECT classification, burn_type, status FROM fire_incident_feedback").fetchall()

    def test_yes_files_a_pending_cropland_controlled_burn_without_alerting_staff(self):
        response = self.http.post(f"/api/fires/incidents/{self.slug}/cropland-answer", json={"answer": "yes", "source": "website"})
        self.assertEqual(response.status_code, 201)
        self.assertEqual(self._feedback_rows(), [("controlled_burn", "cropland", "pending")])
        self.alerts.assert_not_called()
        self.assertEqual(cropland_burn.status()["yes"], 1)

    def test_no_only_teaches_the_model(self):
        response = self.http.post(f"/api/fires/incidents/{self.slug}/cropland-answer", json={"answer": "no"})
        self.assertEqual(response.status_code, 201)
        self.assertEqual(self._feedback_rows(), [])
        self.alerts.assert_not_called()
        self.assertEqual(cropland_burn.status()["answers"], 1)

    def test_repeat_yes_does_not_file_twice(self):
        for _ in range(2):
            self.http.post(f"/api/fires/incidents/{self.slug}/cropland-answer", json={"answer": "yes"})
        self.assertEqual(len(self._feedback_rows()), 1)

    def test_unknown_incident_is_404(self):
        self.assertEqual(self.http.post("/api/fires/incidents/nope/cropland-answer", json={"answer": "yes"}).status_code, 404)

    def test_unsure_is_stored_but_never_alerts_while_other_feedback_still_does(self):
        url = f"/api/fires/incidents/{self.slug}/feedback"
        self.assertEqual(self.http.post(url, json={"classification": "unsure"}).status_code, 201)
        self.alerts.assert_not_called()
        self.assertEqual([row[0] for row in self._feedback_rows()], ["unsure"])
        self.assertEqual(self.http.post(url, json={"classification": "controlled_burn"}).status_code, 201)
        self.alerts.assert_called_once()

    def test_timeline_says_cropland_burn(self):
        self.http.post(f"/api/fires/incidents/{self.slug}/cropland-answer", json={"answer": "yes"})
        self.assertEqual(incident_history.feedback_summary("controlled_burn", "pending", "cropland"),
                         "Public feedback submitted as a cropland burn, awaiting review")
        self.assertEqual(incident_history.feedback_summary("controlled_burn", "pending"),
                         "Public feedback submitted as a controlled burn, awaiting review")


if __name__ == "__main__":
    unittest.main()
