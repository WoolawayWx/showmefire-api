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
from services import incident_history

START = datetime(2026, 11, 20, 20, 0, tzinfo=timezone.utc)


class StructureFireAndFireTypeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "smf.db"
        for module in (database, incident_history):
            patcher = patch.object(module, "get_db_path", return_value=self.db)
            patcher.start()
            self.addCleanup(patcher.stop)
        database.init_database()
        for name, value in (("notify_staff_alert", None), ("verify_turnstile", (True, "ok"))):
            p = patch.object(fires, name, return_value=value) if value else patch.object(fires, name)
            p.start()
            self.addCleanup(p.stop)
        patch.object(fires, "_reverse_geocode", return_value="").start()
        self.addCleanup(patch.stopall)
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

    def _report(self, **extra):
        body = {
            "latitude": 38.6, "longitude": -92.5,
            "occurred_at": (datetime.now(timezone.utc) - timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "description": "House fully involved, flames through the roof.",
            "reporter_contact": "a@example.com", "reporter_name": "Pat",
            "consent_acknowledged": True, "turnstile_token": "t",
        }
        body.update(extra)
        return self.http.post("/api/fires/reports", json=body)

    def test_structure_fire_needs_no_acres_or_fuel(self):
        response = self._report(fire_type="structure")
        self.assertEqual(response.status_code, 201, response.text)
        with closing(sqlite3.connect(self.db)) as conn:
            row = conn.execute("SELECT fire_type, acres FROM fire_events WHERE source='user_submission'").fetchone()
        self.assertEqual(row, ("structure", None))

    def test_wildland_still_requires_acres_and_fuel(self):
        self.assertEqual(self._report().status_code, 422)
        ok = self._report(acres=2, fuel_types=["grass"])
        self.assertEqual(ok.status_code, 201, ok.text)
        with closing(sqlite3.connect(self.db)) as conn:
            self.assertEqual(conn.execute("SELECT fire_type FROM fire_events WHERE source='user_submission'").fetchone()[0], "wildland")

    def test_feedback_fire_type_is_optional_and_stored(self):
        url = f"/api/fires/incidents/{self.slug}/feedback"
        self.assertEqual(self.http.post(url, json={"classification": "confirmed_fire"}).status_code, 201)
        self.assertEqual(self.http.post(url, json={"classification": "confirmed_fire", "fire_type": "structure"}).status_code, 201)
        self.assertEqual(self.http.post(url, json={"classification": "confirmed_fire", "fire_type": "bogus"}).status_code, 422)
        with closing(sqlite3.connect(self.db)) as conn:
            rows = conn.execute("SELECT fire_type FROM fire_incident_feedback ORDER BY id").fetchall()
        self.assertEqual(rows, [(None,), ("structure",)])


if __name__ == "__main__":
    unittest.main()
