import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core import database
from services import fire_labeling


class NotAFireTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary.name) / "showmefire.db"
        self.patch = patch.object(database, "get_db_path", return_value=self.db_path)
        self.patch.start()
        database.init_database()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        self.incident_id = database.find_or_create_incident_for_detection(
            cursor, 37.0, -90.0, "2026-09-28T15:00:00Z", "29207", "Stoddard", radius_km=0.1)
        cursor.execute(
            """INSERT INTO fire_events (source, external_id, status, latitude, longitude, occurred_at, incident_id, frp)
               VALUES ('viirs', 'a', 'pending', 37.0, -90.0, '2026-09-28T15:00:00Z', ?, 10)""",
            (self.incident_id,))
        conn.commit()
        conn.close()

    def tearDown(self):
        self.patch.stop()
        try:
            self.temporary.cleanup()
        except PermissionError:
            pass

    def _row(self, sql):
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(sql, (self.incident_id,)).fetchone()

    def test_mark_then_purge_after_seven_days(self):
        fire_labeling.apply_feedback_label(self.incident_id, "not_a_fire", reviewed_by="admin")
        database.set_incident_review_decision(self.incident_id, "rejected", "admin", reason="not_a_fire")
        due = database.schedule_incident_deletion(self.incident_id, days=7)
        self.assertIsNotNone(due)
        self.assertEqual(self._row("SELECT status FROM fire_events WHERE incident_id = ?")[0], "rejected")
        self.assertEqual(self._row("SELECT cause_category FROM fire_events WHERE incident_id = ?")[0], "not_a_fire")

        self.assertEqual(database.purge_due_fire_incidents(), 0)  # not due yet
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE fire_incidents SET delete_after = datetime('now', '-1 minute') WHERE id = ?", (self.incident_id,))
        self.assertEqual(database.purge_due_fire_incidents(), 1)
        self.assertEqual(self._row("SELECT status FROM fire_incidents WHERE id = ?")[0], "deleted")
        self.assertEqual(self._row("SELECT status FROM fire_events WHERE incident_id = ?")[0], "deleted")
        self.assertEqual(database.purge_due_fire_incidents(), 0)

    def test_schedule_unknown_incident(self):
        self.assertIsNone(database.schedule_incident_deletion(99999))


if __name__ == "__main__":
    unittest.main()
