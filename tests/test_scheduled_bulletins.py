import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from core import database
from core.security import require_admin
from routers import bulletins
from services import bulletin_send


def _iso(delta):
    return (datetime.now(timezone.utc) + delta).strftime("%Y-%m-%dT%H:%M:%SZ")


class ScheduledBulletinTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        patcher = patch.object(database, "get_db_path", return_value=Path(self.tmp.name) / "smf.db")
        patcher.start()
        self.addCleanup(patcher.stop)
        database.init_database()
        app = FastAPI()
        app.include_router(bulletins.router)
        app.dependency_overrides[require_admin] = lambda: "tester"
        self.http = TestClient(app)
        self.id = database.create_bulletin("Hello", "<p>Hi</p>", "Hi", "", ["a@example.com"])["id"]

    def tearDown(self):
        try:
            self.tmp.cleanup()
        except PermissionError:
            pass

    def test_schedule_change_and_cancel(self):
        when = _iso(timedelta(hours=2))
        r = self.http.put(f"/api/admin/bulletins/{self.id}/schedule", json={"send_at": when})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["scheduled_for"], when)
        self.assertEqual(r.json()["status"], "draft")
        later = _iso(timedelta(hours=3))
        self.assertEqual(self.http.put(f"/api/admin/bulletins/{self.id}/schedule", json={"send_at": later}).json()["scheduled_for"], later)
        self.assertIsNone(self.http.delete(f"/api/admin/bulletins/{self.id}/schedule").json()["scheduled_for"])

    def test_rejects_past_times_missing_timezone_and_no_audience(self):
        url = f"/api/admin/bulletins/{self.id}/schedule"
        self.assertEqual(self.http.put(url, json={"send_at": _iso(timedelta(minutes=-5))}).status_code, 422)
        self.assertEqual(self.http.put(url, json={"send_at": "2099-01-01T10:00:00"}).status_code, 422)
        empty = database.create_bulletin("Empty", "<p>x</p>", "x", "", [])["id"]
        self.assertEqual(self.http.put(f"/api/admin/bulletins/{empty}/schedule", json={"send_at": _iso(timedelta(hours=1))}).status_code, 409)

    def test_job_sends_only_due_bulletins_once(self):
        other = database.create_bulletin("Later", "<p>x</p>", "x", "", ["b@example.com"])["id"]
        database.schedule_bulletin(self.id, _iso(timedelta(seconds=-30)), "tester")
        database.schedule_bulletin(other, _iso(timedelta(hours=1)), "tester")
        with patch.object(bulletin_send, "send_bulletin_email") as mail:
            self.assertEqual(bulletin_send.deliver_scheduled_bulletins(), 1)
            self.assertEqual(bulletin_send.deliver_scheduled_bulletins(), 0)
        self.assertEqual(mail.call_count, 1)
        self.assertEqual(database.get_bulletin(self.id)["status"], "sent")
        self.assertEqual(database.get_bulletin(other)["status"], "draft")
        # a sent bulletin can no longer be scheduled
        self.assertEqual(self.http.put(f"/api/admin/bulletins/{self.id}/schedule", json={"send_at": _iso(timedelta(hours=1))}).status_code, 409)

    def test_failed_scheduled_send_is_not_retried_and_alerts_staff(self):
        database.schedule_bulletin(self.id, _iso(timedelta(seconds=-30)), "tester")
        with patch.object(bulletin_send, "send_bulletin_email", side_effect=RuntimeError("boom")) as mail, \
                patch("services.discord_notifier.notify_staff_alert") as alert:
            self.assertEqual(bulletin_send.deliver_scheduled_bulletins(), 0)
            self.assertEqual(bulletin_send.deliver_scheduled_bulletins(), 0)
        self.assertEqual(mail.call_count, 1)
        alert.assert_called_once()
        row = database.get_bulletin(self.id)
        self.assertEqual((row["status"], row["scheduled_for"]), ("draft", None))
        self.assertIn("boom", row["last_error"])


if __name__ == "__main__":
    unittest.main()
