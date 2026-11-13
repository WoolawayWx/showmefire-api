import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from core import database
from services import incident_history, mobile_push, official_incidents

NOW = datetime(2026, 11, 13, 18, 0, tzinfo=timezone.utc)  # noon Central: outside push quiet hours


def _feature(irwin="{AAA-1}", name="Rockhouse Fire", acres=5.0, pct=None, lat=36.61, lon=-93.78, fips="29009"):
    return {
        "attributes": {"IrwinID": irwin, "IncidentName": name, "POOCounty": "Barry", "POOFips": fips, "IncidentSize": acres,
                       "PercentContained": pct, "FireDiscoveryDateTime": int((NOW - timedelta(hours=3)).timestamp() * 1000),
                       "CreatedOnDateTime_dt": int((NOW - timedelta(hours=1)).timestamp() * 1000)},
        "geometry": {"x": lon, "y": lat},
    }


class IncidentHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary.name) / "showmefire.db"
        self.patches = [
            patch.object(module, "get_db_path", return_value=self.db_path)
            for module in (database, mobile_push, incident_history, official_incidents)
        ]
        for item in self.patches:
            item.start()
        database.init_database()

    def tearDown(self):
        for item in self.patches:
            item.stop()
        self.temporary.cleanup()

    def _detection(self, n, minutes_ago=180, lat=36.61, lon=-93.78):
        when = (NOW - timedelta(minutes=minutes_ago)).isoformat().replace("+00:00", "Z")
        return database.upsert_detection_event(
            source="viirs", external_id=f"d{n}", latitude=lat, longitude=lon, occurred_at=when,
            county_fips="29009", county_name="Barry", frp=10.0, confidence="n", satellite="N20",
        )

    def _incident_id(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            return connection.execute("SELECT id FROM fire_incidents ORDER BY id LIMIT 1").fetchone()[0]

    def _sync_official(self, features, now=NOW):
        parsed = [official_incidents.parse_incident(f) for f in features]
        with patch.object(official_incidents, "fetch_usfs_missouri_incidents", return_value=parsed):
            return official_incidents.sync_official_incidents(now)

    def _subscribe(self):
        mobile_push.upsert_subscription(
            installation_id="11111111-1111-4111-8111-111111111111",
            expo_push_token="ExpoPushToken[abcdefghijklmnopqrstuv]", platform="ios", app_version="1.2.0",
            forecast=False, sitrep=False, fire_weather=False, fire_detection=True, county_fips=["29009"],
        )

    def test_official_listing_joins_the_detection_incident_and_writes_its_timeline(self):
        self._detection(1)
        self._detection(2, minutes_ago=170)
        self._sync_official([])                                   # first run: baseline only
        self.assertEqual(self._sync_official([_feature()]), 0)    # nobody subscribed yet, nothing to send
        incident_id = self._incident_id()
        with closing(sqlite3.connect(self.db_path)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM fire_incidents").fetchone()[0], 1)
            self.assertEqual(connection.execute("SELECT incident_id FROM fire_events WHERE source = 'nifc_wfigs'").fetchone()[0], incident_id)

        official = official_incidents.official_source_for(incident_id)
        self.assertEqual((official["name"], official["agency_name"], official["acres"], official["active"]),
                         ("Rockhouse Fire", "U.S. Forest Service", 5.0, True))

        self._sync_official([_feature(acres=12.0, pct=20.0, name="Rockhouse Fire")], NOW + timedelta(minutes=10))
        self._sync_official([], NOW + timedelta(minutes=20))
        summaries = [entry["summary"] for entry in incident_history.list_history(incident_id)]
        self.assertTrue(any(s.startswith("Listed by the U.S. Forest Service as the Rockhouse Fire (5 acres)") for s in summaries))
        self.assertIn("Size updated from 5 acres to 12 acres", summaries)
        self.assertIn("Containment updated to 20%", summaries)
        self.assertTrue(any(s.startswith("No longer on the Forest Service") for s in summaries))
        self.assertFalse(official_incidents.official_source_for(incident_id)["active"])

    def test_new_listing_reuses_the_fire_detection_push_for_opted_in_county(self):
        self._subscribe()
        self._detection(1)
        self._sync_official([])  # baseline
        with patch.object(mobile_push, "_send_batch", return_value=[{"status": "ok", "id": "t1"}]) as send:
            self.assertEqual(self._sync_official([_feature()]), 1)
            message = send.call_args.args[0][0]
            self.assertEqual(message["title"], "Forest Service is working the Rockhouse Fire")
            self.assertEqual(message["data"]["eventType"], "fire_detection")
            self.assertTrue(message["data"]["url"].startswith("/incidents/"))
            self.assertIn("near Barry County", message["body"])
            self.assertEqual(self._sync_official([_feature()], NOW + timedelta(minutes=10)), 0)   # once per record
            self.assertEqual(send.call_count, 1)
        incident_id = self._incident_id()
        notes = [e for e in incident_history.list_history(incident_id) if e["kind"] == "notification"]
        self.assertEqual([e["summary"] for e in notes], ["Told 1 person near Barry County the Forest Service listed it"])
        # It shows up in this device's notification history too.
        self.assertEqual(len(mobile_push.notification_history("11111111-1111-4111-8111-111111111111")), 1)

    def test_feedback_prompt_push_lands_on_the_timeline(self):
        self._subscribe()
        self._detection(1)
        incident_id = self._incident_id()
        mobile_push.record_event("fire_detection:baseline:v2", "fire_detection", {})
        candidate = {"incident_id": incident_id, "incident_slug": "abc", "county_fips": "29009", "county_name": "Barry",
                     "confidence_label": "high", "latitude": 36.61, "longitude": -93.78}
        with patch.object(mobile_push, "_send_batch", return_value=[{"status": "ok", "id": "t9"}]):
            self.assertEqual(mobile_push.notify_feedback_candidates([candidate], now=NOW), 1)
        notes = [e for e in incident_history.list_history(incident_id) if e["kind"] == "notification"]
        self.assertEqual([e["summary"] for e in notes], ["Asked 1 person near Barry County to help verify it"])
        self.assertIsNone(notes[0]["status"])

    def test_staff_approval_is_a_confirmed_entry_even_if_recorded_before_statuses_existed(self):
        self._detection(1)
        incident_id = self._incident_id()
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.execute(
                "UPDATE fire_incidents SET review_decision = 'approved', review_label = 'wildfire', "
                "review_decided_at = '2026-11-13 00:18:00' WHERE id = ?", (incident_id,))
            connection.execute(  # the old shape: no status, older wording
                "INSERT INTO fire_incident_history (incident_id, occurred_at, kind, summary, dedupe_key) "
                "VALUES (?, '2026-11-13T00:18:00Z', 'review', 'Reviewed by Show Me Fire staff and labelled a wildfire', "
                "'review:2026-11-13 00:18:00')", (incident_id,))
        incident_history.sync_incident_history(NOW)
        review = [e for e in incident_history.list_history(incident_id) if e["kind"] == "review"]
        self.assertEqual([(e["summary"], e["status"]) for e in review],
                         [("Confirmed by Show Me Fire staff as a wildfire", "approved")])

    def test_listing_with_no_matching_detection_starts_an_incident(self):
        self._sync_official([])
        self._sync_official([_feature(irwin="{BBB-2}", name="Lone Fire", lat=36.9, lon=-91.5, fips="29203")])
        with closing(sqlite3.connect(self.db_path)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM fire_incidents").fetchone()[0], 1)

    def test_history_sync_records_changes_feedback_and_reports_once(self):
        for n in range(3):
            self._detection(n, minutes_ago=200 - n)
        incident_id = self._incident_id()
        self.assertGreaterEqual(incident_history.sync_incident_history(NOW), 1)
        kinds = [entry["kind"] for entry in incident_history.list_history(incident_id)]
        self.assertEqual(kinds.count("first_detected"), 1)

        for n in range(3, 8):
            self._detection(n, minutes_ago=100 - n)
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.execute(
                "INSERT INTO fire_incident_feedback (incident_id, classification, note, status, reviewed_at) "
                "VALUES (?, 'confirmed_fire', 'secret note', 'approved', '2026-11-13 17:00:00')", (incident_id,))
            connection.execute(
                "INSERT INTO fire_incident_feedback (incident_id, classification, note, status) "
                "VALUES (?, 'not_a_fire', 'unreviewed', 'pending')", (incident_id,))
        incident_history.sync_incident_history(NOW + timedelta(minutes=5))
        incident_history.sync_incident_history(NOW + timedelta(minutes=10))   # idempotent
        history = incident_history.list_history(incident_id)
        summaries = [entry["summary"] for entry in history]
        self.assertTrue(any(s.startswith("Satellite detections grew from 3 to 8") for s in summaries))
        self.assertEqual(summaries.count("Public feedback submitted as a fire, approved"), 1)
        self.assertEqual(summaries.count("Public feedback submitted as not a fire, awaiting review"), 1)
        # What it was submitted as is shown at every stage; comment text never is.
        self.assertNotIn("secret note", str(history))
        self.assertNotIn("unreviewed", str(history))
        statuses = {entry["summary"]: entry["status"] for entry in history if entry["kind"] == "feedback"}
        self.assertEqual(statuses["Public feedback submitted as a fire, approved"], "approved")
        self.assertEqual(statuses["Public feedback submitted as not a fire, awaiting review"], "pending")

        # Review changes the same entry in place: pending -> rejected.
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.execute("UPDATE fire_incident_feedback SET status = 'rejected' WHERE note = 'unreviewed'")
        incident_history.sync_incident_history(NOW + timedelta(minutes=15))
        after = [e for e in incident_history.list_history(incident_id) if e["kind"] == "feedback"]
        self.assertEqual(len(after), 2)
        self.assertEqual(sorted(e["status"] for e in after), ["approved", "rejected"])
        self.assertIn("Public feedback submitted as not a fire, not accepted", [e["summary"] for e in after])

    def test_online_reports_appear_on_the_timeline_without_their_content(self):
        self._detection(1, minutes_ago=120)
        incident_id = self._incident_id()
        when = (NOW - timedelta(minutes=60)).isoformat().replace("+00:00", "Z")
        event, _created = database.upsert_ingest_report(
            source="muse", external_id="post-1", title="Smoke west of town", description="secret post text",
            source_url="https://facebook.com/some/post", source_name="Some Facebook Group", latitude=36.61, longitude=-93.78,
            occurred_at=when, occurred_at_precision="minute", acres=None, county_fips="29009", county_name="Barry",
            origin={"kind": "osint_api", "feed": "Muse monitoring"},
        )
        incident_history.sync_incident_history(NOW)
        incident_history.sync_incident_history(NOW + timedelta(minutes=5))        # idempotent
        history = [e for e in incident_history.list_history(incident_id) if e["kind"] == "osint"]
        self.assertEqual([(e["summary"], e["status"], e["marker"]) for e in history],
                         [("An online report was received, awaiting review", "pending", "public_report")])
        text = str(incident_history.list_history(incident_id))
        for private in ("secret post text", "facebook.com", "Some Facebook Group", "Muse"):
            self.assertNotIn(private, text)

        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.execute("UPDATE fire_events SET status = 'approved' WHERE id = ?", (event["id"],))
        incident_history.sync_incident_history(NOW + timedelta(minutes=10))
        after = [e for e in incident_history.list_history(incident_id) if e["kind"] == "osint"]
        self.assertEqual([(e["summary"], e["status"]) for e in after], [("An online report was added", "approved")])


if __name__ == "__main__":
    unittest.main()


class AssignMarkersTests(unittest.TestCase):
    def entry(self, at, kind, status=None):
        return {"at": at, "kind": kind, "summary": kind, "status": status, "data": {}}

    def test_blue_before_staff_confirm_and_green_after(self):
        from services.incident_history import assign_markers

        entries = [
            self.entry("2026-11-13T09:40:00Z", "dispatch"),
            self.entry("2026-11-13T09:15:00Z", "detections"),
            self.entry("2026-11-13T08:50:00Z", "review", "approved"),
            self.entry("2026-11-12T23:05:00Z", "report", "pending"),
            self.entry("2026-11-12T22:30:00Z", "confidence"),
            self.entry("2026-11-12T22:28:00Z", "notification"),
            self.entry("2026-11-12T22:26:00Z", "first_detected"),
        ]
        markers = [e["marker"] for e in assign_markers(entries)]
        self.assertEqual(markers, ["update_confirmed", "update_confirmed", "confirmed", "public_report", "update", "update", "first"])

    def test_dispatch_started_incident_marks_its_first_entry_and_rejected_reports_stay_red(self):
        from services.incident_history import assign_markers

        entries = [self.entry("2026-11-13T10:00:00Z", "report", "rejected"), self.entry("2026-11-13T09:00:00Z", "dispatch")]
        self.assertEqual([e["marker"] for e in assign_markers(entries)], ["rejected", "first"])

    def test_a_report_after_confirmation_is_just_a_green_update(self):
        from services.incident_history import assign_markers

        entries = [self.entry("2026-11-13T11:00:00Z", "report", "approved"), self.entry("2026-11-13T10:00:00Z", "review", "approved"),
                   self.entry("2026-11-13T09:00:00Z", "first_detected")]
        self.assertEqual([e["marker"] for e in assign_markers(entries)], ["update_confirmed", "confirmed", "first"])
