import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core import database
from services import incident_area_review as review


class IncidentAreaReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary.name) / "showmefire.db"
        self.patches = [
            patch.object(database, "get_db_path", return_value=self.db_path),
            patch.object(review, "get_db_path", return_value=self.db_path),
        ]
        for p in self.patches:
            p.start()
        database.init_database()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        try:
            self.temporary.cleanup()
        except PermissionError:
            pass

    def _incident(self, lat, lon, when="2026-10-01T12:00:00Z", county="29207", members=1, source="viirs"):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        incident_id = database.find_or_create_incident_for_detection(
            cursor, lat, lon, when, county, "Stoddard", radius_km=0.01,
        )
        ids = []
        for i in range(members):
            cursor.execute(
                """INSERT INTO fire_events (source, external_id, status, latitude, longitude, occurred_at, incident_id)
                   VALUES (?, ?, 'pending', ?, ?, ?, ?)""",
                (source, f"{lat}:{lon}:{when}:{i}", lat, lon, when, incident_id),
            )
            ids.append(cursor.lastrowid)
        conn.commit()
        conn.close()
        return incident_id, ids

    def test_labels_follow_ingest_order(self):
        incident_id, ids = self._incident(36.90, -90.00, members=3)
        area = review.build_area(incident_id)
        labels = [m["label"] for m in area["incidents"][0]["members"]]
        self.assertEqual(labels, [f"{incident_id}_001", f"{incident_id}_002", f"{incident_id}_003"])
        self.assertEqual(review.resolve_seed_incident(ids[1]), incident_id)

    def test_nearby_recent_incident_is_related_and_far_one_is_not(self):
        seed, _ = self._incident(36.90, -90.00)
        near, _ = self._incident(36.93, -90.00, "2026-10-01T18:00:00Z")   # ~3.3 km
        far, _ = self._incident(37.40, -90.00, "2026-10-01T18:00:00Z")    # ~55 km
        area = review.build_area(seed)
        found = {entry["incident_id"] for entry in area["incidents"]}
        self.assertIn(near, found)
        self.assertNotIn(far, found)
        self.assertEqual(area["incidents"][0]["incident_id"], seed)

    def test_stale_incident_outside_window_is_excluded(self):
        seed, _ = self._incident(36.90, -90.00, "2026-10-01T12:00:00Z")
        old, _ = self._incident(36.91, -90.00, "2026-09-01T12:00:00Z")
        area = review.build_area(seed)
        self.assertNotIn(old, {entry["incident_id"] for entry in area["incidents"]})

    def test_unattached_recent_report_is_listed(self):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        seed, _ = self._incident(36.90, -90.00, now)
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO fire_events (source, external_id, status, latitude, longitude, occurred_at) "
            "VALUES ('user_submission', 'u1', 'pending', 36.91, -90.00, ?)", (now,),
        )
        conn.commit()
        conn.close()
        area = review.build_area(seed)
        self.assertEqual(area["summary"]["unassigned_count"], 1)
        self.assertEqual(area["unassigned_reports"][0]["source"], "user_submission")

    def test_label_supports_three_digits(self):
        self.assertEqual(review.sub_incident_label(7, 1), "7_001")
        self.assertEqual(review.sub_incident_label(7, 123), "7_123")
        self.assertEqual(review.sub_incident_label(7, 1234), "7_1234")

    def test_settings_persist_validate_and_reset(self):
        self.assertEqual(review.get_settings()["radius_km"], review.SETTING_SPECS["radius_km"][0])
        saved = review.save_settings({"radius_km": 5, "threshold": 0.7}, updated_by="tester")
        self.assertEqual((saved["radius_km"], saved["threshold"]), (5.0, 0.7))
        with self.assertRaises(ValueError):
            review.save_settings({"radius_km": 999})
        with self.assertRaises(ValueError):
            review.save_settings({"nope": 1})
        self.assertEqual(review.get_settings()["radius_km"], 5.0)  # bad update changed nothing
        self.assertEqual(review.reset_settings()["radius_km"], review.SETTING_SPECS["radius_km"][0])

    def test_radius_setting_drives_default_area(self):
        seed, _ = self._incident(36.90, -90.00)
        near, _ = self._incident(36.93, -90.00, "2026-10-01T18:00:00Z")  # ~3.3 km
        review.save_settings({"radius_km": 1})
        self.assertNotIn(near, {e["incident_id"] for e in review.build_area(seed)["incidents"]})
        review.reset_settings()
        self.assertIn(near, {e["incident_id"] for e in review.build_area(seed)["incidents"]})

    def test_pair_labels_show_in_area_and_can_be_cleared(self):
        seed, _ = self._incident(36.90, -90.00)
        near, _ = self._incident(36.93, -90.00, "2026-10-01T18:00:00Z")
        saved = review.record_pair_label(near, seed, "same", decided_by="tester")  # order must not matter
        self.assertEqual(saved["label"], "same")
        entry = [e for e in review.build_area(seed)["incidents"] if e["incident_id"] == near][0]
        self.assertEqual(entry["pair_label"], "same")
        review.record_pair_label(seed, near, "different")
        entry = [e for e in review.build_area(seed)["incidents"] if e["incident_id"] == near][0]
        self.assertEqual(entry["pair_label"], "different")
        review.record_pair_label(seed, near, "clear")
        entry = [e for e in review.build_area(seed)["incidents"] if e["incident_id"] == near][0]
        self.assertIsNone(entry["pair_label"])
        with self.assertRaises(ValueError):
            review.record_pair_label(seed, seed, "same")
        with self.assertRaises(ValueError):
            review.record_pair_label(seed, near, "maybe")
        self.assertIsNone(review.record_pair_label(seed, 99999, "same"))

    def test_fit_needs_enough_labels_then_learns_distance(self):
        import json
        self.assertFalse(review.fit_weights()["ready"])
        conn = review._pairs_conn()
        for i in range(15):  # close pairs the admin calls the same fire
            conn.execute("INSERT INTO area_review_pair_labels VALUES (?, ?, 'same', 0.8, ?, 't', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
                         (i, 100 + i, json.dumps({"nearest_member_km": 2.0, "time_gap_hours": 5, "same_county": True, "join_areas_touch": False})))
        for i in range(15):  # far pairs the admin calls different fires
            conn.execute("INSERT INTO area_review_pair_labels VALUES (?, ?, 'different', 0.4, ?, 't', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
                         (200 + i, 300 + i, json.dumps({"nearest_member_km": 11.0, "time_gap_hours": 5, "same_county": True, "join_areas_touch": False})))
        conn.commit()
        conn.close()
        fit = review.fit_weights()
        self.assertTrue(fit["ready"])
        self.assertGreaterEqual(fit["accuracy_suggested"], fit["accuracy_current"])
        self.assertGreater(fit["suggested"]["w_distance"], review.SETTING_SPECS["w_distance"][0])


if __name__ == "__main__":
    unittest.main()


class PublicReportIncidentTests(unittest.TestCase):
    """A public report joins an incident when it arrives, and stays pending for review."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary.name) / "showmefire.db"
        self.patch = patch.object(database, "get_db_path", return_value=self.db_path)
        self.patch.start()
        database.init_database()

    def tearDown(self):
        self.patch.stop()
        try:
            self.temporary.cleanup()
        except PermissionError:
            pass

    def _report(self, lat, lon, when="2026-10-01T12:00:00Z"):
        return database.create_fire_report(
            latitude=lat, longitude=lon, occurred_at=when, occurred_at_precision="minute", acres=1.0,
            acres_is_estimate=True, fuel_types=[], description="smoke", out_of_ordinary="", reporter_contact="",
            submitter_ip_hash="h", consent_version="1", captcha_verdict="ok", county_fips="29207", county_name="Stoddard")

    def _detection_incident(self, lat, lon, when="2026-10-01T12:00:00Z"):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        incident_id = database.find_or_create_incident_for_detection(conn.cursor(), lat, lon, when, "29207", "Stoddard")
        conn.execute("INSERT INTO fire_events (source, external_id, status, latitude, longitude, occurred_at, incident_id) "
                     "VALUES ('viirs', 'd1', 'pending', ?, ?, ?, ?)", (lat, lon, when, incident_id))
        conn.commit()
        conn.close()
        return incident_id

    def test_report_near_a_detection_joins_its_incident_and_stays_pending(self):
        incident_id = self._detection_incident(36.90, -90.00)
        report = self._report(36.9005, -90.0005)
        self.assertEqual(report["incident_id"], incident_id)
        self.assertEqual(report["status"], "pending")

    def test_lone_report_starts_its_own_incident(self):
        report = self._report(37.50, -92.00)
        self.assertIsNotNone(report["incident_id"])
        self.assertEqual(report["status"], "pending")

    def test_backfill_helper_is_idempotent(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("INSERT INTO fire_events (source, status, latitude, longitude, occurred_at) "
                     "VALUES ('user_submission', 'pending', 36.9, -90.0, '2026-10-01T12:00:00Z')")
        event_id = conn.execute("SELECT max(id) FROM fire_events").fetchone()[0]
        conn.commit()
        conn.close()
        first = database.correlate_report_with_incident(event_id)
        self.assertEqual(database.correlate_report_with_incident(event_id), first)

    def test_form_answers_are_stored_and_optional(self):
        answered = database.create_fire_report(
            latitude=37.5, longitude=-92.0, occurred_at="2026-10-01T12:00:00Z", occurred_at_precision="minute",
            acres=3.0, acres_is_estimate=True, fuel_types=[], description="smoke", out_of_ordinary="",
            reporter_contact="", submitter_ip_hash="h", consent_version="1", captcha_verdict="ok",
            reported_fire_status="contained", reported_structures_threatened=False, reported_responders_on_scene=True)
        self.assertEqual((answered["reported_fire_status"], answered["reported_structures_threatened"],
                          answered["reported_responders_on_scene"]), ("contained", 0, 1))
        skipped = self._report(37.6, -92.1)
        self.assertEqual((skipped["reported_fire_status"], skipped["reported_structures_threatened"]), (None, None))

    def test_admin_report_list_counts_photos_but_not_documents(self):
        report = self._report(37.5, -92.0)
        database.add_fire_event_media(report["id"], "a.jpg", "a.jpg", "image/jpeg", 10, "h1")
        database.add_fire_event_media(report["id"], "b.jpg", "b.jpg", "image/jpeg", 10, "h2")
        database.add_fire_event_media(report["id"], "c.pdf", "c.pdf", "application/pdf", 10, "h3", kind="document")
        listed = database.list_fire_events(status="pending", admin=True)
        self.assertEqual(listed[0]["photo_count"], 2)
        self.assertEqual(len(database.get_fire_event(report["id"], admin=True)["media"]), 3)
