import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core import database
from services import detection_validation as dv


class DetectionValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "smf.db"
        patcher = patch.object(database, "get_db_path", return_value=self.db)
        patcher.start()
        self.addCleanup(patcher.stop)
        database.init_database()
        self.conn = sqlite3.connect(self.db)
        self.addCleanup(self.conn.close)

    def tearDown(self):
        self.conn.close()
        try:
            self.tmp.cleanup()
        except PermissionError:
            pass

    def _event(self, source, lat, lon, when, status="approved", acres=None, precision="minute"):
        self.conn.execute(
            "INSERT INTO fire_events (source, external_id, status, latitude, longitude, occurred_at, acres, "
            "occurred_at_precision) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (source, f"{source}-{lat}-{when}", status, lat, lon, when, acres, precision))
        self.conn.commit()

    def test_a_nearby_detection_after_the_report_counts_and_reports_the_lead_time(self):
        self._event("user_submission", 37.0, -92.0, "2026-10-01T12:00:00Z", acres=40)
        self._event("viirs", 37.01, -92.0, "2026-10-01T18:30:00Z", status="pending")  # ~1.1 km, 6.5 h later
        result = dv.compare(self.conn)
        self.assertEqual((result["reference_fires"], result["detected"]), (1, 1))
        self.assertAlmostEqual(result["median_lead_hours"], 6.5)
        self.assertEqual(result["by_size"]["10-100 acres"], {"fires": 1, "detected": 1})
        self.assertEqual(result["detected_by_source"], {"viirs": 1})

    def test_too_far_too_late_deleted_and_unapproved_things_do_not_match(self):
        self._event("user_submission", 37.0, -92.0, "2026-10-01T12:00:00Z", acres=0.5)
        self._event("viirs", 37.1, -92.0, "2026-10-01T13:00:00Z")                      # 11 km away
        self._event("viirs", 37.0, -92.0, "2026-10-03T13:00:00Z")                      # 49 h later
        self._event("viirs", 37.0, -92.0, "2026-10-01T13:00:00Z", status="deleted")    # deleted
        self._event("user_submission", 38.0, -93.0, "2026-10-01T12:00:00Z", status="pending")  # not confirmed: not a reference
        result = dv.compare(self.conn)
        self.assertEqual((result["reference_fires"], result["detected"]), (1, 0))
        self.assertEqual(result["by_size"], {"under 1 acre": {"fires": 1, "detected": 0}})

    def test_a_detection_before_the_report_counts_inside_the_window_and_negative_lead_means_seen_first(self):
        self._event("official", 37.0, -92.0, "2026-10-01T12:00:00Z")
        self._event("ngfs", 37.0, -92.0, "2026-10-01T10:00:00Z")
        self.assertAlmostEqual(dv.compare(self.conn)["median_lead_hours"], -2.0)
        self.assertEqual(dv.compare(self.conn, before_hours=1)["detected"], 0)

    def test_a_day_precision_report_gets_twelve_hours_of_slack(self):
        self._event("official", 37.0, -92.0, "2020-04-02T12:00:00Z", precision="day")
        self._event("viirs", 37.0, -92.0, "2020-04-01T03:00:00Z")   # 33 h before: outside 6 h, inside 6 + 12 + day
        self.assertEqual(dv.compare(self.conn, before_hours=22)["detected"], 1)
        self.assertEqual(dv.compare(self.conn, before_hours=20)["detected"], 0)

    def test_coverage_reports_each_sources_date_range_and_the_satellite_span(self):
        self._event("viirs", 37.0, -92.0, "2026-03-27T07:14:00Z")
        self._event("ngfs", 37.0, -92.0, "2026-09-05T18:11:21Z", status="rejected")
        self._event("official", 37.0, -92.0, "2019-05-05T12:00:00Z")
        cov = dv.coverage(self.conn)
        self.assertEqual((cov["satellite_first"][:10], cov["satellite_last"][:10]), ("2026-03-27", "2026-09-05"))
        official = next(s for s in cov["sources"] if s["source"] == "official")
        self.assertEqual(official["first"][:4], "2019")

    def test_nothing_to_compare_is_not_an_error(self):
        self.assertEqual(dv.compare(self.conn)["reference_fires"], 0)
        self.assertIsNone(dv.compare(self.conn)["detection_rate"])


if __name__ == "__main__":
    unittest.main()
