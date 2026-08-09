import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core import database


class ExportFireLabelsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary.name) / "showmefire.db"
        self.database_patch = patch.object(database, "get_db_path", return_value=self.db_path)
        self.database_patch.start()
        database.init_database()

    def tearDown(self):
        self.database_patch.stop()
        self.temporary.cleanup()

    def _report(self, contact="chief@example.com"):
        return database.create_fire_report(
            latitude=38.9517, longitude=-92.3341,
            occurred_at="2026-08-01T14:30:00Z", occurred_at_precision="minute",
            acres=3.0, acres_is_estimate=True, fuel_types=["grass"],
            description="Grass fire along the fence line near the highway.",
            out_of_ordinary="", reporter_contact=contact,
            submitter_ip_hash="a" * 64, consent_version="v1", captcha_verdict="success",
            county_fips="29019", county_name="Boone",
        )

    def _approve(self, event_id, tier="admin_reviewed", ref=""):
        return database.set_fire_event_status(
            event_id, to_status="approved", actor="staff@showmefire.org",
            to_tier=tier, official_source_ref=ref,
        )

    def test_min_tier_official_excludes_admin_reviewed_and_unverified(self):
        official = self._report()
        self._approve(official["id"], tier="official_source_confirmed", ref="MDC-1")
        reviewed = self._report()
        self._approve(reviewed["id"], tier="admin_reviewed")

        rows = database.export_fire_labels(min_tier="official_source_confirmed")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["verification_tier"], "official_source_confirmed")

    def test_min_tier_admin_reviewed_includes_both_reviewed_tiers(self):
        official = self._report()
        self._approve(official["id"], tier="official_source_confirmed", ref="MDC-1")
        reviewed = self._report()
        self._approve(reviewed["id"], tier="admin_reviewed")

        rows = database.export_fire_labels(min_tier="admin_reviewed")
        tiers = {row["verification_tier"] for row in rows}
        self.assertEqual(tiers, {"official_source_confirmed", "admin_reviewed"})

    def test_unverified_rows_never_exported(self):
        # A detection-style row, never moderated - stays unverified.
        database.upsert_detection_event(
            source="viirs", external_id="v1", latitude=38.9, longitude=-92.3,
            occurred_at="2026-08-01T12:00:00Z",
        )
        rows = database.export_fire_labels(min_tier="admin_reviewed")
        self.assertEqual(len(rows), 0)

    def test_pending_and_rejected_rows_never_exported_regardless_of_tier(self):
        pending = self._report()
        rejected = self._report()
        database.set_fire_event_status(rejected["id"], to_status="rejected", actor="staff", reason="duplicate")

        rows = database.export_fire_labels(min_tier="unverified")
        ids = {row["event_id"] for row in rows}
        self.assertNotIn(pending["id"], ids)
        self.assertNotIn(rejected["id"], ids)

    def test_deleted_rows_never_exported(self):
        report = self._report()
        self._approve(report["id"], tier="official_source_confirmed", ref="MDC-2")
        database.delete_fire_event(report["id"], actor="staff", reason="bad data")

        rows = database.export_fire_labels(min_tier="unverified")
        self.assertEqual(len(rows), 0)

    def test_fuel_types_is_a_list_not_a_group_concat_string(self):
        report = database.create_fire_report(
            latitude=38.9517, longitude=-92.3341,
            occurred_at="2026-08-01T14:30:00Z", occurred_at_precision="minute",
            acres=3.0, acres_is_estimate=True, fuel_types=["grass", "brush"],
            description="Grass and brush fire near the tree line by the road.",
            out_of_ordinary="", reporter_contact="",
            submitter_ip_hash="b" * 64, consent_version="v1", captcha_verdict="success",
        )
        self._approve(report["id"], tier="official_source_confirmed", ref="MDC-3")
        rows = database.export_fire_labels(min_tier="official_source_confirmed")
        self.assertIsInstance(rows[0]["fuel_types"], list)
        self.assertEqual(sorted(rows[0]["fuel_types"]), ["brush", "grass"])

    def test_no_pii_columns_in_any_exported_dict(self):
        report = self._report(contact="chief@example.com")
        self._approve(report["id"], tier="official_source_confirmed", ref="MDC-4")
        rows = database.export_fire_labels(min_tier="official_source_confirmed")
        for leaked_key in ("reporter_contact", "submitter_ip_hash", "captcha_verdict"):
            self.assertNotIn(leaked_key, rows[0])

    def test_since_and_until_bound_the_window_inclusively(self):
        report = self._report()
        self._approve(report["id"], tier="official_source_confirmed", ref="MDC-5")

        in_window = database.export_fire_labels(
            min_tier="official_source_confirmed", since="2026-08-01T00:00:00Z", until="2026-08-01T23:59:59Z"
        )
        self.assertEqual(len(in_window), 1)

        out_of_window = database.export_fire_labels(
            min_tier="official_source_confirmed", since="2026-08-02T00:00:00Z"
        )
        self.assertEqual(len(out_of_window), 0)


if __name__ == "__main__":
    unittest.main()
