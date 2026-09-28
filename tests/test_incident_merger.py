import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core import database
from services import incident_merger


def _square(lon, lat, size=0.018):
    """~2 km GOES-like pixel footprint."""
    return json.dumps({"type": "Polygon", "coordinates": [[
        [lon, lat], [lon + size, lat], [lon + size, lat + size], [lon, lat + size], [lon, lat],
    ]]})


class IncidentMergerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary.name) / "showmefire.db"
        self.patches = [
            patch.object(database, "get_db_path", return_value=self.db_path),
            patch.object(incident_merger, "get_db_path", return_value=self.db_path),
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

    def _incident(self, lon, lat, when="2026-09-28T15:00:00Z", size=0.018):
        """One incident with one NGFS footprint member; returns (incident_id, slug)."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        incident_id = database.find_or_create_incident_for_detection(
            cursor, lat + size / 2, lon + size / 2, when, "29207", "Stoddard",
            radius_km=0.1,  # force one incident per call so tests control grouping
        )
        cursor.execute(
            """INSERT INTO fire_events (source, external_id, status, latitude, longitude, occurred_at,
                                        footprint_geojson, incident_id, frp)
               VALUES ('ngfs', ?, 'approved', ?, ?, ?, ?, ?, 10)""",
            (f"{lon}:{lat}:{when}", lat + size / 2, lon + size / 2, when, _square(lon, lat, size), incident_id),
        )
        slug = cursor.execute("SELECT public_slug FROM fire_incidents WHERE id = ?", (incident_id,)).fetchone()[0]
        conn.commit()
        conn.close()
        return incident_id, slug

    def _status(self, incident_id):
        conn = sqlite3.connect(self.db_path)
        try:
            return conn.execute(
                "SELECT status, merged_into_id, detection_count FROM fire_incidents WHERE id = ?", (incident_id,)
            ).fetchone()
        finally:
            conn.close()

    def test_adjacent_incidents_merge_into_oldest(self):
        a, slug_a = self._incident(-90.030, 36.900, "2026-09-28T12:00:00Z")
        b, slug_b = self._incident(-90.012, 36.900, "2026-09-28T15:00:00Z")  # touching pixel
        result = incident_merger.merge_touching_incidents()
        self.assertEqual(result["merged_away"], 1)
        self.assertEqual(self._status(a), ("active", None, 2))
        self.assertEqual(self._status(b)[:2], ("merged", a))
        # old link still resolves to the survivor
        self.assertEqual(database.get_public_fire_incident(slug_b)["id"], a)
        # merged incident is not listed
        self.assertEqual([i["id"] for i in database.list_fire_incidents()], [a])

    def test_distant_incidents_do_not_merge(self):
        a, _ = self._incident(-90.030, 36.900)
        b, _ = self._incident(-89.700, 36.900)  # ~30 km away
        result = incident_merger.merge_touching_incidents()
        self.assertEqual(result["merged_away"], 0)
        self.assertEqual(self._status(b)[0], "active")

    def test_far_apart_in_time_do_not_merge(self):
        a, _ = self._incident(-90.030, 36.900, "2026-09-20T12:00:00Z")
        b, _ = self._incident(-90.012, 36.900, "2026-09-28T15:00:00Z")
        self.assertEqual(incident_merger.merge_touching_incidents()["merged_away"], 0)

    def test_feedback_moves_to_survivor_and_pass_is_idempotent(self):
        a, _ = self._incident(-90.030, 36.900, "2026-09-28T12:00:00Z")
        b, _ = self._incident(-90.012, 36.900, "2026-09-28T15:00:00Z")
        database.create_fire_incident_feedback(b, "confirmed_fire", "", "", "hash")
        incident_merger.merge_touching_incidents()
        self.assertEqual(incident_merger.merge_touching_incidents()["merged_away"], 0)
        conn = sqlite3.connect(self.db_path)
        try:
            ids = [r[0] for r in conn.execute("SELECT incident_id FROM fire_incident_feedback")]
        finally:
            conn.close()
        self.assertEqual(ids, [a])

    def test_oversized_group_is_reported_not_merged(self):
        ids = [self._incident(-90.030 + 0.018 * i, 36.900)[0] for i in range(4)]  # chain of touching pixels
        result = incident_merger.merge_touching_incidents(max_extent_km=3.0)
        self.assertEqual(result["merged_away"], 0)
        self.assertEqual(len(result["oversized"]), 1)
        self.assertTrue(all(self._status(i)[0] == "active" for i in ids))

    def test_dry_run_writes_nothing(self):
        a, _ = self._incident(-90.030, 36.900, "2026-09-28T12:00:00Z")
        b, _ = self._incident(-90.012, 36.900, "2026-09-28T15:00:00Z")
        result = incident_merger.merge_touching_incidents(dry_run=True)
        self.assertEqual(result["merged_away"], 1)
        self.assertEqual(self._status(b)[0], "active")


if __name__ == "__main__":
    unittest.main()
