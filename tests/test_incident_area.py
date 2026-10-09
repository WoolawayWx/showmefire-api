import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shapely.geometry import Point, shape

from core import database
from services import incident_merger
from services.incident_area import buffer_km, extent_km


def _square(lon, lat, size=0.018):
    return json.dumps({"type": "Polygon", "coordinates": [[
        [lon, lat], [lon + size, lat], [lon + size, lat + size], [lon, lat + size], [lon, lat],
    ]]})


class IncidentAreaTests(unittest.TestCase):
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

    def _detect(self, n, lon, lat, when="2026-10-01T12:00:00"):
        return database.upsert_detection_event("viirs", f"d{n}", lat, lon, when, county_fips="29051", county_name="Cole")

    def _incident_ids(self):
        conn = sqlite3.connect(self.db_path)
        try:
            return [r[0] for r in conn.execute("SELECT DISTINCT incident_id FROM fire_events ORDER BY incident_id")]
        finally:
            conn.close()

    def test_spreading_chain_stays_one_incident_and_far_detection_starts_another(self):
        # ~1 km apart along a line, 8 long: the later points are far from the running centroid.
        for i in range(8):
            self._detect(i, -92.0 + i * 0.011, 38.5, when=f"2026-10-01T12:{i:02d}:00")
        self.assertEqual(len(self._incident_ids()), 1, "chain should follow the shape into one incident")

        self._detect(99, -92.0 + 3 * 0.011, 38.5 + 0.05)  # ~5.5 km north of the line
        self.assertEqual(len(self._incident_ids()), 2)

    def test_area_follows_shape_and_has_buffer(self):
        for i in range(6):
            self._detect(i, -92.0 + i * 0.011, 38.5)
        conn = sqlite3.connect(self.db_path)
        raw, = conn.execute("SELECT join_area_geojson FROM fire_incidents").fetchone()
        conn.close()
        area = shape(json.loads(raw))
        self.assertEqual(area.geom_type, "Polygon")
        self.assertTrue(area.contains(Point(-92.0 + 0.027, 38.5)))          # on the line
        self.assertTrue(area.contains(Point(-92.0 + 0.027, 38.5 + 0.006)))  # ~0.7 km off the line: inside the buffer
        self.assertFalse(area.contains(Point(-92.0 + 0.027, 38.5 + 0.04)))  # ~4.4 km off: outside
        # Elongated along the line rather than a circle.
        minx, miny, maxx, maxy = area.bounds
        self.assertGreater((maxx - minx) * 0.78, (maxy - miny) * 1.5)

    def test_old_incidents_get_an_area_by_backfill(self):
        self._detect(1, -92.0, 38.5)
        conn = sqlite3.connect(self.db_path)
        conn.execute("UPDATE fire_incidents SET join_area_geojson = NULL, join_min_lat = NULL")
        conn.commit()
        conn.close()
        self.assertEqual(database.backfill_incident_areas(), 1)
        conn = sqlite3.connect(self.db_path)
        self.assertIsNotNone(conn.execute("SELECT join_area_geojson FROM fire_incidents").fetchone()[0])
        conn.close()

    def test_merge_rebuilds_area_with_buffer_around_combined_shape(self):
        def incident(lon, lat, tag):
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            incident_id = database.find_or_create_incident_for_detection(
                cursor, lat + 0.009, lon + 0.009, "2026-10-01T12:00:00", "29051", "Cole", radius_km=0.1)
            cursor.execute(
                """INSERT INTO fire_events (source, external_id, status, latitude, longitude, occurred_at,
                                            footprint_geojson, incident_id, frp)
                   VALUES ('ngfs', ?, 'approved', ?, ?, '2026-10-01T12:00:00', ?, ?, 10)""",
                (tag, lat + 0.009, lon + 0.009, _square(lon, lat), incident_id))
            conn.commit()
            conn.close()
            return incident_id

        a = incident(-92.00, 38.50, "a")
        b = incident(-91.98, 38.50, "b")   # touching neighbour
        self.assertNotEqual(a, b)
        summary = incident_merger.merge_touching_incidents()
        self.assertEqual(summary["merged_away"], 1)

        conn = sqlite3.connect(self.db_path)
        survivor, raw = conn.execute(
            "SELECT id, join_area_geojson FROM fire_incidents WHERE status = 'active'").fetchone()
        conn.close()
        area = shape(json.loads(raw))
        union_outline = shape(json.loads(_square(-92.00, 38.50))).union(shape(json.loads(_square(-91.98, 38.50))))
        self.assertTrue(area.contains(union_outline), "area covers both original footprints")
        self.assertGreater(extent_km(area), extent_km(union_outline) + 1.5, "buffer added around the merged shape")


if __name__ == "__main__":
    unittest.main()
