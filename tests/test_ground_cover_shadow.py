import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from core import database
from services import ground_cover, ground_cover_shadow as shadow
from services.incident_confidence import CROPLAND_PENALTY, WATER_PENALTY
from tests.test_ground_cover import _cell_centre_lonlat, _polygon, _write_bundle

NOW = datetime(2026, 11, 13, 18, 0, tzinfo=timezone.utc)


def _members(land_cover=None, n=6):
    return [{"occurred_at": (NOW - timedelta(minutes=10 * (n - i))).isoformat(), "frp": 20.0, "source": "viirs",
             "confidence": "nominal", "land_cover": land_cover} for i in range(n)]


class ScorePairTests(unittest.TestCase):
    def test_cropland_and_water_stand_ins_lower_the_shadow_score(self):
        members = _members()
        crops = shadow.score_pair({}, members, "Cropland:100")
        self.assertEqual(crops["live"]["score"] - crops["shadow"]["score"], CROPLAND_PENALTY)
        water = shadow.score_pair({}, members, "Water:80,Trees:20")
        self.assertEqual(water["live"]["score"] - water["shadow"]["score"], round(WATER_PENALTY * 0.8))
        woods = shadow.score_pair({}, members, "Trees:100")
        self.assertEqual(woods["live"]["score"], woods["shadow"]["score"])

    def test_inputs_are_never_modified(self):
        members, incident = _members(), {"approved_feedback_counts": {"confirmed_fire": 1}}
        shadow.score_pair(incident, members, "Cropland:100")
        self.assertTrue(all(m["land_cover"] is None for m in members))
        self.assertEqual(incident, {"approved_feedback_counts": {"confirmed_fire": 1}})

    def test_stand_in_string_format(self):
        self.assertEqual(
            shadow.stand_in_land_cover([{"group": "Trees", "pct": 55}, {"group": "Cropland", "pct": 40},
                                        {"group": "Other", "pct": 5}]),
            "Trees:55,Cropland:40")
        self.assertEqual(shadow.stand_in_land_cover(None), "")


class ScanTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary.name) / "showmefire.db"
        self.bundle = Path(self.temporary.name) / "gc.nc"
        self.x0, self.y1 = _write_bundle(self.bundle)
        self.patches = [
            patch.object(database, "get_db_path", return_value=self.db_path),
            patch.dict(os.environ, {"GROUND_COVER_STATIC_BUNDLE": str(self.bundle)}),
        ]
        for p in self.patches:
            p.start()
        ground_cover._RUNTIME = None
        ground_cover._cached.cache_clear()
        database.init_database()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        ground_cover._RUNTIME = None
        ground_cover._cached.cache_clear()
        try:
            self.temporary.cleanup()
        except PermissionError:
            pass

    def _incident(self, lon, lat, footprint=None, land_cover=None, source="viirs"):
        when = (NOW - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        incident_id = database.find_or_create_incident_for_detection(cursor, lat, lon, when, "29207", "Stoddard", radius_km=0.1)
        for i in range(4):
            stamp = (NOW - timedelta(hours=1, minutes=-10 * i)).strftime("%Y-%m-%dT%H:%M:%SZ")
            cursor.execute(
                "INSERT INTO fire_events (source, external_id, status, latitude, longitude, occurred_at, footprint_geojson, "
                "incident_id, frp, land_cover, confidence) VALUES (?, ?, 'approved', ?, ?, ?, ?, ?, 20, ?, 'nominal')",
                (source, f"{lon}:{lat}:{i}", lat, lon, stamp, footprint, incident_id, land_cover))
        conn.commit()
        conn.close()
        return incident_id

    def _rows(self, table):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in conn.execute(f"SELECT * FROM {table} ORDER BY id")]
        finally:
            conn.close()

    def test_records_only_incidents_without_a_noaa_string(self):
        lon, lat = _cell_centre_lonlat(self.x0, self.y1, 7, 5)
        crops = self._incident(lon, lat, footprint=_polygon(self.x0, self.y1, 6, 9, 3, 7, 10))
        noaa_lon, noaa_lat = _cell_centre_lonlat(self.x0, self.y1, 7, 2)
        self._incident(noaa_lon, noaa_lat, land_cover="Trees:90,Cropland:10", source="ngfs")
        self._incident(-80.0, 30.0)  # off the ground cover grid
        summary = shadow.run_shadow_scan(now=NOW)
        self.assertEqual(summary, {"considered": 3, "has_noaa_string": 1, "no_ground_cover": 1,
                                   "unchanged": 0, "recorded": 1})
        (row,) = self._rows("incident_confidence_shadow")
        self.assertEqual(row["incident_id"], crops)
        self.assertEqual(row["stand_in_land_cover"], "Cropland:100")
        self.assertEqual(row["delta"], -CROPLAND_PENALTY)
        self.assertEqual(row["live_score"] - row["shadow_score"], CROPLAND_PENALTY)
        self.assertEqual((row["basis"], row["cropland_pct"], row["water_pct"]), ("footprint", 100.0, 0.0))
        self.assertGreaterEqual(row["cells"], 5)

    def test_never_changes_live_incident_or_detection_rows(self):
        lon, lat = _cell_centre_lonlat(self.x0, self.y1, 7, 5)
        self._incident(lon, lat)
        before = (self._rows("fire_incidents"), self._rows("fire_events"))
        shadow.run_shadow_scan(now=NOW)
        self.assertEqual((self._rows("fire_incidents"), self._rows("fire_events")), before)
        self.assertTrue(all(r["land_cover"] is None for r in self._rows("fire_events")))

    def test_rerun_does_not_duplicate_unchanged_results(self):
        lon, lat = _cell_centre_lonlat(self.x0, self.y1, 7, 5)
        self._incident(lon, lat)
        shadow.run_shadow_scan(now=NOW)
        again = shadow.run_shadow_scan(now=NOW + timedelta(minutes=30))
        self.assertEqual((again["recorded"], again["unchanged"]), (0, 1))
        self.assertEqual(len(self._rows("incident_confidence_shadow")), 1)

    def test_can_be_switched_off_and_is_inert_without_the_product(self):
        lon, lat = _cell_centre_lonlat(self.x0, self.y1, 7, 5)
        self._incident(lon, lat)
        with patch.dict(os.environ, {"SMF_GROUND_COVER_SHADOW": "0"}):
            self.assertEqual(shadow.run_shadow_scan(now=NOW)["recorded"], 0)
        ground_cover._RUNTIME = None
        ground_cover._cached.cache_clear()
        with patch.object(ground_cover, "_locate_bundle", side_effect=FileNotFoundError("none")):
            summary = shadow.run_shadow_scan(now=NOW)
        self.assertEqual((summary["recorded"], summary["no_ground_cover"]), (0, 1))
        self.assertEqual(self._rows("incident_confidence_shadow"), [])

    def test_review_summarises_changes_and_readiness(self):
        lon, lat = _cell_centre_lonlat(self.x0, self.y1, 7, 5)
        self._incident(lon, lat)
        shadow.run_shadow_scan(now=NOW)
        report = shadow.review(now=NOW + timedelta(minutes=5))
        self.assertEqual(report["incidents"], 1)
        self.assertFalse(report["ready_for_review"])  # one incident, minutes of data
        self.assertEqual(report["score_change"]["lowered"], 1)
        self.assertEqual(report["largest_changes"][0]["delta"], -CROPLAND_PENALTY)
        self.assertEqual(report["by_basis"], {"point": 1})
        empty = shadow.review(days=0, now=NOW + timedelta(days=30))
        self.assertEqual((empty["incidents"], empty["score_change"]["mean"]), (0, None))

    def test_job_is_registered_in_a_group(self):
        from core.job_groups import API, JOB_GROUPS

        self.assertEqual(JOB_GROUPS["ground_cover_shadow"], API)


if __name__ == "__main__":
    unittest.main()
