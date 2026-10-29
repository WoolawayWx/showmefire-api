import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core import database
from services import fire_ingest


SATDET_FEATURE = {
    "type": "Feature",
    "geometry": {"type": "Point", "coordinates": [-92.3341, 38.9517]},
    "properties": {
        "SOURCE": "FIRMS",
        "TYPENAME": "VIIRS_SNPP_NRT",
        "SOURCE_ID": "viirs:SNPP:2026-08-01T14:30:00Z:38.952:-92.334",
        "ACQ_DATE_TIME": "2026-08-01T14:30:00Z",
        "FRP": 12.4,
        "CONFIDENCE": "nominal",
        "SATELLITE": "SNPP",
    },
}

NGFS_FEATURE = {
    "type": "Feature",
    "geometry": {"type": "Point", "coordinates": [-92.5, 39.0]},
    "properties": {
        "event_id": "2026-08-01_10-00-00_1234567",
        "event_datetime": "2026-08-01T10:00:00Z",
        "location": {"county": "Boone County", "state": "Missouri"},
        "fire_info": {"frp": 5.2, "type": "wildfire"},
    },
}


class FireIngestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary.name) / "showmefire.db"
        self.database_patch = patch.object(database, "get_db_path", return_value=self.db_path)
        self.database_patch.start()
        database.init_database()

        self.gis_dir = Path(self.temporary.name) / "gis"
        self.data_dir = Path(self.temporary.name) / "data"
        self.gis_dir.mkdir()
        self.data_dir.mkdir()
        self.satdet_path = self.gis_dir / "satfiredetection.geojson"
        self.ngfs_path = self.data_dir / "missouri_fires.geojson"

    def tearDown(self):
        self.database_patch.stop()
        self.temporary.cleanup()

    def _write(self, path: Path, features):
        with open(path, "w") as f:
            json.dump({"type": "FeatureCollection", "features": features}, f)

    def _paths(self):
        return {"satdet": self.satdet_path, "ngfs": self.ngfs_path}

    def test_ingests_viirs_feature_as_pending_unverified(self):
        self._write(self.satdet_path, [SATDET_FEATURE])
        self._write(self.ngfs_path, [])
        result = fire_ingest.ingest_detection_files(paths=self._paths())
        self.assertEqual(result["inserted"], 1)

        events = database.list_fire_events(admin=True)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["source"], "viirs")
        # Raw satellite detections require admin moderation before they are
        # public, same as user-submitted reports; see upsert_detection_event.
        self.assertEqual(events[0]["status"], "pending")
        self.assertEqual(events[0]["verification_tier"], "unverified")

    def test_detections_only_ingest_creates_no_incident(self):
        self._write(self.satdet_path, [SATDET_FEATURE])
        self._write(self.ngfs_path, [])
        result = fire_ingest.ingest_detection_files(paths=self._paths(), cluster_incidents=False)
        self.assertEqual(result["inserted"], 1)
        self.assertIsNone(database.list_fire_events(admin=True)[0]["incident_id"])
        conn = sqlite3.connect(self.db_path)
        try:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM fire_incidents").fetchone()[0], 0)
        finally:
            conn.close()

    def test_ingest_csv_adds_only_the_listed_detections_with_no_incident(self):
        from services import firms_backfill
        csv_path = Path(self.temporary.name) / "firms_neris_overlap.csv"
        csv_path.write_text(
            ",".join(firms_backfill.HISTORY_COLUMNS) + ",neris_matched\n"  # extra columns (the overlap's neris_*) are ignored
            "VIIRS_SNPP_NRT:N:2026-08-01T10:00:00Z:39.000:-92.500,viirs,VIIRS_SNPP_NRT,N,2026-08-01T10:00:00Z,39.0,-92.5,5.2,nominal,335.2,290.1,0.14,D,True\n")
        result = firms_backfill.ingest_csv(csv_path)
        self.assertEqual(result["ingest"]["inserted"], 1)
        event = database.list_fire_events(admin=True)[0]
        self.assertEqual((event["source"], event["bright_t7"], event["bright_t13"], event["daynight"]), ("viirs", 335.2, 290.1, "D"))
        self.assertIsNone(event["incident_id"])
        self.assertEqual(firms_backfill.ingest_csv(csv_path)["ingest"]["inserted"], 0)  # re-running updates, never duplicates
        self.assertEqual(len(database.list_fire_events(admin=True)), 1)

    def test_default_ingest_still_clusters_into_an_incident(self):
        self._write(self.satdet_path, [SATDET_FEATURE])
        self._write(self.ngfs_path, [])
        fire_ingest.ingest_detection_files(paths=self._paths())
        self.assertIsNotNone(database.list_fire_events(admin=True)[0]["incident_id"])

    def test_reingesting_is_idempotent(self):
        self._write(self.satdet_path, [SATDET_FEATURE])
        self._write(self.ngfs_path, [])
        fire_ingest.ingest_detection_files(paths=self._paths())
        result = fire_ingest.ingest_detection_files(paths=self._paths())
        self.assertEqual(result["inserted"], 0)
        self.assertEqual(result["updated"], 1)
        events = database.list_fire_events(admin=True)
        self.assertEqual(len(events), 1)

    def test_admin_tier_promotion_and_position_correction_survive_reingest(self):
        self._write(self.satdet_path, [SATDET_FEATURE])
        self._write(self.ngfs_path, [])
        fire_ingest.ingest_detection_files(paths=self._paths())
        events = database.list_fire_events(admin=True)
        event_id = events[0]["id"]

        database.update_fire_event(
            event_id, actor="staff@showmefire.org", edit_reason="Promoted after ground confirmation",
            latitude=39.5, longitude=-93.0, verification_tier="official_source_confirmed",
            official_source_ref="MDC-2026-0099",
        )

        # Re-run ingest; the ON CONFLICT clause must not clobber the correction.
        fire_ingest.ingest_detection_files(paths=self._paths())

        updated_event = database.get_fire_event(event_id, admin=True)
        self.assertEqual(updated_event["latitude"], 39.5)
        self.assertEqual(updated_event["longitude"], -93.0)
        self.assertEqual(updated_event["verification_tier"], "official_source_confirmed")

    def test_ingests_ngfs_feature_with_event_id_as_external_id(self):
        self._write(self.satdet_path, [])
        self._write(self.ngfs_path, [NGFS_FEATURE])
        result = fire_ingest.ingest_detection_files(paths=self._paths())
        self.assertEqual(result["inserted"], 1)

        events = database.list_fire_events(admin=True)
        self.assertEqual(events[0]["source"], "ngfs")
        self.assertEqual(events[0]["external_id"], "2026-08-01_10-00-00_1234567")

    def test_feature_without_coordinates_is_skipped_and_counted_in_skipped(self):
        broken = {"type": "Feature", "geometry": {"type": "Point", "coordinates": []}, "properties": {}}
        self._write(self.satdet_path, [broken])
        self._write(self.ngfs_path, [])
        result = fire_ingest.ingest_detection_files(paths=self._paths())
        self.assertEqual(result["skipped"], 1)
        self.assertEqual(result["inserted"], 0)

    def test_missing_input_file_is_skipped_without_raising(self):
        # Neither file was written at all.
        result = fire_ingest.ingest_detection_files(paths=self._paths())
        self.assertEqual(result["inserted"], 0)
        self.assertEqual(result["errors"], [])

    def test_dry_run_writes_nothing(self):
        self._write(self.satdet_path, [SATDET_FEATURE])
        self._write(self.ngfs_path, [NGFS_FEATURE])
        fire_ingest.ingest_detection_files(paths=self._paths(), dry_run=True)
        events = database.list_fire_events(admin=True)
        self.assertEqual(len(events), 0)


if __name__ == "__main__":
    unittest.main()
