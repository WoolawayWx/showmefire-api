import json
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

    def test_ingests_viirs_feature_as_approved_unverified(self):
        self._write(self.satdet_path, [SATDET_FEATURE])
        self._write(self.ngfs_path, [])
        result = fire_ingest.ingest_detection_files(paths=self._paths())
        self.assertEqual(result["inserted"], 1)

        events = database.list_fire_events(admin=True)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["source"], "viirs")
        self.assertEqual(events[0]["status"], "approved")
        self.assertEqual(events[0]["verification_tier"], "unverified")

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
