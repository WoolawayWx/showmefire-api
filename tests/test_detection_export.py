import csv
import importlib.util
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import database
from services import detection_review as review

_spec = importlib.util.spec_from_file_location("export_detection_training_data",
                                               Path(__file__).resolve().parent.parent / "scripts" / "export_detection_training_data.py")
exporter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exporter)


class ExportReviewTablesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "showmefire.db"
        self.patch = patch.object(database, "get_db_path", return_value=self.db_path)
        self.patch.start()
        database.init_database()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def _event(self):
        return database.upsert_detection_event(source="viirs", external_id="a", latitude=38.0, longitude=-92.0,
                                               occurred_at="2026-09-01T18:00:00Z", frp=5.0, cluster_incidents=False)["event_id"]

    def test_review_tables_exported_with_pseudonymous_reviewer(self):
        eid = self._event()
        review.submit_outcome(eid, "not_vegetation_fire", "staff@example.org", reason="routine_industrial_heat", db_path=self.db_path)
        manifest = exporter.export(db_path=self.db_path, out_dir=Path(self.tmp.name) / "out")
        out = manifest.parent
        for name in ("review_outcomes.csv", "review_selection.csv", "feature_snapshots.csv", "score_history.csv",
                     "recurring_sources.csv", "exclusion_zones.csv"):
            self.assertTrue((out / name).exists(), name)
        rows = list(csv.DictReader(open(out / "review_outcomes.csv", encoding="utf-8")))
        self.assertEqual(rows[0]["outcome"], "not_vegetation_fire")
        self.assertNotIn("@", rows[0]["reviewer"])
        self.assertEqual(len(rows[0]["reviewer"]), 10)

    def test_older_database_without_the_new_tables_still_exports(self):
        self._event()
        conn = sqlite3.connect(self.db_path)
        for table in ("detection_review_outcome", "detection_review_selection", "detection_feature_snapshot", "detection_score_history"):
            conn.execute(f"DROP TABLE {table}")
        conn.commit()
        conn.close()
        manifest = exporter.export(db_path=self.db_path, out_dir=Path(self.tmp.name) / "out")
        rows = list(csv.reader(open(manifest.parent / "review_outcomes.csv", encoding="utf-8")))
        self.assertEqual(len(rows), 1)  # header only


if __name__ == "__main__":
    unittest.main()
