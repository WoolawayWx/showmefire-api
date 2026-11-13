import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from services import station_feed_snapshot as snapshot

NOW = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)
STATIONS = [{"stid": "KCOU", "latitude": 38.8, "longitude": -92.2}]


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.patches = [
            patch.object(snapshot, "SNAPSHOT_DIR", Path(self.temporary.name)),
            patch.dict(os.environ, {"SMF_STATION_SNAPSHOT": "1"}),   # snapshots are skipped under pytest unless asked for
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in self.patches:
            item.stop()
        self.temporary.cleanup()

    def test_write_is_atomic_and_read_rejects_old_data(self):
        self.assertTrue(snapshot.write_snapshot("stations", {"stations": STATIONS, "last_updated": "x"}, now=NOW))
        self.assertEqual([p.name for p in Path(self.temporary.name).iterdir()], ["station_feed.json"])   # no temp file left
        self.assertEqual(snapshot.read_snapshot("stations", NOW + timedelta(minutes=10))["stations"], STATIONS)
        self.assertIsNone(snapshot.read_snapshot("stations", NOW + timedelta(minutes=31)))                # too old to forecast from
        self.assertIsNone(snapshot.read_snapshot("raws", NOW))                                            # never written

    def test_nothing_is_written_for_an_empty_feed_or_when_disabled(self):
        self.assertFalse(snapshot.write_snapshot("stations", {"stations": []}, now=NOW))
        with patch.dict(os.environ, {"SMF_STATION_SNAPSHOT": "0"}):
            self.assertFalse(snapshot.write_snapshot("stations", {"stations": STATIONS}, now=NOW))

    def test_jobs_use_live_data_when_there_is_some_and_the_snapshot_otherwise(self):
        snapshot.write_snapshot("stations", {"stations": STATIONS, "last_updated": "snap"}, now=datetime.now(timezone.utc))
        with patch("services.synoptic.get_station_data", return_value={"stations": [{"stid": "LIVE"}]}):
            self.assertEqual(snapshot.station_data_for_job()["stations"][0]["stid"], "LIVE")
        with patch("services.synoptic.get_station_data", return_value={"stations": None, "error": None}):
            shared = snapshot.station_data_for_job()          # the heavy worker: its in-memory feed is always empty
            self.assertEqual(shared["stations"], STATIONS)
            self.assertTrue(shared["from_snapshot"])
        with patch("services.synoptic.get_station_data", return_value={"stations": None, "error": "boom"}), \
                patch.object(snapshot, "read_snapshot", return_value=None):
            self.assertIsNone(snapshot.station_data_for_job()["stations"])   # stale/missing: the job skips its tick

    def test_raws_feed_the_same_way(self):
        from core import scheduler

        snapshot.write_snapshot("raws", {"stations": [{"stid": "RAWS1"}], "last_updated": "r"}, now=datetime.now(timezone.utc))
        with patch.dict(scheduler.raws_station_data, {"stations": None}):
            self.assertEqual(snapshot.raws_data_for_job()["stations"], [{"stid": "RAWS1"}])


if __name__ == "__main__":
    unittest.main()
