import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

API_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(API_DIR))

from maps import station_history as sh

CHICAGO_TZ = sh.CHICAGO_TZ


class StationHistoryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "test_station_history.sqlite"

    def tearDown(self):
        self._tmp.cleanup()

    def test_record_and_read_today_extreme(self):
        base = datetime(2026, 7, 15, 18, 0, tzinfo=timezone.utc)  # ~1pm CDT
        stations = [{"stid": "S1", "observations": {"air_temp": {"value": 80.0}}}]
        fields = {"air_temp_f": lambda s: s["observations"]["air_temp"]["value"]}

        sh.record_observations(stations, fields, run_time_utc=base, db_path=self.db_path)
        stations[0]["observations"]["air_temp"]["value"] = 90.0
        sh.record_observations(stations, fields, run_time_utc=base + timedelta(minutes=5), db_path=self.db_path)
        stations[0]["observations"]["air_temp"]["value"] = 70.0
        sh.record_observations(stations, fields, run_time_utc=base + timedelta(minutes=10), db_path=self.db_path)

        extreme = sh.get_today_extreme("S1", "air_temp_f", now_utc=base + timedelta(minutes=10), db_path=self.db_path)
        self.assertEqual(extreme["min"], 70.0)
        self.assertEqual(extreme["max"], 90.0)

    def test_day_rollover_starts_fresh_extreme(self):
        day1 = datetime(2026, 7, 15, 23, 0, tzinfo=timezone.utc)  # 6pm CDT, still July 15 local
        day2 = datetime(2026, 7, 16, 6, 0, tzinfo=timezone.utc)   # 1am CDT July 16 local - new day
        stations = [{"stid": "S1", "observations": {"air_temp": {"value": 95.0}}}]
        fields = {"air_temp_f": lambda s: s["observations"]["air_temp"]["value"]}

        sh.record_observations(stations, fields, run_time_utc=day1, db_path=self.db_path)
        stations[0]["observations"]["air_temp"]["value"] = 60.0
        sh.record_observations(stations, fields, run_time_utc=day2, db_path=self.db_path)

        extreme = sh.get_today_extreme("S1", "air_temp_f", now_utc=day2, db_path=self.db_path)
        self.assertEqual(extreme["min"], 60.0)
        self.assertEqual(extreme["max"], 60.0)

    def test_missing_station_returns_none(self):
        self.assertIsNone(sh.get_today_extreme("NOPE", "air_temp_f", db_path=self.db_path))

    def test_value_n_hours_ago_within_tolerance(self):
        now = datetime(2026, 7, 15, 18, 0, tzinfo=timezone.utc)
        three_hours_ago = now - timedelta(hours=3, minutes=2)  # within default 20-min tolerance
        stations = [{"stid": "S1", "observations": {"air_temp": {"value": 65.0}}}]
        fields = {"air_temp_f": lambda s: s["observations"]["air_temp"]["value"]}
        sh.record_observations(stations, fields, run_time_utc=three_hours_ago, db_path=self.db_path)

        value = sh.get_value_n_hours_ago("S1", "air_temp_f", 3, now_utc=now, db_path=self.db_path)
        self.assertEqual(value, 65.0)

    def test_value_n_hours_ago_outside_tolerance_is_none(self):
        now = datetime(2026, 7, 15, 18, 0, tzinfo=timezone.utc)
        way_off = now - timedelta(hours=5)
        stations = [{"stid": "S1", "observations": {"air_temp": {"value": 65.0}}}]
        fields = {"air_temp_f": lambda s: s["observations"]["air_temp"]["value"]}
        sh.record_observations(stations, fields, run_time_utc=way_off, db_path=self.db_path)

        value = sh.get_value_n_hours_ago("S1", "air_temp_f", 3, now_utc=now, db_path=self.db_path)
        self.assertIsNone(value)

    def test_get_change_computes_delta(self):
        now = datetime(2026, 7, 15, 18, 0, tzinfo=timezone.utc)
        three_hours_ago = now - timedelta(hours=3)
        stations = [{"stid": "S1", "observations": {"air_temp": {"value": 60.0}}}]
        fields = {"air_temp_f": lambda s: s["observations"]["air_temp"]["value"]}
        sh.record_observations(stations, fields, run_time_utc=three_hours_ago, db_path=self.db_path)

        change = sh.get_change("S1", "air_temp_f", 3, current_value=75.0, now_utc=now, db_path=self.db_path)
        self.assertEqual(change, 15.0)

    def test_rolling_extreme_trailing_window(self):
        now = datetime(2026, 7, 15, 18, 0, tzinfo=timezone.utc)
        fields = {"dewpoint_f": lambda s: s["observations"]["dewpoint"]["value"]}
        for hours_ago, value in [(23, 60.0), (10, 75.0), (1, 65.0)]:
            sh.record_observations(
                [{"stid": "S1", "observations": {"dewpoint": {"value": value}}}],
                fields, run_time_utc=now - timedelta(hours=hours_ago), db_path=self.db_path,
            )
        # 25 hours ago - outside the 24h window, must not affect the max/min.
        sh.record_observations(
            [{"stid": "S1", "observations": {"dewpoint": {"value": 99.0}}}],
            fields, run_time_utc=now - timedelta(hours=25), db_path=self.db_path,
        )

        extreme = sh.get_rolling_extreme("S1", "dewpoint_f", 24, now_utc=now, db_path=self.db_path)
        self.assertEqual(extreme["max"], 75.0)
        self.assertEqual(extreme["min"], 60.0)

    def test_rolling_extreme_none_when_empty(self):
        now = datetime(2026, 7, 15, 18, 0, tzinfo=timezone.utc)
        self.assertIsNone(sh.get_rolling_extreme("NOPE", "dewpoint_f", 24, now_utc=now, db_path=self.db_path))

    def test_get_change_none_when_no_history(self):
        now = datetime(2026, 7, 15, 18, 0, tzinfo=timezone.utc)
        change = sh.get_change("S1", "air_temp_f", 24, current_value=75.0, now_utc=now, db_path=self.db_path)
        self.assertIsNone(change)

    def test_today_extremes_batch_matches_per_station(self):
        now = datetime(2026, 7, 15, 18, 0, tzinfo=timezone.utc)
        fields = {"air_temp_f": lambda s: s["observations"]["air_temp"]["value"]}
        for stid, value in [("S1", 70.0), ("S2", 90.0)]:
            sh.record_observations([{"stid": stid, "observations": {"air_temp": {"value": value}}}], fields, run_time_utc=now, db_path=self.db_path)

        batch = sh.get_today_extremes_batch("air_temp_f", now_utc=now, db_path=self.db_path)
        self.assertEqual(set(batch.keys()), {"S1", "S2"})
        self.assertEqual(batch["S1"]["max"], 70.0)
        self.assertEqual(batch["S2"]["max"], 90.0)

    def test_rolling_extremes_batch_respects_window_per_station(self):
        now = datetime(2026, 7, 15, 18, 0, tzinfo=timezone.utc)
        fields = {"dewpoint_f": lambda s: s["observations"]["dewpoint"]["value"]}
        sh.record_observations([{"stid": "S1", "observations": {"dewpoint": {"value": 60.0}}}], fields, run_time_utc=now - timedelta(hours=10), db_path=self.db_path)
        sh.record_observations([{"stid": "S1", "observations": {"dewpoint": {"value": 99.0}}}], fields, run_time_utc=now - timedelta(hours=30), db_path=self.db_path)
        sh.record_observations([{"stid": "S2", "observations": {"dewpoint": {"value": 55.0}}}], fields, run_time_utc=now - timedelta(hours=5), db_path=self.db_path)

        batch = sh.get_rolling_extremes_batch("dewpoint_f", 24, now_utc=now, db_path=self.db_path)
        self.assertEqual(batch["S1"]["max"], 60.0)  # the 30h-old 99.0 sample is outside the window
        self.assertEqual(batch["S2"]["max"], 55.0)

    def test_values_n_hours_ago_batch_nearest_match_per_station(self):
        now = datetime(2026, 7, 15, 18, 0, tzinfo=timezone.utc)
        fields = {"air_temp_f": lambda s: s["observations"]["air_temp"]["value"]}
        sh.record_observations([{"stid": "S1", "observations": {"air_temp": {"value": 71.0}}}], fields, run_time_utc=now - timedelta(hours=3, minutes=15), db_path=self.db_path)
        sh.record_observations([{"stid": "S1", "observations": {"air_temp": {"value": 65.0}}}], fields, run_time_utc=now - timedelta(hours=3, minutes=2), db_path=self.db_path)
        sh.record_observations([{"stid": "S2", "observations": {"air_temp": {"value": 80.0}}}], fields, run_time_utc=now - timedelta(hours=5), db_path=self.db_path)

        batch = sh.get_values_n_hours_ago_batch("air_temp_f", 3, now_utc=now, db_path=self.db_path)
        self.assertEqual(batch["S1"], 65.0)  # nearer to exactly 3h ago than the 3h15m sample
        self.assertNotIn("S2", batch)  # 5h ago sample is outside the 3h lookup's tolerance window

    def test_changes_batch_computes_delta_per_station(self):
        now = datetime(2026, 7, 15, 18, 0, tzinfo=timezone.utc)
        fields = {"air_temp_f": lambda s: s["observations"]["air_temp"]["value"]}
        sh.record_observations([{"stid": "S1", "observations": {"air_temp": {"value": 60.0}}}], fields, run_time_utc=now - timedelta(hours=3), db_path=self.db_path)

        changes = sh.get_changes_batch("air_temp_f", 3, {"S1": 75.0, "S2": 80.0}, now_utc=now, db_path=self.db_path)
        self.assertEqual(changes, {"S1": 15.0})  # S2 has no history, so it's omitted rather than guessed at

    def test_prune_old_removes_stale_rows_only(self):
        now = datetime(2026, 7, 15, 18, 0, tzinfo=timezone.utc)
        old = now - timedelta(hours=48)
        recent = now - timedelta(hours=1)
        stations = [{"stid": "S1", "observations": {"air_temp": {"value": 50.0}}}]
        fields = {"air_temp_f": lambda s: s["observations"]["air_temp"]["value"]}
        sh.record_observations(stations, fields, run_time_utc=old, db_path=self.db_path)
        sh.record_observations(stations, fields, run_time_utc=recent, db_path=self.db_path)

        deleted = sh.prune_old(retention_hours=36, now_utc=now, db_path=self.db_path)
        self.assertEqual(deleted, 1)
        # The recent sample should still be retrievable ~1 hour back.
        self.assertIsNotNone(sh.get_value_n_hours_ago("S1", "air_temp_f", 1, now_utc=now, db_path=self.db_path))

    def test_skips_none_values_and_missing_stid(self):
        stations = [
            {"stid": "S1", "observations": {"air_temp": {"value": None}}},
            {"observations": {"air_temp": {"value": 70.0}}},  # no stid
        ]
        fields = {"air_temp_f": lambda s: s["observations"]["air_temp"]["value"]}
        recorded = sh.record_observations(stations, fields, db_path=self.db_path)
        self.assertEqual(recorded, 0)


if __name__ == "__main__":
    unittest.main()
