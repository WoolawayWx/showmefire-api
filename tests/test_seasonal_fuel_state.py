import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from services import seasonal_fuel_state as sfs


def _write_raw_data(raw_root: Path, day: date, temps_f: list) -> None:
    path = raw_root / f"raw_data_{day.strftime('%Y%m%d')}.json"
    payload = {"STATION": [{"STID": "TEST1", "OBSERVATIONS": {"air_temp_set_1": temps_f}}]}
    path.write_text(json.dumps(payload), encoding="utf-8")


class SeasonalFuelStateTests(unittest.TestCase):
    def setUp(self):
        self.state_dir = tempfile.TemporaryDirectory()
        self.raw_dir = tempfile.TemporaryDirectory()
        self.state_path_patch = patch.object(sfs, "STATE_PATH", Path(self.state_dir.name) / "seasonal-fuel-state.json")
        self.state_path_patch.start()
        sfs._cache["state"] = None
        sfs._cache["loaded_at"] = None

    def tearDown(self):
        self.state_path_patch.stop()
        self.state_dir.cleanup()
        self.raw_dir.cleanup()

    def test_daily_mean_temp_missing_archive_returns_none(self):
        raw_root = Path(self.raw_dir.name)
        self.assertIsNone(sfs.daily_mean_temp_c_from_archive(date(2026, 6, 1), raw_root))

    def test_daily_mean_temp_converts_fahrenheit_to_celsius(self):
        raw_root = Path(self.raw_dir.name)
        _write_raw_data(raw_root, date(2026, 6, 1), [68.0, 68.0])  # 68F == 20C
        mean_c = sfs.daily_mean_temp_c_from_archive(date(2026, 6, 1), raw_root)
        self.assertAlmostEqual(mean_c, 20.0, places=3)

    def test_update_daily_gdd_accumulates_above_base_temp(self):
        raw_root = Path(self.raw_dir.name)
        _write_raw_data(raw_root, date(2026, 3, 5), [68.0])  # 20C -> 10 GDD above 10C base
        state = sfs.update_daily_gdd(date(2026, 3, 5), raw_root)
        self.assertAlmostEqual(state["gdd_accum_since_mar1"], 10.0, places=3)
        self.assertEqual(state["last_updated_date"], "2026-03-05")

    def test_update_daily_gdd_is_idempotent_for_the_same_day(self):
        raw_root = Path(self.raw_dir.name)
        _write_raw_data(raw_root, date(2026, 3, 5), [68.0])
        sfs.update_daily_gdd(date(2026, 3, 5), raw_root)
        state = sfs.update_daily_gdd(date(2026, 3, 5), raw_root)
        self.assertAlmostEqual(state["gdd_accum_since_mar1"], 10.0, places=3)

    def test_update_daily_gdd_resets_at_new_season_start(self):
        raw_root = Path(self.raw_dir.name)
        _write_raw_data(raw_root, date(2025, 6, 1), [86.0])  # 30C -> 20 GDD, prior season
        sfs.update_daily_gdd(date(2025, 6, 1), raw_root)
        _write_raw_data(raw_root, date(2026, 3, 2), [68.0])  # new season, 10 GDD
        state = sfs.update_daily_gdd(date(2026, 3, 2), raw_root)
        self.assertAlmostEqual(state["gdd_accum_since_mar1"], 10.0, places=3)
        self.assertEqual(state["season_start"], "2026-03-01")

    def test_update_daily_gdd_clips_below_base_temp_to_zero(self):
        raw_root = Path(self.raw_dir.name)
        _write_raw_data(raw_root, date(2026, 3, 5), [32.0])  # 0C, below the 10C base
        state = sfs.update_daily_gdd(date(2026, 3, 5), raw_root)
        self.assertAlmostEqual(state["gdd_accum_since_mar1"], 0.0, places=3)

    def test_update_daily_gdd_missing_observations_leaves_state_unchanged(self):
        raw_root = Path(self.raw_dir.name)
        _write_raw_data(raw_root, date(2026, 3, 5), [68.0])
        sfs.update_daily_gdd(date(2026, 3, 5), raw_root)
        state = sfs.update_daily_gdd(date(2026, 3, 6), raw_root)  # no archive file for this date
        self.assertAlmostEqual(state["gdd_accum_since_mar1"], 10.0, places=3)
        self.assertEqual(state["last_updated_date"], "2026-03-05")

    def test_current_gdd_accum_returns_none_when_stale(self):
        sfs._persist_state({
            "season_start": "2026-03-01",
            "gdd_accum_since_mar1": 500.0,
            "last_updated_date": "2020-01-01",
        })
        self.assertIsNone(sfs.current_gdd_accum(max_age_days=3))

    def test_current_gdd_accum_returns_value_when_fresh(self):
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).date().isoformat()
        sfs._persist_state({
            "season_start": "2026-03-01",
            "gdd_accum_since_mar1": 250.0,
            "last_updated_date": today,
        })
        self.assertAlmostEqual(sfs.current_gdd_accum(max_age_days=3), 250.0, places=3)


if __name__ == "__main__":
    unittest.main()
