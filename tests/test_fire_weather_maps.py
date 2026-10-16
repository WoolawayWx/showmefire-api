"""Hourly cadence, archive and summaries for the fire-weather maps (no downloads)."""
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.fire_weather import history
from services.fire_weather import hrrr_profiles as hp
from services.fire_weather import summary as fs


def hour(h):
    return datetime(2026, 10, 16, h, tzinfo=timezone.utc)


class CycleSelectionTests(unittest.TestCase):
    def test_nothing_to_do_when_all_processed(self):
        available = [hour(17), hour(18)]
        self.assertEqual(history.select_hours(available, {"2026101617", "2026101618"}), [])

    def test_backfills_missing_hours_oldest_first(self):
        available = [hour(15), hour(16), hour(17)]
        self.assertEqual(history.select_hours(available, {"2026101616"}), [hour(15), hour(17)])

    def test_budget_keeps_the_newest_hour(self):
        available = [hour(h) for h in range(12, 19)]
        self.assertEqual(history.select_hours(available, set(), budget=3), [hour(12), hour(13), hour(18)])

    def test_available_hours_use_previous_run_f01_and_need_both_files(self):
        requested = []

        def head(url):
            requested.append(url)
            if "f01" not in url:
                return 404
            return 404 if ("t16z" in url and "prs" in url) or "t19z" in url else 200

        hours = hp.available_hours(2, now=hour(18) + timedelta(minutes=55), head=head)
        # Valid 17z needs the 16z run, whose prs file is missing; the 18z run's f01 makes 19z available early.
        self.assertEqual(hours, [hour(16), hour(18), hour(19)])
        self.assertTrue(all("f01.grib2.idx" in url for url in requested))


class LedgerTests(unittest.TestCase):
    def test_mark_processed_keeps_recent_keys_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "processed.json"
            path.write_text(json.dumps(["2026090100", "2026101617"]))
            history.mark_processed("2026101618", path=path, now=hour(18))
            self.assertEqual(history.load_ledger(path), {"2026101617", "2026101618"})


def _fields(shape=(4, 5)):
    lat, lon = np.meshgrid(np.linspace(36.5, 40.0, shape[0]), np.linspace(-94.5, -90.5, shape[1]), indexing="ij")
    ones = np.ones(shape)
    return {
        "lat": lat, "lon": lon,
        "mixing_height_ft": 3000 * ones, "holzworth_ft": 2800 * ones,
        "transport_u_ms": 0 * ones, "transport_v_ms": -5 * ones,  # from the north
        "transport_speed_mph": 11.2 * ones, "transport_dir_deg": 0 * ones,
        "ventilation_kt_ft": np.where(lat > 38.5, 120000.0, 30000.0),
        "ventilation_category": np.where(lat > 38.5, 3.0, 0.0),
        "inversion_strength_f": np.where(lon < -92.5, 3.0, 0.0), "inversion_top_ft": 300 * ones,
        "lapse_rate_c_km": 6 * ones, "haines": 4 * ones, "hdw": 50 * ones,
    }


class SummaryTests(unittest.TestCase):
    def test_statewide_and_regional_summary(self):
        fields = _fields()
        result = fs.summarize(fields, hour(18), in_state=np.ones(fields["lat"].shape, bool))
        state = result["statewide"]
        self.assertEqual(result["valid_time"], "2026-10-16T18:00:00Z")
        self.assertEqual(state["mixing_height_ft"], {"min": 3000, "median": 3000, "max": 3000})
        self.assertEqual(state["transport_direction"], "north")
        self.assertEqual(state["ventilation_category_percent"]["Poor"] + state["ventilation_category_percent"]["Very Good"], 100.0)
        self.assertGreater(state["surface_inversion_percent"], 0)
        self.assertEqual(set(result["regions"]), {"Northwest", "Northeast", "Central", "Southwest", "Southeast"})

    def test_station_values_use_nearest_cell(self):
        fields = _fields()
        stations = [{"stid": "KTEST", "latitude": 39.9, "longitude": -94.4, "state": "MO", "name": "Test"},
                    {"stid": "KOUT", "latitude": 39.9, "longitude": -94.4, "state": "KS"}]
        values = fs.station_values(fields, stations)
        self.assertEqual(list(values), ["KTEST"])
        self.assertEqual(values["KTEST"]["ventilation_kt_ft"], 120000.0)


class ArchiveTests(unittest.TestCase):
    def test_record_grid_and_read_back(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(history, "HISTORY_DIR", Path(directory) / "history"), \
                mock.patch.object(history, "GRID_DIR", Path(directory) / "grids"):
            fields = _fields()
            summary = fs.summarize(fields, hour(17), in_state=np.ones(fields["lat"].shape, bool))
            history.write_record(hour(17), summary, {})
            grid_path = history.write_grids(hour(17), fields)
            records = history.read_records(3, now=hour(18))
            self.assertEqual([record["valid_time"] for record in records], ["2026-10-16T17:00:00Z"])
            with xr.open_dataset(grid_path) as grids:
                self.assertAlmostEqual(float(grids["transport_speed_mph"].values[0, 0]), 11.2, places=1)
                self.assertEqual(float(grids["ventilation_kt_ft"].values[-1, 0]), 120000.0)


class FakeHerbie:
    """Returns cfgrib-style datasets for each search, like Herbie.xarray."""

    def __init__(self, date, **kwargs):
        self.product = kwargs["product"]

    def xarray(self, search, remove_grib=True):
        shape = (3, 3)
        lat, lon = np.meshgrid(np.linspace(37, 39, 3), np.linspace(-94, -91, 3), indexing="ij")
        coords = {"latitude": (("y", "x"), lat), "longitude": (("y", "x"), lon + 360)}
        if self.product == "prs":
            levels = np.array(hp.PRESSURE_LEVELS, dtype=float)
            heights = np.linspace(100, 3500, levels.size)[:, None, None] * np.ones(shape)
            temps = (288.0 - 0.0065 * heights)
            data = {"t": temps, "dpt": temps - 10, "u": 5 + 0 * heights, "v": 0 * heights, "gh": heights + 200}
            return xr.Dataset({k: (("isobaricInhPa", "y", "x"), v) for k, v in data.items()},
                              coords={"isobaricInhPa": levels, **coords})
        ones = np.ones(shape)
        if "HPBL" in search:
            return xr.Dataset({"blh": (("y", "x"), 1000 * ones), "sp": (("y", "x"), 99000 * ones),
                               "orog": (("y", "x"), 200 * ones)}, coords=coords)
        if "2 m" in search:
            return xr.Dataset({"t2m": (("y", "x"), 290 * ones), "d2m": (("y", "x"), 280 * ones)}, coords=coords)
        if "10 m" in search:
            return xr.Dataset({"u10": (("y", "x"), 3 * ones), "v10": (("y", "x"), 0 * ones)}, coords=coords)
        return xr.Dataset({"u": (("y", "x"), 4 * ones), "v": (("y", "x"), 0 * ones)}, coords=coords)


class FetchTests(unittest.TestCase):
    def test_fetch_and_derive_with_fake_herbie(self):
        with mock.patch.object(hp, "crop", lambda dataset: dataset):
            profiles = hp.fetch_profiles(hour(18), herbie_factory=FakeHerbie)
        # 1000 mb lies below the 990 mb surface: masked out.
        self.assertTrue(np.isnan(profiles.temp_c[0]).all())
        self.assertTrue(np.isfinite(profiles.temp_c[1]).all())
        fields = hp.derive_fields(profiles)
        self.assertAlmostEqual(fields["mixing_height_ft"][1, 1], 1000 * 3.28084, places=1)
        self.assertGreater(fields["transport_speed_mph"][1, 1], 0)
        self.assertTrue(np.isfinite(fields["haines"]).all())
        self.assertTrue(np.isfinite(fields["hdw"]).all())


if __name__ == "__main__":
    unittest.main()
