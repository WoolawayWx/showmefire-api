"""Capture-only model collection (services/data_capture): what is due, bounded work, disk guard, no partial runs."""
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xarray as xr

from services import day_archive
from services.data_capture import capture
from services.ensemble_fire_danger import grib_idx

NOW = datetime(2026, 11, 18, 20, 0, tzinfo=timezone.utc)


def model(**extra):
    base = {"kind": "grib", "dir": "cache/rrfs", "prefix": "rrfs", "url": "https://x/{date}/{hh}/f{fxx:03d}",
            "cycles": [0, 6, 12, 18], "leads": {"start": 1, "end": 3, "step": 1}, "latency_hours": 3.5,
            "required": ["t2m"], "searches": {"t2m": "T", "r2": "R", "apcp": "A"}}
    base.update(extra)
    return base


def config(**models):
    return {"keep_days": 14, "lookback_hours": 30, "models": models}


def fake_dataset(*_args, **_kwargs):
    return xr.Dataset({"t2m": (("step", "y", "x"), np.ones((3, 2, 2), dtype="float32"))})


class PlanningTests(unittest.TestCase):
    def test_due_cycles_respect_latency_lookback_and_cycle_hours(self):
        cycles = capture.due_cycles(model(), NOW, 30)
        self.assertEqual(cycles[0], datetime(2026, 11, 18, 12, tzinfo=timezone.utc))      # 18z + 3.5 h is not out yet
        self.assertTrue(all(c.hour in (0, 6, 12, 18) for c in cycles))
        self.assertGreaterEqual(min(cycles), datetime(2026, 11, 17, 14, tzinfo=timezone.utc))

    def test_output_path_uses_the_training_layout(self):
        path = capture.output_path(Path("/root"), model(), datetime(2026, 11, 18, 12, tzinfo=timezone.utc))
        self.assertEqual(path.as_posix(), "/root/cache/rrfs/rrfs_20261118_12z_f01-03.nc")

    def test_the_real_config_loads_and_every_grib_model_has_required_fields(self):
        cfg = capture.load_config()
        for name, m in cfg["models"].items():
            if m["kind"] == "grib":
                self.assertTrue(set(m["required"]) <= set(m["searches"]), name)
                self.assertTrue(capture.leads_for(m), name)
                self.assertIn("{fxx:03d}", m["url"], name)


class TickTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.written = []

    def writer(self, ds, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"nc")
        self.written.append(path.name)
        return path

    def run_tick(self, cfg, **kwargs):
        with patch.object(capture, "enough_disk", return_value=True):
            return capture.tick(NOW, config=cfg, root=self.root, crop=lambda ds: ds, capture=kwargs.pop("capture", fake_dataset),
                                writer=self.writer, mrms_fetch=kwargs.pop("mrms_fetch", None), **kwargs)

    def test_captures_newest_first_within_the_budget_and_never_twice(self):
        first = self.run_tick(config(rrfs=model()), max_runs=2)
        self.assertEqual(first["captured"], ["rrfs_20261118_12z_f01-03.nc", "rrfs_20261118_06z_f01-03.nc"])
        second = self.run_tick(config(rrfs=model()), max_runs=10)
        self.assertNotIn("rrfs_20261118_12z_f01-03.nc", second["captured"])
        self.assertTrue(second["captured"])

    def test_unpublished_runs_wait_and_real_errors_are_reported_without_stopping_other_models(self):
        def flaky(m, cycle, crop, tmp):
            if m["prefix"] == "rrfs":
                raise grib_idx.SourceUnavailable("not yet")
            raise KeyError("required field missing")
        result = self.run_tick(config(rrfs=model(), nbm=model(dir="cache/nbm", prefix="nbm", cycles=[13])), capture=flaky, max_runs=10)
        self.assertTrue(result["waiting"])
        self.assertTrue(any(name.startswith("nbm_") for name in result["failed"]))
        self.assertEqual(self.written, [])

    def test_disk_guard_skips_everything(self):
        with patch.object(capture, "enough_disk", return_value=False):
            result = capture.tick(NOW, config=config(rrfs=model()), root=self.root, crop=lambda ds: ds,
                                  capture=fake_dataset, writer=self.writer)
        self.assertIn("free disk", result["skipped"])
        self.assertEqual(self.written, [])

    def test_dry_run_lists_without_fetching_or_writing(self):
        def boom(*a, **k):
            raise AssertionError("dry run must not fetch")
        result = capture.tick(NOW, config=config(rrfs=model()), root=self.root, dry_run=True, capture=boom, writer=boom)
        self.assertIn("rrfs_20261118_12z_f01-03.nc", result["would_capture"])

    def test_disabled_models_are_skipped(self):
        result = self.run_tick(config(rrfs=model(enabled=False)))
        self.assertEqual(result["captured"], [])

    def test_mrms_hours_are_fetched_into_the_mrms_folder(self):
        seen = []
        def fetch(hour, folder):
            seen.append(hour)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"mrms_{hour:%Y%m%d_%H}z.nc").write_bytes(b"x")
        result = self.run_tick(config(mrms={"kind": "mrms", "dir": "cache/mrms"}), mrms_fetch=fetch, max_runs=2)
        self.assertEqual(len(seen), 2)
        self.assertEqual(result["captured"][0], "mrms_20261118_19z.nc")

    def test_prune_removes_only_old_dated_files_of_its_own_prefix(self):
        folder = self.root / "cache" / "rrfs"
        folder.mkdir(parents=True)
        for name in ("rrfs_20261101_12z_f01-48.nc", "rrfs_20261117_12z_f01-48.nc", "RRFS_20260212_12z_f04-15.nc", "other.nc"):
            (folder / name).write_bytes(b"x")
        removed = capture.prune(folder, "rrfs", 14, NOW)
        self.assertEqual(removed, 1)
        self.assertEqual(sorted(p.name for p in folder.iterdir()), ["RRFS_20260212_12z_f04-15.nc", "other.nc", "rrfs_20261117_12z_f01-48.nc"])


class CaptureRunTests(unittest.TestCase):
    """capture_run on synthetic GRIB results (no network)."""

    def fields_for(self, missing=()):
        lat = np.array([[38.0, 38.0], [39.0, 39.0]])
        lon = np.array([[-92.0, -91.0], [-92.0, -91.0]])
        def fetch_fields(session, url, searches, tmp_dir):
            lead = int(url.rsplit("f", 1)[1])
            found = {key: [] for key in searches}
            for key in searches:
                if key in missing:
                    continue
                if key == "apcp":
                    entry = grib_idx.IdxEntry("1", 0, f":APCP:surface:{lead - 1}-{lead} hour acc fcst:")
                    found[key] = [(entry, np.full((2, 2), 2.0))]
                else:
                    found[key] = [(grib_idx.IdxEntry("1", 0, ":X"), np.full((2, 2), float(lead)))]
            return found, lat, lon
        return fetch_fields

    def run_capture(self, missing=(), required=("t2m",)):
        with patch.object(grib_idx, "fetch_idx", return_value=[grib_idx.IdxEntry("1", 0, ":X")]), \
                patch.object(grib_idx, "fetch_fields", side_effect=self.fields_for(missing)):
            return capture.capture_run(model(required=list(required)), datetime(2026, 11, 18, 12, tzinfo=timezone.utc),
                                       lambda ds: ds, Path(tempfile.gettempdir()), session=object())

    def test_fields_are_stacked_by_lead_with_hourly_precip(self):
        ds = self.run_capture()
        self.assertEqual(list(ds["step"].values), [1, 2, 3])
        self.assertEqual(float(ds["t2m"].isel(step=2)[0, 0]), 3.0)
        self.assertEqual(float(ds["precip_rate_mmh"].isel(step=0)[0, 0]), 2.0)
        self.assertEqual(ds.attrs["model"], "rrfs")

    def test_alternate_row_scanning_is_undone_only_for_models_that_declare_it(self):
        truth = np.arange(20, dtype="float64").reshape(4, 5)
        stored = truth.copy()
        stored[1::2] = stored[1::2, ::-1]                       # how a boustrophedon grid arrives from the decoder
        np.testing.assert_array_equal(capture.unscan_alternate_rows(stored), truth)
        self.assertFalse(np.array_equal(stored, truth))
        self.assertTrue(capture.load_config()["models"]["nbm"]["alternate_row_scan"])
        self.assertFalse(capture.load_config()["models"]["rrfs"].get("alternate_row_scan"))

    def test_a_missing_optional_field_is_nan_and_recorded(self):
        ds = self.run_capture(missing=("r2",))
        self.assertTrue(np.isnan(ds["r2"].values).all())
        self.assertIn("r2", ds.attrs["missing_fields"])

    def test_a_missing_required_field_fails_the_run(self):
        with self.assertRaises(KeyError):
            self.run_capture(missing=("t2m",))

    def test_an_unpublished_last_lead_means_the_run_is_not_ready(self):
        with patch.object(grib_idx, "fetch_idx", side_effect=grib_idx.SourceUnavailable("404")):
            with self.assertRaises(grib_idx.SourceUnavailable):
                capture.capture_run(model(), datetime(2026, 11, 18, 12, tzinfo=timezone.utc), lambda ds: ds,
                                    Path(tempfile.gettempdir()), session=object())

    def test_written_files_are_compressed_and_readable(self):
        ds = self.run_capture()
        with tempfile.TemporaryDirectory() as tmp:
            path = capture.write_atomic(ds, Path(tmp) / "cache" / "rrfs" / "rrfs_20261118_12z_f01-03.nc")
            with xr.open_dataset(path) as back:
                self.assertTrue(back["t2m"].encoding.get("zlib"))
                np.testing.assert_allclose(back["t2m"].values, ds["t2m"].values)
            self.assertFalse(any(p.suffix == ".part" for p in path.parent.iterdir()))


class BundleTests(unittest.TestCase):
    def test_captured_models_are_bundle_components_from_the_training_layout_folders(self):
        for name in ("rrfs", "gefs", "nbm", "mrms"):
            self.assertIn(name, day_archive.FILE_COMPONENTS)
            self.assertIn(name, day_archive.PRUNABLE)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"SMF_ARCHIVE_ROOT": tmp}):
            folder = Path(tmp) / "cache" / "rrfs"
            folder.mkdir(parents=True)
            (folder / "rrfs_20261118_12z_f01-48.nc").write_bytes(b"x")
            os.utime(folder / "rrfs_20261118_12z_f01-48.nc", (0, 0))                  # long settled
            found = day_archive.FILE_COMPONENTS["rrfs"]("2026-11-18", 10**10)
        self.assertEqual([arc for _, arc in found], ["cache/rrfs/rrfs_20261118_12z_f01-48.nc"])


if __name__ == "__main__":
    unittest.main()
