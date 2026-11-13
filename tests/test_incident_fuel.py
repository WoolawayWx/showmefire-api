import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xarray as xr

from services import fire_behavior_static, ground_cover, incident_fuel


def _bundle(path: Path, fuel_rows):
    """A small statewide-like grid: ~1 km cells; `fuel_rows` gives the fuel code of each row (south to north)."""
    n = len(fuel_rows)
    lat = np.linspace(36.0, 36.0 + 0.009 * (n - 1), n)[:, None] * np.ones(n)
    lon = np.linspace(-93.9, -93.9 + 0.011 * (n - 1), n)[None, :] * np.ones((n, 1))
    fuel = np.array(fuel_rows, dtype="float32")[:, None] * np.ones((1, n), dtype="float32")
    full = lambda value: (("y", "x"), np.full((n, n), value, dtype="float32"))
    ds = xr.Dataset(
        {
            "slope_degrees": full(5.0), "aspect_sin": full(0.0), "aspect_cos": full(1.0),
            "canopy_cover_pct": full(40.0), "canopy_height_m": full(10.0),
            "latitude": (("y", "x"), lat.astype("float32")), "longitude": (("y", "x"), lon.astype("float32")),
            "static_valid_mask": full(1.0), "fuel_model_fbfm40": (("y", "x"), fuel),
        },
        coords={"x": np.arange(n) * 1000.0, "y": np.arange(n) * 1000.0},
        attrs={"bundle_version": "test"},
    )
    ds.to_netcdf(path, engine="netcdf4")


class IncidentFuelTests(unittest.TestCase):
    def setUp(self):
        # These tests are about the coarse 3 km bundle; keep the 480 m product (if this machine has one) out of them.
        patcher = patch.object(ground_cover, "_locate_bundle", side_effect=FileNotFoundError("none"))
        patcher.start()
        self.addCleanup(patcher.stop)
        ground_cover._RUNTIME = None
        ground_cover._cached.cache_clear()
        self.temporary = tempfile.TemporaryDirectory()
        self.bundle = Path(self.temporary.name) / "bundle.nc"
        fire_behavior_static._RUNTIME = None
        fire_behavior_static._open_bundle.cache_clear()
        incident_fuel._cached.cache_clear()

    def tearDown(self):
        fire_behavior_static._RUNTIME = None
        fire_behavior_static._open_bundle.cache_clear()
        incident_fuel._cached.cache_clear()
        self.temporary.cleanup()

    def _use(self, fuel_rows):
        _bundle(self.bundle, fuel_rows)
        return patch.dict("os.environ", {"FIRE_BEHAVIOR_STATIC_BUNDLE": str(self.bundle)})

    def test_groups_the_fuel_models_around_a_point(self):
        # Grass (GR2) in the south half, timber litter (TL3) in the north half, straddling the point.
        rows = [102] * 10 + [183] * 10
        with self._use(rows):
            result = incident_fuel.fuel_context(36.09, -93.8)
        self.assertIsNotNone(result)
        labels = {g["group"]: g["pct"] for g in result["groups"]}
        self.assertEqual(set(labels), {"Grass", "Timber litter"})
        self.assertEqual(sum(labels.values()), 100)
        self.assertEqual(result["canopy_cover_pct"], 40.0)
        self.assertIn(result["fuel_model_fbfm40"], (102, 183))
        self.assertEqual(result["source"], incident_fuel.SOURCE)
        self.assertGreaterEqual(result["cells"], 2)
        self.assertAlmostEqual(result["resolution_km"], 1.0, delta=0.2)      # the synthetic grid is ~1 km

    def test_a_coarse_grid_reports_that_it_rests_on_one_cell(self):
        # ~3 km cells like the production bundle: a 1.5 km radius reaches a single cell, so the summary says so.
        n = 12
        lat = np.linspace(36.0, 36.0 + 0.028 * (n - 1), n)[:, None] * np.ones(n)
        lon = np.linspace(-93.9, -93.9 + 0.034 * (n - 1), n)[None, :] * np.ones((n, 1))
        full = lambda value: (("y", "x"), np.full((n, n), value, dtype="float32"))
        ds = xr.Dataset(
            {"slope_degrees": full(5.0), "aspect_sin": full(0.0), "aspect_cos": full(1.0), "canopy_cover_pct": full(60.0),
             "canopy_height_m": full(10.0), "latitude": (("y", "x"), lat.astype("float32")),
             "longitude": (("y", "x"), lon.astype("float32")), "static_valid_mask": full(1.0), "fuel_model_fbfm40": full(189.0)},
            coords={"x": np.arange(n) * 3000.0, "y": np.arange(n) * 3000.0}, attrs={"bundle_version": "coarse"})
        ds.to_netcdf(self.bundle, engine="netcdf4")
        with patch.dict("os.environ", {"FIRE_BEHAVIOR_STATIC_BUNDLE": str(self.bundle)}):
            result = incident_fuel.fuel_context(36.15, -93.75)
        self.assertEqual(result["cells"], 1)
        self.assertEqual(result["groups"], [{"group": "Timber litter", "pct": 100}])
        self.assertGreaterEqual(result["resolution_km"], 2.5)

    def test_non_burnable_codes_read_as_land_cover(self):
        with self._use([91] * 6 + [93] * 14):
            result = incident_fuel.fuel_context(36.02, -93.85, radius_km=3.0)
        self.assertEqual({g["group"] for g in result["groups"]} - {"Developed", "Cropland"}, set())

    def test_points_off_the_grid_and_missing_bundles_return_none(self):
        with self._use([102] * 10):
            self.assertIsNone(incident_fuel.fuel_context(41.0, -80.0))
        self.assertIsNone(incident_fuel.fuel_context("x", None))
        incident_fuel._cached.cache_clear()
        fire_behavior_static._RUNTIME = None
        with patch.dict("os.environ", {"FIRE_BEHAVIOR_STATIC_BUNDLE": "/does/not/exist.nc"}), \
                patch.object(fire_behavior_static, "_load_from_registry", side_effect=FileNotFoundError("no bundle")):
            self.assertIsNone(incident_fuel.fuel_context(36.02, -93.85))

    def test_fuel_group_names(self):
        self.assertEqual(incident_fuel.fuel_group(101), "Grass")
        self.assertEqual(incident_fuel.fuel_group(183), "Timber litter")
        self.assertEqual(incident_fuel.fuel_group(161), "Timber with understory")
        self.assertEqual(incident_fuel.fuel_group(93), "Cropland")
        self.assertEqual(incident_fuel.fuel_group(0), "Other")


if __name__ == "__main__":
    unittest.main()
