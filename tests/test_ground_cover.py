import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyproj
import xarray as xr

from services import fire_behavior_static, ground_cover, incident_fuel

TO_5070 = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
TO_LONLAT = pyproj.Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
N = 10  # 10 x 10 cells; columns 0-4 timber/forest, columns 5-9 crops
REAL_BUNDLE = Path(os.getenv("GROUND_COVER_TEST_BUNDLE",
                             r"D:\_Development\ShowMeFire\training-data\static\bundles\ground_cover_static_2026.1.nc"))


def _grid():
    x, y = TO_5070.transform(-92.3, 38.6)
    return np.floor(x / 480) * 480 - 5 * 480, np.ceil(y / 480) * 480 + 5 * 480


def _write_bundle(path: Path, valid_hole=None):
    x0, y1 = _grid()
    xs = x0 + (np.arange(N) + 0.5) * 480
    ys = y1 - (np.arange(N) + 0.5) * 480
    zeros = lambda: np.zeros((N, N), dtype="uint8")
    data = {name: zeros() for name, _ in ground_cover.NLCD_VARS + ground_cover.FBFM_VARS}
    data["nlcd_forest_pct"][:, :5] = 100
    data["nlcd_crops_pct"][:, 5:] = 100
    data["fbfm_timber_litter_pct"][:, :5] = 100
    data["fbfm_nb_cropland_pct"][:, 5:] = 100
    mode = np.full((N, N), 183, dtype="int16")
    mode[:, 5:] = 93
    valid = np.full((N, N), 100, dtype="uint8")
    if valid_hole:
        valid[valid_hole] = 0
    data_vars = {k: (("y", "x"), v) for k, v in data.items()}
    data_vars.update(
        fbfm40_mode=(("y", "x"), mode), valid_pct=(("y", "x"), valid),
        canopy_cover_mean_pct=(("y", "x"), np.full((N, N), 40.0, dtype="float32")),
        canopy_height_mean_m=(("y", "x"), np.full((N, N), 12.0, dtype="float32")))
    mode[:, 0] = ground_cover.MODE_FILL  # column 0 has no fuel code, written with a _FillValue like the real bundle
    data_vars["fbfm40_mode"] = (("y", "x"), mode)
    xr.Dataset(data_vars, coords={"x": xs, "y": ys}).to_netcdf(
        path, encoding={"fbfm40_mode": {"_FillValue": ground_cover.MODE_FILL}})
    return x0, y1


def _polygon(x0, y1, col0, col1, row0, row1, inset=0.0):
    """GeoJSON polygon (lon/lat) covering the given cell range in the synthetic grid."""
    left, right = x0 + col0 * 480 + inset, x0 + col1 * 480 - inset
    top, bottom = y1 - row0 * 480 - inset, y1 - row1 * 480 + inset
    ring = [TO_LONLAT.transform(px, py) for px, py in
            [(left, top), (right, top), (right, bottom), (left, bottom), (left, top)]]
    return json.dumps({"type": "Polygon", "coordinates": [[list(p) for p in ring]]})


def _cell_centre_lonlat(x0, y1, col, row):
    return TO_LONLAT.transform(x0 + (col + 0.5) * 480, y1 - (row + 0.5) * 480)


class GroundCoverTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.bundle = Path(self.temporary.name) / "gc.nc"
        self.x0, self.y1 = _write_bundle(self.bundle)
        self._env = patch.dict("os.environ", {"GROUND_COVER_STATIC_BUNDLE": str(self.bundle)})
        self._env.start()
        ground_cover._RUNTIME = None
        ground_cover._cached.cache_clear()

    def tearDown(self):
        self._env.stop()
        ground_cover._RUNTIME = None
        ground_cover._cached.cache_clear()
        self.temporary.cleanup()

    def groups(self, result, key="groups"):
        return {g["group"]: g["pct"] for g in result[key]}


class SamplingTests(GroundCoverTestCase):
    def test_polygon_over_one_land_cover_is_pure(self):
        lat_lon = _cell_centre_lonlat(self.x0, self.y1, 2, 5)
        result = ground_cover.sample(lat_lon[1], lat_lon[0], footprints=[_polygon(self.x0, self.y1, 1, 4, 2, 8, 20)])
        self.assertEqual(self.groups(result), {"Timber litter": 100})
        self.assertEqual(self.groups(result, "ground_cover"), {"Trees": 100})
        self.assertEqual(result["basis"], "footprint")
        self.assertEqual(result["fuel_model_fbfm40"], 183)
        self.assertEqual(result["resolution_km"], 0.48)
        self.assertEqual(result["canopy_cover_pct"], 40.0)
        self.assertEqual(result["canopy_height_m"], 12.0)

    def test_polygon_across_boundary_is_area_weighted(self):
        # 2 columns of forest (3,4) and 4 of crops (5-8): by area one third forest, two thirds crops.
        lon, lat = _cell_centre_lonlat(self.x0, self.y1, 5, 5)
        result = ground_cover.sample(lat, lon, footprints=[_polygon(self.x0, self.y1, 3, 9, 2, 6, 1)])
        self.assertEqual(self.groups(result, "ground_cover"), {"Cropland": 67, "Trees": 33})
        self.assertEqual(sum(self.groups(result).values()), 100)
        self.assertEqual(result["cells"], 24)

    def test_partial_cell_weights_by_overlap(self):
        # A rectangle 1.5 cells wide: a full forest cell and half a crops cell side by side... boundary at col 5.
        lon, lat = _cell_centre_lonlat(self.x0, self.y1, 4, 5)
        left = self.x0 + 4 * 480            # all of column 4 (forest) ...
        right = self.x0 + 5 * 480 + 240      # ... plus half of column 5 (crops)
        top, bottom = self.y1 - 5 * 480 - 10, self.y1 - 6 * 480 + 10
        ring = [list(TO_LONLAT.transform(px, py)) for px, py in
                [(left + 10, top), (right, top), (right, bottom), (left + 10, bottom), (left + 10, top)]]
        polygon = {"type": "Polygon", "coordinates": [ring]}
        result = ground_cover.sample(lat, lon, footprints=[polygon])
        pct = self.groups(result, "ground_cover")
        self.assertAlmostEqual(pct["Trees"], 100 * (470 / 710), delta=2)
        self.assertEqual(pct["Trees"] + pct["Cropland"], 100)

    def test_multiple_footprints_are_unioned_and_duplicates_ignored(self):
        lon, lat = _cell_centre_lonlat(self.x0, self.y1, 2, 2)
        forest = _polygon(self.x0, self.y1, 1, 3, 1, 3, 10)
        crops = _polygon(self.x0, self.y1, 6, 8, 1, 3, 10)
        once = ground_cover.sample(lat, lon, footprints=[forest, crops])
        again = ground_cover.sample(lat, lon, footprints=[forest, crops, forest, crops])
        self.assertEqual(self.groups(once, "ground_cover"), {"Trees": 50, "Cropland": 50})
        self.assertEqual(once, again)

    def test_outline_used_when_there_are_no_footprints(self):
        lon, lat = _cell_centre_lonlat(self.x0, self.y1, 7, 5)
        result = ground_cover.sample(lat, lon, footprints=[None, ""], outline=_polygon(self.x0, self.y1, 6, 9, 3, 7, 10))
        self.assertEqual(result["basis"], "outline")
        self.assertEqual(self.groups(result, "ground_cover"), {"Cropland": 100})

    def test_point_uses_the_three_by_three_block(self):
        lon, lat = _cell_centre_lonlat(self.x0, self.y1, 5, 5)  # first crops column: block spans cols 4-6
        result = ground_cover.sample(lat, lon)
        self.assertEqual(result["basis"], "point")
        self.assertEqual(result["cells"], 9)
        self.assertEqual(self.groups(result, "ground_cover"), {"Cropland": 67, "Trees": 33})

    def test_point_at_grid_edge_uses_fewer_cells(self):
        lon, lat = _cell_centre_lonlat(self.x0, self.y1, 0, 0)
        self.assertEqual(ground_cover.sample(lat, lon)["cells"], 4)

    def test_footprint_off_grid_falls_back_to_the_point(self):
        lon, lat = _cell_centre_lonlat(self.x0, self.y1, 2, 2)
        far = json.dumps({"type": "Polygon", "coordinates": [[[-80, 30], [-79.9, 30], [-79.9, 30.1], [-80, 30]]]})
        result = ground_cover.sample(lat, lon, footprints=[far])
        self.assertEqual(result["basis"], "point")

    def test_off_grid_bad_input_and_no_bundle_return_none(self):
        self.assertIsNone(ground_cover.sample(41.0, -80.0))
        self.assertIsNone(ground_cover.sample("x", None))
        lon, lat = _cell_centre_lonlat(self.x0, self.y1, 2, 2)
        garbage = ground_cover.sample(lat, lon, footprints=["not json", "{}", '{"type": "Point"}'])
        self.assertEqual(garbage["basis"], "point")  # unusable geometry is ignored, not an error
        ground_cover._RUNTIME = None
        ground_cover._cached.cache_clear()
        with patch.dict("os.environ", {"GROUND_COVER_STATIC_BUNDLE": "/does/not/exist.nc"}), \
                patch.object(ground_cover, "_locate_bundle", side_effect=FileNotFoundError("none")):
            self.assertIsNone(ground_cover.sample(lat, lon))

    def test_cells_with_no_valid_data_are_skipped(self):
        _write_bundle(self.bundle, valid_hole=(slice(None), slice(0, 5)))  # forest half has no valid data
        ground_cover._RUNTIME = None
        ground_cover._cached.cache_clear()
        os.utime(self.bundle, (time.time() + 5, time.time() + 5))
        lon, lat = _cell_centre_lonlat(self.x0, self.y1, 5, 5)
        result = ground_cover.sample(lat, lon)
        self.assertEqual(self.groups(result, "ground_cover"), {"Cropland": 100})
        self.assertEqual(result["cells"], 6)  # block is columns 4-6; only 5 and 6 have valid data

    def test_a_changed_bundle_is_reloaded(self):
        lon, lat = _cell_centre_lonlat(self.x0, self.y1, 2, 5)
        self.assertEqual(self.groups(ground_cover.sample(lat, lon), "ground_cover"), {"Trees": 100})
        ds = xr.open_dataset(self.bundle).load()
        ds.close()
        ds["nlcd_forest_pct"][:] = 0
        ds["nlcd_crops_pct"][:] = 100
        self.bundle.unlink()
        ds.to_netcdf(self.bundle)
        os.utime(self.bundle, (time.time() + 10, time.time() + 10))
        ground_cover._cached.cache_clear()
        self.assertEqual(self.groups(ground_cover.sample(lat, lon), "ground_cover"), {"Cropland": 100})

    def test_cells_without_a_fuel_code_do_not_invent_one(self):
        lon, lat = _cell_centre_lonlat(self.x0, self.y1, 0, 5)
        result = ground_cover.sample(lat, lon, footprints=[_polygon(self.x0, self.y1, 0, 1, 2, 8, 20)])
        self.assertIsNone(result["fuel_model_fbfm40"])
        self.assertEqual(self.groups(result, "ground_cover"), {"Trees": 100})

    def test_labels_match_incident_fuel_groups(self):
        labels = {label for _, label in ground_cover.FBFM_VARS} - {"Other"}
        self.assertEqual(labels, {incident_fuel.fuel_group(code) for code in (91, 92, 93, 98, 99, 101, 121, 141, 161, 181, 201)})


class FallbackChainTests(GroundCoverTestCase):
    def setUp(self):
        super().setUp()
        fire_behavior_static._RUNTIME = None
        fire_behavior_static._open_bundle.cache_clear()
        incident_fuel._cached.cache_clear()

    def tearDown(self):
        fire_behavior_static._RUNTIME = None
        fire_behavior_static._open_bundle.cache_clear()
        incident_fuel._cached.cache_clear()
        super().tearDown()

    def test_prefers_the_480m_product(self):
        lon, lat = _cell_centre_lonlat(self.x0, self.y1, 2, 2)
        result = incident_fuel.fuel_context(lat, lon, footprints=[_polygon(self.x0, self.y1, 1, 4, 1, 4, 10)])
        self.assertEqual(result["basis"], "footprint")
        self.assertEqual(result["resolution_km"], 0.48)
        self.assertGreaterEqual(result["cells"], 5)  # enough for the panel to draw a fuel bar

    def test_incident_context_exposes_the_new_fields(self):
        from routers.fires_v2 import _land_cover_context

        lon, lat = _cell_centre_lonlat(self.x0, self.y1, 7, 5)
        members = [{"footprint_geojson": _polygon(self.x0, self.y1, 6, 9, 3, 7, 10), "land_cover": None,
                    "fuel_model_fbfm40": None, "canopy_cover_pct": None}]
        incident = {"centroid_latitude": lat, "centroid_longitude": lon, "shape_geojson": None}
        context = _land_cover_context(members, incident)
        self.assertEqual(context["fuel_basis"], "footprint")
        self.assertEqual(context["ground_cover"], [{"group": "Cropland", "pct": 100}])
        self.assertEqual(context["fuel_groups"], [{"group": "Cropland", "pct": 100}])
        self.assertGreaterEqual(context["fuel_cells"], 5)
        self.assertEqual(context["fuel_resolution_km"], 0.48)
        self.assertEqual(context["fuel_model_fbfm40"], 93)
        self.assertEqual(context["land_cover"], None)  # NOAA's own string is never replaced by ours

    def test_falls_back_to_the_coarse_bundle_then_none(self):
        lon, lat = _cell_centre_lonlat(self.x0, self.y1, 2, 2)
        coarse = {"fuel_model_code": np.full((3, 3), 183), "lat": np.full((3, 3), lat), "lon": np.full((3, 3), lon),
                  "valid_mask": np.ones((3, 3), dtype=bool), "canopy_cover_pct": np.full((3, 3), 50.0),
                  "bundle_version": "coarse"}
        with patch.dict("os.environ", {"GROUND_COVER_STATIC_BUNDLE": "/does/not/exist.nc"}), \
                patch.object(ground_cover, "_locate_bundle", side_effect=FileNotFoundError("none")), \
                patch("services.fire_behavior_static.load_static_fields", return_value=coarse), \
                patch("services.fire_behavior_static.diagnostics", return_value={"bundle_version": "coarse"}):
            result = incident_fuel.fuel_context(lat, lon)
            self.assertEqual(result["source"], incident_fuel.SOURCE)
            self.assertNotIn("basis", result)
        with patch.dict("os.environ", {"GROUND_COVER_STATIC_BUNDLE": "/does/not/exist.nc"}), \
                patch.object(ground_cover, "_locate_bundle", side_effect=FileNotFoundError("none")), \
                patch("services.fire_behavior_static.load_static_fields", side_effect=FileNotFoundError("none")):
            incident_fuel._cached.cache_clear()
            self.assertIsNone(incident_fuel.fuel_context(lat, lon))


@unittest.skipUnless(REAL_BUNDLE.is_file(), f"real ground cover bundle not found: {REAL_BUNDLE}")
class RealBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._env = patch.dict("os.environ", {"GROUND_COVER_STATIC_BUNDLE": str(REAL_BUNDLE)})
        cls._env.start()
        ground_cover._RUNTIME = None
        ground_cover._cached.cache_clear()
        assert ground_cover.sample(39.0997, -94.5786) is not None  # loads the bundle

    @classmethod
    def tearDownClass(cls):
        cls._env.stop()
        ground_cover._RUNTIME = None
        ground_cover._cached.cache_clear()

    def test_known_places(self):
        kc = ground_cover.sample(39.0997, -94.5786)
        self.assertEqual(kc["ground_cover"][0]["group"], "Developed")
        woods = ground_cover.sample(36.80, -91.25)
        self.assertEqual(woods["ground_cover"][0]["group"], "Trees")
        self.assertGreaterEqual(woods["canopy_cover_pct"], 30)
        self.assertLess(woods["canopy_height_m"], 60)  # metres, not tenths of metres
        self.assertEqual(ground_cover.sample(36.20, -90.05)["ground_cover"][0]["group"], "Cropland")

    def test_shares_sum_to_100(self):
        result = ground_cover.sample(38.6, -92.3)
        for key in ("groups", "nlcd_groups", "ground_cover"):
            self.assertEqual(sum(g["pct"] for g in result[key]), 100, key)

    def test_a_ngfs_sized_footprint_gives_a_real_mix(self):
        # ~3 x 3 km pixel, like a GOES detection footprint: should span several 480 m cells.
        x, y = TO_5070.transform(-92.3, 38.6)
        half = 1500
        ring = [list(TO_LONLAT.transform(x + dx, y + dy)) for dx, dy in
                [(-half, -half), (half, -half), (half, half), (-half, half), (-half, -half)]]
        result = ground_cover.sample(38.6, -92.3, footprints=[{"type": "Polygon", "coordinates": [ring]}])
        self.assertEqual(result["basis"], "footprint")
        self.assertGreaterEqual(result["cells"], 25)

    def test_performance_budget(self):
        x, y = TO_5070.transform(-92.0, 38.0)
        half = 1500
        ring = [list(TO_LONLAT.transform(x + dx, y + dy)) for dx, dy in
                [(-half, -half), (half, -half), (half, half), (-half, half), (-half, -half)]]
        footprint = {"type": "Polygon", "coordinates": [ring]}
        start = time.perf_counter()
        ground_cover.sample(38.0, -92.0, footprints=[footprint])
        cold = time.perf_counter() - start
        start = time.perf_counter()
        ground_cover.sample(38.0, -92.0, footprints=[footprint])
        warm = time.perf_counter() - start
        self.assertLess(cold, 0.1)
        self.assertLess(warm, 0.02)


if __name__ == "__main__":
    unittest.main()
