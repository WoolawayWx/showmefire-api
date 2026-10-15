import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio

from maps.station_rbf import build_fill_products, idw_grid, mask_to_missouri, smooth_nan


def _mesh(lons, lats):
    return np.meshgrid(np.asarray(lons, dtype=float), np.asarray(lats, dtype=float))


class IdwTests(unittest.TestCase):
    def test_reproduces_station_values_and_stays_in_range(self):
        lon, lat = _mesh(np.linspace(-93, -90, 31), np.linspace(37.5, 38.5, 11))
        points = [(-92.0, 38.0, 1.0), (-91.0, 38.0, 0.0), (-91.5, 37.6, 0.4)]
        grid = idw_grid(points, lon, lat, max_km=300)
        self.assertGreaterEqual(np.nanmin(grid), 0.0)   # never negative (an RBF would be)
        self.assertLessEqual(np.nanmax(grid), 1.0)      # never above the observations
        at_station = grid[np.argmin(abs(lat[:, 0] - 38.0)), np.argmin(abs(lon[0] - -92.0))]
        self.assertAlmostEqual(at_station, 1.0, places=2)

    def test_dry_area_is_zero_not_missing(self):
        lon, lat = _mesh([-92.0, -92.1], [38.0, 38.1])
        grid = idw_grid([(-92.05, 38.05, 0.0), (-92.2, 38.0, 0.0)], lon, lat, max_km=50)
        self.assertTrue(np.all(grid == 0.0))

    def test_cells_beyond_max_km_are_no_data(self):
        lon, lat = _mesh([-92.0, -89.5], [38.0])  # second cell is ~220 km from the only station
        grid = idw_grid([(-92.0, 38.0, 2.0)], lon, lat, max_km=80)
        self.assertEqual(grid[0, 0], 2.0)
        self.assertTrue(np.isnan(grid[0, 1]))

    def test_no_points_gives_all_nan(self):
        lon, lat = _mesh([-92.0], [38.0])
        self.assertTrue(np.all(np.isnan(idw_grid([], lon, lat))))

    def test_nearer_station_dominates(self):
        lon, lat = _mesh([-92.0], [38.0])
        grid = idw_grid([(-92.05, 38.0, 1.0), (-91.0, 38.0, 0.0)], lon, lat, max_km=200)
        self.assertGreater(grid[0, 0], 0.9)


class SmoothAndMaskTests(unittest.TestCase):
    def test_smoothing_ignores_no_data_edges(self):
        grid = np.full((20, 20), 2.0)
        grid[:, 10:] = np.nan
        smoothed = smooth_nan(grid, 3.0)
        self.assertTrue(np.allclose(smoothed[:, :10], 2.0))   # no fade toward zero at the edge
        self.assertTrue(np.all(np.isnan(smoothed[:, 10:])))

    def test_mask_keeps_missouri_only(self):
        lon, lat = _mesh([-92.5, -89.0], [38.5, 40.0])  # (-92.5, 38.5) MO; (-89.0, 40.0) Illinois
        grid = mask_to_missouri(np.ones(lon.shape), lon, lat)
        self.assertEqual(grid[0, 0], 1.0)
        self.assertTrue(np.isnan(grid[1, 1]))


class BuildFillProductsTests(unittest.TestCase):
    SPEC = dict(name='t_precip', field='precip_24h_in', cmap='Blues', vmin=0, vmax=3.0, max_km=80,
                description='test', legend='test legend')

    def test_writes_a_geotiff_with_missouri_data_and_transparent_outside(self):
        records = [
            {'longitude': -92.3, 'latitude': 38.6, 'precip_24h_in': 0.0},   # dry stations are data
            {'longitude': -91.0, 'latitude': 38.0, 'precip_24h_in': 1.2},
            {'longitude': -93.0, 'latitude': 37.5, 'precip_24h_in': None},  # unknown: not zero
        ]
        with tempfile.TemporaryDirectory() as tmp:
            written = build_fill_products(records, [self.SPEC], out_dir=tmp)
            self.assertEqual(written, ['t_precip'])
            with rasterio.open(Path(tmp) / 't_precip.tif') as src:
                self.assertEqual(src.count, 4)
                alpha = src.read(4)
                self.assertGreater(alpha.max(), 0)           # data inside Missouri
                self.assertEqual(alpha[0, 0], 0)             # NW corner of the extent: outside MO
                self.assertIn('CREATED', src.tags())          # conditions.py reads this

    def test_no_stations_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            written = build_fill_products([{'longitude': -92.0, 'latitude': 38.0, 'precip_24h_in': None}],
                                          [self.SPEC], out_dir=tmp)
            self.assertEqual(written, [])
            self.assertFalse((Path(tmp) / 't_precip.tif').exists())


if __name__ == '__main__':
    unittest.main()
