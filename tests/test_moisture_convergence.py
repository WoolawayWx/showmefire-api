import sys
import unittest
from pathlib import Path

import numpy as np

API_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(API_DIR))

from maps.moisture_convergence import compute_moisture_convergence


class ComputeMoistureConvergenceTests(unittest.TestCase):
    def setUp(self):
        lon = np.linspace(-95.0, -89.0, 60)
        lat = np.linspace(36.0, 40.0, 60)
        self.lon_mesh, self.lat_mesh = np.meshgrid(lon, lat)

    def test_uniformly_diverging_wind_gives_negative_convergence(self):
        # u increasing eastward (constant specific humidity) => pure
        # divergence everywhere => convergence (our output) should be
        # negative in the interior (edges use one-sided differences and can
        # differ slightly).
        u = self.lon_mesh - self.lon_mesh.min()  # increases eastward
        v = np.zeros_like(self.lon_mesh)
        q = np.full_like(self.lon_mesh, 0.01)

        convergence = compute_moisture_convergence(u, v, q, self.lon_mesh, self.lat_mesh)
        interior = convergence[5:-5, 5:-5]
        self.assertTrue(np.all(interior < 0))

    def test_uniformly_converging_wind_gives_positive_convergence(self):
        # u decreasing eastward => pure convergence everywhere.
        u = -(self.lon_mesh - self.lon_mesh.min())
        v = np.zeros_like(self.lon_mesh)
        q = np.full_like(self.lon_mesh, 0.01)

        convergence = compute_moisture_convergence(u, v, q, self.lon_mesh, self.lat_mesh)
        interior = convergence[5:-5, 5:-5]
        self.assertTrue(np.all(interior > 0))

    def test_zero_wind_gives_zero_convergence(self):
        u = np.zeros_like(self.lon_mesh)
        v = np.zeros_like(self.lon_mesh)
        q = np.full_like(self.lon_mesh, 0.01)

        convergence = compute_moisture_convergence(u, v, q, self.lon_mesh, self.lat_mesh)
        np.testing.assert_allclose(convergence, 0.0, atol=1e-12)

    def test_output_shape_matches_input(self):
        u = np.random.default_rng(0).normal(size=self.lon_mesh.shape)
        v = np.random.default_rng(1).normal(size=self.lon_mesh.shape)
        q = np.full_like(self.lon_mesh, 0.01)
        convergence = compute_moisture_convergence(u, v, q, self.lon_mesh, self.lat_mesh)
        self.assertEqual(convergence.shape, self.lon_mesh.shape)


if __name__ == "__main__":
    unittest.main()
