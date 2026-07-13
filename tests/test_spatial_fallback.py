import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from services import spatial_fm


class SpatialFallbackTests(unittest.TestCase):
    def test_missing_assets_returns_none_and_records_fallback(self):
        with patch.object(spatial_fm, "_load_runtime", side_effect=FileNotFoundError("no spatial stable")):
            result = spatial_fm.try_predict(object(), [(-93.0, 38.0, 8), (-92.0, 38.5, 9), (-91.0, 39.0, 10)])
        self.assertIsNone(result); self.assertTrue(spatial_fm.diagnostics()["fallback"])
        self.assertIn("no spatial stable", spatial_fm.diagnostics()["fallback_reason"])

    def test_antecedent_allows_two_causal_fills_but_rejects_three(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); cache = root / "cache" / "rtma"; cache.mkdir(parents=True)
            init = pd.Timestamp("2026-07-12T12:00:00Z"); lat = np.array([[38.0]]); lon = np.array([[-92.0]])
            for offset in range(-12, 1):
                stamp = init + pd.Timedelta(hours=offset)
                ds = xr.Dataset({name: (("y", "x"), np.ones((1, 1), "float32")) for name in ("t2m", "r2", "u10", "v10")},
                                coords={"latitude": (("y", "x"), lat), "longitude": (("y", "x"), lon)})
                ds.to_netcdf(cache / f"rtma_{stamp:%Y%m%d_%H}z.nc")
            for count in range(3):
                if count:
                    stamp = init + pd.Timedelta(hours=-10 + count)
                    (cache / f"rtma_{stamp:%Y%m%d_%H}z.nc").unlink()
                frames, missing, _ = spatial_fm._antecedent_rtma(root, init, lat, lon, 2)
                self.assertEqual(frames.shape, (13, 4, 1, 1)); self.assertEqual(len(missing), count)
            stamp = init + pd.Timedelta(hours=-7); (cache / f"rtma_{stamp:%Y%m%d_%H}z.nc").unlink()
            with self.assertRaisesRegex(ValueError, "maximum is 2"):
                spatial_fm._antecedent_rtma(root, init, lat, lon, 2)


if __name__ == "__main__": unittest.main()
