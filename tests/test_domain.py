import unittest

import numpy as np
import xarray as xr

from core.domain import MO_BUFFERED_BBOX, crop


def _synthetic_conus_grid():
    """A coarse lon/lat grid spanning well beyond Missouri on all sides,
    with 2D (y, x) coordinates like a real HRRR/Lambert-conformal grid."""
    lons_1d = np.arange(-125.0, -65.0, 1.0)  # west coast to east coast
    lats_1d = np.arange(20.0, 50.0, 1.0)  # gulf to canada
    lon2d, lat2d = np.meshgrid(lons_1d, lats_1d)
    t2m = np.zeros_like(lon2d)
    return xr.Dataset(
        {"t2m": (["y", "x"], t2m)},
        coords={"longitude": (["y", "x"], lon2d), "latitude": (["y", "x"], lat2d)},
    )


class CropTests(unittest.TestCase):
    def test_crops_a_full_conus_grid_down_to_the_buffered_missouri_box(self):
        ds = _synthetic_conus_grid()
        cropped = crop(ds)

        west, east, south, north = MO_BUFFERED_BBOX
        self.assertTrue((cropped.longitude.values >= west - 1.0).all())
        self.assertTrue((cropped.longitude.values <= east + 1.0).all())
        self.assertTrue((cropped.latitude.values >= south - 1.0).all())
        self.assertTrue((cropped.latitude.values <= north + 1.0).all())
        # Strictly smaller than the input on both grid dimensions.
        self.assertLess(cropped.sizes["y"], ds.sizes["y"])
        self.assertLess(cropped.sizes["x"], ds.sizes["x"])

    def test_crops_a_realistic_merged_forecast_dataset_with_a_step_dimension(self):
        # Mirrors the real shape DailyForecast.py's ds_full has at crop time:
        # multiple leads (step) merged from separate FastHerbie calls, with
        # 2D longitude/latitude coords that carry no step dimension of their
        # own - exactly what ds_rh_temp.merge(ds_wind).merge(ds_precip)
        # produces before to_netcdf.
        ds = _synthetic_conus_grid()
        n_steps = 12
        ds = ds.expand_dims(step=n_steps).assign(
            t2m=(["step", "y", "x"], np.zeros((n_steps, *ds.t2m.shape))),
            r2=(["step", "y", "x"], np.zeros((n_steps, *ds.t2m.shape))),
        )
        cropped = crop(ds)

        self.assertEqual(cropped.sizes["step"], n_steps)
        self.assertLess(cropped.sizes["y"], ds.sizes["y"])
        self.assertLess(cropped.sizes["x"], ds.sizes["x"])
        self.assertEqual(cropped.t2m.dims, ("step", "y", "x"))
        self.assertEqual(cropped.t2m.shape, (n_steps, cropped.sizes["y"], cropped.sizes["x"]))

    def test_raises_when_the_grid_does_not_intersect_missouri(self):
        lons_1d = np.arange(-20.0, -10.0, 1.0)  # nowhere near North America
        lats_1d = np.arange(10.0, 20.0, 1.0)
        lon2d, lat2d = np.meshgrid(lons_1d, lats_1d)
        ds = xr.Dataset(
            {"t2m": (["y", "x"], np.zeros_like(lon2d))},
            coords={"longitude": (["y", "x"], lon2d), "latitude": (["y", "x"], lat2d)},
        )
        with self.assertRaises(ValueError):
            crop(ds)


if __name__ == "__main__":
    unittest.main()
