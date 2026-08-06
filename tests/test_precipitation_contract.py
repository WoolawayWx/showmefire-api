import unittest

import numpy as np
import xarray as xr

from core.precipitation import decode_forecast_precipitation, final_accumulation_mm, normalize_to_mm


class PrecipitationContractTests(unittest.TestCase):
    def test_accumulation_is_not_summed_across_leads(self):
        array = xr.DataArray(
            np.array([1.0, 3.0, 6.0])[:, None, None],
            dims=("step", "y", "x"),
            coords={"step": np.array([1, 2, 3], dtype="timedelta64[h]")},
            attrs={"units": "kg m**-2", "GRIB_stepType": "accum"},
        )
        dataset = xr.Dataset({"tp": array})
        decoded = decode_forecast_precipitation(dataset)
        np.testing.assert_allclose(decoded.interval_mm.values[:, 0, 0], [1, 2, 3])
        self.assertEqual(float(final_accumulation_mm(dataset).values[0, 0]), 6.0)

    def test_units_fail_closed(self):
        with self.assertRaises(ValueError): normalize_to_mm(np.array([1.0]), None)
        with self.assertRaises(ValueError): normalize_to_mm(np.array([1.0]), "unknown")


if __name__ == "__main__":
    unittest.main()
