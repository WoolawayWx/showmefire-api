import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from services import rtma_verification as rv
from services.rtma_capture import cache_path_for_hour
from services.rtma_peak import _hours_for_local_date


def _write_hour(cache_dir, hour, t_k=300.0, rh=40.0, u=3.0, v=4.0):
    lat, lon = np.meshgrid(np.linspace(37, 39, 5), np.linspace(-93, -91, 5), indexing='ij')
    shape = lat.shape
    ds = xr.Dataset(
        {
            't2m': (('y', 'x'), np.full(shape, t_k)),
            'r2': (('y', 'x'), np.full(shape, rh)),
            'u10': (('y', 'x'), np.full(shape, u)),
            'v10': (('y', 'x'), np.full(shape, v)),
        },
        coords={'latitude': (('y', 'x'), lat), 'longitude': (('y', 'x'), lon)},
    )
    path = cache_path_for_hour(hour, cache_dir=cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def _fd(fm, rh, wind_kts):
    return 1


class SampleRtmaTests(unittest.TestCase):
    def test_samples_cached_hours_and_converts_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            hours = _hours_for_local_date(date(2025, 7, 12))
            for hour in hours[:3]:
                _write_hour(tmp, hour)
            df = rv.sample_rtma_at_points(
                date(2025, 7, 12), {'AAA': (38.0, -92.0), 'BAD': (None, None)}, _fd, cache_dir=Path(tmp)
            )
        self.assertEqual(len(df), 3)
        self.assertEqual(set(df['stid']), {'AAA'})
        row = df.iloc[0]
        self.assertAlmostEqual(row['rtma_temp'], 300.0 - 273.15)
        self.assertAlmostEqual(row['rtma_wind'], 5.0)
        self.assertAlmostEqual(row['rtma_fm'], 3.0 + 0.25 * 40.0)
        self.assertEqual(row['rtma_fire_danger'], 1)

    def test_no_cache_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = rv.sample_rtma_at_points(date(2025, 7, 12), {'AAA': (38.0, -92.0)}, _fd, cache_dir=Path(tmp))
        self.assertTrue(df.empty)


class StationPointMetricsTests(unittest.TestCase):
    def test_bias_is_first_minus_rtma(self):
        ts = pd.Timestamp('2025-07-12 18:00', tz='UTC')
        merged = pd.DataFrame([{
            'stid': 'AAA', 'timestamp': ts,
            'pred_temp': 30.0, 'obs_temp': 28.0, 'pred_rh': 40.0, 'obs_rh': 45.0,
            'pred_wind': 5.0, 'obs_wind': 4.0, 'pred_fm': 10.0, 'obs_fm': 9.0,
            'pred_fire_danger': 2, 'obs_fire_danger': 1,
        }])
        rtma = pd.DataFrame([{
            'stid': 'AAA', 'timestamp': ts, 'rtma_temp': 27.0, 'rtma_rh': 50.0,
            'rtma_wind': 4.5, 'rtma_fm': 12.0, 'rtma_fire_danger': 1,
        }])
        out = rv.station_point_metrics(merged, rtma)
        self.assertEqual(out['record_count'], 1)
        self.assertEqual(out['forecast_vs_rtma']['Temperature (C)']['bias'], 3.0)
        self.assertEqual(out['station_vs_rtma']['Temperature (C)']['bias'], 1.0)
        self.assertEqual(out['forecast_vs_rtma']['Fire Danger Index']['bias'], 1.0)
        self.assertIn('Fuel Moisture (%)', out['derived_variables'])

    def test_empty_inputs_return_none(self):
        self.assertIsNone(rv.station_point_metrics(pd.DataFrame(), pd.DataFrame()))

    def test_build_never_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            block = rv.build_rtma_metrics(
                pd.DataFrame(), {'AAA': (38.0, -92.0)}, '2025-07-12', _fd,
                cache_dir=Path(tmp), gis_dir=Path(tmp),
            )
        self.assertFalse(block['available'])


if __name__ == '__main__':
    unittest.main()
