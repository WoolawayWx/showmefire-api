from __future__ import annotations

from datetime import datetime

import numpy as np
import xarray as xr

from services.mrms_capture import _normalize_coordinates, cache_path_for_hour
from services.verification_rainfall import load_mrms_grid


def test_normalize_mrms_1d_coordinates():
    raw = xr.Dataset(
        {"unknown": (("latitude", "longitude"), np.ones((2, 3), dtype=np.float32))},
        coords={"latitude": [35.0, 36.0], "longitude": [-95.0, -94.0, -93.0]},
    )
    normalized = _normalize_coordinates(raw)
    assert normalized["precipitation"].attrs["units"] == "mm"
    assert normalized["latitude"].size == 2
    assert normalized["longitude"].size == 3


def test_mrms_cache_path_is_utc_hourly(tmp_path):
    from datetime import timezone

    path = cache_path_for_hour(datetime(2026, 9, 15, 16, 37, tzinfo=timezone.utc), tmp_path)
    assert path.name == "mrms_20260915_16z.nc"


def test_mrms_grid_loader_reads_normalized_netcdf(tmp_path, monkeypatch):
    path = tmp_path / "mrms_20260915_16z.nc"
    xr.Dataset(
        {
            "precipitation": (("latitude", "longitude"), np.ones((2, 2), dtype=np.float32)),
        },
        coords={"latitude": [35.0, 36.0], "longitude": [-95.0, -94.0]},
    ).to_netcdf(path)
    monkeypatch.setenv("VERIFICATION_MRMS_ROOT", str(tmp_path))
    values, lon, lat, metadata = load_mrms_grid(
        datetime(2026, 9, 15, 16)
    )
    assert values.shape == (2, 2)
    assert lon.shape == (2,)
    assert lat.shape == (2,)
    assert metadata["provider"] == "mrms"
