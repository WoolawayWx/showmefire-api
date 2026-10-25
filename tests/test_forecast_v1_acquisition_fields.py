import numpy as np
import pytest
import xarray as xr

from forecast_v1 import acquisition as aq


def _isobaric(shortnames=("t", "u", "v")):
    levels = [700, 850]
    data = {name: (("isobaricInhPa", "y", "x"), np.ones((2, 3, 3), dtype="float32"), {"GRIB_shortName": name}) for name in shortnames}
    return xr.Dataset(data, coords={"isobaricInhPa": levels})


def test_canonicalizing_twice_keeps_the_upper_air_fields():
    # _fetch_member canonicalizes the already-canonical merged groups again; that used to drop all six level fields.
    once = aq._canonicalize_variables(_isobaric())
    assert set(once.data_vars) == {"t700", "t850", "u700", "u850", "v700", "v850"}
    assert set(aq._canonicalize_variables(once).data_vars) == set(once.data_vars)


def test_rrfs_and_gefs_take_fields_from_the_product_that_carries_them(monkeypatch):
    monkeypatch.delenv("SMF_RRFS_UPPER_PRODUCT", raising=False)
    specs = {spec.public_name: spec for spec in aq.default_specs()}

    assert specs["hrrr"].extras == () and specs["hrrr"].upper_air_on_main
    assert specs["rrfs"].extras == (("prslev.13km", aq.UPPER_AIR_SEARCH),) and not specs["rrfs"].upper_air_on_main
    assert dict(specs["gefs"].extras) == {"atmos.5": aq.UPPER_AIR_SEARCH, "atmos.5b": aq.PBL_SEARCH}
    assert aq.PBL_SEARCH not in specs["gefs"].searches and aq.PBL_SEARCH in specs["hrrr"].searches


def test_rrfs_upper_air_product_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("SMF_RRFS_UPPER_PRODUCT", "")
    rrfs = {spec.public_name: spec for spec in aq.default_specs()}["rrfs"]
    assert rrfs.extras == () and rrfs.upper_air_on_main


def _grid(step, value):
    x = np.arange(-96.0, -94.0 + 1e-6, step)
    y = np.arange(41.0, 39.0 - 1e-6, -step)
    field = np.broadcast_to(x[None, :] * 0 + value, (1, len(y), len(x))).copy()
    return xr.Dataset({"t850": (("time", "y", "x"), field)}, coords={"time": [0], "x": x, "y": y})


def test_coarser_grid_is_put_on_the_surface_grid_without_nan():
    surface, coarse = _grid(0.25, 1.0), _grid(0.5, 7.0)
    aligned = aq._align_to_grid(coarse, surface)
    assert aligned.sizes["x"] == surface.sizes["x"] and aligned.sizes["y"] == surface.sizes["y"]
    assert not bool(aligned.t850.isnull().any()) and float(aligned.t850.mean()) == pytest.approx(7.0)
    merged = xr.merge([surface.rename(t850="t2m"), aligned], join="outer")
    assert merged.sizes["x"] == surface.sizes["x"]


def test_same_grid_is_only_relabelled():
    surface = _grid(0.25, 1.0)
    other = _grid(0.25, 3.0)
    other = other.assign_coords(x=other.x + 1e-9)
    aligned = aq._align_to_grid(other, surface)
    assert np.array_equal(aligned.x.values, surface.x.values)


def test_a_query_that_returns_no_variables_is_a_miss():
    class Client:
        def xarray(self, search, remove_grib=False):
            return xr.Dataset(coords={"latitude": ("y", [40.0]), "longitude": ("x", [-95.0])})

    with pytest.raises(FileNotFoundError):
        aq._query_leads(lambda leads: Client(), (3,), aq.PBL_SEARCH, __import__("datetime").datetime(2026, 10, 24, 12))
