"""Tests for services/fire_outlook history.py (causal replay + MRMS fill) and mrms.py, on synthetic RTMA cache files."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import xarray as xr

from services.fire_outlook import conditioning, history, moisture, mrms

END = datetime(2026, 10, 29, 12, tzinfo=timezone.utc)
HOURS, MIN = 10, 5
# RTMA-like source grid covering lon -96..-93 (stored 0-360 like the real cache), lat 36..40
SRC_LAT, SRC_LON_180 = np.meshgrid(np.linspace(36, 40, 17), np.linspace(-96, -93, 13), indexing="ij")
# target grid: inside the hull, plus a strip west of it (the real public grid is wider than the RTMA crop)
TLAT, TLON = np.meshgrid(np.linspace(37, 39, 5), np.linspace(-96.6, -94, 9), indexing="ij")


def write_hour(dirpath, stamp, t_k=293.15, rh=40.0, precip=0.0, missing=False, lon=None, drop_apcp=False):
    shape = SRC_LAT.shape
    ds = xr.Dataset({
        "t2m": (("y", "x"), np.full(shape, t_k, "float32")), "r2": (("y", "x"), np.full(shape, rh, "float32")),
        "latitude": (("y", "x"), SRC_LAT), "longitude": (("y", "x"), (SRC_LON_180 if lon is None else lon) % 360.0),
    })
    if not drop_apcp:
        ds["apcp"] = (("y", "x"), np.full(shape, 0.0 if missing else precip, "float32"))
        if missing:
            ds["apcp"].attrs["missing"] = "true"
    ds.to_netcdf(dirpath / f"rtma_{stamp:%Y%m%d_%H}z.nc")


def stamps():
    base = END.replace(tzinfo=None)
    return [base - timedelta(hours=h) for h in range(HOURS - 1, -1, -1)]


def cache(tmp_path, per_hour=None):
    """Write HOURS files; per_hour maps hour index -> kwargs for write_hour."""
    per_hour = per_hour or {}
    rtma = tmp_path / "rtma"
    rtma.mkdir()
    for i, s in enumerate(stamps()):
        write_hour(rtma, s, **per_hour.get(i, {}))
    return rtma


def run(tmp_path, rtma, **kw):
    return history.condition(TLAT, TLON, END, rtma_dir=rtma, cache_dir=tmp_path / "c", hours=HOURS, min_hours=MIN, **kw)


def manual(precips, skip=()):
    """Reference replay with constant 20 C / 40 % weather using moisture.advance, seeded from EMC at the first used hour."""
    t, rh = np.full(TLAT.shape, 20.0), np.full(TLAT.shape, 40.0)
    state = None
    for i, p in enumerate(precips):
        if i in skip:
            continue
        if state is None:
            e = moisture.nelson_emc(t, rh)
            state = {"fm1": e.copy(), "fm10": e.copy(), "fm100": e.copy()}
        else:
            state = moisture.advance(state, t, rh, np.full(TLAT.shape, p))
    return state


def test_replay_matches_the_moisture_step_and_the_existing_conditioning_math(tmp_path):
    precips = [0, 0, 3.0, 0, 0, 0, 1.2, 0, 0, 0]
    rtma = cache(tmp_path, {i: {"precip": p} for i, p in enumerate(precips)})
    out = run(tmp_path, rtma)
    want = manual(precips)
    for k, key in (("fm1", "fm1_pct"), ("fm10", "fm10_pct"), ("fm100", "fm100_pct")):
        np.testing.assert_allclose(out[key], want[k], rtol=1e-6)
    from services import spread_rate_moisture as srm          # same loop as the existing production conditioner
    ref = None
    for p in precips:
        e = srm.nelson_emc(np.full(TLAT.shape, 20.0), np.full(TLAT.shape, 40.0))
        ref = {"fm1": e.copy(), "fm10": e.copy(), "fm100": e.copy()} if ref is None else srm._advance_moisture(
            ref, np.full(TLAT.shape, 20.0), np.full(TLAT.shape, 40.0), np.full(TLAT.shape, p))
    np.testing.assert_allclose(out["fm10_pct"], ref["fm10"], rtol=1e-6)
    a = out["precip_audit"]
    assert a["hours_replayed"] == HOURS and a["precip_rtma_hours"] == HOURS and a["precip_unfilled_hours"] == 0


def test_flagged_missing_precip_is_filled_from_mrms_not_replayed_as_dry(tmp_path):
    rtma = cache(tmp_path, {4: {"missing": True}})
    rain_hour = stamps()[4]
    asked = []

    def fill(hour):
        asked.append(hour)
        return np.full(TLAT.shape, 8.0, "float32"), "MultiSensor_QPE_01H_Pass2_00.00"
    out = run(tmp_path, rtma, precip_fill=fill)
    assert asked == [rain_hour]                                         # only the flagged hour is looked up
    want = manual([0, 0, 0, 0, 8.0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(out["fm10_pct"], want["fm10"], rtol=1e-6)
    dry = manual([0] * HOURS)
    assert (out["fm10_pct"] > dry["fm10"] + 0.5).all()                   # a plain replay would have left fuels drier
    a = out["precip_audit"]
    assert a["precip_mrms_hours"] == 1 and a["precip_mrms_products"] == {"MultiSensor_QPE_01H_Pass2_00.00": 1}
    assert a["precip_rtma_hours"] == HOURS - 1


def test_unfilled_hour_is_skipped_never_dry_and_limit_enforced(tmp_path):
    rtma = cache(tmp_path, {4: {"missing": True}, 7: {"missing": True}})
    out = run(tmp_path, rtma, precip_fill=lambda h: None)
    np.testing.assert_allclose(out["fm10_pct"], manual([0] * HOURS, skip=(4, 7))["fm10"], rtol=1e-6)
    assert out["precip_audit"]["precip_unfilled_hours"] == 2 and len(out["precip_audit"]["unfilled_hours"]) == 2
    assert out["precip_audit"]["hours_replayed"] == HOURS - 2
    with pytest.raises(RuntimeError, match="no usable precipitation"):
        run(tmp_path, rtma, precip_fill=None, max_gap_hours=1)


def test_edge_cells_outside_rtma_hull_take_nearest_inside_value_and_are_counted(tmp_path):
    out = run(tmp_path, cache(tmp_path))
    assert np.isfinite(out["fm10_pct"]).all()                            # TLON reaches -96.6, west of the -96 hull edge
    assert out["precip_audit"]["edge_filled_cells"] > 0
    np.testing.assert_allclose(out["fm10_pct"][:, 0], out["fm10_pct"][:, 3], rtol=1e-6)  # uniform fields: edge == interior


def test_missing_files_counted_and_too_little_history_raises(tmp_path):
    rtma = cache(tmp_path)
    (rtma / f"rtma_{stamps()[2]:%Y%m%d_%H}z.nc").unlink()
    out = run(tmp_path, rtma)
    assert out["conditioning_hours_missing"] == 1 and out["precip_audit"]["hours_file_missing"] == 1
    for s in stamps()[:6]:
        (rtma / f"rtma_{s:%Y%m%d_%H}z.nc").unlink(missing_ok=True)
    with pytest.raises(RuntimeError, match="insufficient RTMA history"):
        run(tmp_path, rtma)


def test_nothing_after_the_analysis_hour_is_read(tmp_path):
    rtma = cache(tmp_path)
    write_hour(rtma, END.replace(tzinfo=None) + timedelta(hours=1), t_k=330.0, rh=2.0, precip=0.0)   # a future hour
    write_hour(rtma, END.replace(tzinfo=None) + timedelta(hours=5), t_k=330.0, rh=2.0, precip=50.0)
    np.testing.assert_allclose(run(tmp_path, rtma)["fm10_pct"], manual([0] * HOURS)["fm10"], rtol=1e-6)
    assert history.condition.__defaults__ is not None and "analysis_hour" in history.condition.__code__.co_varnames
    with pytest.raises(ValueError):
        history.condition(TLAT, TLON, None, rtma_dir=rtma, cache_dir=tmp_path / "c")        # never anchors on the clock


def test_condition_initial_state_uses_the_replay_and_records_the_audit(tmp_path):
    rtma = cache(tmp_path, {3: {"missing": True}})
    cond = history.make_condition(None, rtma_dir=rtma, cache_dir=tmp_path / "c", hours=HOURS, min_hours=MIN)
    state, q = conditioning.condition_initial_state(END, TLAT, TLON, None, condition=cond)
    assert q["level"] == conditioning.LEVEL_CONDITIONED and q["precip_audit"]["precip_unfilled_hours"] == 1
    assert all(np.isfinite(v).all() for v in state.values())
    cond_strict = history.make_condition(None, rtma_dir=rtma, cache_dir=tmp_path / "c", hours=HOURS, min_hours=MIN, max_gap_hours=0)
    with pytest.raises(conditioning.ConditioningUnavailable):
        conditioning.condition_initial_state(END, TLAT, TLON, None, condition=cond_strict)


# --- mrms ------------------------------------------------------------------------------------------------------

def test_mrms_url_and_hour_formatting():
    u = mrms.url("MultiSensor_QPE_01H_Pass2_00.00", datetime(2026, 10, 28, 12, tzinfo=timezone.utc))
    assert u.endswith("/MultiSensor_QPE_01H_Pass2_00.00/20261028/MRMS_MultiSensor_QPE_01H_Pass2_00.00_20261028-120000.grib2.gz")


def test_sample_to_grid_area_mean_and_nodata():
    lat1 = np.linspace(40, 36, 41)                      # descending like MRMS
    lon1 = np.linspace(-96, -92, 41)
    grid = np.full((41, 41), 2.0)
    sub_lat, sub_lon = mrms.subcell_samples(TLAT, TLON)
    assert len(sub_lat) == 9 and sub_lat[0].shape == TLAT.shape
    np.testing.assert_allclose(mrms.sample_to_grid(grid, lat1, lon1, sub_lat, sub_lon), 2.0, rtol=1e-6)
    holed = grid.copy()
    holed[:, :] = np.nan                                 # no coverage anywhere -> NaN, never 0
    assert np.isnan(mrms.sample_to_grid(holed, lat1, lon1, sub_lat, sub_lon)).all()
    ramp = np.tile(np.linspace(0, 8, 41), (41, 1))       # increases eastward: mean of the cell == value at its centre
    got = mrms.sample_to_grid(ramp, lat1, lon1, sub_lat, sub_lon)
    centre = mrms.sample_to_grid(ramp, lat1, lon1, [TLAT], [TLON])
    np.testing.assert_allclose(got, centre, atol=0.05)


class _Resp:
    def __init__(self, code, content=b"x"):
        self.status_code, self.content = code, content


class _Session:
    def __init__(self, codes):
        self.codes, self.calls = codes, []

    def get(self, url, timeout=0):
        product = next(p for p in mrms.PRODUCTS if p in url)
        self.calls.append(product)
        return _Resp(self.codes[product])


def _patch_decode(monkeypatch):
    monkeypatch.setattr(mrms, "decode", lambda b: (np.full((60, 80), 1.5, "float64"),
                                                   np.linspace(41, 33, 60), np.linspace(-98, -86, 80)))


BBOX = (34.0, 41.0, -97.0, -87.0)


def test_fetch_prefers_pass2_falls_back_to_pass1_caches_and_returns_none_when_absent(tmp_path, monkeypatch):
    _patch_decode(monkeypatch)
    hour = datetime(2026, 10, 28, 12, tzinfo=timezone.utc)
    s = _Session({mrms.PRODUCTS[0]: 404, mrms.PRODUCTS[1]: 200})
    got = mrms.fetch_hour(hour, tmp_path, BBOX, session=s)
    assert got["product"] == mrms.PRODUCTS[1] and s.calls == list(mrms.PRODUCTS) and got["mm"].dtype == np.float32
    again = mrms.fetch_hour(hour, tmp_path, BBOX, session=_Session({}))   # cache hit: no network at all
    assert again["product"] == mrms.PRODUCTS[1]
    both = _Session({p: 404 for p in mrms.PRODUCTS})
    assert mrms.fetch_hour(hour + timedelta(hours=1), tmp_path, BBOX, session=both) is None
    err = _Session({mrms.PRODUCTS[0]: 500, mrms.PRODUCTS[1]: 500})
    assert mrms.fetch_hour(hour + timedelta(hours=2), tmp_path, BBOX, session=err) is None
    (tmp_path / f"mrms_{hour:%Y%m%d%H}.npz").write_bytes(b"corrupt")       # corrupt cache is refetched
    assert mrms.fetch_hour(hour, tmp_path, BBOX, session=_Session({mrms.PRODUCTS[0]: 200}))["product"] == mrms.PRODUCTS[0]


def test_make_filler_regrids_fetched_hour_and_reports_product(tmp_path, monkeypatch):
    _patch_decode(monkeypatch)
    sub_lat, sub_lon = mrms.subcell_samples(TLAT, TLON)
    fill = mrms.make_filler(tmp_path, BBOX, sub_lat, sub_lon, session=_Session({mrms.PRODUCTS[0]: 200}))
    grid, product = fill(datetime(2026, 10, 28, 12, tzinfo=timezone.utc))
    assert grid.shape == TLAT.shape and product == mrms.PRODUCTS[0]
    np.testing.assert_allclose(grid, 1.5, rtol=1e-5)
    none = mrms.make_filler(tmp_path / "n", BBOX, sub_lat, sub_lon, session=_Session({p: 404 for p in mrms.PRODUCTS}))
    assert none(datetime(2026, 10, 28, 13, tzinfo=timezone.utc)) is None
