"""Tests for services/fire_outlook/acquisition.py with fake fetchers (no network)."""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from services.fire_outlook import acquisition as aq, driver, moisture

CYCLE = datetime(2026, 10, 29, 0, tzinfo=timezone.utc)
LAT1 = np.arange(40.0, 36.99, -0.25)       # descending, like GRIB
LON1 = np.arange(-94.0, -90.99, 0.25)
TLAT, TLON = np.meshgrid(np.linspace(38.0, 39.0, 5), np.linspace(-93.0, -92.0, 6), indexing="ij")


def bucket_start(lead):
    return ((lead - 1) // 6) * 6


def fake_fetch(session, cycle, member, lead, domain, tmp_dir):
    """Per-lead fields: T rises 0.1 K/h, precip 1 mm/h (as GEFS 6 h reset buckets)."""
    yy, xx = np.meshgrid(LAT1, LON1, indexing="ij")
    base = {"rows_lat": LAT1, "cols_lon": LON1,
            "t2m": (273.15 + 0.1 * lead + 0 * yy).astype("float32"), "r2": (50.0 + 0 * yy).astype("float32"),
            "u10": (2.0 + yy * 0).astype("float32"), "v10": (0.0 * yy).astype("float32"),
            "swe": (0.0 * yy).astype("float32")}
    if lead == 0:
        base["apcp"] = None
    else:
        start = bucket_start(lead)
        base["apcp"] = (start, ((lead - start) * 1.0 + 0 * yy).astype("float32"))
    return base


def test_grib_url_member_and_cycle_formatting():
    assert aq.grib_url(CYCLE, "c00", 6).endswith("/gefs.20261029/00/atmos/pgrb2sp25/gec00.t00z.pgrb2s.0p25.f006")
    c12 = datetime(2026, 1, 2, 12, tzinfo=timezone.utc)
    assert aq.grib_url(c12, "p05", 168).endswith("/gefs.20260102/12/atmos/pgrb2sp25/gep05.t12z.pgrb2s.0p25.f168")
    assert len(aq.MEMBERS) == 31 and aq.MEMBERS[0] == "c00" and aq.MEMBERS[-1] == "p30" and len(aq.LEADS) == 57


def test_three_hour_precip_differences_inside_reset_buckets():
    per_lead = {}
    for lead in aq.LEADS[1:]:
        start = bucket_start(lead)
        per_lead[lead] = (start, np.full((2, 2), float(lead - start), dtype="float32"))
    out = aq.three_hour_precip(per_lead)
    assert out.shape == (56, 2, 2)
    np.testing.assert_allclose(out, 3.0)         # 1 mm/h x 3 h everywhere, both halves of every bucket


def test_three_hour_precip_rejects_misaligned_buckets():
    per_lead = {lead: (bucket_start(lead), np.zeros((1, 1), "float32")) for lead in aq.LEADS[1:]}
    per_lead[9] = (0, np.zeros((1, 1), "float32"))   # bucket start should be 6
    with pytest.raises(aq.MemberIncomplete):
        aq.three_hour_precip(per_lead)


def test_fetch_member_caches_and_resumes(tmp_path):
    calls = []

    def counting(*a):
        calls.append(a[3])
        return fake_fetch(*a)
    dom = aq.Domain(36.0, 41.0, -95.0, -90.0)
    p = aq.fetch_member(CYCLE, "c00", tmp_path, dom, fetch=counting)
    assert len(calls) == 57 and p.exists() and not list(tmp_path.glob("*.tmp*"))
    aq.fetch_member(CYCLE, "c00", tmp_path, dom, fetch=counting)
    assert len(calls) == 57                                 # cached: no refetch
    p.write_bytes(b"truncated")                             # corrupt entry is refetched
    aq.fetch_member(CYCLE, "c00", tmp_path, dom, fetch=counting)
    assert len(calls) == 114


def test_missing_lead_fails_the_member_and_leaves_no_cache(tmp_path):
    def flaky(session, cycle, member, lead, domain, tmp_dir):
        if lead == 90:
            raise aq.MemberIncomplete("p07 f090 not published")
        return fake_fetch(session, cycle, member, lead, domain, tmp_dir)
    with pytest.raises(aq.MemberIncomplete):
        aq.fetch_member(CYCLE, "p07", tmp_path, aq.Domain(36, 41, -95, -90), fetch=flaky)
    assert list(tmp_path.iterdir()) == []


def test_regridder_reproduces_linear_field_with_descending_lat():
    rg = aq.Regridder(LAT1, LON1, TLAT, TLON)
    yy, xx = np.meshgrid(LAT1, LON1, indexing="ij")
    field = 3.0 * yy + 2.0 * xx
    np.testing.assert_allclose(rg(field), 3.0 * TLAT + 2.0 * TLON, rtol=1e-5)
    assert rg.shape == TLAT.shape
    with pytest.raises(aq.MemberIncomplete):
        aq.Regridder(LAT1, LON1, TLAT + 10.0, TLON)         # target outside the cropped domain


def _cached(tmp_path):
    return aq.fetch_member(CYCLE, "c00", tmp_path, aq.Domain(36, 41, -95, -90), fetch=fake_fetch)


def test_member_dataset_matches_driver_contract_and_conserves_precip(tmp_path):
    ds = aq.member_dataset(CYCLE, _cached(tmp_path), lambda l1, l2: aq.Regridder(l1, l2, TLAT, TLON))
    assert dict(ds.sizes) == {"time": 169, "y": 5, "x": 6}
    f = driver._fields(ds, CYCLE)                             # the driver accepts it as is
    assert set(driver.REQUIRED) <= set(f)
    np.testing.assert_allclose(ds.temperature_2m.values[:, 0, 0], 0.1 * np.arange(169), atol=1e-4)  # K -> degC, linear in time
    np.testing.assert_allclose(ds.temperature_2m.values[1, 0, 0] - ds.temperature_2m.values[0, 0, 0], 0.1, atol=1e-4)
    p = ds.precipitation_increment.values[:, 0, 0]
    assert p[0] == 0 and np.allclose(p[1:], 1.0)              # even allocation of 3 mm over 3 h
    assert p.sum() == pytest.approx(168.0, rel=1e-5)          # nothing lost or invented
    assert ds.attrs["precip_source_interval_hours"] == 3


def test_member_dataset_runs_through_the_driver(tmp_path):
    from services.fire_outlook import rules
    ds = aq.member_dataset(CYCLE, _cached(tmp_path), lambda l1, l2: aq.Regridder(l1, l2, TLAT, TLON))
    categorize, dampen, _ = rules.rule_functions(gdd_accum=None)
    init = moisture.initial_state(ds.temperature_2m.values[0], ds.relative_humidity_2m.values[0])
    out = driver.run_member(ds, CYCLE, init, categorize, dampen)
    assert out["max_category"].shape == (28, 5, 6) and out["precip_6h_mm"][0, 0, 0] == pytest.approx(6.0, rel=1e-4)


class FakeSession:
    def __init__(self, missing=(), modified="Wed, 28 Oct 2026 04:59:18 GMT"):
        self.missing, self.modified = set(missing), modified

    def head(self, url, **kw):
        member = url.rsplit("/", 1)[1].split(".")[0][2:]   # ".../gep12.t00z..." -> "p12"
        ok = member not in self.missing
        return type("R", (), {"status_code": 200 if ok else 404, "headers": {"Last-Modified": self.modified} if ok else {}})()


def test_inputs_ready_requires_all_31_members():
    ready, cov = aq.inputs_ready(CYCLE, session=FakeSession())
    assert ready and cov["members_ready"] == 31 and cov["available_utc"] == datetime(2026, 10, 28, 4, 59, 18, tzinfo=timezone.utc)
    ready, cov = aq.inputs_ready(CYCLE, session=FakeSession(missing=["p12", "p30"]))
    assert not ready and cov["members_ready"] == 29 and cov["missing_members"] == ["p12", "p30"] and cov["available_utc"] is None


def test_domain_crop_and_overlap_check():
    dom = aq.Domain(37.5, 39.5, -93.5, -91.5)
    lat2d, lon2d = np.meshgrid(LAT1, LON1, indexing="ij")
    rows, cols = dom.crop_indices(lat2d, lon2d)
    assert LAT1[rows].min() >= 37.5 and LAT1[rows].max() <= 39.5 and LON1[cols].min() >= -93.5
    with pytest.raises(aq.MemberIncomplete):
        aq.Domain(10, 11, -60, -59).crop_indices(lat2d, lon2d)
