"""Tests for services/fire_outlook rules.py and assemble.py, including an end-to-end tiny run."""
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from services.fire_outlook import assemble, driver, rules, windows
from tests.test_fire_outlook_driver import GRID, member

CYCLE = datetime(2026, 10, 29, 0, tzinfo=timezone.utc)


def _result(n=31, e_fill=0.0):
    shape = (28,) + GRID
    e = np.full(shape, e_fill, dtype="float32")
    return {
        "agreement_elevated": e, "agreement_critical": e * 0.5, "nodata": np.zeros(shape, bool), "n_members": n,
        "fm10_min_p10": np.full(shape, 5, "float32"), "fm10_min_p50": np.full(shape, 8, "float32"),
        "fm10_min_p90": np.full(shape, 11, "float32"),
        "mean_hours_elevated": np.zeros(shape, "float32"), "mean_hours_critical": np.zeros(shape, "float32"),
        "mean_precip_6h_mm": np.zeros(shape, "float32"), "temp_max_c_envelope": np.zeros(shape, "float32"),
        "rh_min_pct_envelope": np.zeros(shape, "float32"), "wind_max_kts_envelope": np.zeros(shape, "float32"),
    }


def _write(tmp_path, result=None, cycle=CYCLE, **kw):
    return assemble.write_run(
        tmp_path, cycle, result or _result(), grid={"id": "test", "shape": list(GRID)},
        source_cycles={"gefs": "2026-10-29T00Z"}, rule_info={"rule_spec_version": "x"},
        conditioning_quality={"level": "conditioned"}, **kw)


# --- rules ------------------------------------------------------------------

def test_rules_record_versions_and_missing_seasonal_state():
    _, dampen, info = rules.rule_functions(gdd_accum=None)
    assert info["seasonal_state"] == "missing_no_dampening" and info["green_factor"] == 0.0
    cat = np.array([2, 3, 0], dtype="int8")
    z = np.zeros(3)
    np.testing.assert_array_equal(dampen(cat, z, z, z), cat)
    assert info["rule_spec_sha256"] and info["rule_spec_version"]


def test_dampening_never_raises_a_category():
    _, dampen, info = rules.rule_functions(gdd_accum=0.0)  # deepest green-up
    assert info["green_factor"] == 1.0 and info["seasonal_state"] == "applied"
    rng = np.random.default_rng(0)
    cat = rng.integers(0, 5, 500).astype("int8")
    out = dampen(cat, rng.uniform(2, 15, 500), rng.uniform(5, 60, 500), rng.uniform(0, 30, 500))
    assert (out <= cat).all() and (out >= cat - 1).all()


# --- assemble ---------------------------------------------------------------

def test_write_read_roundtrip_and_checksums(tmp_path):
    run = _write(tmp_path, _result(e_fill=10 / 31))
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["run_id"] == "20261029T00Z" and manifest["member_count"] == 31
    assert len(manifest["windows"]) == 28 and manifest["windows"][27]["end_hour"] == 168
    assert manifest["windows"][1]["start_utc"] == "2026-10-29T06:00:00Z"
    assert manifest["label"] == "scenario agreement" and "weather-scenario" in manifest["label_note"]
    pct = assemble.read_agreement_percent(run, "elevated")
    np.testing.assert_allclose(pct, 100 * 10 / 31, rtol=1e-5)
    crit = assemble.read_agreement_percent(run, "critical")
    assert (crit <= pct + 1e-4).all()
    assert assemble.verify_checksums(run)
    (run / "weather.npz").write_bytes(b"corrupt")
    assert not assemble.verify_checksums(run)


def test_nodata_roundtrips_as_nan(tmp_path):
    r = _result()
    r["agreement_elevated"][3, 0, 0] = np.nan
    r["agreement_critical"][3, 0, 0] = np.nan
    r["nodata"][3, 0, 0] = True
    run = _write(tmp_path, r)
    pct = assemble.read_agreement_percent(run)
    assert np.isnan(pct[3, 0, 0]) and pct[3, 1, 1] == 0.0
    assert json.loads((run / "manifest.json").read_text())["nodata_cells_any_window"] == 1


def test_invalid_result_writes_nothing(tmp_path):
    bad = _result(e_fill=0.2)
    bad["agreement_critical"] = bad["agreement_elevated"] + 0.1
    with pytest.raises(ValueError):
        _write(tmp_path, bad)
    assert list(tmp_path.iterdir()) == []
    short = _result()
    short["agreement_elevated"] = short["agreement_elevated"][:27]
    with pytest.raises(ValueError):
        _write(tmp_path, short)
    assert list(tmp_path.iterdir()) == []


def test_runs_are_immutable_and_no_temp_left(tmp_path):
    _write(tmp_path)
    with pytest.raises(FileExistsError):
        _write(tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == ["20261029T00Z"]


def test_stale_after_24_hours(tmp_path):
    manifest = json.loads((_write(tmp_path) / "manifest.json").read_text())
    assert not assemble.is_stale(manifest, CYCLE + timedelta(hours=24))
    assert assemble.is_stale(manifest, CYCLE + timedelta(hours=24, minutes=1))


def test_end_to_end_tiny_run(tmp_path):
    from core.fire_danger import RULE_SPEC
    categorize, dampen, info = rules.rule_functions(gdd_accum=None)
    wind_ms = (RULE_SPEC["thresholds"]["critical_wind"] + 6) / driver.MPS_TO_KNOTS
    hot = member(temp=30.0, rh=RULE_SPEC["thresholds"]["critical_rh"] - 3, wind=wind_ms)
    members = [hot] * 7 + [member()] * 24

    def init(ds):
        return driver.seed_state(driver._fields(ds, CYCLE))[0]
    result = driver.run_ensemble(iter(members), CYCLE, 31, GRID, init, categorize, dampen)
    run = _write(tmp_path, result, driver_params={"wind_factor": 1.0})
    np.testing.assert_allclose(assemble.read_agreement_percent(run, "critical"), 100 * 7 / 31, rtol=1e-5)
    assert assemble.verify_checksums(run)
