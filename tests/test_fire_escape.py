"""Fire escape shadow maps: the gridMET inputs, the day's computation on a synthetic grid, the record, the tracking ingest."""
import json
import sqlite3
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import fire_danger_tracking as tracking
from services.fire_escape import features as ft
from services.fire_escape import gridmet, hrrr_day, job, predict, render

DAY = date(2026, 3, 14)
NY, NX, HOURS = 12, 14, 12


def tiny_bundle(directory: Path, names) -> None:
    """A real (if meaningless) two-target bundle in the layout model-training writes: drier and windier scores higher."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({name: rng.normal(size=400) for name in names})
    frame["rh_min"], frame["wind_ms"] = rng.uniform(5, 95, 400), rng.uniform(0, 12, 400)
    label = ((frame["wind_ms"] / 12 - frame["rh_min"] / 95 + rng.normal(0, 0.2, 400)) > 0).astype(int)
    booster = xgb.train({"objective": "binary:logistic", "max_depth": 2, "seed": 1},
                        xgb.DMatrix(frame.to_numpy("float32"), label=label, feature_names=list(names)), 20)
    directory.mkdir(parents=True)
    targets = {}
    for target, acres in (("escape_10ac", 10.0), ("escape_100ac", 100.0)):
        booster.save_model(directory / f"{target}.json")
        targets[target] = {"model": f"{target}.json", "trees": 20, "threshold_acres": acres, "platt": [1.0, -1.0 if acres == 100 else 0.0]}
    (directory / "bundle.json").write_text(json.dumps({"version": "test", "features": list(names), "targets": targets}))


def identity_adapter() -> dict:
    """Each adapted feature read straight from one HRRR day value."""
    source = {"tmax_c": "tmax_c", "rh_min": "rh_min", "rh_max": "rh_max", "wind_ms": "wind_mean_ms", "vpd_kpa": "vpd_mean_kpa",
              "srad_wm2": "clear_sky_wm2", "precip_day_mm": "precip_mm", "stn_wind_peak_ms": "wind_peak_ms",
              "stn_gust_peak_ms": "gust_peak_ms", "stn_rh_min": "rh_min"}
    return {"predictors": list(hrrr_day.PREDICTORS),
            "features": {name: {"intercept": 0.0, "low": -1e9, "high": 1e9,
                                "coefficients": {p: float(p == source[name]) for p in hrrr_day.PREDICTORS}} for name in hrrr_day.ADAPTED}}


@pytest.fixture
def release(tmp_path):
    root = tmp_path / "release"
    tiny_bundle(root / "county", ft.WEATHER_ONLY_FEATURES)
    tiny_bundle(root / "fire", ft.FEATURES)
    (root / "hrrr_adapter.json").write_text(json.dumps(identity_adapter()))
    lat, lon = np.arange(35.9, 40.71, 0.5), np.arange(-95.9, -88.99, 0.5)
    shape = (lat.size, lon.size)
    np.savez(root / "fuel_grid.npz", lat=lat, lon=lon, frac_grass=np.full(shape, 0.6, "float32"), frac_timber=np.full(shape, 0.3, "float32"),
             frac_nonburnable=np.full(shape, 0.1, "float32"), canopy_cover_pct=np.full(shape, 25.0, "float32"))
    return root


def hrrr(rh=25.0, wind=8.0, rain=None):
    lat, lon = np.meshgrid(np.linspace(36.2, 40.4, NY), np.linspace(-95.4, -89.6, NX), indexing="ij")
    full = lambda value: np.full((HOURS, NY, NX), value, dtype="float32")
    east_dry = full(rh)
    east_dry[:, :, NX // 2:] = rh / 2                                 # the eastern half is drier
    return {"t2m": full(293.15), "r2": east_dry, "u10": full(wind), "v10": full(0.0), "gust": full(wind * 1.5),
            "tp": None if rain is None else full(rain), "lat": lat, "lon": lon,
            "valid_time": np.arange("2026-03-14T16", "2026-03-15T04", dtype="datetime64[h]").astype("datetime64[m]")}


def state(lag=1):
    lat, lon = np.arange(40.9, 35.69, -0.25), np.arange(-96.1, -88.79, 0.25)
    shape = (lat.size, lon.size)
    return {"rain": np.zeros((gridmet.RAIN_DAYS,) + shape, "float32"), "rain_days_known": 30, "lag_days": lag, "latest_day": "2026-03-13",
            "lat": lat, "lon": lon, "fm100_prev": np.full(shape, 11.0, "float32"), "fm1000_prev": np.full(shape, 15.0, "float32"),
            "erc_prev": np.full(shape, 40.0, "float32")}


# --- gridMET ---------------------------------------------------------------------------------------

def test_state_uses_the_latest_day_and_counts_missing_days_as_no_rain():
    days = np.arange("2026-02-10", "2026-03-13", dtype="datetime64[D]")            # through Mar 12: two days behind Mar 14
    rain = np.zeros((days.size, 2, 2), "float32")
    rain[-1] = 7.0
    fuel_days = days[-4:]
    fuel = {name: np.arange(4, dtype="float32")[:, None, None] * np.ones((4, 2, 2), "float32") for name in gridmet.STATE_VARS}
    out = gridmet.state_from_arrays(DAY, days, rain, {name: fuel_days for name in gridmet.STATE_VARS}, fuel, [40.0, 39.0], [-94.0, -93.0])
    assert out["lag_days"] == 2 and out["latest_day"] == "2026-03-12" and out["rain_days_known"] == 29
    assert out["rain"].shape == (30, 2, 2) and out["rain"][-2, 0, 0] == 7.0 and out["rain"][-1, 0, 0] == 0.0   # Mar 13 unknown -> 0
    assert out["fm100_prev"][0, 0] == 3.0


def test_state_refuses_stale_gridmet():
    days = np.arange("2026-02-01", "2026-03-05", dtype="datetime64[D]")            # ten days behind
    fuel = {name: np.zeros((days.size, 1, 1), "float32") for name in gridmet.STATE_VARS}
    assert gridmet.state_from_arrays(DAY, days, np.zeros((days.size, 1, 1)), {n: days for n in gridmet.STATE_VARS}, fuel, [40.0], [-94.0]) is None


def test_to_cells_takes_the_nearest_value_and_nan_outside():
    field = np.arange(6.0).reshape(2, 3)                                # lat 40, 39 (descending); lon -94, -93, -92
    out = gridmet.to_cells(field, [40.0, 39.0], [-94.0, -93.0, -92.0], np.array([39.1, 40.0, 30.0]), np.array([-92.2, -94.4, -93.0]))
    assert out[0] == 5.0 and out[1] == 0.0 and np.isnan(out[2])
    stacked = gridmet.to_cells(np.stack([field, field + 10]), [40.0, 39.0], [-94.0, -93.0, -92.0], np.array([[39.0]]), np.array([[-93.0]]))
    assert stacked.shape == (2, 1, 1) and stacked[1, 0, 0] == 14.0


# --- HRRR day values --------------------------------------------------------------------------------

def test_aggregate_and_adapter():
    data = hrrr(rh=30.0, wind=6.0, rain=4.0)
    values = hrrr_day.aggregate(data["t2m"], data["r2"], data["u10"], data["v10"], data["tp"], data["gust"])
    assert values["tmax_c"][0, 0] == pytest.approx(20.0, abs=1e-3) and values["rh_min"][0, 0] == 30.0 and values["rh_min"][0, -1] == 15.0
    assert values["wind_peak_ms"][0, 0] == 6.0 and values["gust_peak_ms"][0, 0] == 9.0 and values["precip_mm"][0, 0] == 4.0
    assert values["vpd_mean_kpa"][0, 0] == pytest.approx(hrrr_day.saturation_vapor_pressure_kpa(20.0) * 0.7, rel=1e-4)
    no_gust = hrrr_day.aggregate(data["t2m"], data["r2"], data["u10"], data["v10"])
    assert no_gust["gust_peak_ms"][0, 0] == pytest.approx(6.0 * hrrr_day.NEUTRAL_GUST_FACTOR) and np.isnan(no_gust["precip_mm"]).all()
    adapter = {"predictors": ["rh_min"], "features": {"rh_min": {"intercept": 5.0, "coefficients": {"rh_min": 2.0}, "low": 0.0, "high": 50.0}}}
    assert hrrr_day.adapt(adapter, {"rh_min": np.array([10.0, 40.0])})["rh_min"].tolist() == [25.0, 50.0]     # 85 is held at the fit's range


def test_county_mean_ignores_cells_outside_and_missing():
    values = np.array([[1.0, 3.0], [np.nan, 100.0]])
    index = np.array([[0, 0], [0, -1]])
    out = hrrr_day.county_mean(values, index, 2)
    assert out[0] == 2.0 and np.isnan(out[1])


# --- the day ----------------------------------------------------------------------------------------

def test_compute_gives_every_product_and_drier_is_never_safer(release):
    data = hrrr(rain=0.0)
    fips, cell_county = job.county_cells(data["lat"], data["lon"])
    assert len(fips) == 115 and (cell_county >= 0).any() and (cell_county == -1).any()
    out = job.compute(data, state(), job.load_release(release), DAY, fips, cell_county)
    assert set(job.GRID_FIELDS) == set(out["grid"])
    inside = cell_county >= 0
    for name in job.GRID_FIELDS:
        assert out["grid"][name].shape == (NY, NX)
        assert np.isfinite(out["grid"][name][inside]).all() and np.isnan(out["grid"][name][~inside]).all()
    assert ((out["grid"]["fire_p10"][inside] >= 0) & (out["grid"]["fire_p10"][inside] <= 1)).all()
    assert set(np.unique(out["grid"]["pfdi_level"][inside])) <= {0.0, 1.0, 2.0, 3.0, 4.0}
    west, east = inside & (np.arange(NX) < NX // 2)[None, :], inside & (np.arange(NX) >= NX // 2)[None, :]
    assert out["grid"]["pfdi_flame_ft"][east].mean() > out["grid"]["pfdi_flame_ft"][west].mean()      # the drier half burns hotter
    county = out["county"]
    known = np.isfinite(county["p_10ac"])
    assert known.sum() > 50 and ((county["p_10ac"][known] >= 0) & (county["p_10ac"][known] <= 1)).all()
    assert (county["p_100ac"][known] < county["p_10ac"][known]).all()                                 # the bundle's calibration offset
    assert list(out["features"].columns) == list(ft.FEATURES) and len(out["features"]) == 115


def test_missing_rain_field_is_unknown_not_dry(release):
    data = hrrr(rain=None)
    fips, cell_county = job.county_cells(data["lat"], data["lon"])
    out = job.compute(data, state(), job.load_release(release), DAY, fips, cell_county)
    assert out["features"]["precip_day_mm"].isna().all() and out["features"]["rh_min"].notna().any()
    assert np.isfinite(out["grid"]["pfdi_flame_ft"][cell_county >= 0]).all()      # the physical index does not need the rain
    assert np.isnan(out["county"]["p_10ac"]).all()               # the county model is not run on a row with a missing input


def test_run_writes_the_record_and_skips_cleanly(release, tmp_path, monkeypatch):
    import xarray as xr

    data = hrrr(rain=0.0)
    steps = (np.arange(4, 16) * 3600 * 1e9).astype("timedelta64[ns]")
    path = tmp_path / "hrrr_20260314_12z_f04-15.nc"
    xr.Dataset({name: (("step", "y", "x"), data[name]) for name in ("t2m", "r2", "u10", "v10", "gust")} | {"tp": (("step", "y", "x"), data["tp"])},
               coords={"step": steps, "latitude": (("y", "x"), data["lat"]), "longitude": (("y", "x"), data["lon"] + 360.0),
                       "valid_time": ("step", np.datetime64("2026-03-14T12") + steps)}).to_netcdf(path)
    out = job.run(DAY, path, release, tmp_path / "out", state=state(lag=2), render_maps=False)
    assert out["status"] == "ok" and out["gridmet_lag_days"] == 2
    record = json.loads(Path(out["record"]).read_text())
    assert record["valid_local_date"] == "2026-03-14" and record["not_for_operations"] is True and len(record["county_fips"]) == 115
    assert len(record["p_10ac"]) == 115 and set(record["features"]) == set(ft.WEATHER_ONLY_FEATURES)
    with np.load(tmp_path / "out" / "20260314_12z.fire_escape.npz") as grids:
        assert set(job.GRID_FIELDS) <= set(grids.files) and grids["lon"].max() < 0          # longitudes brought to -180..180

    assert job.run(DAY, path, tmp_path / "nowhere", tmp_path / "out")["reason"].startswith("no bundle")
    assert "no cached HRRR file" in job.run(DAY, tmp_path / "missing.nc", release, tmp_path / "out")["reason"]
    monkeypatch.setattr(gridmet, "latest_state", lambda day, cache: None)
    assert "gridMET" in job.run(DAY, path, release, tmp_path / "out")["reason"]


def test_find_cache_file_takes_the_cycle_or_one_hour_late(tmp_path):
    for name in ("hrrr_20260314_09z_f04-15.nc", "hrrr_20260314_13z_f04-15.nc", "hrrr_20260313_12z_f04-15.nc"):
        (tmp_path / name).write_bytes(b"")
    assert job.find_cache_file(DAY, tmp_path).name == "hrrr_20260314_13z_f04-15.nc"
    assert job.find_cache_file(date(2026, 3, 15), tmp_path) is None


def test_odds_wording():
    assert render.odds(0.025) == "1 in 40" and render.odds(0.5) == "1 in 2" and render.odds(0.00001) == "under 1 in 10,000"
    assert render.odds(float("nan")) == "n/a"
    assert [render.percent(p) for p in (1.0, 0.146, 0.025, 0.0028, 0.00004)] == ["100%", "15%", "2.5%", "0.28%", "0.004%"]
    assert all(len(levels) == len(render.PROBABILITY_COLORS) + 1 for levels in render.SCALES.values())


# --- tracking ---------------------------------------------------------------------------------------

def test_tracking_ingests_the_county_probability_as_a_score(tmp_path):
    evidence = tmp_path / "fire-escape"
    evidence.mkdir()
    (evidence / "20260314_12z.fire_escape.json").write_text(json.dumps({
        "run_id": "20260314_12z", "valid_local_date": "2026-03-14", "recorded_at": "2026-03-14T13:00:00Z",
        "county_fips": ["29019", "29510"], "p_10ac": [0.031, None]}))
    db = tmp_path / "t.db"
    with sqlite3.connect(db) as conn:
        conn.execute('''CREATE TABLE fire_danger_model_scores (model TEXT, run_id TEXT, valid_date TEXT, county_fips TEXT, score REAL,
                        category INTEGER, probability REAL, produced_at TEXT, source TEXT,
                        UNIQUE (model, run_id, valid_date, county_fips))''')
    empty = tmp_path / "empty"
    empty.mkdir()
    dirs = {"fire_weather_index": empty, "risk_fusion_glm": empty, "ensemble": empty, "fire_escape": evidence}
    assert tracking.ingest_scores(db, dirs)["added"] == {"fire_escape_county": 2}
    assert tracking.ingest_scores(db, dirs)["added"] == {}                              # idempotent
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT county_fips, score, category, probability FROM fire_danger_model_scores ORDER BY county_fips").fetchall()
    assert rows == [("29019", 0.031, None, None), ("29510", None, None, None)]
    assert "fire_escape_county" not in tracking.PROBABILITY_OF_FIRE and "fire_escape_county" in tracking.LABELS


# --- Testbed ----------------------------------------------------------------------------------------

@pytest.fixture
def testbed(tmp_path, monkeypatch):
    from services import beta_products
    root = tmp_path / "testbed"
    monkeypatch.setattr(beta_products, "BETA_ROOT", root)
    monkeypatch.setattr(beta_products, "BETA_MANIFEST_PATH", root / "manifest.json")
    return root


def test_publish_copies_the_maps_and_lists_them_in_the_testbed_manifest(release, testbed, tmp_path):
    data = hrrr(rain=0.0)
    fips, cell_county = job.county_cells(data["lat"], data["lon"])
    result = job.compute(data, state(), job.load_release(release), DAY, fips, cell_county)
    images = {}
    for kind in (*render.MAPS, "overview"):
        images[kind] = tmp_path / f"{kind}.png"
        images[kind].write_bytes(kind.encode())
    record = {"valid_local_date": "2026-03-14", "recorded_at": "2026-03-14T13:00:00Z", "run_id": "20260314_12z",
              "gridmet_lag_days": 1, "bundle": {"county": "test"}}
    (testbed).mkdir()
    (testbed / "manifest.json").write_text(json.dumps({"products": {"spread_rate": {"status": "ready"}}}))
    job.publish_to_testbed(images, result, record)
    manifest = json.loads((testbed / "manifest.json").read_text())
    product = manifest["products"]["fire_escape"]
    assert manifest["products"]["spread_rate"] == {"status": "ready"}                       # other products are left alone
    assert product["not_for_operations"] is True and product["valid_local_date"] == "2026-03-14"
    assert set(product["previews"]) == {*render.MAPS, "overview"} and set(product["summary"]) == set(render.MAPS)
    assert (testbed / product["previews"]["overview"]).read_bytes() == b"overview"
    assert "1 in" in product["summary"]["county_10ac"] and "ft" in product["summary"]["pfdi"]


# --- model overview ---------------------------------------------------------------------------------

def scores_db(path: Path, rows) -> Path:
    with sqlite3.connect(path) as conn:
        conn.execute('''CREATE TABLE fire_danger_model_scores (model TEXT, run_id TEXT, valid_date TEXT, county_fips TEXT, score REAL,
                        category INTEGER, probability REAL, produced_at TEXT, source TEXT,
                        UNIQUE (model, run_id, valid_date, county_fips))''')
        conn.executemany("INSERT INTO fire_danger_model_scores VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    return path


def test_overview_collects_each_models_latest_run_for_the_day(tmp_path):
    from services import model_overview
    rows = [
        ("public_rule", "early", "2026-03-14", "29019", None, 1, None, "2026-03-14 11:00:00", "x"),
        ("public_rule", "late", "2026-03-14", "29019", None, 3, None, "2026-03-14 13:00:00", "x"),
        ("public_rule", "late", "2026-03-14", "29510", None, 2, None, "2026-03-14 13:00:00", "x"),
        ("public_rule", "other-day", "2026-03-15", "29019", None, 4, None, "2026-03-15 13:00:00", "x"),
        ("risk_fusion_glm", "g", "2026-03-14", "29019", 0.4, None, 0.33, "2026-03-14T12:00:00Z", "x"),
    ]
    models = model_overview.collect(DAY, scores_db(tmp_path / "t.db", rows), ingest=False)
    assert set(models) == {"public_rule", "risk_fusion_glm"}
    assert models["public_rule"]["run_id"] == "late" and models["public_rule"]["category"] == {"29019": 3.0, "29510": 2.0}
    assert models["risk_fusion_glm"]["probability"] == {"29019": 0.33} and models["risk_fusion_glm"]["category"] == {}
    assert model_overview.collect(date(2026, 1, 1), tmp_path / "t.db", ingest=False) == {}


def test_overview_panel_summaries():
    from services import model_overview
    assert model_overview.panel_summary({"a": 0, "b": 2, "c": 2}, "level") == "Low 1 | Elevated 2"
    assert model_overview.panel_summary({"a": 0.5, "b": 0.1, "c": 0.02}, "any_fire") == "Highest 50% (1 in 2) | median 10%"
    assert model_overview.panel_summary({}, "level") == "No data for this day"
    assert all(len(levels) == len(render.PROBABILITY_COLORS) + 1 for levels in model_overview.SCALES.values())


def test_overview_draws_publishes_and_skips_an_empty_day(tmp_path, testbed):
    from services import model_overview
    fips = [f"29{n:03d}" for n in range(1, 230, 2)]
    rows = [("public_rule", "r", "2026-03-14", f, None, i % 5, None, "2026-03-14 13:00:00", "x") for i, f in enumerate(fips)]
    rows += [("ensemble:members", "e", "2026-03-14", f, 0.01 * (i % 90), i % 4, 0.01 * (i % 90), "2026-03-14T15:00:00Z", "x") for i, f in enumerate(fips)]
    db = scores_db(tmp_path / "t.db", rows)
    out = model_overview.run(DAY, tmp_path / "out", db, publish_to_testbed=True)
    assert out["status"] == "ok" and out["models"] == ["ensemble:members", "public_rule"]
    assert "risk_fusion_glm" in out["missing"] and Path(out["image"]).stat().st_size > 50_000
    product = json.loads((testbed / "manifest.json").read_text())["products"]["model_overview"]
    assert product["not_for_operations"] is True and (testbed / product["preview"]).is_file()
    assert set(product["models"]) == {"ensemble:members", "public_rule"} and "fire_weather_index" in product["missing"]
    assert model_overview.run(date(2026, 1, 1), tmp_path / "out", db)["reason"].startswith("no model has county scores")


# --- installing a release ---------------------------------------------------------------------------

def pack(release_dir: Path, archive: Path, tamper=None, extra=None, tag="fire_escape-vtest-00000000") -> Path:
    """An archive in the layout model-training's fire_escape/publish.py writes."""
    import hashlib
    import zipfile

    from services.fire_escape import install
    files = {p.relative_to(release_dir).as_posix(): p.read_bytes() for p in sorted(release_dir.rglob("*")) if p.is_file()}
    manifest = {"schema": install.SCHEMA, "tag": tag, "models": {"county": {"version": "test"}},
                "files": {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)} for name, data in files.items()}}
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, data in files.items():
            bundle.writestr(name, b"tampered" if name == tamper else data)
        for name, data in (extra or {}).items():
            bundle.writestr(name, data)
        bundle.writestr(install.MANIFEST_NAME, json.dumps(manifest))
    return archive


def test_install_verifies_swaps_in_and_rolls_back(release, tmp_path):
    from services.fire_escape import install
    target = tmp_path / "server" / "models" / "fire_escape_release"
    assert install.installed(target) is None
    first = install.install_archive(pack(release, tmp_path / "a.zip", tag="fire_escape-vtest-aaaaaaaa"), target)
    assert first["tag"] == "fire_escape-vtest-aaaaaaaa" and install.installed(target)["tag"] == first["tag"]
    assert job.load_release(target) is not None and not target.with_name(target.name + ".incoming").exists()

    install.install_archive(pack(release, tmp_path / "b.zip", tag="fire_escape-vtest-bbbbbbbb"), target)
    assert install.installed(target)["tag"] == "fire_escape-vtest-bbbbbbbb"
    assert install.installed(target.with_name(target.name + ".previous"))["tag"] == "fire_escape-vtest-aaaaaaaa"
    assert install.rollback(target)["tag"] == "fire_escape-vtest-aaaaaaaa"
    assert install.rollback(target)["tag"] == "fire_escape-vtest-bbbbbbbb"            # twice undoes it


def test_a_bad_release_is_refused_and_the_running_one_is_untouched(release, tmp_path):
    from services.fire_escape import install
    target = tmp_path / "server" / "fire_escape_release"
    install.install_archive(pack(release, tmp_path / "good.zip", tag="fire_escape-vtest-good0000"), target)
    bad = {
        "checksum mismatch": pack(release, tmp_path / "tampered.zip", tamper="county/escape_10ac.json"),
        "disagree": pack(release, tmp_path / "extra.zip", extra={"surprise.py": b"print(1)"}),
        "unsafe path": pack(release, tmp_path / "escape.zip", extra={"../outside.txt": b"x"}),
    }
    for reason, archive in bad.items():
        with pytest.raises(install.InstallError, match=reason):
            install.install_archive(archive, target)
    (tmp_path / "junk.zip").write_bytes(b"not a zip")
    with pytest.raises(install.InstallError, match="not a readable archive"):
        install.install_archive(tmp_path / "junk.zip", target)
    assert install.installed(target)["tag"] == "fire_escape-vtest-good0000"           # still the good one
    assert not (tmp_path / "outside.txt").exists() and not target.with_name(target.name + ".incoming").exists()
    with pytest.raises(install.InstallError, match="no previous release"):
        install.rollback(target)


def test_a_release_whose_models_need_unknown_features_is_refused(release, tmp_path):
    from services.fire_escape import install
    bundle = json.loads((release / "county" / "bundle.json").read_text())
    bundle["features"] = [*bundle["features"], "a_feature_from_the_future"]
    (release / "county" / "bundle.json").write_text(json.dumps(bundle))
    with pytest.raises(install.InstallError):
        install.install_archive(pack(release, tmp_path / "future.zip"), tmp_path / "target")
    assert not (tmp_path / "target").exists()


def test_install_without_the_model_libraries_says_so_or_installs_on_checksums(release, tmp_path, monkeypatch):
    import builtins

    from services.fire_escape import install
    real_import = builtins.__import__

    def no_pandas(name, *args, **kwargs):
        if name == "pandas":
            raise ModuleNotFoundError("No module named 'pandas'")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_pandas)
    target = tmp_path / "target"
    with pytest.raises(install.InstallError, match="cannot load the models"):
        install.install_archive(pack(release, tmp_path / "a.zip"), target)
    assert not target.exists()
    manifest = install.install_archive(pack(release, tmp_path / "a.zip"), target, load_test=False)
    assert manifest["load_tested"] is False and (target / "county" / "bundle.json").is_file()
    with pytest.raises(install.InstallError, match="checksum mismatch"):          # checksums still apply
        install.install_archive(pack(release, tmp_path / "bad.zip", tamper="fuel_grid.npz"), target, load_test=False)
