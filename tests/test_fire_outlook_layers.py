"""Tests for the supporting layers (layers.py), scenario means (driver), manifest layer metadata, older-run compatibility,
and the contour file route."""
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers import fire_outlook
from services.fire_outlook import assemble, contours, driver, layers, point
from tests.test_fire_outlook_contours_point import GRID, N, _counts
from tests.test_fire_outlook_driver import GRID as DGRID, member
from tests.test_fire_outlook_assemble import _result

CYCLE = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(hours=2)
SHAPE = (28, GRID["height"], GRID["width"])


def full_result(n=N, seed=0):
    r = _result(n)
    rng = np.random.default_rng(seed)
    counts = np.stack([_counts(seed=w) for w in range(28)])
    r["agreement_elevated"] = (counts / n).astype("float32")
    r["agreement_critical"] = (np.minimum(counts, 5) / n).astype("float32")
    for k in ("fm10_min_p10", "fm10_min_p90", "mean_hours_elevated", "mean_hours_critical", "temp_max_c_envelope",
              "rh_min_pct_envelope", "wind_max_kts_envelope"):
        r[k] = np.full(SHAPE, 7.0, "float32")                                 # the base helper uses a smaller grid
    r["nodata"] = np.zeros(SHAPE, bool)
    r["fm10_min_p50"] = rng.uniform(3, 28, SHAPE).astype("float32")
    r["temp_max_c_mean"] = rng.uniform(-5, 38, SHAPE).astype("float32")
    r["rh_min_pct_mean"] = rng.uniform(8, 95, SHAPE).astype("float32")
    r["wind_max_kts_mean"] = rng.uniform(0, 30, SHAPE).astype("float32")
    r["mean_precip_6h_mm"] = rng.uniform(0, 30, SHAPE).astype("float32")
    return r


# --- driver: scenario means ---------------------------------------------------------------------------------

def test_driver_returns_scenario_means_between_the_extremes():
    members = [member(temp=10.0 + 4 * i) for i in range(3)]               # 10, 14, 18 C
    init = lambda ds: driver.seed_state(driver._fields(ds, CYCLE.replace(year=2026, month=10, day=29, hour=0)))[0]  # noqa: E731
    out = driver.run_ensemble(iter(members), CYCLE.replace(year=2026, month=10, day=29, hour=0), 3, DGRID, init,
                              lambda fm, rh, w: np.zeros(np.broadcast(fm, rh, w).shape, "int8"))
    np.testing.assert_allclose(out["temp_max_c_mean"], 14.0, atol=1e-4)
    np.testing.assert_allclose(out["temp_max_c_envelope"], 18.0, atol=1e-4)
    assert (out["rh_min_pct_mean"] >= 0).all() and out["wind_max_kts_mean"].shape == (28,) + DGRID


# --- layers --------------------------------------------------------------------------------------------------

def test_display_values_units_and_wind_factor():
    r = {"fm10_min_p50": np.full((1, 2, 2), 9.0), "temp_max_c_mean": np.full((1, 2, 2), 20.0),
         "rh_min_pct_mean": np.full((1, 2, 2), 35.0), "wind_max_kts_mean": np.full((1, 2, 2), 10.0),
         "mean_precip_6h_mm": np.full((1, 2, 2), 25.4)}
    v = layers.display_values(r)
    assert v["temp"][0, 0, 0] == pytest.approx(68.0) and v["precip"][0, 0, 0] == pytest.approx(1.0)
    assert v["wind"][0, 0, 0] == pytest.approx(11.5078, rel=1e-4) and v["fm10"][0, 0, 0] == 9.0
    assert layers.display_values(r, wind_factor=0.8)["wind"][0, 0, 0] == pytest.approx(11.5078 / 0.8, rel=1e-4)   # undo the rule's factor
    with pytest.raises(KeyError, match="temp_max_c_mean"):
        layers.display_values({k: v for k, v in r.items() if k != "temp_max_c_mean"})


def test_value_bands_edges_open_ends_and_nan():
    edges = [10, 20, 30]
    vals = np.array([[9.99, 10.0, 19.9, 20.0, 30.0, 99.0, np.nan]])
    assert layers.value_band_raster(vals, edges)[0].tolist() == [1, 2, 2, 3, 4, 4, 0]    # edge value belongs to the upper band
    labels = layers.band_labels(edges)
    assert labels[1] == {"lower": None, "upper": 10} and labels[4] == {"lower": 30, "upper": None} and len(labels) == 4


def test_layer_files_cover_every_window_and_match_the_field():
    from rasterio import features
    from rasterio.transform import from_origin
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import transform as shp_transform
    r = full_result()
    r["fm10_min_p50"][3, 2, 2] = np.nan                                       # NoData must never be drawn
    files = layers.layer_files(r, GRID, simplify_cells=0, precision=7)
    assert len(files) == len(layers.LAYER_IDS) * 28 and "contours/fm10/w27.geojson" in files and "contours/precip/w00.geojson" in files
    fc = json.loads(files["contours/fm10/w03.geojson"])
    assert fc["properties"]["layer"] == "fm10" and fc["properties"]["edges"] == [4, 6, 8, 10, 12, 15, 20, 25]
    # rasterize the polygons back: they reproduce the band raster exactly, and the NaN cell is empty
    to_utm = Transformer.from_crs(4326, GRID["crs"], always_xy=True).transform
    tf = from_origin(GRID["west"], GRID["north"], GRID["resolution_m"], GRID["resolution_m"])
    back = np.zeros(SHAPE[1:], "uint8")
    for f in fc["features"]:
        g = shp_transform(to_utm, shape(f["geometry"]))
        m = features.rasterize([(g, 1)], out_shape=SHAPE[1:], transform=tf, fill=0, dtype="uint8")
        back[m == 1] = f["properties"]["band"]
    want = layers.value_band_raster(layers.display_values(r)["fm10"][3], [4, 6, 8, 10, 12, 15, 20, 25])
    np.testing.assert_array_equal(back, want)
    assert back[2, 2] == 0
    assert {"lower", "upper", "band"} <= set(fc["features"][0]["properties"])


def test_manifest_layers_describe_every_file_set():
    meta = layers.manifest_layers()
    assert [m["id"] for m in meta] == ["elevated", "critical", *layers.LAYER_IDS]
    assert meta[0]["kind"] == "agreement" and meta[0]["edges"] == [10, 30, 50, 70, 90] and meta[2]["kind"] == "value"
    assert all({"id", "label", "group", "unit", "edges", "kind"} <= set(m) for m in meta)


def test_value_layers_are_filled_edge_to_edge_where_there_is_data():
    r = full_result()
    bands = layers.value_band_raster(layers.display_values(r)["temp"][0], [32, 40, 50, 60, 70, 80, 90, 100])
    assert (bands > 0).all()                                                     # unlike agreement, every valid cell is drawn


# --- run writer / point / router ---------------------------------------------------------------------------

def _write(tmp_path, result=None, with_layers=True):
    result = result or full_result()
    extra = {}
    if with_layers:
        extra = contours.contour_files(assemble._counts(result["agreement_elevated"], N),
                                       assemble._counts(result["agreement_critical"], N), N, GRID)
        extra.update(layers.layer_files(result, GRID))
    return assemble.write_run(tmp_path / "runs", CYCLE, result, grid=GRID, source_cycles={}, rule_info={},
                              conditioning_quality={"level": "conditioned"}, extra_files=extra,
                              layers=layers.manifest_layers() if with_layers else None)


def test_manifest_lists_layers_and_weather_npz_has_the_means(tmp_path):
    run = _write(tmp_path)
    manifest = json.loads((run / "manifest.json").read_text())
    assert [m["id"] for m in manifest["layers"]][:3] == ["elevated", "critical", "fm10"]
    assert "contours/wind/w05.geojson" in manifest["files"] and assemble.verify_checksums(run)
    with np.load(run / "weather.npz") as z:
        assert {"temp_max_c_mean", "rh_min_pct_mean", "wind_max_kts_mean"} <= set(z.files)


def test_point_has_scenario_means_and_older_runs_still_work(tmp_path):
    run = _write(tmp_path)
    out = point.point_series(run, *_latlon(5, 6))
    w = out["windows"][4]
    assert w["temp_max_c_mean"] is not None and w["rh_min_pct_mean"] is not None and w["wind_max_kts_mean"] is not None
    old = full_result()
    for k in ("temp_max_c_mean", "rh_min_pct_mean", "wind_max_kts_mean"):
        del old[k]                                                              # a run written before the means existed
    point._arrays.cache_clear()
    old_run = assemble.write_run(tmp_path / "old", CYCLE - timedelta(hours=12), old, grid=GRID, source_cycles={}, rule_info={},
                                 conditioning_quality={})
    got = point.point_series(old_run, *_latlon(5, 6))["windows"][4]
    assert got["temp_max_c_mean"] is None and got["temp_max_c"] is not None


def _latlon(r, c):
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(GRID["crs"], 4326, always_xy=True).transform(
        GRID["west"] + (c + 0.5) * 3000, GRID["north"] - (r + 0.5) * 3000)
    return lat, lon


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("SMF_FIRE_OUTLOOK_DIR", str(tmp_path))
    monkeypatch.setenv("SMF_FIRE_OUTLOOK_PUBLIC", "true")
    for limiter in (fire_outlook.runs_limiter, fire_outlook.manifest_limiter, fire_outlook.point_limiter, fire_outlook.file_limiter):
        limiter.reset()
    app = FastAPI()
    app.include_router(fire_outlook.router)
    return TestClient(app)


def test_file_route_serves_only_contour_files_with_the_right_cache(tmp_path, monkeypatch):
    run = _write(tmp_path)
    (tmp_path / "latest.json").write_text(json.dumps({"run_id": run.name}))
    c = _client(tmp_path, monkeypatch)
    ok = c.get(f"/api/fire-outlook/runs/{run.name}/files/contours/fm10/w03.geojson")
    assert ok.status_code == 200 and ok.headers["content-type"].startswith("application/geo+json")
    assert "immutable" in ok.headers["cache-control"] and ok.json()["type"] == "FeatureCollection"
    latest = c.get("/api/fire-outlook/runs/latest/files/contours/elevated/w00.geojson")
    assert latest.status_code == 200 and latest.headers["cache-control"] == "public, max-age=60"
    for bad in ("manifest.json", "weather.npz", "contours/evil/w00.geojson", "contours/fm10/w28.geojson",
                "contours/fm10/../../manifest.json", "..%2Fmanifest.json", "contours/fm10/w03.geojson.bak"):
        assert c.get(f"/api/fire-outlook/runs/{run.name}/files/{bad}").status_code in (404, 400), bad
    assert c.get("/api/fire-outlook/runs/20200101T00Z/files/contours/fm10/w03.geojson").status_code == 404
    # a run without the supporting layers (older run): 404, not a crash
    old = _write(tmp_path / "x", with_layers=False)
    (tmp_path / "x" / "latest.json").write_text(json.dumps({"run_id": old.name}))
    monkeypatch.setenv("SMF_FIRE_OUTLOOK_DIR", str(tmp_path / "x"))
    assert c.get(f"/api/fire-outlook/runs/{old.name}/files/contours/fm10/w03.geojson").status_code == 404
