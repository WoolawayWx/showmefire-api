from datetime import datetime, timezone, timedelta
import json
import zlib
from pathlib import Path
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from services import fire_weather_hybrid_store as store
from services.fire_weather_hybrid.contracts import Grid
from core.security import require_admin
from routers import fire_weather_hybrid_admin as admin


def write(root, *, complete=True):
    grid = Grid(id="small-test",width=3,height=2,west=500000,north=4300000)
    issued = datetime(2026,11,6,12,tzinfo=timezone.utc)
    times = [issued-timedelta(hours=1),issued]
    hour = {"danger_category":np.array([[0,1,2],[3,4,np.nan]],"float32"),"fm1000_pct":np.full((2,3),20.,"float32")}
    return store.write_run(root,issued,times,iter([hour,hour] if complete else [hour]),{"model_version":"test"},grid=grid)


def test_streamed_formats_and_point_values_agree(tmp_path):
    import netCDF4
    import rasterio
    manifest = write(tmp_path)
    root = store.resolve_run(tmp_path,manifest["cycle"])
    meta = json.loads((root/"forecast.zarr/.zmetadata").read_text())
    assert meta["metadata"]["danger_category/.zarray"]["shape"] == [2,2,3]
    raw = np.frombuffer(zlib.decompress((root/"forecast.zarr/danger_category/0.0.0").read_bytes()),"<f4").reshape(2,3)
    with netCDF4.Dataset(root/"forecast.nc") as nc, rasterio.open(root/"rasters/danger_category.tif") as raster:
        np.testing.assert_allclose(raw,np.ma.filled(nc["danger_category"][0],np.nan),equal_nan=True)
        np.testing.assert_allclose(raw,np.ma.filled(raster.read(1,masked=True),np.nan),equal_nan=True)
        lat,lon = float(nc["latitude"][0,1]),float(nc["longitude"][0,1])
    point = store.point(root,lat,lon)
    assert point["series"]["danger_category"] == [1,1]
    assert point["step_kinds"] == ["analysis","forecast"]
    assert manifest["public"] is False
    with pytest.raises(ValueError,match="immutable"):
        write(tmp_path)


def test_incomplete_run_never_advances_latest(tmp_path):
    with pytest.raises(ValueError,match="incomplete"):
        write(tmp_path,complete=False)
    assert not (tmp_path/"latest.json").exists()


def test_every_asset_route_requires_staff_auth(tmp_path,monkeypatch):
    monkeypatch.setenv("SMF_FIRE_WEATHER_HYBRID_DIR",str(tmp_path))
    manifest = write(tmp_path)
    admin.limiter.reset()
    app = FastAPI()
    app.include_router(admin.router)
    client = TestClient(app)
    prefix = "/api/admin/fire-weather-hybrid"
    for route in ["/status","/runs",f"/runs/{manifest['cycle']}",f"/runs/{manifest['cycle']}/zarr/.zmetadata",
                  f"/runs/{manifest['cycle']}/files/forecast.nc"]:
        assert client.get(prefix+route).status_code == 401
    app.dependency_overrides[require_admin] = lambda: "test@example.com"
    response = client.get(prefix+"/runs/latest")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-store"
    response = client.get(prefix+f"/runs/{manifest['cycle']}/zarr/.zmetadata")
    assert response.status_code == 200 and response.headers["cache-control"] == "private, no-store"
    assert client.get(prefix+f"/runs/{manifest['cycle']}/files/../../worker.log").status_code == 404
    assert client.post(prefix+"/run").status_code == 409


def test_runtime_mirror_is_byte_identical():
    import hashlib
    directory = Path(__file__).resolve().parents[1]/"services/fire_weather_hybrid"
    for filename,expected in json.loads((directory/"runtime_manifest.json").read_text()).items():
        assert hashlib.sha256((directory/filename).read_bytes()).hexdigest() == expected


def test_native_adapter_requires_explicit_wind_reference_and_analysis_is_causal():
    from services.fire_weather_hybrid_inputs import normalized_hour, analysis_hour
    with pytest.raises(ValueError,match="declare"):
        normalized_hour(None,datetime.now(timezone.utc),wind_reference="unknown")
    now = datetime.now(timezone.utc)
    with pytest.raises(ValueError,match="precede"):
        analysis_hour({},None,valid_time=now,issued_at=now)


def test_bundle_rejects_changed_assets_before_loading_models(tmp_path):
    from services.fire_weather_hybrid.trees import Bundle
    from services.fire_weather_hybrid.contracts import FEATURES,GRID
    (tmp_path/"risk_scale.json").write_text('{}')
    (tmp_path/"bundle.json").write_text(json.dumps({"schema":"fire-weather-hybrid-v1","advisory_only":True,
        "features":list(FEATURES),"grid_fingerprint":GRID.fingerprint,"tasks":{"ros_m_min":{},"intensity_kw_m":{}},
        "checksums":{"risk_scale.json":"wrong"}}))
    with pytest.raises(ValueError,match="checksum"):
        Bundle(tmp_path)


def test_verification_excludes_elapsed_analysis_and_marks_missing_classes(tmp_path):
    import netCDF4
    import pandas as pd
    from scripts.verify_fire_weather_hybrid import verify
    manifest = write(tmp_path)
    directory = store.resolve_run(tmp_path,manifest["cycle"])
    with netCDF4.Dataset(directory/"forecast.nc") as ds:
        lat,lon = float(ds["latitude"][0,1]),float(ds["longitude"][0,1])
    path = tmp_path/"observations.csv"
    pd.DataFrame({"station_id":["TEST","TEST"],"valid_time":manifest["valid_times_utc"],"lat":[lat,lat],"lon":[lon,lon],
                  "qc_pass":[True,True],"fm1000_pct":[0,20]}).to_csv(path,index=False)
    report = verify(tmp_path,manifest["cycle"],path,now=datetime(2026,11,7,tzinfo=timezone.utc))
    assert report["metrics"]["fm1000_pct"]["samples"] == 1
    assert report["metrics"]["fm1000_pct"]["mae"] == 0
    assert report["metrics"]["fm10_pct"]["status"] == "deferred"


def test_weathernext_reads_google_credentials_path_from_dotenv_without_reading_key(tmp_path, monkeypatch):
    import os
    from services.fire_weather_hybrid.weathernext import Config
    key = tmp_path/"key.json"
    key.write_text("not read", encoding="utf-8")
    env = tmp_path/".env"
    env.write_text("OTHER=1\nGOOGLE_APPLICATION_CREDENTIALS=" + key.as_posix() + "\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    Config().apply_credentials()
    assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == str(key.resolve())
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "already-set")  # the process environment wins
    Config().apply_credentials()
    assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == "already-set"
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS")
    env.write_text("GOOGLE_APPLICATION_CREDENTIALS=missing.json\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        Config().apply_credentials()


def test_already_corrected_flag_ignores_nan_cells_outside_the_domain():
    import numpy as np
    from services.fire_weather_hybrid.runtime import _applied
    flag = np.array([[1., 1., np.nan], [1., np.nan, np.nan]], "float32")
    assert _applied({"weather_bias_applied": flag}, "weather_bias_applied")  # NaN outside domain must not defeat the flag
    assert not _applied({"weather_bias_applied": np.full((2, 2), np.nan, "float32")}, "weather_bias_applied")
    assert not _applied({"weather_bias_applied": np.array([1., 0.], "float32")}, "weather_bias_applied")
    assert not _applied({}, "weather_bias_applied")


def test_physics_predictor_uses_absolute_flame_length_levels():
    import numpy as np
    from datetime import datetime, timezone
    from services.fire_weather_hybrid.runtime import PhysicsPredictor
    shape = (1, 3)
    fields = {"temp_c": np.full(shape, 20.), "rh_pct": np.full(shape, 25.), "wind_speed_ms": np.array([[1., 9., 9.]]),
              "wind_u_ms": np.zeros(shape), "wind_v_ms": np.array([[-1., -9., -9.]]),
              "fm1_pct": np.full(shape, 6.), "fm10_pct": np.full(shape, 7.), "fm100_pct": np.full(shape, 9.),
              "burnable_fuel_code": np.array([[102., 102., np.nan]]), "fuel_model_code": np.array([[102., 102., 93.]]),
              "burnable_fraction": np.array([[1., 1., 0.]]), "missouri_mask": np.ones(shape),
              "slope_deg": np.zeros(shape), "canopy_cover_pct": np.zeros(shape), "canopy_height_m": np.zeros(shape)}
    when = datetime(2026, 11, 6, 20, tzinfo=timezone.utc)
    out = PhysicsPredictor().predict_hour(fields, when, when)
    assert out["ros_m_min"][0, 1] > out["ros_m_min"][0, 0] > 0           # wind drives cured grass harder
    assert out["danger_category"][0, 1] > out["danger_category"][0, 0]
    assert np.isnan(out["danger_category"][0, 2])                         # nothing burnable
    expected = np.searchsorted([0.3048, 1.2192, 2.4384, 3.3528], out["flame_length_m"][0, :2], side="right")
    assert np.array_equal(out["danger_category"][0, :2], expected)
