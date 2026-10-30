"""Tests for routers/fire_outlook.py against runs written by the real assemble/contours code."""
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pyproj import Transformer

from routers import fire_outlook
from services.fire_outlook import assemble, contours
from tests.test_fire_outlook_contours_point import GRID, N, _counts
from tests.test_fire_outlook_assemble import _result

def _cycle(hours_ago):
    """A cycle time relative to the real clock, so stale/fresh assertions never depend on today's date."""
    return datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(hours=hours_ago)


CYCLE = _cycle(3)                    # fresh (3 h old)
OLD = _cycle(72)                     # stale
RID, OLD_RID = assemble.run_id_for(CYCLE), assemble.run_id_for(OLD)


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("SMF_FIRE_OUTLOOK_DIR", str(tmp_path))
    monkeypatch.setenv("SMF_FIRE_OUTLOOK_PUBLIC", "true")      # the public routes are gated off by default (see test_fire_outlook_admin)
    for limiter in (fire_outlook.status_limiter, fire_outlook.runs_limiter, fire_outlook.manifest_limiter,
                    fire_outlook.point_limiter):
        limiter.reset()
    app = FastAPI()
    app.include_router(fire_outlook.router)
    return TestClient(app)


def _write(tmp_path, cycle=CYCLE, latest=True):
    shape = (28, GRID["height"], GRID["width"])
    r = _result()
    counts = np.stack([_counts(seed=w) for w in range(28)])
    r["agreement_elevated"] = (counts / N).astype("float32")
    r["agreement_critical"] = (np.minimum(counts, 5) / N).astype("float32")
    for k in ("fm10_min_p10", "fm10_min_p50", "fm10_min_p90", "mean_hours_elevated", "mean_hours_critical",
              "mean_precip_6h_mm", "temp_max_c_envelope", "rh_min_pct_envelope", "wind_max_kts_envelope"):
        r[k] = np.full(shape, 7.0, "float32")
    r["nodata"] = np.zeros(shape, bool)
    run = assemble.write_run(tmp_path / "runs", cycle, r, grid=GRID, source_cycles={"gefs": "x"}, rule_info={},
                             conditioning_quality={"level": "conditioned"},
                             extra_files=contours.contour_files(assemble._counts(r["agreement_elevated"], N),
                                                                assemble._counts(r["agreement_critical"], N), N, GRID))
    if latest:
        (tmp_path / "latest.json").write_text(json.dumps({"run_id": run.name}))
    return run, counts


def _center(r, c):
    lon, lat = Transformer.from_crs(GRID["crs"], 4326, always_xy=True).transform(
        GRID["west"] + (c + 0.5) * 3000, GRID["north"] - (r + 0.5) * 3000)
    return {"lat": lat, "lon": lon}


def test_empty_state(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    assert c.get("/api/fire-outlook/status").json() == {"state": "never_run"}
    assert c.get("/api/fire-outlook/runs").json() == {"dataset": "fire-outlook", "latest": None, "runs": []}
    assert c.get("/api/fire-outlook/runs/latest/manifest").status_code == 404
    assert c.get(f"/api/fire-outlook/runs/{RID}/manifest").status_code == 404


def test_runs_listing_newest_first_with_latest_and_stale(tmp_path, monkeypatch):
    _write(tmp_path, OLD, latest=False)
    _write(tmp_path, CYCLE)
    (tmp_path / "runs" / ".partial").mkdir()                    # temp / junk dirs are ignored
    (tmp_path / "runs" / "garbage").mkdir()
    body = _client(tmp_path, monkeypatch).get("/api/fire-outlook/runs").json()
    assert [r["run_id"] for r in body["runs"]] == [RID, OLD_RID] and body["latest"] == RID
    assert body["runs"][0]["is_latest"] and not body["runs"][1]["is_latest"]
    assert [r["stale"] for r in body["runs"]] == [False, True]    # 3 h old vs 72 h old
    assert body["runs"][0]["windows"] == 28 and body["runs"][0]["member_count"] == 31


def test_manifest_by_id_and_latest_with_serving_block(tmp_path, monkeypatch):
    _write(tmp_path)
    c = _client(tmp_path, monkeypatch)
    a = c.get(f"/api/fire-outlook/runs/{RID}/manifest").json()
    b = c.get("/api/fire-outlook/runs/latest/manifest").json()
    assert a == b and a["run_id"] == RID and len(a["windows"]) == 28
    assert a["label"] == "scenario agreement" and a["serving"]["stale_after_hours"] == 24 and a["serving"]["stale"] is False
    assert "contours/elevated/w00.geojson" in a["files"]


def test_bad_and_unknown_run_ids(tmp_path, monkeypatch):
    _write(tmp_path)
    c = _client(tmp_path, monkeypatch)
    assert c.get("/api/fire-outlook/runs/not-a-run/manifest").status_code == 400
    assert c.get("/api/fire-outlook/runs/..%2F..%2Fetc/manifest").status_code in (400, 404)
    assert c.get("/api/fire-outlook/runs/20260101T00Z/point", params=_center(5, 6)).status_code == 404


def test_point_matches_stored_grid(tmp_path, monkeypatch):
    _, counts = _write(tmp_path)
    c = _client(tmp_path, monkeypatch)
    body = c.get("/api/fire-outlook/runs/latest/point", params=_center(5, 6)).json()
    assert body["in_domain"] and len(body["windows"]) == 28 and body["serving"]["stale"] is False
    for w in (0, 9, 27):
        assert body["windows"][w]["elevated_member_count"] == int(counts[w, 5, 6])
        assert body["windows"][w]["critical_agreement_pct"] <= body["windows"][w]["elevated_agreement_pct"]
    assert "scenario" in body["label_note"]


def test_point_outside_domain_and_validation(tmp_path, monkeypatch):
    _write(tmp_path)
    c = _client(tmp_path, monkeypatch)
    out = c.get("/api/fire-outlook/runs/latest/point", params={"lat": 0, "lon": 0})
    assert out.status_code == 200 and out.json()["in_domain"] is False
    assert c.get("/api/fire-outlook/runs/latest/point", params={"lat": 95, "lon": 0}).status_code == 422
    assert c.get("/api/fire-outlook/runs/latest/point", params={"lat": 38}).status_code == 422


def test_status_exposes_safe_subset_only(tmp_path, monkeypatch):
    (tmp_path).mkdir(exist_ok=True)
    (tmp_path / "status.json").write_text(json.dumps({
        "state": "failed", "stage": "calculating", "cycle": "2026-10-29T00:00:00Z", "cycle_age_hours": 7.5,
        "runtime_s": 412.0, "peak_rss_mb": 2400.0, "threads": 1, "memory_limit_gb": 8.0,
        "errors": [{"stage": "calculating", "type": "IncompleteMember", "message": "secret path C:\\data\\x"}],
        "source_coverage": {"members_expected": 31, "members_ready": 29, "missing_members": ["p12", "p30"],
                            "last_lead_hours": 168},
        "last_complete": {"run_id": "20261028T12Z"}}))
    body = _client(tmp_path, monkeypatch).get("/api/fire-outlook/status").json()
    assert body["state"] == "failed" and body["failed_stage"] == "calculating" and body["error_count"] == 1
    assert body["source_coverage"] == {"members_expected": 31, "members_ready": 29, "last_lead_hours": 168}
    assert "secret" not in json.dumps(body) and "errors" not in body
    assert body["last_complete"]["run_id"] == "20261028T12Z"


def test_listed_in_api_index_and_registered():
    from routers import api_index
    paths = {r[1] for r in api_index._ROUTES}
    assert {"/fire-outlook/status", "/fire-outlook/runs", "/fire-outlook/runs/{run_id}/manifest",
            "/fire-outlook/runs/{run_id}/point"} <= paths
    # main.py imports Linux-only modules (fcntl), so check the registration in its source instead of importing it
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
    assert "app.include_router(fire_outlook.router)" in source
    imported = [name.strip() for line in source.splitlines() if line.startswith("from routers import ")
                for name in line[len("from routers import "):].split(",")]
    assert "fire_outlook" in imported


def test_stale_flag_flips_after_24_hours(tmp_path, monkeypatch):
    _write(tmp_path, OLD)
    c = _client(tmp_path, monkeypatch)
    assert c.get("/api/fire-outlook/runs/latest/manifest").json()["serving"]["stale"] is True
    assert c.get("/api/fire-outlook/runs/latest/point", params=_center(5, 6)).json()["serving"]["stale"] is True
