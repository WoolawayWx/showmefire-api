from __future__ import annotations

import json
from datetime import datetime, timezone

import geopandas as gpd
import numpy as np
import pytest
import shapely
from pyproj import Transformer
from shapely.geometry import Point, shape
from shapely.ops import unary_union

from forecast import export_fire_danger_gis as base
from forecast import export_smooth_polygons as smooth


def _model_grid():
    """A 3 km Lambert-conformal style grid (curved in lon/lat) over Missouri, with a
    smooth 0-4 risk field that is NaN outside the state, like the production input."""
    lcc = Transformer.from_crs(
        "EPSG:4326",
        "+proj=lcc +lat_1=38.5 +lat_2=38.5 +lat_0=38.5 +lon_0=-92.5 +datum=WGS84",
        always_xy=True,
    )
    inverse = Transformer.from_crs(lcc.target_crs, "EPSG:4326", always_xy=True)
    x0, y0 = lcc.transform(-92.5, 38.5)
    xs = x0 + np.arange(-340_000, 340_001, 3000)
    ys = y0 + np.arange(-300_000, 300_001, 3000)
    xx, yy = np.meshgrid(xs, ys)
    lon, lat = inverse.transform(xx, yy)
    field = 2.0 + 1.9 * np.sin((lon + 92.5) * 1.3) * np.cos((lat - 38.5) * 1.1) + 0.6 * ((lat - 38.5) / 2.5)
    field = np.clip(field, -0.4, 4.4)
    state = gpd.read_file(base.STATE_BOUNDARY_SHP).to_crs("EPSG:4326")
    inside = shapely.contains_xy(unary_union(state.geometry).buffer(0.01), lon, lat)
    return np.where(inside, field, np.nan), lon, lat, state


@pytest.fixture(scope="module")
def regions():
    field, lon, lat, state = _model_grid()
    run = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
    return smooth.build_smooth_danger_regions(field, lon, lat, run_date=run, resolution_m=1000), field, lon, lat, state


def test_bands_tile_missouri_without_gaps_or_overlaps(regions):
    found, _, _, _, state = regions
    utm = lambda g: gpd.GeoSeries([g], crs="EPSG:4326").to_crs("EPSG:32615").iloc[0]
    state_geom = utm(unary_union(state.geometry))
    bands = [utm(r["geometry"]) for r in found]
    union = unary_union(bands)
    assert state_geom.difference(union).area / state_geom.area < 0.005      # no gaps
    assert union.difference(state_geom).area / state_geom.area < 0.001      # nothing outside Missouri
    assert sum(b.area for b in bands) == pytest.approx(union.area, rel=0.002)  # no overlaps


def test_polygons_agree_with_the_field(regions):
    found, field, lon, lat, _ = regions
    rng = np.random.default_rng(7)
    rows, cols = np.where(np.isfinite(field))
    checked = wrong = 0
    for idx in rng.choice(len(rows), 400, replace=False):
        r, c = rows[idx], cols[idx]
        expected = int(np.clip(np.rint(field[r, c]), 0, 4))
        point = Point(lon[r, c], lat[r, c])
        level = [x["danger_level"] for x in found if x["geometry"].contains(point)]
        if not level:        # within a few hundred metres of the border
            continue
        checked += 1
        wrong += level[0] != expected
    assert checked > 300
    assert wrong / checked < 0.03       # only cells sitting right on a threshold may differ


def test_boundaries_are_smooth_not_blocky(regions):
    found, *_ = regions
    # A 3 km cell dissolve has long straight axis-aligned runs; contours do not.
    assert all(r["geometry"].is_valid for r in found)
    assert sum(len(r["geometry"].exterior.coords) if r["geometry"].geom_type == "Polygon"
               else sum(len(p.exterior.coords) for p in r["geometry"].geoms) for r in found) > 500


def test_export_writes_geojson_atomically(tmp_path):
    field, lon, lat, _ = _model_grid()
    out = tmp_path / "smooth.geojson"
    assert smooth.export_smooth_geojson_polygons(field, lon, lat, out, run_date=datetime(2026, 10, 1, 12, tzinfo=timezone.utc))
    payload = json.loads(out.read_text())
    assert payload["type"] == "FeatureCollection"
    levels = {f["properties"]["danger_level"] for f in payload["features"]}
    assert levels <= {0, 1, 2, 3, 4} and len(levels) >= 3
    assert all(shape(f["geometry"]).is_valid for f in payload["features"])
    assert oct(out.stat().st_mode & 0o777) in {"0o644", "0o666"}   # readable by the qgis-server container
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]  # no temp files left


def test_failure_is_swallowed_and_leaves_no_file(tmp_path):
    out = tmp_path / "smooth.geojson"
    assert smooth.export_smooth_geojson_polygons(np.full((5, 5), np.nan), np.zeros((5, 5)), np.zeros((5, 5)), out) is False
    assert not out.exists()


def test_can_be_disabled(monkeypatch):
    monkeypatch.setenv("SMF_SMOOTH_POLYGONS", "0")
    assert smooth.enabled() is False
