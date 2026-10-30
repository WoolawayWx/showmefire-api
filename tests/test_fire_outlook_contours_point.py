"""Tests for services/fire_outlook contours.py and point.py (and write_run's extra files)."""
import json
from datetime import datetime, timezone

import numpy as np
import pytest
from pyproj import Transformer
from rasterio import features
from rasterio.transform import from_origin
from shapely.geometry import shape
from shapely.ops import transform as shp_transform

from services.fire_outlook import assemble, contours, point
from tests.test_fire_outlook_assemble import _result

CYCLE = datetime(2026, 10, 29, 0, tzinfo=timezone.utc)
GRID = {"id": "t", "crs": "EPSG:32615", "width": 12, "height": 10, "resolution_m": 3000.0,
        "west": 500_000.0, "north": 4_300_000.0}
N = 31


def _rasterize(features_, bands_shape):
    """Rasterize contour features back onto the grid, one band at a time."""
    to_utm = Transformer.from_crs(4326, GRID["crs"], always_xy=True).transform
    transform = from_origin(GRID["west"], GRID["north"], GRID["resolution_m"], GRID["resolution_m"])
    out = np.zeros(bands_shape, dtype="uint8")
    for f in features_:
        geom = shp_transform(to_utm, shape(f["geometry"]))
        mask = features.rasterize([(geom, 1)], out_shape=bands_shape, transform=transform, fill=0, dtype="uint8")
        out[mask == 1] = f["properties"]["band"]
    return out


def _counts(h=10, w=12, seed=0):
    return np.random.default_rng(seed).integers(0, N + 1, (h, w)).astype("uint8")


def test_band_edges_are_exact_with_31_members():
    counts = np.array([[0, 3, 4, 9, 10, 15, 16, 21, 22, 27, 28, 31]], dtype="uint8")
    # pct: 0, 9.7, 12.9, 29.0, 32.3, 48.4, 51.6, 67.7, 71.0, 87.1, 90.3, 100
    bands = contours.band_raster(counts, N)
    assert bands[0].tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]


def test_exactly_ten_percent_boundary_uses_integer_math():
    # n=10: count 1 is exactly 10% -> band 1; no float rounding error possible
    assert contours.band_raster(np.array([[0, 1, 3, 5, 7, 9, 10]], dtype="uint8"), 10)[0].tolist() == [0, 1, 2, 3, 4, 5, 5]


def test_nodata_never_drawn_or_bridged():
    counts = np.full((10, 12), 20, dtype="uint8")
    counts[4, 4] = 255
    counts[:, 8] = 255                          # a full NoData column splits the domain
    bands = contours.band_raster(counts, N)
    assert bands[4, 4] == 0 and (bands[:, 8] == 0).all()
    feats = contours.band_polygons(bands, west=GRID["west"], north=GRID["north"], resolution_m=3000, simplify_cells=0)
    back = _rasterize(feats, bands.shape)
    np.testing.assert_array_equal(back, bands)  # matches values exactly, incl. hole at (4,4) and the split
    assert len(feats) == 2                      # two polygons: left of and right of the column


def test_polygons_match_values_disjoint_and_close_when_simplified():
    counts = _counts()
    bands = contours.band_raster(counts, N)
    exact = _rasterize(contours.band_polygons(bands, west=GRID["west"], north=GRID["north"], resolution_m=3000,
                                              simplify_cells=0), bands.shape)
    np.testing.assert_array_equal(exact, bands)
    simp = _rasterize(contours.band_polygons(bands, west=GRID["west"], north=GRID["north"], resolution_m=3000,
                                             simplify_cells=0.2), bands.shape)
    assert (simp != bands).mean() < 0.05
    # disjointness: no two features overlap in area
    feats = contours.band_polygons(bands, west=GRID["west"], north=GRID["north"], resolution_m=3000)
    geoms = [shape(f["geometry"]) for f in feats]
    total = sum(g.area for g in geoms)
    from shapely.ops import unary_union
    # Not exactly 0: reprojecting to lon/lat bends a straight edge slightly away from a neighbour's polyline
    # that carries an extra vertex, leaving sub-metre slivers (~1e-5 of area). Measured identical with
    # simplification off, so it is the projection, not the simplifier.
    assert unary_union(geoms).area == pytest.approx(total, rel=1e-4)


def test_holes_preserved():
    counts = np.full((10, 12), 20, dtype="uint8")
    counts[4:6, 4:6] = 0
    bands = contours.band_raster(counts, N)
    feats = contours.band_polygons(bands, west=GRID["west"], north=GRID["north"], resolution_m=3000, simplify_cells=0)
    assert len(feats) == 1 and len(feats[0]["geometry"]["coordinates"]) == 2  # exterior + one hole


def test_empty_and_full_domain():
    empty = contours.band_polygons(contours.band_raster(np.zeros((10, 12), "uint8"), N), west=0, north=0, resolution_m=3000)
    assert empty == []
    full = contours.band_polygons(contours.band_raster(np.full((10, 12), 31, "uint8"), N), west=GRID["west"],
                                  north=GRID["north"], resolution_m=3000)
    assert len(full) == 1 and full[0]["properties"]["band"] == 5 and full[0]["properties"]["lower_pct"] == 90


def test_critical_bands_never_exceed_elevated_bands():
    e = _counts(seed=1)
    c = np.minimum(_counts(seed=2), e)
    assert (contours.band_raster(c, N) <= contours.band_raster(e, N)).all()


# --- end to end through write_run + point -----------------------------------

def _run(tmp_path):
    shape_ = (28, GRID["height"], GRID["width"])
    r = _result()
    elevated_counts = np.stack([_counts(seed=w) for w in range(28)])
    critical_counts = np.minimum(elevated_counts, np.stack([_counts(seed=100 + w) for w in range(28)]))
    r["agreement_elevated"] = (elevated_counts / N).astype("float32")
    r["agreement_critical"] = (critical_counts / N).astype("float32")
    for k in ("fm10_min_p10", "fm10_min_p50", "fm10_min_p90", "mean_hours_elevated", "mean_hours_critical",
              "mean_precip_6h_mm", "temp_max_c_envelope", "rh_min_pct_envelope", "wind_max_kts_envelope"):
        r[k] = np.full(shape_, 7.0, "float32")
    r["nodata"] = np.zeros(shape_, bool)
    r["agreement_elevated"][2, 3, 4] = np.nan
    r["agreement_critical"][2, 3, 4] = np.nan
    r["nodata"][2, 3, 4] = True
    files = contours.contour_files(assemble._counts(r["agreement_elevated"], N), assemble._counts(r["agreement_critical"], N),
                                   N, GRID)
    run = assemble.write_run(tmp_path, CYCLE, r, grid=GRID, source_cycles={}, rule_info={},
                             conditioning_quality={}, extra_files=files)
    return run, elevated_counts


def test_contour_files_are_in_manifest_and_checksummed(tmp_path):
    run, _ = _run(tmp_path)
    manifest = json.loads((run / "manifest.json").read_text())
    assert "contours/elevated/w00.geojson" in manifest["files"] and "contours/critical/w27.geojson" in manifest["files"]
    assert "masks.npz" not in manifest["files"] and assemble.verify_checksums(run)
    fc = json.loads((run / "contours/elevated/w05.geojson").read_text())
    assert fc["type"] == "FeatureCollection" and fc["properties"]["label"] == "scenario agreement"


def test_extra_file_cannot_escape_run_dir(tmp_path):
    with pytest.raises(ValueError):
        assemble.write_run(tmp_path, CYCLE, _result(), grid=GRID, source_cycles={}, rule_info={},
                           conditioning_quality={}, extra_files={"../evil.txt": b"x"})
    assert not (tmp_path.parent / "evil.txt").exists() and list(tmp_path.iterdir()) == []


def test_point_matches_grid_values_and_nodata(tmp_path):
    run, counts = _run(tmp_path)
    to_ll = Transformer.from_crs(GRID["crs"], 4326, always_xy=True).transform

    def center(r, c):
        lon, lat = to_ll(GRID["west"] + (c + 0.5) * 3000, GRID["north"] - (r + 0.5) * 3000)
        return lat, lon
    lat, lon = center(5, 6)
    out = point.point_series(run, lat, lon)
    assert out["in_domain"] and (out["row"], out["col"]) == (5, 6) and "nonburnable" not in out
    assert len(out["windows"]) == 28
    for w in (0, 7, 27):
        assert out["windows"][w]["elevated_member_count"] == int(counts[w, 5, 6])
        assert out["windows"][w]["elevated_agreement_pct"] == round(100 * int(counts[w, 5, 6]) / N, 1)
        assert out["windows"][w]["critical_agreement_pct"] <= out["windows"][w]["elevated_agreement_pct"]
    assert out["windows"][3]["fm10_min_modeled_pct"]["p50"] == 7.0
    nd = point.point_series(run, *center(3, 4))
    assert nd["windows"][2]["nodata"] and nd["windows"][2]["elevated_agreement_pct"] is None
    assert point.point_series(run, 0.0, 0.0)["in_domain"] is False


def test_public_grid_dict_matches_contract():
    g = point.public_grid_dict()
    assert (g["width"], g["height"], g["resolution_m"]) == (267, 264, 3000)
    assert point.cell_for(38.5, -92.5, g) is not None and point.cell_for(45.0, -92.5, g) is None
