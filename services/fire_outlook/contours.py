"""
Filled, disjoint contour bands of scenario agreement for the map.

Bands at 10/30/50/70/90 %: [10,30), [30,50), [50,70), [70,90), [90,100]. Agreement below 10 % is not drawn.
Bands are derived by polygonizing an integer band raster, so they are disjoint by construction (no
opacity stacking), match the underlying grid values exactly, preserve holes, and never cover NoData
cells (excluded before polygonizing; 4-connectivity, so diagonal cells are not bridged). There is
deliberately no nonburnable mask: the grid is 3 km and agreement is drawn everywhere it has data.
Display simplification (Douglas-Peucker via a shapely coverage, default 0.2 cell) is applied
in projected metres; the numerical grids stay the source for queries. Empty and full-domain cases work.
Bands are stair-stepped at cell edges on purpose: smoothing would move edges off the data.
"""
from __future__ import annotations

import json
from typing import Dict, Optional

import numpy as np

BAND_EDGES_PCT = (10, 30, 50, 70, 90)


def band_labels() -> list[dict]:
    edges = list(BAND_EDGES_PCT) + [100]
    return [{"band": i + 1, "lower_pct": edges[i], "upper_pct": edges[i + 1],
             "upper_inclusive": i == len(BAND_EDGES_PCT) - 1} for i in range(len(BAND_EDGES_PCT))]


def band_raster(counts: np.ndarray, n_members: int, nodata_count: int = 255) -> np.ndarray:
    """(y, x) uint8 member counts -> band index 0..5 (0 = not drawn). Integer arithmetic: no float edge errors."""
    valid = counts != nodata_count
    scaled = counts.astype("int32") * 100
    out = np.zeros(counts.shape, dtype="uint8")
    for i, edge in enumerate(BAND_EDGES_PCT, start=1):
        out[scaled >= edge * n_members] = i
    out[~valid] = 0
    return out


def band_polygons(bands: np.ndarray, *, west: float, north: float, resolution_m: float, crs: str = "EPSG:32615",
                  simplify_cells: float = 0.2, precision: int = 6, labels: Optional[dict] = None) -> list[dict]:
    """Band raster -> GeoJSON Features (lon/lat). Holes preserved; one Feature per connected polygon.
    labels: {band index: properties dict} for a value layer (default: the scenario-agreement bands)."""
    from pyproj import Transformer
    from rasterio import features
    from rasterio.transform import from_origin
    import shapely
    from shapely.geometry import mapping, shape
    from shapely.ops import transform as shp_transform

    transform = from_origin(west, north, resolution_m, resolution_m)
    to_ll = Transformer.from_crs(crs, 4326, always_xy=True).transform
    meta = labels if labels is not None else {
        b["band"]: {"lower_pct": b["lower_pct"], "upper_pct": b["upper_pct"]} for b in band_labels()}
    raw = [(shape(geom), int(value)) for geom, value in
           features.shapes(bands, mask=bands > 0, transform=transform, connectivity=4)]
    polys = [p for p, _ in raw]
    if polys and simplify_cells > 0:
        # Simplify all bands as ONE coverage so shared edges move together: independent per-polygon
        # simplification leaves slivers (overlaps = opacity stacking, gaps = fake holes) between bands.
        polys = list(shapely.coverage_simplify(np.array(polys, dtype=object), simplify_cells * resolution_m))
    out = []
    for poly, (_, value) in zip(polys, raw):
        if poly.is_empty:
            continue
        poly = shp_transform(to_ll, poly)
        gj = mapping(poly)
        gj = json.loads(json.dumps(gj, default=list))  # tuples -> lists
        gj["coordinates"] = _round(gj["coordinates"], precision)
        band = int(value)
        out.append({"type": "Feature", "geometry": gj,
                    "properties": {"band": band, **meta[band]}})
    return out


def _round(coords, precision):
    if isinstance(coords, (list, tuple)) and coords and isinstance(coords[0], (int, float)):
        return [round(c, precision) for c in coords]
    return [_round(c, precision) for c in coords]


def window_geojson(counts: np.ndarray, n_members: int, *, layer: str, window_index: int, grid: dict,
                   nodata_count: int = 255, simplify_cells: float = 0.2) -> bytes:
    bands = band_raster(counts, n_members, nodata_count)
    feats = band_polygons(bands, west=grid["west"], north=grid["north"], resolution_m=grid["resolution_m"],
                          crs=grid.get("crs", "EPSG:32615"), simplify_cells=simplify_cells)
    payload = {"type": "FeatureCollection", "features": feats,
               "properties": {"layer": layer, "window_index": window_index, "member_count": n_members,
                              "bands": band_labels(), "label": "scenario agreement"}}
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def contour_files(elevated_counts: np.ndarray, critical_counts: np.ndarray, n_members: int, grid: dict,
                  **kwargs) -> Dict[str, bytes]:
    """All windows x both layers -> {'contours/elevated/w00.geojson': bytes, ...} for assemble.write_run."""
    files = {}
    for layer, counts in (("elevated", elevated_counts), ("critical", critical_counts)):
        for w in range(counts.shape[0]):
            files[f"contours/{layer}/w{w:02d}.geojson"] = window_geojson(
                counts[w], n_members, layer=layer, window_index=w, grid=grid, **kwargs)
    return files
