"""
Point queries over a published outlook run. Reads the precomputed arrays only (no forecast calculation),
so the answer is exactly what the map's grids hold: same counts, same NoData.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from services.fire_outlook.assemble import NODATA_COUNT


def public_grid_dict() -> dict:
    from forecast_v1.contracts import PUBLIC_GRID
    return {"id": PUBLIC_GRID.id, "crs": PUBLIC_GRID.crs, "width": PUBLIC_GRID.width, "height": PUBLIC_GRID.height,
            "resolution_m": PUBLIC_GRID.resolution_m, "west": PUBLIC_GRID.west, "north": PUBLIC_GRID.north}


def cell_for(lat: float, lon: float, grid: dict) -> Optional[tuple[int, int]]:
    """(row, col) of the grid cell containing lat/lon, or None outside the grid."""
    from pyproj import Transformer
    x, y = Transformer.from_crs(4326, grid.get("crs", "EPSG:32615"), always_xy=True).transform(lon, lat)
    if not (np.isfinite(x) and np.isfinite(y)):  # far from the UTM zone: projects to inf
        return None
    col = int(np.floor((x - grid["west"]) / grid["resolution_m"]))
    row = int(np.floor((grid["north"] - y) / grid["resolution_m"]))
    if 0 <= row < grid["height"] and 0 <= col < grid["width"]:
        return row, col
    return None


def _num(value) -> Optional[float]:
    value = float(value)
    return None if not np.isfinite(value) else round(value, 3)


_ARRAY_FILES = {
    "elevated": ("agreement_elevated.npz", ("counts",)),
    "critical": ("agreement_critical.npz", ("counts",)),
    "fm": ("fm10_min.npz", ("p10", "p50", "p90")),
    "wx": ("weather.npz", ("mean_hours_elevated", "mean_hours_critical", "mean_precip_6h_mm",
                           "temp_max_c_envelope", "rh_min_pct_envelope", "wind_max_kts_envelope",
                           "temp_max_c_mean", "rh_min_pct_mean", "wind_max_kts_mean")),
    "nodata": ("nodata.npz", ("mask",)),
}


@lru_cache(maxsize=2)
def _arrays(run_dir: str, stamp: int) -> dict:
    """Every stored array of one run, decompressed ONCE (an NpzFile re-decompresses the whole array on each key
    access: indexing it per window made a point request take ~7 s). Runs are immutable; `stamp` (the manifest's
    mtime) only guards against a directory being replaced. Two runs at ~80 MB each stay in memory."""
    out = {}
    for group, (filename, keys) in _ARRAY_FILES.items():
        with np.load(Path(run_dir) / filename) as z:
            out[group] = {k: z[k] for k in keys if k in z.files}   # runs written before the scenario means lack some keys
    return out


def point_series(run_dir: Path, lat: float, lon: float) -> dict:
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cell = cell_for(lat, lon, manifest["grid"])
    if cell is None:
        return {"run_id": manifest["run_id"], "in_domain": False, "lat": lat, "lon": lon}
    r, c = cell
    n = manifest["member_count"]
    a = _arrays(str(run_dir), manifest_path.stat().st_mtime_ns)
    elev, crit = a["elevated"]["counts"][:, r, c], a["critical"]["counts"][:, r, c]
    fm = {k: v[:, r, c] for k, v in a["fm"].items()}
    wx = {k: v[:, r, c] for k, v in a["wx"].items()}
    nodata = a["nodata"]["mask"][:, r, c]

    def pct(count):
        return None if count == NODATA_COUNT else round(100.0 * int(count) / n, 1)

    windows = []
    for w, meta in enumerate(manifest["windows"]):
        windows.append({
            "index": w, "start_utc": meta["start_utc"], "end_utc": meta["end_utc"], "nodata": bool(nodata[w]),
            "elevated_agreement_pct": pct(elev[w]), "critical_agreement_pct": pct(crit[w]),
            "elevated_member_count": None if elev[w] == NODATA_COUNT else int(elev[w]),
            "critical_member_count": None if crit[w] == NODATA_COUNT else int(crit[w]),
            "fm10_min_modeled_pct": {k: _num(fm[k][w]) for k in ("p10", "p50", "p90")},
            "mean_hours_elevated": _num(wx["mean_hours_elevated"][w]),
            "mean_hours_critical": _num(wx["mean_hours_critical"][w]),
            "mean_precip_6h_mm": _num(wx["mean_precip_6h_mm"][w]),
            # envelope = the most extreme member; mean = the typical scenario (absent in runs written before 2026-10-30)
            "temp_max_c": _num(wx["temp_max_c_envelope"][w]),
            "rh_min_pct": _num(wx["rh_min_pct_envelope"][w]),
            "wind_max_kts": _num(wx["wind_max_kts_envelope"][w]),
            "temp_max_c_mean": _num(wx["temp_max_c_mean"][w]) if "temp_max_c_mean" in wx else None,
            "rh_min_pct_mean": _num(wx["rh_min_pct_mean"][w]) if "rh_min_pct_mean" in wx else None,
            "wind_max_kts_mean": _num(wx["wind_max_kts_mean"][w]) if "wind_max_kts_mean" in wx else None,
        })
    return {"run_id": manifest["run_id"], "in_domain": True, "lat": lat, "lon": lon, "row": r, "col": c,
            "member_count": n, "label": manifest["label"],
            "label_note": manifest["label_note"], "cycle_time_utc": manifest["cycle_time_utc"],
            "windows": windows}