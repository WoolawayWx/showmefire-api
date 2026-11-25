"""Winter weather maps for Missouri: the National Weather Service snow and ice forecasts and WPC's Winter Storm Severity Index.

Same look as maps/spc_wpc_outlook_maps.py (the Missouri basemap, title / valid time / "Data Source" block, legend at the
right), one PNG per product in images/ named mo-winter-<key>.png; maps.sh uploads new images to the CDN, and this script
also uploads what it redraws.

Products
- snow-72hr / ice-72hr: total snowfall and total ice accumulation for the next 72 hours, summed from the NWS National Digital
  Forecast Database (NDFD) 6-hour grids (tgftp.nws.noaa.gov, ds.snow.bin / ds.iceaccum.bin). NDFD stores snow in metres and
  ice in millimetres of liquid; the maps show inches.
- wssi-day1 / day2 / day3 / days1-3: WPC's Winter Storm Severity Index, overall impact (WPC's own map service, colors as
  published): Winter Weather Area, then Minor to Extreme impacts.
- wssi-snow / snow-load / ice / blowing: the four WSSI components for Days 1-3.

maps.sh runs every script in this folder every 15 minutes, but these products change a few times a day. So each source is
fetched, compared with a hash of what was last drawn, and only redrawn (and pushed to the CDN) when it changed. A source that
is down is reported and skipped; the others still update. In summer the WSSI has no polygons and the maps say so.
"""
import argparse
import re
import sys
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_DIR))

from maps import raster_maps as rm
from maps import station_plot as sp
from maps.spc_wpc_outlook_maps import (CENTRAL, HEADERS, STATUS_DIR, IMAGES_DIR, _clock, _prop, _stamp, clip_to_missouri,
                                        fetch_url, read_polygons, signature)

FILE_PREFIX = "mo-winter"
NDFD = "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.001-003"
WSSI = "https://mapservices.weather.noaa.gov/vector/rest/services/outlooks/wpc_wssi/MapServer"
# Only polygons near Missouri: the national WSSI layer is large.
WSSI_QUERY = ("query?where=1%3D1&outFields=*&returnGeometry=true&outSR=4326&f=geojson"
              "&geometry=-98,34.5,-88,42&geometryType=esriGeometryEnvelope&inSR=4326&spatialRel=esriSpatialRelIntersects")
NWS_SOURCE = "National Weather Service (NDFD)"
WSSI_SOURCE = "NOAA Weather Prediction Center"

METRES_TO_INCHES = 1 / 0.0254
MM_TO_INCHES = 1 / 25.4
# NDFD grids are national; keep a box around Missouri for drawing.
GRID_BOX = (-97.5, 34.0, -88.0, 41.5)        # lon_min, lat_min, lon_max, lat_max
SNOW_LEVELS = [0.1, 1, 2, 3, 4, 6, 8, 12, 18, 24, 36]
SNOW_COLORS = ["#c3e1f5", "#a6d0ee", "#74b0e0", "#4a8ccc", "#2a5fb0", "#5b3fa8", "#8a2fa6", "#c4259a", "#e0447a", "#a50f15"]
ICE_LEVELS = [0.01, 0.1, 0.25, 0.5, 0.75, 1.0]
ICE_COLORS = ["#ffd6ec", "#ff9ccf", "#e9529f", "#b32a7c", "#6d1060"]
# Straight from WPC's own map service renderer (value, label, fill).
WSSI_CLASSES = {
    "WINTER WEATHER AREA": (1, "Winter Weather Area", "#D2DFE7"),
    "MINOR": (2, "Minor Impacts", "#FAF5A3"),
    "MODERATE": (3, "Moderate Impacts", "#F7962F"),
    "MAJOR": (4, "Major Impacts", "#E61F26"),
    "EXTREME": (5, "Extreme Impacts", "#7853A1"),
}


@dataclass(frozen=True)
class Product:
    key: str                    # file is mo-winter-<key>.png
    title: str
    kind: str                   # ndfd_snow | ndfd_ice | wssi
    note: str                   # what the map shows, in the description block
    source: str
    url: str = ""               # wssi: the GeoJSON query URL
    group: str = "Forecast"     # how the website and the app group it


def _wssi(key: str, title: str, layer: int, note: str, group: str) -> Product:
    return Product(key, title, "wssi", note, WSSI_SOURCE, f"{WSSI}/{layer}/{WSSI_QUERY}", group)


OVERALL_NOTE = "How much a winter storm is likely to disrupt daily life: snow, ice, snow load and blowing snow combined."
PRODUCTS: tuple[Product, ...] = (
    Product("snow-72hr", "Missouri NWS Snowfall Forecast\n(Next 72 Hours)", "ndfd_snow",
            "Total snowfall expected over the next 72 hours.", NWS_SOURCE),
    Product("ice-72hr", "Missouri NWS Ice Accumulation Forecast\n(Next 72 Hours)", "ndfd_ice",
            "Total freezing-rain ice accumulation expected over the next 72 hours.", NWS_SOURCE),
    _wssi("wssi-day1", "Missouri Winter Storm\nSeverity Index: Day 1", 1, OVERALL_NOTE, "Day 1"),
    _wssi("wssi-day2", "Missouri Winter Storm\nSeverity Index: Day 2", 2, OVERALL_NOTE, "Day 2"),
    _wssi("wssi-day3", "Missouri Winter Storm\nSeverity Index: Day 3", 3, OVERALL_NOTE, "Day 3"),
    _wssi("wssi-days1-3", "Missouri Winter Storm\nSeverity Index: Days 1-3", 4, OVERALL_NOTE, "Days 1-3"),
    _wssi("wssi-snow", "Missouri WSSI Snow Amount\nImpacts (Days 1-3)", 9,
          "How much the expected snowfall is likely to disrupt travel and daily life.", "Days 1-3"),
    _wssi("wssi-snow-load", "Missouri WSSI Snow Load\nImpacts (Days 1-3)", 14,
          "Weight of snow on trees and power lines: tree damage and power outages.", "Days 1-3"),
    _wssi("wssi-ice", "Missouri WSSI Ice Accumulation\nImpacts (Days 1-3)", 19,
          "How much expected ice is likely to disrupt travel and damage trees and power lines.", "Days 1-3"),
    _wssi("wssi-blowing-snow", "Missouri WSSI Blowing Snow\nImpacts (Days 1-3)", 27,
          "Reduced visibility and drifting from wind-blown snow.", "Days 1-3"),
)
BY_KEY = {product.key: product for product in PRODUCTS}


# --- NDFD snow and ice ---------------------------------------------------------------------------------------------------
def read_ndfd_total(payload: bytes, scale: float) -> dict:
    """Total accumulation (inches) over all forecast steps, on the box around Missouri.

    NDFD accumulation grids hold one value per period (the 6-hour period ending at each valid time), so the total over the
    window is the sum of the steps. Returns lon/lat (2-D, -180..180), the total, the model reference time and the last
    valid time. Imported lazily: cfgrib/xarray are heavy and only this product needs them."""
    import tempfile

    import cfgrib

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "ndfd.grib2"
        path.write_bytes(payload)
        dataset = cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""})[0]
        values = np.nan_to_num(dataset["unknown"].values.astype("float64"), nan=0.0)       # (step, y, x)
        lon = dataset["longitude"].values.astype("float64")
        lon = np.where(lon > 180, lon - 360, lon)
        lat = dataset["latitude"].values.astype("float64")
        valid = [datetime.fromisoformat(str(t)[:19]).replace(tzinfo=timezone.utc) for t in dataset["valid_time"].values]
        reference = datetime.fromisoformat(str(dataset["time"].values)[:19]).replace(tzinfo=timezone.utc)
    total = np.clip(values, 0, None).sum(axis=0) * scale
    inside = (lon >= GRID_BOX[0]) & (lon <= GRID_BOX[2]) & (lat >= GRID_BOX[1]) & (lat <= GRID_BOX[3])
    rows, cols = np.where(inside)
    window = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
    return {"lon": lon[window], "lat": lat[window], "total": total[window], "reference": reference, "end": max(valid)}


def max_over_missouri(grid: dict) -> float:
    """Largest total inside the state outline, so a storm just across the border is not reported as Missouri's."""
    state = sp._missouri_geometry()
    if state is None:
        return float(np.nanmax(grid["total"]))
    import shapely

    inside = shapely.contains_xy(state, grid["lon"], grid["lat"])
    return float(np.nanmax(grid["total"][inside])) if inside.any() else 0.0


def draw_ndfd(product: Product, payload: bytes, out_png: Path, status_path: Path) -> dict:
    started = time.time()
    snow = product.kind == "ndfd_snow"
    grid = read_ndfd_total(payload, METRES_TO_INCHES if snow else MM_TO_INCHES)
    levels, colors = (SNOW_LEVELS, SNOW_COLORS) if snow else (ICE_LEVELS, ICE_COLORS)
    fig, ax, data_crs, _map_crs, _dpi = sp.generate_basemap()
    cmap = ListedColormap(colors)
    cmap.set_over(colors[-1])
    norm = BoundaryNorm(levels, cmap.N)
    filled = ax.contourf(grid["lon"], grid["lat"], grid["total"], levels=levels, cmap=cmap, norm=norm, extend="max",
                         transform=data_crs, zorder=6)    # norm: one color per bin (not per value)
    rm.clip_to_state(ax, filled)
    sp.redraw_boundaries(ax, data_crs)                      # draw county and state lines over the fill
    rm.left_colorbar(fig, cmap, norm, "Total snowfall (inches)" if snow else "Total ice accumulation (inches)",
                     ticks=levels, tick_labels=[f"{level:g}" for level in levels], extend="max")

    peak = max_over_missouri(grid)
    threshold = levels[0]
    if peak < threshold:
        headline = "No snow expected in Missouri." if snow else "No ice accumulation expected in Missouri."
    else:
        headline = f"Highest total in Missouri: {peak:.1f} in." if snow else f"Highest total in Missouri: {peak:.2f} in."
    start, end = grid["reference"].astimezone(CENTRAL), grid["end"].astimezone(CENTRAL)
    subtitle = f"Forecast from {_stamp(start)} through {_stamp(end)} CT"
    description = "\n".join([f"Data Source: {product.source}", "", textwrap.fill(product.note, 40), "", headline,
                             "", textwrap.fill("Forecast amounts change as the storm gets closer. Check weather.gov for "
                                               "warnings and advisories.", 40)])
    sp.add_branding(fig, ax, title=product.title, subtitle=subtitle, description=description,
                    update_schedule="Redrawn whenever the NWS forecast is updated")
    status = {"product": f"winter_{product.key}", "status": "updated", "valid": subtitle,
              "peak_inches_over_missouri": round(peak, 2), "last_update": datetime.now(CENTRAL).strftime("%Y-%m-%d %H:%M CT"),
              "runtime_sec": round(time.time() - started, 2)}
    sp.save_and_record_status(fig, out_png, status_path, status)
    return status


# --- WSSI ----------------------------------------------------------------------------------------------------------------
def _wssi_time(text) -> datetime | None:
    """'12Z 11/25/26' (hour, month/day/two-digit year, UTC) or '2026-11-25 1315Z'."""
    if not text:
        return None
    text = str(text).strip()
    for pattern, rule in ((r"(\d{4})-(\d{2})-(\d{2}) (\d{2})(\d{2})Z", "iso"), (r"(\d{1,2})Z (\d{1,2})/(\d{1,2})/(\d{2})", "us")):
        found = re.fullmatch(pattern, text)
        if found:
            nums = [int(part) for part in found.groups()]
            try:
                if rule == "iso":
                    return datetime(nums[0], nums[1], nums[2], nums[3], nums[4], tzinfo=timezone.utc)
                return datetime(2000 + nums[3], nums[1], nums[2], nums[0], tzinfo=timezone.utc)
            except ValueError:
                return None
    return None


def wssi_valid(properties: dict) -> tuple[str, str | None]:
    start, end = _wssi_time(_prop(properties, "start_time")), _wssi_time(_prop(properties, "end_time"))
    issued = _wssi_time(_prop(properties, "issue_time"))
    valid = "Valid: current outlook"
    if start and end:
        start, end = start.astimezone(CENTRAL), end.astimezone(CENTRAL)
        valid = f"Valid: {_stamp(start)} – {_stamp(end)} CT"
    return valid, (f"Issued: {_stamp(issued.astimezone(CENTRAL))} CT" if issued else None)


def draw_wssi(product: Product, payload: bytes, out_png: Path, status_path: Path) -> dict:
    started = time.time()
    frame = read_polygons(payload)
    properties = dict(frame.iloc[0].drop("geometry")) if not frame.empty else {}
    frame = clip_to_missouri(frame)
    fig, ax, data_crs, _map_crs, _dpi = sp.generate_basemap()

    entries: dict[int, tuple[str, str]] = {}
    drawn = []
    for _, row in frame.iterrows():
        level, label, color = WSSI_CLASSES.get(str(_prop(row, "impact", default="")).upper().strip(), (0, "", ""))
        if level:
            drawn.append((level, label, color, row["geometry"]))
    for level, label, color, geometry in sorted(drawn, key=lambda item: item[0]):
        # Opaque, lowest first: the classes are nested, and a see-through fill would blend each with the one beneath it.
        ax.add_geometries([geometry], crs=data_crs, facecolor=color, edgecolor="#555555", alpha=1.0, linewidth=1.0, zorder=7)
        entries.setdefault(level, (label, color))
    if entries:
        handles = [Patch(facecolor=color, edgecolor="#555555", linewidth=0.6, label=label) for _level, (label, color) in sorted(entries.items())]
        rm.category_legend(fig, handles, "Impact Level", y=0.36, fontsize=12)
    status_line = None if entries else "No winter weather impacts\nare forecast for Missouri."

    valid, issued = wssi_valid(properties)
    lines = [f"Data Source: {product.source}", "", textwrap.fill(product.note, 40), "", issued, "" if status_line else None, status_line]
    sp.add_branding(fig, ax, title=product.title, subtitle=valid,
                    description="\n".join(line for line in lines if line is not None),
                    update_schedule="Redrawn whenever WPC updates the index")
    status = {"product": f"winter_{product.key}", "status": "updated", "valid": valid, "issued": issued,
              "levels_over_missouri": [entries[level][0] for level in sorted(entries)],
              "last_update": datetime.now(CENTRAL).strftime("%Y-%m-%d %H:%M CT"), "runtime_sec": round(time.time() - started, 2)}
    sp.save_and_record_status(fig, out_png, status_path, status)
    return status


# --- fetching, change detection, upload -----------------------------------------------------------------------------------
def source_url(product: Product) -> str:
    return f"{NDFD}/ds.snow.bin" if product.kind == "ndfd_snow" else f"{NDFD}/ds.iceaccum.bin" if product.kind == "ndfd_ice" else product.url


def refresh(products=PRODUCTS, force: bool = False, fetch_fn=fetch_url, images_dir: Path | None = None,
            status_dir: Path | None = None) -> tuple[list[Path], list[str]]:
    """Redraw each product whose source changed. Returns (PNGs written, keys that failed)."""
    images_dir, status_dir = images_dir or IMAGES_DIR, status_dir or STATUS_DIR
    status_dir.mkdir(parents=True, exist_ok=True)
    changed, failed = [], []
    for product in products:
        out_png = images_dir / f"{FILE_PREFIX}-{product.key}.png"
        sig_path = status_dir / f"winter_{product.key}.signature"
        try:
            payload = fetch_fn(source_url(product))
            digest = signature(payload)
            try:
                if not force and out_png.is_file() and sig_path.read_text() == digest:
                    print(f"{product.key}: unchanged; skipping.")
                    continue
            except OSError:
                pass
            draw = draw_ndfd if product.kind.startswith("ndfd") else draw_wssi
            status = draw(product, payload, out_png, status_dir / f"winter_{product.key}.json")
            sig_path.write_text(digest)
            changed.append(out_png)
            print(f"{product.key}: updated ({status['valid']}, {status['runtime_sec']:.1f}s)")
        except Exception as error:  # noqa: BLE001 - one source being down must not stop the others
            failed.append(product.key)
            print(f"{product.key}: FAILED ({type(error).__name__}: {str(error)[:160]})")
    return changed, failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true", help="redraw even if the source has not changed")
    parser.add_argument("--only", action="append", choices=sorted(BY_KEY), help="just this product (repeatable)")
    parser.add_argument("--no-upload", action="store_true", help="draw only; skip the CDN upload")
    args = parser.parse_args()
    products = tuple(BY_KEY[key] for key in args.only) if args.only else PRODUCTS
    changed, failed = refresh(products, force=args.force)
    if changed and not args.no_upload:
        try:
            from scripts.upload_cdn import upload_to_cdn
            upload_to_cdn(changed, [f"latest/{path.name}" for path in changed], ["image/png"] * len(changed),
                          ["max-age=300"] * len(changed))
        except Exception as exc:  # noqa: BLE001 - maps.sh's own upload pass will also try
            print(f"Winter map CDN upload failed: {exc}")
    if failed and len(failed) == len(products):
        sys.exit(1)             # nothing could be fetched at all: let maps.sh log the failure


if __name__ == "__main__":
    main()
