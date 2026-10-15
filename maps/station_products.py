"""Render helpers for scripts that produce several station-based maps in one process.

Each maps/*.py script normally renders one PNG, paying ~4s of imports
(cartopy/geopandas/scipy) plus its own basemap work every maps.sh cycle. The
pressure/wind/station-plot scripts instead render a whole family of products
here in a single process: one import, one station fetch, one set of history
queries, and the cached boundaries/fill grid in station_plot reused by every
map. A failure in one product never stops the rest (see `run_products`).
"""
from __future__ import annotations

import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from maps import station_plot as sp  # noqa: E402

CENTRAL = ZoneInfo("America/Chicago")
STATUS_DIR = PROJECT_DIR / "data" / "current_conditions" / "status"
IMAGES_DIR = PROJECT_DIR / "images"
DEFAULT_SOURCE = "Data Source: Official Sites Only from Synoptic Weather\nAWOS, RAWS, Missouri Mesonet"


def now_central() -> datetime:
    return datetime.now(CENTRAL)


def valid_time_label(when: Optional[datetime] = None) -> str:
    return (when or now_central()).strftime("%Y-%m-%d %H:%M CT")


def obs_value(station: dict, name: str) -> Any:
    return ((station.get("observations") or {}).get(name) or {}).get("value")


def set_value(station: dict, name: str, value: Any) -> None:
    """Attach a derived value as observations.<name>.value, the shape the
    station_plot field paths read."""
    station.setdefault("observations", {})[name] = {"value": value}


def set_values(stations: list[dict], name: str, values: dict[str, float], *, default: Any = None) -> int:
    """Attach a stid-keyed dict of values to every station; returns how many
    stations received a value."""
    count = 0
    for station in stations:
        value = values.get(station.get("stid"), default)
        set_value(station, name, value)
        count += value is not None
    return count


def field_path(name: str) -> str:
    return f"observations.{name}.value"


def render_map(
    stations: list[dict], *, product: str, png_name: str, title: str, description: str,
    field: Optional[str] = None, fill: Optional[dict] = None, formatter: Callable[[Any], str] = lambda v: f"{v:.0f}",
    label_field: Optional[str] = None, overlay: Optional[Callable] = None,
    status_note: str = "", source: str = DEFAULT_SOURCE, ok: bool = True,
    min_label_dist: float = sp.LABEL_DECLUTTER_MIN_DIST, extra_status: Optional[dict] = None,
) -> dict:
    """One station map: basemap, optional interpolated fill of `field`
    (`fill` holds plot_filled_field's color/range kwargs), value labels,
    optional overlay callable(fig, ax, data_crs, stations), branding, atomic PNG
    save and a per-product status JSON. Returns the status payload."""
    started = time.time()
    fig, ax, data_crs, _map_crs, _dpi = sp.generate_basemap()
    fill_count = 0
    if field and fill:
        fill_count = sp.plot_filled_field(fig, ax, data_crs, stations, field_path=field_path(field), **fill)
    plotted = 0
    labels = label_field or field
    if labels and formatter is not None:
        plotted = sp.plot_station_values(
            ax, data_crs, stations, field_path=field_path(labels), formatter=formatter, min_label_dist=min_label_dist,
        )
    if overlay is not None:
        # An overlay may return how many stations it drew (for the status/log).
        drawn = overlay(fig, ax, data_crs, stations)
        if isinstance(drawn, int):
            plotted = drawn

    sp.add_branding(
        fig, ax, title=title, subtitle=f"Valid Time: {valid_time_label()}",
        description=f"{source}\n\n{description}{status_note}",
    )
    runtime = time.time() - started
    status = {
        "product": product,
        "status": "updated" if ok else "insufficient_history",
        "last_update": valid_time_label(),
        "runtime_sec": round(runtime, 2),
        "stations_plotted": plotted,
        "stations_filled": fill_count,
        **(extra_status or {}),
    }
    sp.save_and_record_status(fig, IMAGES_DIR / png_name, STATUS_DIR / f"{product}.json", status)
    print(f"{title} updated ({plotted} stations, {runtime:.2f}s)")
    return status


def plot_labels(
    ax, data_crs, stations: list[dict], text_fn: Callable[[dict], Optional[str]], *,
    min_label_dist: float = sp.LABEL_DECLUTTER_MIN_DIST, color: str = "black", fontsize: int = 9, dot: bool = True,
    dy: float = 0.03,
) -> int:
    """Dot + a custom text label per station (for values that combine several
    observations, e.g. "18G32"), decluttered like plot_station_values."""
    placed: list[np.ndarray] = []
    plotted = 0
    for station in stations:
        text = text_fn(station)
        lon, lat = station.get("longitude"), station.get("latitude")
        if text is None or lon is None or lat is None:
            continue
        pos = np.array([lon, lat])
        if any(np.linalg.norm(pos - other) <= min_label_dist for other in placed):
            continue
        placed.append(pos)
        if dot:
            ax.scatter(lon, lat, transform=data_crs, color=color, s=20, zorder=10, edgecolor="white", linewidth=1)
        ax.text(
            lon, lat + dy, text, transform=data_crs, fontsize=fontsize, color=color, zorder=11,
            ha="center", va="bottom", path_effects=[sp.path_effects.withStroke(linewidth=2, foreground="white")],
        )
        plotted += 1
    return plotted


def plot_wind_arrows(
    ax, data_crs, stations: list[dict], *, speed_name: str = "wind_speed", dir_name: str = "wind_direction",
    color: str = "black",
) -> int:
    """Fixed-length arrows pointing the way the wind blows (direction only;
    the speed is carried by the fill and labels). Meteorological directions
    are "from", so the vector points the opposite way."""
    lons, lats, us, vs = [], [], [], []
    for station in stations:
        speed, direction = obs_value(station, speed_name), obs_value(station, dir_name)
        lon, lat = station.get("longitude"), station.get("latitude")
        if None in (speed, direction, lon, lat) or speed < 1:
            continue
        lons.append(lon)
        lats.append(lat)
        us.append(-np.sin(np.radians(direction)))
        vs.append(-np.cos(np.radians(direction)))
    if not lons:
        return 0
    # Quiver in the map's own projected coordinates would rotate with the
    # projection; pass lon/lat + transform so arrows point true compass
    # directions on the Lambert map.
    ax.quiver(
        np.array(lons), np.array(lats), np.array(us), np.array(vs), transform=data_crs,
        color=color, scale=38, width=0.0016, headwidth=4.5, headlength=5, pivot="mid", zorder=10,
        regrid_shape=None,
    )
    return len(lons)


def run_products(products: list[tuple[str, Callable[[], Any]]], *, label: str) -> None:
    """Run each (name, callable); log + continue on failure, exit non-zero at
    the end if any failed so maps.sh still reports the script as failed."""
    started = time.time()
    failed = []
    for name, build in products:
        try:
            build()
        except Exception:  # noqa: BLE001 - one bad product must not block the others
            failed.append(name)
            print(f"[ERROR] {name} failed:")
            traceback.print_exc()
    print(f"{label}: {len(products) - len(failed)}/{len(products)} maps updated in {time.time() - started:.2f}s")
    if failed:
        print("Failed: " + ", ".join(failed))
        sys.exit(1)
