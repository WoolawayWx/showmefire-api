"""Hourly HRRR fire-weather maps: mixing height, transport wind, ventilation
(smoke dispersal), surface inversions, low-level lapse rate, Haines Index and
Hot-Dry-Windy Index, plus a 48-hour trend chart.

Each map is valid for the current clock hour, from the previous hour's HRRR
run at forecast hour 1 (see services/fire_weather/hrrr_profiles.py for why).
maps.sh runs this every 15 minutes, but each valid hour is processed exactly
once: a ledger of finished hours (services/fire_weather/history.py) makes the
usual run a couple of cheap HEAD requests. When a new hour is published it is
downloaded (~100 MB of byte-range GRIB subsets), archived (hourly JSON record +
compressed grids) and mapped; hours missed while the job was paused (forecast
lock, outages) are backfilled into the archive, oldest first.
Metric definitions and their sources are in services/fire_weather/indices.py.
"""
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.patches import Patch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from maps import raster_maps as rm
from maps import station_plot as sp
from maps import station_products as spx
from services.fire_weather import history
from services.fire_weather import hrrr_profiles as hp
from services.fire_weather import indices as ix
from services.fire_weather import summary as fs

SOURCE = "Data Source: NOAA HRRR (3 km) 1-hour forecast,\nrefreshed hourly from pressure-level profiles"
STATUS_PATH = spx.STATUS_DIR / "fire_weather_maps.json"
ARROW_STRIDE = 9  # grid cells (~27 km) between transport-wind arrows

HAINES_CLASSES = [
    (2, "2  Very Low", "#1a9850"), (3, "3  Very Low", "#91cf60"), (4, "4  Low", "#fee08b"),
    (5, "5  Moderate", "#fc8d59"), (6, "6  High", "#d73027"),
]
LAPSE_CMAP = LinearSegmentedColormap.from_list(
    "lapse", ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c", "#7f0000"])


def _valid_label(valid: datetime) -> str:
    return "Valid Time: " + spx.valid_time_label(valid.astimezone(spx.CENTRAL))


def _mesh(ax, fields, values, **kwargs):
    drawn = ax.pcolormesh(fields["lon"], fields["lat"], np.ma.masked_invalid(values), transform=ccrs.PlateCarree(),
                          shading="nearest", zorder=7, **kwargs)
    rm.clip_to_state(ax, drawn)
    return drawn


def _save(fig, ax, *, key: str, png: str, title: str, valid: datetime, description: str, started: float, **extra) -> dict:
    sp.add_branding(fig, ax, title=title, subtitle=_valid_label(valid), description=f"{SOURCE}\n\n{description}",
                    update_schedule="Updates hourly")
    status = {"product": f"fire_weather_{key}", "status": "updated", "valid_time": valid.strftime("%Y-%m-%dT%H:00:00Z"),
              "last_update": spx.valid_time_label(), "runtime_sec": round(time.time() - started, 2), **extra}
    sp.save_and_record_status(fig, spx.IMAGES_DIR / png, spx.STATUS_DIR / f"fire_weather_{key}.json", status)
    print(f"{title} updated ({status['runtime_sec']:.2f}s)")
    return status


def continuous_map(fields, valid, *, key, png, title, field, cmap, vmin, vmax, label, ticks, description,
                   norm=None, extend="both", overlay=None):
    started = time.time()
    fig, ax, data_crs, _map_crs, _dpi = sp.generate_basemap()
    colors = rm.blend_cmap(cmap)
    norm = norm or Normalize(vmin=vmin, vmax=vmax, clip=False)
    _mesh(ax, fields, fields[field], cmap=colors, norm=norm)
    rm.left_colorbar(fig, colors, norm, label, ticks=ticks, extend=extend)
    if overlay:
        overlay(ax, data_crs)
    return _save(fig, ax, key=key, png=png, title=title, valid=valid, description=description, started=started)


def category_map(fields, valid, *, key, png, title, values, classes, legend_title, description):
    """`classes` is [(label, color)] for values 0..n-1."""
    started = time.time()
    fig, ax, _data_crs, _map_crs, _dpi = sp.generate_basemap()
    colors = ListedColormap([color for _label, color in classes])
    norm = BoundaryNorm(np.arange(-0.5, len(classes)), len(classes))
    _mesh(ax, fields, np.where(values >= 0, values, np.nan), cmap=colors, norm=norm)
    rm.category_legend(fig, [Patch(facecolor=color, edgecolor="#555555", linewidth=0.4, label=label)
                             for label, color in classes], legend_title, y=0.36, fontsize=12)
    return _save(fig, ax, key=key, png=png, title=title, valid=valid, description=description, started=started)


def _transport_arrows(fields, in_state):
    """Pseudo-stations on a thinned grid so station_products.plot_wind_arrows can draw them."""
    points = []
    for i in range(ARROW_STRIDE // 2, fields["lat"].shape[0], ARROW_STRIDE):
        for j in range(ARROW_STRIDE // 2, fields["lat"].shape[1], ARROW_STRIDE):
            if in_state[i, j] and np.isfinite(fields["transport_dir_deg"][i, j]):
                spd = float(fields["transport_speed_mph"][i, j])
                points.append({"latitude": float(fields["lat"][i, j]), "longitude": float(fields["lon"][i, j]),
                               "observations": {"wind_speed": {"value": spd},
                                                "wind_direction": {"value": float(fields["transport_dir_deg"][i, j])}}})

    def overlay(ax, data_crs):
        spx.plot_wind_arrows(ax, data_crs, points)

    return overlay


def render_maps(fields, valid, summary, in_state) -> list:
    state = summary["statewide"]
    vent_classes = [
        (f"{name}  ({'<40' if index == 0 else (f'{low:.0f}+' if index == len(ix.VENTILATION_CATEGORIES) - 1 else f'{low:.0f}-{ix.VENTILATION_CATEGORIES[index + 1][1]:.0f}')})", color)
        for index, (name, low, color) in enumerate(ix.VENTILATION_CATEGORIES)
    ]
    return [
        ("Mixing height", lambda: continuous_map(
            fields, valid, key="mixing_height", png="mo-fw-mixing-height.png", title="Missouri Mixing Height",
            field="mixing_height_ft", cmap="YlGnBu", vmin=0, vmax=8000, label="Mixing height (ft above ground)",
            ticks=range(0, 8001, 1000), extend="max",
            description=("Depth of the turbulent mixed layer that smoke\nspreads through (HRRR boundary-layer height).\n"
                         "Low values trap smoke near the ground.\n"
                         f"Statewide median: {state['mixing_height_ft']['median']:,} ft"))),
        ("Transport wind", lambda: continuous_map(
            fields, valid, key="transport_wind", png="mo-fw-transport-wind.png", title="Missouri Transport Winds",
            field="transport_speed_mph", cmap="YlOrRd", vmin=0, vmax=40, label="Transport wind (mph)",
            ticks=range(0, 41, 5), extend="max", overlay=_transport_arrows(fields, in_state),
            description=("Average wind through the mixed layer: the\nspeed and direction smoke is carried away.\n"
                         f"Statewide median: {state['transport_speed_mph']['median']} mph"
                         + (f" from the {state['transport_direction']}" if state["transport_direction"] else "")))),
        ("Ventilation", lambda: category_map(
            fields, valid, key="ventilation", png="mo-fw-ventilation.png", title="Missouri Smoke Dispersal",
            values=fields["ventilation_category"], classes=vent_classes,
            legend_title="Ventilation rate (x1000 kt-ft)",
            description=("Ventilation rate = transport wind (kt) x mixing\nheight (ft), in NWS Springfield's smoke\n"
                         "dispersal categories. Higher is better\ndispersal of smoke."))),
        ("Surface inversion", lambda: continuous_map(
            fields, valid, key="inversion", png="mo-fw-inversion.png", title="Missouri Surface Inversions",
            field="inversion_strength_f", cmap="PuBu", vmin=0, vmax=10, label="Surface inversion strength (°F)",
            ticks=range(0, 11, 2), extend="max",
            description=("How much warmer the air gets above the ground\n(0 = no surface-based inversion). Inversions\n"
                         "trap smoke and keep it low overnight and in\nthe morning. "
                         f"Inversion over {state['surface_inversion_percent']}% of Missouri."))),
        ("Lapse rate", lambda: continuous_map(
            fields, valid, key="lapse_rate", png="mo-fw-lapse-rate.png", title="Missouri Low-Level Lapse Rate",
            field="lapse_rate_c_km", cmap=LAPSE_CMAP, vmin=-6, vmax=12, label="0-1 km lapse rate (°C/km)",
            ticks=[-6, -3, 0, 3, 6, 9.8, 12],
            description=("Temperature drop over the lowest 1 km. Near\n9.8 °C/km (dry adiabatic) the air mixes freely;\n"
                         "below zero is an inversion that suppresses\nmixing and smoke lift."))),
        ("Haines", lambda: category_map(
            fields, valid, key="haines", png="mo-fw-haines.png", title="Missouri Haines Index",
            values=np.where(np.isfinite(fields["haines"]), fields["haines"] - 2, -1),
            classes=[(label, color) for _value, label, color in HAINES_CLASSES], legend_title="Haines (low elevation)",
            description=("Lower-atmosphere stability (950-850 mb) plus\ndryness (850 mb) for plume-dominated fire\n"
                         "growth (Haines 1988). Supplemental only:\nits skill is debated (Potter 2018)."))),
        ("HDW", lambda: continuous_map(
            fields, valid, key="hdw", png="mo-fw-hdw.png", title="Missouri Hot-Dry-Windy Index",
            field="hdw", cmap="YlOrRd", vmin=0, vmax=300, label="Hot-Dry-Windy Index (hPa m/s)",
            ticks=range(0, 301, 50), extend="max",
            description=("Max vapor pressure deficit x max wind in the\nlowest ~50 hPa (Srock et al. 2018). Raw\n"
                         "values; no local climatology yet, so compare\nagainst recent days rather than a threshold."))),
    ]


def render_trend(now: datetime) -> dict:
    """48-hour trend of statewide (bold) and regional medians."""
    started = time.time()
    records = history.read_records(48, now)
    by_time = {record["valid_time"]: record for record in records}
    end = now.replace(minute=0, second=0, microsecond=0)
    hours = [end - timedelta(hours=back) for back in range(48, -1, -1)]
    # One slot per hour; a missing hour stays empty so the lines break there
    # instead of drawing a straight line across the gap.
    slots = [by_time.get(hour.strftime("%Y-%m-%dT%H:00:00Z"), {"statewide": {}, "regions": {}}) for hour in hours]
    sp._ensure_fonts_loaded()
    fig = plt.figure(figsize=(sp.PIXEL_WIDTH / sp.MAP_DPI, sp.PIXEL_HEIGHT / sp.MAP_DPI), dpi=sp.MAP_DPI,
                     facecolor=sp.MAP_BACKGROUND)
    panels = [
        ("mixing_height_ft", "Mixing height (ft)", 1.0),
        ("transport_speed_mph", "Transport wind (mph)", 1.0),
        ("ventilation_kt_ft", "Ventilation (x1000 kt-ft)", 0.001),
    ]
    axes = [fig.add_axes([0.07, 0.70 - index * 0.29, 0.58, 0.24]) for index in range(3)]
    times = [hour.astimezone(spx.CENTRAL) for hour in hours]
    region_colors = {"Northwest": "#1b9e77", "Northeast": "#d95f02", "Central": "#7570b3",
                     "Southwest": "#e7298a", "Southeast": "#66a61e"}
    for ax, (key, label, scale) in zip(axes, panels):
        ax.set_facecolor("white")
        if key == "ventilation_kt_ft":
            edges = [low for _name, low, _color in ix.VENTILATION_CATEGORIES] + [400]
            for (name, low, color), high in zip(ix.VENTILATION_CATEGORIES, edges[1:]):
                ax.axhspan(low, high, color=color, alpha=0.18, zorder=0)
                ax.text(1.005, (low + min(high, 250)) / 2, name, transform=ax.get_yaxis_transform(), fontsize=8,
                        va="center", color="#444444")
            ax.set_ylim(0, 250)
        for region, color in region_colors.items():
            values = [_median(record["regions"].get(region, {}), key, scale) for record in slots]
            ax.plot(times, values, color=color, linewidth=1.2, marker="o", markersize=2.5, label=region)
        ax.plot(times, [_median(record["statewide"], key, scale) for record in slots], color="black",
                linewidth=2.6, marker="o", markersize=3.5, label="Statewide")
        ax.set_ylabel(label, fontsize=10)
        ax.set_xlim(times[0], times[-1])
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %-I %p", tz=spx.CENTRAL))
        if key != "ventilation_kt_ft":
            ax.set_ylim(bottom=0)
    axes[0].legend(loc="upper left", ncol=6, fontsize=8, frameon=False, bbox_to_anchor=(0, 1.22))
    sp.add_branding(fig, axes[-1], title="Missouri Dispersion Trend",
                    subtitle=f"Last 48 hours through {spx.valid_time_label(now.astimezone(spx.CENTRAL))}",
                    description=(f"{SOURCE}\n\nHourly median mixing height, transport\nwind and ventilation, statewide (black)\n"
                                 "and by region. Shading shows NWS Springfield\nsmoke-dispersal categories."),
                    update_schedule="Updates hourly")
    status = {"product": "fire_weather_trend", "status": "updated", "hours": len(records),
              "last_update": spx.valid_time_label(), "runtime_sec": round(time.time() - started, 2)}
    sp.save_and_record_status(fig, spx.IMAGES_DIR / "mo-fw-trend.png", spx.STATUS_DIR / "fire_weather_trend.json", status)
    print(f"Dispersion trend updated ({len(records)} hours)")
    return status


def _median(summary: dict, key: str, scale: float):
    value = (summary.get(key) or {}).get("median")
    return np.nan if value is None else value * scale


def process_hour(valid: datetime, stations: list, draw_maps: bool) -> list:
    started = time.time()
    profiles = hp.fetch_profiles(valid)
    fields = hp.derive_fields(profiles)
    in_state = fs.missouri_mask(fields["lat"], fields["lon"])
    summary = fs.summarize(fields, valid, in_state)
    history.write_record(valid, summary, fs.station_values(fields, stations))
    history.write_grids(valid, fields)
    history.mark_processed(hp.hour_key(valid))
    print(f"HRRR valid {valid:%Y-%m-%d %H}z archived ({time.time() - started:.1f}s)")
    if not draw_maps:
        return []
    fs.write_latest(summary)
    return render_maps(fields, valid, summary, in_state)


def main() -> None:
    available = hp.available_hours(6)
    todo = history.select_hours(available, history.load_ledger())
    if not todo:
        print("Fire-weather maps: no new HRRR hour; skipping.")
        return
    newest = available[-1]
    try:
        stations = sp.fetch_stations()
    except Exception as exc:  # noqa: BLE001 - station values are a nice-to-have in the archive
        print(f"Station list unavailable ({exc}); archiving grid summaries only.")
        stations = []
    products = []
    for valid in todo:
        products += process_hour(valid, stations, draw_maps=valid == newest)
    now = datetime.now(timezone.utc)
    products.append(("Dispersion trend", lambda: render_trend(now)))
    history.prune(now)
    missing = [c for c in available if hp.hour_key(c) not in history.load_ledger()]
    fs.atomic_write_json(STATUS_PATH, {
        "product": "fire_weather_maps", "last_update": spx.valid_time_label(),
        "processed": [hp.hour_key(c) for c in todo], "newest_available": hp.hour_key(newest),
        "hours_behind": len(missing),
    })
    spx.run_products(products, label="Fire-weather maps")


if __name__ == "__main__":
    main()
