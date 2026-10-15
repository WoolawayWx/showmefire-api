"""Wind maps rendered together in one process (see maps/station_products.py):

  * Wind Speed & Gusts, with Arrows / with Barbs
  * Wind Gusts, 1-hour Maximum Gusts, Today's and Yesterday's Maximum Gusts
  * Mass Convergence
  * 2-meter Wind Speed
  * Dispersion Conditions and Winds

("Wind Speed and Direction" is maps/windmap-fil.py's mo-windfilmap.png.)
"""
import sys
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from maps import station_plot as sp
from maps import station_products as spx
from maps.moisture_convergence import compute_moisture_convergence
from maps.station_history import get_extremes_for_date_batch, get_rolling_extremes_batch, get_today_extremes_batch
from services import derived_met

SPEED_FILL = dict(cmap="RdYlGn_r", vmin=0, vmax=40)
GUST_FILL = dict(cmap="RdYlGn_r", vmin=0, vmax=60)
MAX_SPEED_MPH, MAX_GUST_MPH = 120.0, 150.0
# Sensor heights (m) when a station doesn't report one: ASOS/AWOS and the
# Mesonet at 10 m, RAWS at the standard 20 ft, MoCoMagNet at 3 m.
DEFAULT_WIND_HEIGHT_M = {"ASOS/AWOS": 10.0, "Missouri Mesonet": 10.0, "RAWS": 6.1, "MOCOMAGNET": 3.0}
DISPERSION_LEVELS = [0.0, 3.75, 4.5, 7.0]
DISPERSION_COLORS = ["#2E7D32", "#F9A825", "#C62828"]
CONVERGENCE_SCALE = 1e-5  # s^-1 per displayed unit


def sensor_height_m(station: dict) -> float:
    reported = ((station.get("sensors") or {}).get("wind_speed") or {}).get("height_m")
    return float(reported) if reported else DEFAULT_WIND_HEIGHT_M.get(station.get("network"), 10.0)


def prepare(stations: list[dict]) -> dict:
    """Attach every derived wind field the maps use; returns bookkeeping."""
    now_utc = spx.now_central().astimezone(timezone.utc)
    yesterday = (spx.now_central() - timedelta(days=1)).strftime("%Y-%m-%d")
    rolling_1h = get_rolling_extremes_batch("wind_gust_mph", 1)
    today = get_today_extremes_batch("wind_gust_mph")
    previous_day = get_extremes_for_date_batch("wind_gust_mph", yesterday)

    for station in stations:
        stid = station.get("stid")
        speed = spx.obs_value(station, "wind_speed")
        gust = spx.obs_value(station, "wind_gust")
        if speed is not None and not 0 <= speed <= MAX_SPEED_MPH:
            speed = None
            spx.set_value(station, "wind_speed", None)
        if gust is not None and not 0 <= gust <= MAX_GUST_MPH:
            gust = None
            spx.set_value(station, "wind_gust", None)

        def max_with_current(recorded):
            candidates = [v for v in (recorded, gust) if v is not None and 0 <= v <= MAX_GUST_MPH]
            return max(candidates) if candidates else None

        spx.set_value(station, "gust_max_1hr", max_with_current((rolling_1h.get(stid) or {}).get("max")))
        spx.set_value(station, "gust_max_today", max_with_current((today.get(stid) or {}).get("max")))
        yesterday_max = (previous_day.get(stid) or {}).get("max")
        spx.set_value(station, "gust_max_yesterday", yesterday_max if yesterday_max is not None and 0 <= yesterday_max <= MAX_GUST_MPH else None)

        height = sensor_height_m(station)
        spx.set_value(station, "wind_speed_2m", None if speed is None else float(derived_met.wind_at_height(speed, height, 2.0)))

        stability = None
        lat, lon = station.get("latitude"), station.get("longitude")
        if speed is not None and lat is not None and lon is not None:
            wind_10m_ms = float(derived_met.mph_to_ms(derived_met.wind_at_height(speed, height, 10.0)))
            stability = derived_met.pasquill_stability_index(wind_10m_ms, derived_met.solar_elevation_deg(lat, lon, now_utc))
        spx.set_value(station, "stability_index", stability)

    return {
        "history_1h": bool(rolling_1h), "history_today": bool(today), "history_yesterday": bool(previous_day),
    }


def speed_gust_text(station: dict):
    speed, gust = spx.obs_value(station, "wind_speed"), spx.obs_value(station, "wind_gust")
    if speed is None:
        return None
    return f"{speed:.0f}G{gust:.0f}" if gust is not None and gust >= speed + 3 else f"{speed:.0f}"


def mass_convergence_overlay(fig, ax, data_crs, stations):
    """Interpolate station winds to the fill grid and draw -div(V)
    (positive = converging air, where lift/storm initiation is favored)."""
    points = [
        (s["longitude"], s["latitude"], spx.obs_value(s, "wind_speed"), spx.obs_value(s, "wind_direction"))
        for s in stations if None not in (s.get("longitude"), s.get("latitude"), spx.obs_value(s, "wind_speed"), spx.obs_value(s, "wind_direction"))
    ]
    if len(points) < 4:
        return
    lons, lats, speeds, directions = (np.array(column, dtype=float) for column in zip(*points))
    u = derived_met.mph_to_ms(-speeds * np.sin(np.radians(directions)))
    v = derived_met.mph_to_ms(-speeds * np.cos(np.radians(directions)))
    # Lighter blur than the scalar fills (sigma 4): divergence is a derivative,
    # so over-smoothing would wash out the very gradients it measures.
    u_grid = sp.interpolate_to_grid(lons, lats, u, ax, data_crs, sigma=2)
    v_grid = sp.interpolate_to_grid(lons, lats, v, ax, data_crs, sigma=2)
    lon_mesh, lat_mesh, *_ = sp._fill_grid(sp.FILL_GRID_SIZE, ax.projection, data_crs)
    convergence = compute_moisture_convergence(u_grid, v_grid, np.ones_like(u_grid), lon_mesh, lat_mesh) / CONVERGENCE_SCALE
    sp.draw_filled_grid(
        fig, ax, data_crs, convergence, cmap="BrBG", diverging=True, vmax=8,
        label="Mass Convergence (1e-5 /s)",
    )
    sp.plot_wind_barbs(ax, data_crs, stations)


def dispersion_overlay(fig, ax, data_crs, stations):
    points = [
        (s["longitude"], s["latitude"], spx.obs_value(s, "stability_index"))
        for s in stations if None not in (s.get("longitude"), s.get("latitude"), spx.obs_value(s, "stability_index"))
    ]
    if len(points) >= 4:
        lons, lats, values = (np.array(column, dtype=float) for column in zip(*points))
        sp.draw_filled_grid(
            fig, ax, data_crs, sp.interpolate_to_grid(lons, lats, values, ax, data_crs, sigma=3),
            levels=DISPERSION_LEVELS, colors=DISPERSION_COLORS, tick_labels=["Good", "Fair", "Poor"],
            label="Dispersion (estimated)",
        )
    sp.plot_wind_barbs(ax, data_crs, stations)


def main() -> None:
    stations = sp.fetch_stations()
    info = prepare(stations)
    gust_note = lambda ok: "" if ok else "\n\nAccumulating history - full coverage after a day of recording."  # noqa: E731

    def with_barbs(fig, ax, data_crs, stations):
        sp.plot_wind_barbs(ax, data_crs, stations)

    def with_arrows(fig, ax, data_crs, stations):
        spx.plot_wind_arrows(ax, data_crs, stations)

    def speed_gust_map(kind, overlay):
        return lambda: spx.render_map(
            stations, product=f"wind_speed_gusts_{kind}", png_name=f"mo-wind-speed-gusts-{kind}.png",
            title=f"Wind Speed & Gusts, with {kind.capitalize()}", field="wind_speed", fill=dict(**SPEED_FILL, label="Wind Speed (mph)"),
            formatter=None, overlay=lambda fig, ax, data_crs, st: (
                overlay(fig, ax, data_crs, st),
                spx.plot_labels(ax, data_crs, st, speed_gust_text, dy=0.07, dot=False),
            ),
            description=(
                "Sustained wind speed (mph) filled by color, with a label of speed\n"
                "and gust (e.g. 18G32).\n"
                + ("Arrows point the way the wind blows." if kind == "arrows" else "Barbs show direction; each full flag is 10 kt.")
            ),
        )

    def gust_map(product, png, title, field, description, ok=True):
        return lambda: spx.render_map(
            stations, product=product, png_name=png, title=title, field=field,
            fill=dict(**GUST_FILL, label="Wind Gust (mph)"), formatter=lambda v: f"{v:.0f}",
            description=description, ok=ok, status_note=gust_note(ok),
        )

    spx.run_products([
        ("speed & gusts arrows", speed_gust_map("arrows", with_arrows)),
        ("speed & gusts barbs", speed_gust_map("barbs", with_barbs)),
        ("gusts", gust_map("wind_gust", "mo-windgust.png", "Wind Gusts", "wind_gust",
                           "Current peak wind gust (mph) at each reporting station.")),
        ("1-hour max gusts", gust_map("wind_gust_max_1hr", "mo-windgust-1hr-max.png", "1-Hour Maximum Wind Gusts", "gust_max_1hr",
                                      "Highest wind gust (mph) recorded in the last hour.", info["history_1h"])),
        ("today's max gusts", gust_map("wind_gust_max_today", "mo-windgust-max-today.png", "Today's Maximum Wind Gusts", "gust_max_today",
                                       "Highest wind gust (mph) so far today (since midnight CT).", info["history_today"])),
        ("yesterday's max gusts", gust_map("wind_gust_max_yesterday", "mo-windgust-max-yesterday.png", "Yesterday's Maximum Wind Gusts", "gust_max_yesterday",
                                           "Highest wind gust (mph) during the previous day (CT).", info["history_yesterday"])),
        ("mass convergence", lambda: spx.render_map(
            stations, product="mass_convergence", png_name="mo-massconvergence.png", title="Mass Convergence",
            formatter=None, overlay=mass_convergence_overlay,
            description=(
                "Convergence of the surface wind field (-div V). Positive (green)\n"
                "values mark converging air, where lift and storm development are\n"
                "favored; negative (brown) values mark diverging air. Relative units."
            ),
        )),
        ("2-meter wind", lambda: spx.render_map(
            stations, product="wind_speed_2m", png_name="mo-windspeed-2m.png", title="2-Meter Wind Speed (Estimated)",
            field="wind_speed_2m", fill=dict(**SPEED_FILL, label="2 m Wind Speed (mph)"), formatter=lambda v: f"{v:.0f}",
            description=(
                "Wind speed (mph) adjusted to 2 meters above ground, the level\n"
                "that matters for grass and surface fire spread, from each\n"
                "sensor's own height with a logarithmic wind profile."
            ),
        )),
        ("dispersion & winds", lambda: spx.render_map(
            stations, product="dispersion_winds", png_name="mo-dispersion-winds.png", title="Dispersion Conditions and Winds (Estimated)",
            formatter=None, overlay=dispersion_overlay,
            description=(
                "Estimated smoke/air dispersion from wind speed and sun angle\n"
                "(Pasquill stability class, clear-sky assumption). Good = well\n"
                "mixed; Poor = light wind and a stable air layer, such as\n"
                "calm nights, when smoke lingers near the ground."
            ),
        )),
    ], label="Wind maps")


if __name__ == "__main__":
    main()
