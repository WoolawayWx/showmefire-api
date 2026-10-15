"""Sea-level pressure maps: current and 3-hour change, in inches of mercury and
millibars. All four render in one process (see maps/station_products.py)."""
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from maps import station_plot as sp
from maps import station_products as spx
from maps.station_history import get_changes_batch
from maps.station_rbf import buffer_stations
from services import derived_met

LOOKBACK_HOURS = 3
# Physically plausible sea-level pressure (mb) / 3-hour change (mb); anything
# outside is a bad sensor or bad reduction to sea level, not weather.
SLP_RANGE_MB = (940.0, 1070.0)
MAX_ABS_CHANGE_MB = 15.0
HPA_PER_INHG = derived_met.HPA_PER_INHG


def prepare(stations: list[dict]) -> tuple[dict, bool]:
    current: dict[str, float] = {}
    for station in stations:
        value = spx.obs_value(station, "sea_level_pressure")
        if value is not None and not SLP_RANGE_MB[0] <= value <= SLP_RANGE_MB[1]:
            value = None
        spx.set_value(station, "slp_mb", value)
        spx.set_value(station, "slp_inhg", None if value is None else float(derived_met.hpa_to_inhg(value)))
        if value is not None:
            current[station["stid"]] = value
    changes = {
        stid: change for stid, change in get_changes_batch("sea_level_pressure_mb", LOOKBACK_HOURS, current).items()
        if abs(change) <= MAX_ABS_CHANGE_MB
    }
    for station in stations:
        change = changes.get(station.get("stid"))
        spx.set_value(station, "slp_change_mb", change)
        spx.set_value(station, "slp_change_inhg", None if change is None else change / HPA_PER_INHG)
    return changes, bool(changes)


def anchor_points(buffer: list[dict], field: str) -> list[tuple[float, float, float]]:
    """(lon, lat, value) for neighbor-state stations that have `field`, so the
    fill is anchored at the border rather than extrapolated past it."""
    points = []
    for station in buffer:
        value = spx.obs_value(station, field)
        if value is not None and station.get("longitude") is not None and station.get("latitude") is not None:
            points.append((station["longitude"], station["latitude"], value))
    return points


def main() -> None:
    stations = sp.fetch_stations()
    # Neighbor-state stations anchor the interpolation at the border; markers and
    # labels stay Missouri-only. Their 3-hour history is recorded by the scheduler.
    buffer = buffer_stations()
    changes, have_history = prepare(stations + buffer)
    note = "" if have_history else "\n\nAccumulating history - full coverage 3 hours after the first recording."
    reduction = "Pressure reduced to sea level, so stations at different elevations are comparable."

    def pressure(unit, field, label, vmin, vmax, fmt, png):
        title = f"Sea-Level Pressure ({unit})"
        return lambda: spx.render_map(
            stations, product=f"pressure_{png}", png_name=f"mo-pressure-{png}.png", title=title,
            field=field, fill=dict(cmap="viridis", label=label, vmin=vmin, vmax=vmax, extra_points=anchor_points(buffer, field)),
            formatter=fmt,
            description=f"Sea-level pressure ({unit}) at each reporting station.\n{reduction}",
        )

    def change(unit, field, label, vmax, fmt, png):
        title = f"3-Hour Sea-Level Pressure Change ({unit})"
        return lambda: spx.render_map(
            stations, product=f"pressurechange_3hr_{png}", png_name=f"mo-pressurechange-3hr-{png}.png", title=title,
            field=field, fill=dict(cmap="PRGn", diverging=True, label=label, vmax=vmax, extra_points=anchor_points(buffer, field)),
            formatter=fmt,
            ok=have_history, status_note=note,
            description=(
                f"Change in sea-level pressure ({unit}) over the last 3 hours.\n"
                "Falling pressure often signals an approaching system;\nrapid rises follow a front."
            ),
        )

    spx.run_products([
        ("pressure inHg", pressure("inches of mercury", "slp_inhg", "Pressure (inHg)", 29.40, 30.60, lambda v: f"{v:.2f}", "inhg")),
        ("pressure change inHg", change("inches of mercury", "slp_change_inhg", "3-Hr Change (inHg)", 0.12, lambda v: f"{v:+.2f}", "inhg")),
        ("pressure mb", pressure("millibars", "slp_mb", "Pressure (mb)", 995, 1035, lambda v: f"{v:.1f}", "mb")),
        ("pressure change mb", change("millibars", "slp_change_mb", "3-Hr Change (mb)", 4, lambda v: f"{v:+.1f}", "mb")),
    ], label="Pressure maps")


if __name__ == "__main__":
    main()
