"""Station rainfall maps, rendered together in one process:

  * Today's Rainfall (since midnight CT)
  * Consecutive Days with Less Than 0.10 inches / 0.25 inches Rainfall

The multi-period accumulation maps (1 hour ... 365 days) are NOAA QPE grids and
live in maps/precipitation_graphics.py. Daily station totals come from
maps/station_history.py (seeded by scripts/backfill_station_history.py, kept
current by the 5-minute recorder in core/scheduler.py).
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from maps import station_plot as sp
from maps import station_products as spx
from maps.station_history import get_daily_maxima_batch

HISTORY_FIELD = "precip_since_midnight_in"
MAX_DAILY_TOTAL_IN = 20.0  # a larger single-day gauge total is a bad reading
MAX_STREAK_DAYS = 45  # the backfill window; the scale tops out at 30+
STREAK_FILL = dict(cmap="YlOrBr", vmin=0, vmax=30, clip_min=0)
# Today's rainfall: discrete classes so a trace and a soaking are distinguishable.
RAIN_LEVELS = [0.0, 0.01, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 20.0]
RAIN_COLORS = ["#EFEFEF", "#C7E9C0", "#A1D99B", "#74C476", "#31A354", "#006D2C", "#6BAED6", "#2171B5", "#08306B", "#6A3D9A"]
RAIN_TICKS = ["0", "0.01-0.10", "0.10-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00", "1.00-1.50", "1.50-2.00", "2.00-3.00", "3+"]


def dry_streak(daily_totals: dict[str, float], today: str, threshold: float) -> int | None:
    """Consecutive days ending today with a total under `threshold`. None if
    today has no report. A missing earlier day ends the count (a lower bound,
    never an overcount), as does the first day at or above the threshold."""
    if today not in daily_totals:
        return None
    day = datetime.fromisoformat(today)
    streak = 0
    while streak < MAX_STREAK_DAYS:
        total = daily_totals.get(day.strftime("%Y-%m-%d"))
        if total is None or total >= threshold:
            break
        streak += 1
        day -= timedelta(days=1)
    return streak


def prepare(stations: list[dict]) -> dict:
    today = spx.now_central().strftime("%Y-%m-%d")
    daily = get_daily_maxima_batch(HISTORY_FIELD, MAX_STREAK_DAYS + 1)
    for station in stations:
        stid = station.get("stid")
        totals = {
            day: total for day, total in (daily.get(stid) or {}).items() if 0 <= total <= MAX_DAILY_TOTAL_IN
        }
        # The live reading can be ahead of the last 5-minute recording.
        current = spx.obs_value(station, "precip_accum_since_local_midnight")
        if current is not None and 0 <= current <= MAX_DAILY_TOTAL_IN:
            totals[today] = max(current, totals.get(today, 0.0))
            spx.set_value(station, "rain_today", totals[today])
        else:
            spx.set_value(station, "rain_today", None)
        for name, threshold in (("dry_days_010", 0.10), ("dry_days_025", 0.25)):
            spx.set_value(station, name, dry_streak(totals, today, threshold))
    return {"history": bool(daily)}


def main() -> None:
    stations = sp.fetch_stations()
    info = prepare(stations)
    note = "" if info["history"] else "\n\nAccumulating history - coverage grows as days are recorded."

    def streak_map(name, threshold, png):
        return lambda: spx.render_map(
            stations, product=f"dry_days_{png}", png_name=f"mo-dry-days-{png}.png",
            title=f"Consecutive Days Under {threshold:.2f} in Rainfall", field=name,
            fill=dict(**STREAK_FILL, label="Consecutive days"), formatter=lambda v: "30+" if v >= 30 else f"{v:.0f}",
            ok=info["history"], status_note=note,
            description=(
                f"Number of consecutive days, ending today, with less than\n"
                f"{threshold:.2f} inches of rain at each gauge. Today counts as a\n"
                "day, so any rain at or above the threshold resets it to 0.\n"
                "Counts top out at 30+ days; a day with no report ends the count."
            ),
        )

    spx.run_products([
        ("today's rainfall", lambda: spx.render_map(
            stations, product="rainfall_today", png_name="mo-rainfall-today.png", title="Today's Rainfall", field="rain_today",
            fill=dict(levels=RAIN_LEVELS, colors=RAIN_COLORS, tick_labels=RAIN_TICKS, label="Rainfall (in)", clip_min=0),
            formatter=lambda v: f"{v:.2f}",
            description="Rainfall (inches) since midnight CT at each gauge.\nShading is interpolated between gauges; the values\nare the measured totals.",
        )),
        ("dry days < 0.10", streak_map("dry_days_010", 0.10, "010")),
        ("dry days < 0.25", streak_map("dry_days_025", 0.25, "025")),
    ], label="Rainfall maps")


if __name__ == "__main__":
    main()
