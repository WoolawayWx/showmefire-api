"""Backtest forecast_v1 Days 2-3 against station observations (read-only).

Answers "is the extended forecast good enough to show people?" from what the
database already holds:

* weather - MAE and bias of RH, wind, gust and temperature by forecast day,
  from forecast_verification (forecast minus observed, published values);
* fuel moisture - station_forecast_hours.fuel_moisture_p50 against RAWS
  10-hour fuel sticks in station_observations_v2, matched within 30 minutes;
* fire danger - each run's Day 2/3 peak category at RAWS stations
  (station_forecast_days) against the observed daily peak, where the observed
  category is the canonical rule (core.fire_danger.calculate_fire_danger) on
  hourly observed fuel, RH and wind. Baseline: persistence (the station's
  observed peak on the day before the run).

Everything is split by config_version: beta-1 runs used the run-total rain
bug, so their fuel and danger numbers are not representative of beta-2.

    python scripts/forecast_v1_backtest.py [--db data/showmefire.db] [--since 2026-10-23] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.fire_danger import MPS_TO_KNOTS, calculate_fire_danger  # noqa: E402

LOCAL = ZoneInfo("America/Chicago")
ELEVATED = 2
MIN_OBSERVED_HOURS = 12  # a daily peak needs most of the day observed
WEATHER_VARIABLES = ("relative_humidity_2m", "wind_speed_10m", "wind_gust_10m", "temperature_2m", "fuel_moisture_p50")
LEAD_DAY = {"0-24": "Day 1", "25-48": "Day 2", "49-72": "Day 3"}


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def weather_errors(connection: sqlite3.Connection, since: str) -> list[dict]:
    rows = connection.execute(
        f"""SELECT config_version, variable, lead_bucket, COUNT(*), AVG(ABS(error)), AVG(error),
                   COUNT(DISTINCT station_id)
            FROM forecast_verification
            WHERE qc_eligible = 1 AND error IS NOT NULL AND valid_time_utc >= ?
              AND variable IN ({",".join("?" for _ in WEATHER_VARIABLES)})
            GROUP BY config_version, variable, lead_bucket
            ORDER BY config_version, variable, lead_bucket""",
        (since, *WEATHER_VARIABLES),
    ).fetchall()
    return [
        {"config": config, "variable": variable, "day": LEAD_DAY.get(bucket, bucket), "pairs": pairs,
         "stations": stations, "mae": round(mae, 3), "bias": round(bias, 3)}
        for config, variable, bucket, pairs, mae, bias, stations in rows
    ]


def fuel_errors(connection: sqlite3.Connection, since: str) -> list[dict]:
    """Forecast vs RAWS fuel sticks for every run, including runs from before
    fuel moisture was added to forecast_verification."""
    observed: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    for station, when, value in connection.execute(
        """SELECT station_id, observed_at_utc, fuel_moisture FROM station_observations_v2
           WHERE fuel_moisture IS NOT NULL AND qc_state = 'pass' AND observed_at_utc >= ?""",
        (since,),
    ):
        observed[station].append((_utc(when), float(value)))
    for values in observed.values():
        values.sort()
    sums: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0])
    for config, station, valid, lead, forecast in connection.execute(
        """SELECT r.config_version, h.station_id, h.valid_time_utc, h.lead_hour, h.fuel_moisture_p50
           FROM station_forecast_hours h JOIN forecast_runs r ON r.run_id = h.run_id
           WHERE h.fuel_moisture_p50 IS NOT NULL AND h.valid_time_utc >= ?""",
        (since,),
    ):
        candidates = observed.get(station)
        if not candidates:
            continue
        target = _utc(valid)
        nearest = min(candidates, key=lambda item: abs((item[0] - target).total_seconds()))
        if abs((nearest[0] - target).total_seconds()) > 1800:
            continue
        error = float(forecast) - nearest[1]
        entry = sums[(config, "Day 1" if lead <= 24 else "Day 2" if lead <= 48 else "Day 3")]
        entry[0] += abs(error)
        entry[1] += error
        entry[2] += 1
    return [
        {"config": config, "day": day, "pairs": count, "mae": round(total_abs / count, 2), "bias": round(total / count, 2)}
        for (config, day), (total_abs, total, count) in sorted(sums.items())
    ]


def observed_daily_peaks(connection: sqlite3.Connection, since: str) -> dict[tuple[str, str], int]:
    """(station, local date) -> observed peak category at fuel-stick stations."""
    hourly: dict[tuple[str, datetime], dict[str, float]] = {}
    fuel_stations = {row[0] for row in connection.execute(
        "SELECT DISTINCT station_id FROM station_observations_v2 WHERE fuel_moisture IS NOT NULL AND observed_at_utc >= ?", (since,),
    )}
    for station, when, fuel, rh, wind in connection.execute(
        """SELECT station_id, observed_at_utc, fuel_moisture, relative_humidity, wind_speed_ms
           FROM station_observations_v2 WHERE qc_state = 'pass' AND observed_at_utc >= ?
           ORDER BY observed_at_utc""",
        (since,),
    ):
        if station not in fuel_stations:
            continue
        hour = _utc(when).replace(minute=0, second=0, microsecond=0)
        slot = hourly.setdefault((station, hour), {})
        for name, value in (("fuel", fuel), ("rh", rh), ("wind", wind)):
            if value is not None:
                slot[name] = float(value)  # latest reading in the hour wins
    peaks: dict[tuple[str, str], list[int]] = defaultdict(list)
    for (station, hour), values in hourly.items():
        if len(values) < 3:
            continue
        category = calculate_fire_danger(values["fuel"], values["rh"], values["wind"] * MPS_TO_KNOTS)
        if category is not None:
            peaks[(station, hour.astimezone(LOCAL).date().isoformat())].append(category)
    return {key: max(values) for key, values in peaks.items() if len(values) >= MIN_OBSERVED_HOURS}


def _scores(pairs: list[tuple[int, int]]) -> dict:
    hits = sum(1 for forecast, observed in pairs if forecast >= ELEVATED and observed >= ELEVATED)
    misses = sum(1 for forecast, observed in pairs if forecast < ELEVATED and observed >= ELEVATED)
    false_alarms = sum(1 for forecast, observed in pairs if forecast >= ELEVATED and observed < ELEVATED)
    count = len(pairs)
    ratio = lambda a, b: round(a / b, 3) if b else None  # noqa: E731
    return {
        "stationDays": count,
        "observedElevatedPlus": hits + misses,
        "forecastElevatedPlus": hits + false_alarms,
        "pod": ratio(hits, hits + misses),
        "far": ratio(false_alarms, hits + false_alarms),
        "csi": ratio(hits, hits + misses + false_alarms),
        "exact": ratio(sum(1 for forecast, observed in pairs if forecast == observed), count),
        "withinOne": ratio(sum(1 for forecast, observed in pairs if abs(forecast - observed) <= 1), count),
    }


def danger_skill(connection: sqlite3.Connection, since: str) -> list[dict]:
    observed = observed_daily_peaks(connection, since)
    forecast_pairs: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    persistence_pairs: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    for config, station, local_date, day_index, peak, cycle in connection.execute(
        """SELECT r.config_version, d.station_id, d.local_date, d.day_index, d.peak_category, r.cycle_time_utc
           FROM station_forecast_days d JOIN forecast_runs r ON r.run_id = d.run_id
           WHERE d.day_index IN (2, 3) AND d.peak_category IS NOT NULL AND d.local_date >= ?""",
        (since[:10],),
    ):
        actual = observed.get((station, local_date))
        if actual is None:
            continue
        forecast_pairs[(config, day_index)].append((int(peak), actual))
        before_run = (_utc(cycle).astimezone(LOCAL).date() - timedelta(days=1)).isoformat()
        previous = observed.get((station, before_run))
        if previous is not None:
            persistence_pairs[(config, day_index)].append((previous, actual))
    return [
        {"config": config, "day": f"Day {day}", "forecast": _scores(pairs), "persistence": _scores(persistence_pairs[(config, day)])}
        for (config, day), pairs in sorted(forecast_pairs.items())
    ]


def _table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join("" if value is None else str(value) for value in row) + " |" for row in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", help="SQLite database (default: the API's configured database)")
    parser.add_argument("--since", default="2026-09-01", help="UTC date, YYYY-MM-DD")
    parser.add_argument("--json", help="also write the results here")
    args = parser.parse_args()
    if args.db:
        database = Path(args.db)
    else:
        from core.database import get_db_path
        database = Path(get_db_path())
    since = f"{args.since}T00:00:00Z"
    # Read-only: never create tables or write to the production database.
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        results = {
            "database": str(database), "since": since,
            "weather": weather_errors(connection, since),
            "fuel": fuel_errors(connection, since),
            "danger": danger_skill(connection, since),
        }
    print(f"# forecast_v1 backtest since {args.since}\n")
    print("## Weather (forecast minus observed)\n")
    print(_table(["config", "variable", "day", "pairs", "stations", "MAE", "bias"],
                 [[r["config"], r["variable"], r["day"], r["pairs"], r["stations"], r["mae"], r["bias"]] for r in results["weather"]]))
    print("\n## 10-hour fuel moisture vs RAWS fuel sticks (%)\n")
    print(_table(["config", "day", "pairs", "MAE", "bias"],
                 [[r["config"], r["day"], r["pairs"], r["mae"], r["bias"]] for r in results["fuel"]]))
    print("\n## Peak fire danger at RAWS, Elevated or worse (forecast | persistence)\n")
    print(_table(
        ["config", "day", "station-days", "observed E+", "POD", "FAR", "CSI", "exact", "within 1"],
        [[r["config"], r["day"], r["forecast"]["stationDays"], r["forecast"]["observedElevatedPlus"],
          f'{r["forecast"]["pod"]} \\| {r["persistence"]["pod"]}', f'{r["forecast"]["far"]} \\| {r["persistence"]["far"]}',
          f'{r["forecast"]["csi"]} \\| {r["persistence"]["csi"]}', f'{r["forecast"]["exact"]} \\| {r["persistence"]["exact"]}',
          f'{r["forecast"]["withinOne"]} \\| {r["persistence"]["withinOne"]}'] for r in results["danger"]],
    ))
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
