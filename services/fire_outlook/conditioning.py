"""
Causal initial fuel-moisture state for an outlook cycle.

Wraps services/spread_rate_moisture.py (continuous RTMA replay + RAWS 10-hour anchoring) with
the guarantees the outlook needs:

  * Causality. Nothing valid after `cycle_time` is used. RAWS age is measured against the
    cycle time, not the wall clock (spread_rate_moisture.extract_raws_fuel_points uses now(),
    which would silently drop every observation, or admit future ones, in a backfill), and an
    observation without a timestamp is rejected because causality cannot be shown.
  * Honest quality. The result says whether the state is RAWS-anchored, conditioned but
    unanchored, or (only if the caller opts in) an EMC seed with no history. A live run
    defaults to strict: missing history raises ConditioningUnavailable rather than publishing
    a number that looks conditioned.
  * Observed vs modeled stay distinct: the returned fm10 is MODELED fuel moisture nudged toward
    RAWS; the observations themselves are never copied in as forecast values.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import numpy as np

from services.fire_outlook import moisture

MAX_RAW_AGE_MINUTES = 180
MIN_RAW_STATIONS = 3
LEVEL_ANCHORED = "conditioned_raws_anchored"
LEVEL_CONDITIONED = "conditioned"
LEVEL_EMC_SEED = "emc_seed"


class ConditioningUnavailable(RuntimeError):
    """Not enough causal history to condition fuel moisture, and the caller did not allow an EMC seed."""


def public_lat_lon() -> tuple[np.ndarray, np.ndarray]:
    """2-D (lat, lon) of the 267 x 264 public grid cell centres, shape (height, width)."""
    from pyproj import Transformer
    from forecast_v1.pipeline import public_coordinates
    x, y = public_coordinates()
    xx, yy = np.meshgrid(x, y)
    lon, lat = Transformer.from_crs(32615, 4326, always_xy=True).transform(xx, yy)
    return np.asarray(lat), np.asarray(lon)


def _aware(stamp: datetime) -> datetime:
    return stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp.astimezone(timezone.utc)


def causal_raws_points(raws_payload: Optional[dict], cycle_time: datetime, *, states: Optional[tuple] = ("MO",),
                       max_age_minutes: float = MAX_RAW_AGE_MINUTES) -> list[dict]:
    """Fresh 10-h fuel-moisture observations valid at or before cycle_time (never after)."""
    cycle = _aware(cycle_time)
    points = []
    for station in (raws_payload or {}).get("stations") or []:
        if states is not None and station.get("state") not in states:
            continue
        fuel = (station.get("observations") or {}).get("fuel_moisture")
        if not isinstance(fuel, dict) or fuel.get("value") is None or not fuel.get("time"):
            continue
        try:
            fm = float(fuel["value"])
            stamp = _aware(datetime.fromisoformat(str(fuel["time"]).replace("Z", "+00:00")))
            lat, lon = float(station["latitude"]), float(station["longitude"])
        except (TypeError, ValueError, KeyError):
            continue
        age = (cycle - stamp).total_seconds() / 60.0
        if age < 0 or age > max_age_minutes:  # future (non-causal) or stale
            continue
        if not (moisture.FM_MIN_PERCENT <= fm <= moisture.FM_MAX_PERCENT):
            continue
        points.append({"stid": station.get("stid"), "lat": lat, "lon": lon, "fm10": fm, "age_minutes": age})
    return points


def condition_initial_state(
    cycle_time: datetime,
    lat: np.ndarray,
    lon: np.ndarray,
    raws_payload: Optional[dict] = None,
    *,
    fallback_weather: Optional[tuple[np.ndarray, np.ndarray]] = None,
    raws_states: Optional[tuple] = ("MO",),
    condition: Optional[Callable] = None,
    anchor: Optional[Callable] = None,
) -> tuple[moisture.State, dict]:
    """Return (state {'fm1','fm10','fm100'} on the target grid, quality dict).

    fallback_weather=(temp_c, rh) at the cycle time opts in to an EMC seed when RTMA history is
    insufficient; without it, insufficient history raises ConditioningUnavailable.
    `condition` / `anchor` default to spread_rate_moisture's and are injectable for tests."""
    cycle = _aware(cycle_time)
    if condition is None:
        # outlook replay: interpolated RTMA, MRMS fill for hours with missing RTMA precipitation, never dry-by-default
        from services.fire_outlook import history, mrms
        sub_lat, sub_lon = mrms.subcell_samples(lat, lon)
        margin = mrms.CROP_MARGIN_DEG
        condition = history.make_condition(
            (float(lat.min()) - margin, float(lat.max()) + margin, float(lon.min()) - margin, float(lon.max()) + margin),
            sub_lat, sub_lon)
    if anchor is None:
        from services import spread_rate_moisture as srm
        anchor = srm.apply_raws_10hr_correction
    quality: dict = {"cycle_time": cycle.isoformat().replace("+00:00", "Z"), "observed_vs_modeled":
                     "fm values are modeled; RAWS only anchors fm10"}
    try:
        result = condition(lat, lon, analysis_hour=cycle, raws_payload=None)  # anchor ourselves, causally
    except RuntimeError as exc:
        if fallback_weather is None:
            raise ConditioningUnavailable(str(exc)) from exc
        temp_c, rh = fallback_weather
        state = moisture.initial_state(temp_c, rh)
        quality.update(level=LEVEL_EMC_SEED, reason=str(exc), raws_station_count=0)
        return state, quality

    state = {"fm1": np.asarray(result["fm1_pct"], dtype=float),
             "fm10": np.asarray(result["fm10_pct"], dtype=float),
             "fm100": np.asarray(result["fm100_pct"], dtype=float)}
    quality.update(
        analysis_hour=result.get("analysis_hour"),
        conditioning_hours_available=result.get("conditioning_hours_available"),
        conditioning_hours_missing=result.get("conditioning_hours_missing"),
    )
    if result.get("precip_audit") is not None:
        quality["precip_audit"] = result["precip_audit"]
    points = causal_raws_points(raws_payload, cycle, states=raws_states)
    if len(points) >= MIN_RAW_STATIONS:
        state["fm10"], meta = anchor(state["fm10"], lat, lon, points)
        quality.update(level=LEVEL_ANCHORED, raws_station_count=len(points), raws_correction=meta,
                       newest_raws_age_minutes=min(p["age_minutes"] for p in points))
    else:
        quality.update(level=LEVEL_CONDITIONED, raws_station_count=len(points),
                       reason=f"fewer than {MIN_RAW_STATIONS} causal RAWS fuel-moisture observations")
    return state, quality
