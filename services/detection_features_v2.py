"""Scoring-time features for the vegetation-fire confidence candidate (v2).

Pure pandas/numpy: no database access. The API builds the inputs from SQLite when it records a shadow score, and the
training package loads this same file to rebuild features from an export, so serving and training cannot drift apart.

Causal rule: a feature may only use observations that had ARRIVED by the scoring time (first_seen_at <= as_of) and were
acquired no later than the detection itself (plus a small look-ahead for cross-sensor corroboration on an "updated"
score). Nothing from incidents, feedback, NERIS, weather scoring or the previous confidence score is used.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

FEATURE_SCHEMA = "veg-conf-features-v1"
EARTH_KM = 6371.0088
MAX_RADIUS_KM = 5.0
COUNT_RADII_KM = (1.0, 5.0)
COUNT_WINDOWS_H = {"24h": 24.0, "7d": 24.0 * 7, "365d": 24.0 * 365}
CORROBORATION_RADIUS_KM = 2.0
CORROBORATION_WINDOW_H = 3.0

LAND_COVER_GROUPS = {
    "lc_cropland": ("cropland", "agricult", "crop"),
    "lc_tree": ("tree", "forest", "wood"),
    "lc_grass": ("grass", "pasture", "herbaceous", "hay"),
    "lc_shrub": ("shrub", "scrub"),
    "lc_developed": ("developed", "urban", "built"),
    "lc_water": ("water",),
}

FEATURE_NAMES = (
    ["frp_log", "bright_diff", "bright_t7_norm", "pixel_area_log", "native_confidence", "is_day",
     "solar_zenith_norm", "satellite_zenith_norm", "src_viirs", "src_modis", "src_ngfs"]
    + list(LAND_COVER_GROUPS)
    + ["in_exclusion_zone", "in_recurring_source", "recurring_source_km", "recurring_source_detections"]
    + [f"n_{int(r)}km_{w}" for r in COUNT_RADII_KM for w in COUNT_WINDOWS_H]
    + ["distinct_days_1km_365d", "n_other_sensor_3h", "max_other_frp_3h"]
)

# Not model inputs, by design: location (a model would memorise where NERIS reports), calendar date, incident grouping,
# the previous confidence score, weather scoring output, ingest lag, feedback/review state, NERIS indicators.
EXCLUDED = ("latitude", "longitude", "doy", "month", "incident_id", "detection_confidence_pct", "weather_danger_prob",
            "weather_danger_category", "ingest_lag_h")


def _num(series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _hours(series) -> np.ndarray:
    t = pd.to_datetime(series, utc=True, errors="coerce")
    out = t.astype("int64").to_numpy().astype(float) / 3.6e12
    out[t.isna().to_numpy()] = np.nan
    return out


def native_confidence(value) -> float:
    """Sensor-native confidence mapped to 0-1. VIIRS uses l/n/h, MODIS a percentage, NGFS a class; unknown -> NaN so
    a missing value is never read as 'low'."""
    text = str(value if value is not None else "").strip().lower()
    if text in ("", "nan", "none"):
        return float("nan")
    table = {"l": 0.3, "low": 0.3, "n": 0.6, "nominal": 0.6, "medium": 0.6, "probable": 0.6, "h": 0.9, "high": 0.9}
    if text in table:
        return table[text]
    try:
        return max(0.0, min(1.0, float(text) / 100.0))
    except ValueError:
        return float("nan")


def land_cover_fractions(land_cover) -> dict:
    """'Cropland:80,Trees:20' -> fractions by group. All NaN when the field is empty (not zero)."""
    names = list(LAND_COVER_GROUPS)
    if land_cover is None or (isinstance(land_cover, float) and math.isnan(land_cover)) or not str(land_cover).strip():
        return {n: float("nan") for n in names}
    out = {n: 0.0 for n in names}
    for part in str(land_cover).split(","):
        name, _, pct = part.partition(":")
        try:
            value = float(pct) / 100.0
        except ValueError:
            continue
        key = name.strip().lower()
        for group, words in LAND_COVER_GROUPS.items():
            if any(w in key for w in words):
                out[group] += value
    return out


def build_features(targets: pd.DataFrame, history: pd.DataFrame, recurring: pd.DataFrame | None = None,
                   lookahead_h: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Feature matrix for `targets` (index-aligned) plus an availability frame.

    targets/history need: id, source, occurred_at, latitude, longitude, frp; history also first_seen_at (arrival).
    targets may carry `as_of` (the scoring time); otherwise it is the target's own arrival time, i.e. an "initial"
    score. Targets also use: confidence, bright_t7, bright_t13, pixel_area, solar_zenith_angle,
    satellite_zenith_angle, daynight, land_cover, exclusion_zone_id, recurring_source_id.
    recurring: confirmed recurring heat sources (id, latitude, longitude, detection_count, reviewed_at).
    lookahead_h: how far past the detection's own time cross-sensor corroboration may look (0 for an initial score).

    A missing arrival time falls back to the acquisition time; availability['arrival_known'] says which happened.
    """
    from sklearn.neighbors import BallTree

    t = targets.copy()
    n = len(t)
    X = pd.DataFrame(index=t.index, columns=FEATURE_NAMES, dtype=float)
    avail = pd.DataFrame(index=t.index)

    frp = _num(t.get("frp"))
    t7, t13 = _num(t.get("bright_t7")), _num(t.get("bright_t13"))
    area = _num(t.get("pixel_area"))
    X["frp_log"] = np.where(frp.notna() & (frp > 0), np.log1p(frp.clip(lower=0)), np.where(frp.notna(), 0.0, np.nan))
    X["bright_diff"] = t13 - t7
    X["bright_t7_norm"] = (t7 - 273.0) / 50.0
    X["pixel_area_log"] = np.where(area.notna() & (area > 0), np.log1p(area.clip(lower=0)), np.nan)
    X["native_confidence"] = t.get("confidence", pd.Series(index=t.index, dtype=object)).map(native_confidence)
    dn = t.get("daynight", pd.Series(index=t.index, dtype=object)).astype(str).str.upper()
    X["is_day"] = np.where(dn == "D", 1.0, np.where(dn == "N", 0.0, np.nan))
    X["solar_zenith_norm"] = _num(t.get("solar_zenith_angle")) / 180.0
    X["satellite_zenith_norm"] = _num(t.get("satellite_zenith_angle")) / 90.0
    src = t["source"].astype(str).str.lower()
    for name in ("viirs", "modis", "ngfs"):
        X[f"src_{name}"] = (src == name).astype(float)
    lc = pd.DataFrame([land_cover_fractions(v) for v in t.get("land_cover", pd.Series([None] * n, index=t.index))], index=t.index)
    for g in LAND_COVER_GROUPS:
        X[g] = lc[g]
    X["in_exclusion_zone"] = _num(t.get("exclusion_zone_id")).notna().astype(float)
    X["in_recurring_source"] = _num(t.get("recurring_source_id")).notna().astype(float)

    lat = _num(t["latitude"]).to_numpy(dtype=float)
    lon = _num(t["longitude"]).to_numpy(dtype=float)
    occ = _hours(t["occurred_at"])
    arr = _hours(t["first_seen_at"]) if "first_seen_at" in t else np.full(n, np.nan)
    arrival_known = ~np.isnan(arr)
    arr = np.where(arrival_known, arr, occ)
    as_of = _hours(t["as_of"]) if "as_of" in t else arr
    as_of = np.where(np.isnan(as_of), arr, as_of)
    avail["arrival_known"] = arrival_known.astype(int)
    avail["as_of_h"] = as_of

    # confirmed recurring heat sources known at scoring time
    rec_km = np.full(n, np.nan)
    rec_det = np.full(n, np.nan)
    if recurring is not None and len(recurring):
        rlat, rlon = _num(recurring["latitude"]).to_numpy(float), _num(recurring["longitude"]).to_numpy(float)
        rdet = _num(recurring["detection_count"] if "detection_count" in recurring else pd.Series(0, index=recurring.index)).to_numpy(float)
        rrev = _hours(recurring["reviewed_at"]) if "reviewed_at" in recurring else np.full(len(recurring), np.nan)
        rtree = BallTree(np.radians(np.c_[rlat, rlon]), metric="haversine")
        pts = np.radians(np.c_[lat, lon])
        for i in np.flatnonzero(~(np.isnan(lat) | np.isnan(lon))):
            dist, idx = rtree.query(pts[i:i + 1], k=min(len(recurring), 5))
            for d, j in zip(dist[0] * EARTH_KM, idx[0]):
                if np.isnan(rrev[j]) or rrev[j] <= as_of[i]:  # a source confirmed after scoring is not known yet
                    rec_km[i], rec_det[i] = d, rdet[j]
                    break
    X["recurring_source_km"] = rec_km
    X["recurring_source_detections"] = rec_det

    # prior-detection counts and cross-sensor corroboration, using only what had arrived
    h = history
    hlat, hlon = _num(h["latitude"]).to_numpy(float), _num(h["longitude"]).to_numpy(float)
    hocc = _hours(h["occurred_at"])
    harr = _hours(h["first_seen_at"]) if "first_seen_at" in h else np.full(len(h), np.nan)
    harr = np.where(np.isnan(harr), hocc, harr)
    hsrc = h["source"].astype(str).str.lower().to_numpy()
    hid = _num(h["id"]).to_numpy(float)
    hfrp = _num(h.get("frp", pd.Series(np.nan, index=h.index))).to_numpy(float)
    tid = _num(t["id"]).to_numpy(float)
    tsrc = src.to_numpy()

    count_cols = {(r, w): np.zeros(n) for r in COUNT_RADII_KM for w in COUNT_WINDOWS_H}
    distinct = np.zeros(n)
    other_n = np.zeros(n)
    other_frp = np.full(n, np.nan)
    if len(h):
        tree = BallTree(np.radians(np.c_[hlat, hlon]), metric="haversine")
        neigh, dists = tree.query_radius(np.radians(np.c_[lat, lon]), r=MAX_RADIUS_KM / EARTH_KM, return_distance=True)
        for i in range(n):
            if np.isnan(occ[i]) or np.isnan(lat[i]):
                continue
            idx, d = neigh[i], dists[i] * EARTH_KM
            if not len(idx):
                continue
            keep = (harr[idx] <= as_of[i]) & (hid[idx] != tid[i]) & (hocc[idx] <= occ[i]) & (hocc[idx] >= occ[i] - COUNT_WINDOWS_H["365d"])
            idx, d = idx[keep], d[keep]
            age = occ[i] - hocc[idx]
            for r in COUNT_RADII_KM:
                for wname, w in COUNT_WINDOWS_H.items():
                    count_cols[(r, wname)][i] = np.count_nonzero((d <= r) & (age <= w))
            near_year = (d <= 1.0)
            distinct[i] = len(np.unique(np.floor(hocc[idx][near_year] / 24.0))) if near_year.any() else 0
        # corroboration: a different sensor within 2 km and 3 h, arrived by scoring time
        for i in range(n):
            if np.isnan(occ[i]) or np.isnan(lat[i]):
                continue
            idx, d = neigh[i], dists[i] * EARTH_KM
            ok = ((d <= CORROBORATION_RADIUS_KM) & (harr[idx] <= as_of[i]) & (hsrc[idx] != tsrc[i]) & (hid[idx] != tid[i])
                  & (hocc[idx] >= occ[i] - CORROBORATION_WINDOW_H) & (hocc[idx] <= occ[i] + lookahead_h))
            if ok.any():
                other_n[i] = ok.sum()
                vals = hfrp[idx][ok]
                other_frp[i] = np.nanmax(vals) if np.isfinite(vals).any() else np.nan
    for (r, wname), vals in count_cols.items():
        X[f"n_{int(r)}km_{wname}"] = vals
    X["distinct_days_1km_365d"] = distinct
    X["n_other_sensor_3h"] = other_n
    X["max_other_frp_3h"] = other_frp
    return X[FEATURE_NAMES], avail
