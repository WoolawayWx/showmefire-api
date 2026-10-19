"""Upper-air (radiosonde) soundings in and around Missouri and their fire-weather metrics.

NWS sites come from the Iowa Environmental Mesonet's RAOB archive, which carries
the routine 00Z/12Z launches and any off-hour launches:
  list:    https://mesonet.agron.iastate.edu/cgi-bin/request/raob.py?station=KSGF&sts=...&ets=...
  profile: https://mesonet.agron.iastate.edu/json/raob.py?ts=YYYYMMDDHHMM&station=KSGF
University launches that never reach the NWS data stream come from CW3E's
university radiosonde share (AR Recon CUSP program), one CSV text file per flight:
  https://cw3e-datashare.ucsd.edu/CW3E_Radiosondes/UCRP/<SITE>/Text_Files/
Fire-weather metrics reuse services/fire_weather/indices.py (sources cited there),
so balloon and HRRR values are computed identically.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import requests

from services.fire_weather import indices as ix

IEM = "https://mesonet.agron.iastate.edu"
HEADERS = {"User-Agent": "ShowMeFire/1.0 graphics@showmefire.org"}

# Launch sites in and around Missouri: (name, latitude, longitude).
STATIONS = {
    "KSGF": ("Springfield, MO", 37.23, -93.40),
    "KTOP": ("Topeka, KS", 39.07, -95.63),
    "KOAX": ("Omaha, NE", 41.32, -96.37),
    "KDVN": ("Davenport, IA", 41.61, -90.58),
    "KILX": ("Lincoln, IL", 40.15, -89.34),
    "KLZK": ("Little Rock, AR", 34.83, -92.26),
    "KBNA": ("Nashville, TN", 36.25, -86.56),
    "USCOU": ("Columbia, MO (Mizzou)", 38.94, -92.33),
}
# Sites published by CW3E rather than IEM: station -> folder of per-flight text files.
CW3E_SITES = {
    "USCOU": "https://cw3e-datashare.ucsd.edu/CW3E_Radiosondes/UCRP/USCOU/Text_Files/",
}
# Short names for images/status files; NWS sites drop the leading K (KSGF -> sgf).
SITE_SLUGS = {"USCOU": "cou"}
MIN_TOP_HPA = 400.0  # a profile that stops below this is incomplete (or still arriving)
SOUNDING_THETA_EXCESS_K = 0.5


def site_slug(station: str) -> str:
    return SITE_SLUGS.get(station, station[1:].lower())


def _cw3e_files(station: str) -> dict[datetime, str]:
    """Flight files in a CW3E site folder by launch time. Names vary
    (USCOU_20260315_18.txt, USCOU_20260314_1200.txt, USCOU_USCOU_20260122_21Z.txt)."""
    response = requests.get(CW3E_SITES[station], headers=HEADERS, timeout=60)
    response.raise_for_status()
    pattern = re.compile(rf'href="((?:{station}_)+(\d{{8}})_(\d{{2}})(\d{{2}})?Z?\.txt)"')
    files = {}
    for name, day, hour, minute in pattern.findall(response.text):
        valid = datetime.strptime(day + hour + (minute or "00"), "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
        files.setdefault(valid, name)
    return files


def list_launches(station: str, start: datetime, end: datetime) -> list[datetime]:
    """Launch times (UTC) for `station` between start and end, at any hour."""
    if station in CW3E_SITES:
        return sorted(valid for valid in _cw3e_files(station) if start <= valid <= end)
    response = requests.get(f"{IEM}/cgi-bin/request/raob.py", headers=HEADERS, timeout=60, params={
        "station": station, "sts": start.strftime("%Y-%m-%dT%H:%MZ"), "ets": end.strftime("%Y-%m-%dT%H:%MZ"),
    })
    response.raise_for_status()
    times = {row["validUTC"] for row in csv.DictReader(io.StringIO(response.text)) if row.get("validUTC")}
    return sorted(datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc) for value in times)


@dataclass
class Sounding:
    station: str
    valid: datetime
    pressure_hpa: np.ndarray   # surface first, decreasing
    height_m: np.ndarray       # MSL
    temp_c: np.ndarray
    dewpoint_c: np.ndarray
    u_ms: np.ndarray           # earth-relative, NaN where no wind
    v_ms: np.ndarray

    @property
    def height_agl_m(self) -> np.ndarray:
        return self.height_m - self.height_m[0]


def parse_profile(station: str, valid: datetime, rows: list[dict[str, Any]]) -> Optional[Sounding]:
    """Clean an IEM profile: levels with pressure, height and temperature,
    ordered from the surface (first level with a temperature) upward."""
    levels = [row for row in rows
              if row.get("pres") is not None and row.get("hght") is not None and row.get("tmpc") is not None]
    levels.sort(key=lambda row: -row["pres"])
    if len(levels) < 5 or min(row["pres"] for row in levels) > MIN_TOP_HPA:
        return None
    pressure = np.array([row["pres"] for row in levels], dtype=float)
    keep = np.concatenate([[True], np.diff(pressure) < 0])  # drop duplicate pressures
    levels = [row for row, ok in zip(levels, keep) if ok]

    def column(key):
        return np.array([np.nan if row.get(key) is None else row[key] for row in levels], dtype=float)

    speed = column("sknt") / ix.MS_TO_KT
    direction = np.radians(column("drct"))
    return Sounding(
        station=station, valid=valid, pressure_hpa=column("pres"), height_m=column("hght"),
        temp_c=column("tmpc"), dewpoint_c=column("dwpc"),
        u_ms=-speed * np.sin(direction), v_ms=-speed * np.cos(direction),
    )


def parse_cw3e(station: str, valid: datetime, text: str) -> Optional[Sounding]:
    """CW3E university flight file: CSV of height (m), pressure (hPa), temperature
    and dew point (C), wind direction, wind speed (m/s)."""
    rows = []
    for record in csv.reader(io.StringIO(text)):
        try:
            height, pressure, temp, dewpoint, direction, speed = (float(value) for value in record[:6])
        except (ValueError, TypeError):
            continue  # header or a malformed line
        rows.append({"pres": pressure, "hght": height, "tmpc": temp, "dwpc": dewpoint,
                     "drct": direction, "sknt": speed * ix.MS_TO_KT})
    return parse_profile(station, valid, rows)


def fetch_sounding(station: str, valid: datetime) -> Optional[Sounding]:
    if station in CW3E_SITES:
        name = _cw3e_files(station).get(valid)
        if name is None:
            return None
        response = requests.get(CW3E_SITES[station] + name, headers=HEADERS, timeout=60)
        response.raise_for_status()
        return parse_cw3e(station, valid, response.text)
    response = requests.get(f"{IEM}/json/raob.py", headers=HEADERS, timeout=60,
                            params={"ts": valid.strftime("%Y%m%d%H%M"), "station": station})
    response.raise_for_status()
    profiles = response.json().get("profiles") or []
    return parse_profile(station, valid, profiles[0]["profile"]) if profiles else None


def _log_interp(pressure: np.ndarray, values: np.ndarray, target_hpa: float) -> float:
    ok = np.isfinite(values)
    if not ok.any() or target_hpa > pressure[ok].max() or target_hpa < pressure[ok].min():
        return float("nan")
    # np.interp needs increasing x: use -ln(p), which rises with height.
    return float(np.interp(-np.log(target_hpa), -np.log(pressure[ok]), values[ok]))


def fire_weather_metrics(sounding: Sounding) -> dict[str, Any]:
    """Balloon equivalents of the hourly HRRR map fields (same formulas)."""
    s = sounding
    col = lambda array: array.reshape(-1, 1, 1)  # noqa: E731 - indices work on (level, y, x)
    heights = col(s.height_agl_m)
    theta = ix.potential_temperature_k(s.temp_c, s.pressure_hpa)

    # Balloon profiles resolve sensor-settling wiggles of a few tenths of a degree
    # near the ground; a 0.5 K parcel excess keeps those from capping the mixed layer.
    mixing = ix.mixing_height_holzworth(heights[1:], col(theta)[1:], np.full((1, 1), theta[0]),
                                        theta_excess=SOUNDING_THETA_EXCESS_K)[0, 0]
    winds = np.isfinite(s.u_ms)
    mean_u, mean_v = ix.transport_wind(col(s.height_agl_m[winds]), col(s.u_ms[winds]), col(s.v_ms[winds]),
                                       np.full((1, 1), mixing), bottom=0.0)
    speed = float(ix.wind_speed(mean_u, mean_v)[0, 0])
    mixing_ft = float(mixing * ix.M_TO_FT)
    ventilation = float(ix.ventilation_rate_kt_ft(speed * ix.MS_TO_KT, mixing_ft))
    inversion, inversion_top = ix.surface_inversion(heights, col(s.temp_c))
    lapse = float(ix.lapse_rate_c_per_km(heights, col(s.temp_c))[0, 0])
    haines = ix.haines_low(_log_interp(s.pressure_hpa, s.temp_c, 950), _log_interp(s.pressure_hpa, s.temp_c, 850),
                           _log_interp(s.pressure_hpa, s.dewpoint_c, 850))
    layer = s.pressure_hpa >= s.pressure_hpa[0] - 50.0
    vpd = np.where(layer, ix.vapor_pressure_deficit_hpa(s.temp_c, s.dewpoint_c), np.nan)
    wind_speed = np.where(layer, ix.wind_speed(s.u_ms, s.v_ms), np.nan)
    hdw = float(ix.hot_dry_windy(col(vpd), col(wind_speed))[0, 0])
    category = int(ix.ventilation_category(ventilation))
    return {
        "mixing_height_ft": round(mixing_ft),
        "mixing_height_hpa": round(_pressure_at_height(s, mixing), 1),
        "transport_speed_mph": round(speed * ix.MS_TO_MPH, 1),
        "transport_dir_deg": round(float(ix.wind_direction_from(mean_u, mean_v)[0, 0])),
        "ventilation_kt_ft": round(ventilation),
        "ventilation_category": ix.VENTILATION_CATEGORIES[category][0] if category >= 0 else None,
        "inversion_strength_f": round(float(inversion[0, 0]) * 1.8, 1),
        "inversion_top_ft": round(float(inversion_top[0, 0]) * ix.M_TO_FT),
        "lapse_rate_c_km": round(lapse, 1) if np.isfinite(lapse) else None,
        "haines": None if not np.isfinite(haines) else int(haines),
        "hdw": round(hdw) if np.isfinite(hdw) else None,
        "surface_pressure_hpa": float(s.pressure_hpa[0]),
        "surface_temp_f": round(float(s.temp_c[0]) * 1.8 + 32, 1),
    }


def _pressure_at_height(s: Sounding, height_agl_m: float) -> float:
    if not np.isfinite(height_agl_m):
        return float("nan")
    return float(np.exp(np.interp(height_agl_m, s.height_agl_m, np.log(s.pressure_hpa))))


def thermodynamic_metrics(sounding: Sounding) -> dict[str, Any]:
    """Standard stability values via MetPy: surface-based CAPE/CIN, LCL, precipitable water."""
    import metpy.calc as mpcalc
    from metpy.units import units

    s = sounding
    ok = np.isfinite(s.dewpoint_c)
    p = s.pressure_hpa[ok] * units.hPa
    t = s.temp_c[ok] * units.degC
    td = s.dewpoint_c[ok] * units.degC
    result: dict[str, Any] = {}
    try:
        cape, cin = mpcalc.surface_based_cape_cin(p, t, td)
        result["sbcape_jkg"] = round(float(cape.m))
        result["sbcin_jkg"] = round(float(cin.m))
    except Exception:  # noqa: BLE001 - MetPy can fail on odd profiles; the plot still renders
        pass
    try:
        lcl_p, _lcl_t = mpcalc.lcl(p[0], t[0], td[0])
        result["lcl_hpa"] = round(float(lcl_p.m))
        result["lcl_ft"] = round(float(np.interp(-np.log(lcl_p.m), -np.log(s.pressure_hpa), s.height_agl_m)) * ix.M_TO_FT)
    except Exception:  # noqa: BLE001
        pass
    try:
        result["pwat_in"] = round(float(mpcalc.precipitable_water(p, td).to("inch").m), 2)
    except Exception:  # noqa: BLE001
        pass
    return result


# --- Model omega at the launch sites ------------------------------------------
# Radiosondes do not measure vertical motion, so omega (dp/dt) comes from the
# HRRR at the launch time: the analysis (f00), or the previous run's 1-hour
# forecast if the analysis is not out yet. One download serves every site.
OMEGA_LEVELS = tuple(range(1000, 99, -25))
OMEGA_SEARCH = ":VVEL:(?:" + "|".join(str(p) for p in OMEGA_LEVELS) + ") mb:"
OMEGA_BOX = (-98.0, 33.5, -85.5, 42.5)  # west, south, east, north: covers every site


def _nearest_index(lat: np.ndarray, lon: np.ndarray, site_lat: float, site_lon: float) -> tuple[int, int]:
    distance = (lat - site_lat) ** 2 + ((lon - site_lon) * np.cos(np.radians(site_lat))) ** 2
    return np.unravel_index(int(np.argmin(distance)), lat.shape)


def fetch_hrrr_omega(valid: datetime, stations: list[str], herbie_factory=None) -> dict[str, dict[str, Any]]:
    """{station: {"pressure_hpa": [...], "omega_pa_s": [...], "source": ...}} at the
    HRRR grid point nearest each site, valid at `valid`. Empty when unavailable."""
    import shutil
    import tempfile
    from datetime import timedelta

    if herbie_factory is None:
        from herbie import Herbie as herbie_factory
    workdir = tempfile.mkdtemp(prefix="smf-omega-")
    try:
        for run, fxx, label in ((valid, 0, "HRRR analysis"), (valid - timedelta(hours=1), 1, "HRRR 1-hour forecast")):
            try:
                herbie = herbie_factory(run.replace(tzinfo=None), model="hrrr", product="prs", fxx=fxx,
                                        save_dir=workdir, verbose=False, priority=["aws"])
                dataset = herbie.xarray(OMEGA_SEARCH, remove_grib=True)
            except Exception:  # noqa: BLE001 - not published yet / missing: try the next option
                continue
            dataset = dataset[0] if isinstance(dataset, list) else dataset
            lat = np.asarray(dataset["latitude"].values, dtype=float)
            lon = np.asarray(dataset["longitude"].values, dtype=float)
            lon = np.where(lon > 180, lon - 360, lon)
            rows, cols = np.where((lon >= OMEGA_BOX[0]) & (lon <= OMEGA_BOX[2]) & (lat >= OMEGA_BOX[1]) & (lat <= OMEGA_BOX[3]))
            window = dict(y=slice(rows.min(), rows.max() + 1), x=slice(cols.min(), cols.max() + 1))
            name = "w" if "w" in dataset.data_vars else next(iter(dataset.data_vars))
            omega = np.asarray(dataset[name].isel(**window).values, dtype=float)
            pressures = np.asarray(dataset["isobaricInhPa"].values, dtype=float)
            lat, lon = lat[window["y"], window["x"]], lon[window["y"], window["x"]]
            order = np.argsort(-pressures)
            result = {}
            for station in stations:
                _name, site_lat, site_lon = STATIONS[station]
                i, j = _nearest_index(lat, lon, site_lat, site_lon)
                result[station] = {"pressure_hpa": pressures[order].tolist(),
                                   "omega_pa_s": [round(float(v), 4) for v in omega[order, i, j]],
                                   "source": label}
            return result
        return {}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# --- Which launches to keep -----------------------------------------------------
KEEP_LATEST = 2


def kept_launches(valid_times: list[datetime]) -> list[datetime]:
    """Newest first: the KEEP_LATEST most recent launches, whatever time they went up."""
    return sorted(set(valid_times), reverse=True)[:KEEP_LATEST]


def sounding_from_record(record: dict[str, Any]) -> Sounding:
    """Rebuild a Sounding from an archived JSON record."""
    profile = record["profile"]
    array = lambda key: np.array([np.nan if v is None else v for v in profile[key]], dtype=float)  # noqa: E731
    return Sounding(
        station=record["station"], valid=datetime.fromisoformat(record["valid_time"]),
        pressure_hpa=array("pressure_hpa"), height_m=array("height_m"), temp_c=array("temp_c"),
        dewpoint_c=array("dewpoint_c"), u_ms=array("u_ms"), v_ms=array("v_ms"),
    )
