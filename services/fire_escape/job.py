"""One day's fire escape products: features from HRRR and gridMET, the three models, the record and the maps.

`compute` is pure given its inputs (arrays in, arrays out) and is what the tests exercise. `run` wraps it with
the files: it never raises into its caller, and a day it cannot do is reported with the reason.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from services import fire_escape
from services.fire_escape import features as ft
from services.fire_escape import gridmet, hrrr_day, install, predict
from services.fire_weather import physical_danger
from services.fire_weather_hybrid import risk

logger = logging.getLogger(__name__)
API_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_NAME = re.compile(r"hrrr_(\d{8})_(\d{2})z_f\d{2}-\d{2}\.nc$")
MIN_HOURS = 10                       # the cached window is 12 hours; fewer than this is not the burning period
GRID_FIELDS = ("pfdi_flame_ft", "pfdi_level", "pfdi_index", "fire_p10", "fire_p100")


def load_release(directory: Path) -> Optional[dict]:
    """The trained bundles, the HRRR adapter and the fuel grid; None when the folder is not there."""
    directory = Path(directory)
    if not (directory / "county" / "bundle.json").is_file():
        return None
    with np.load(directory / "fuel_grid.npz") as fuel:
        fuel_grid = {name: fuel[name] for name in fuel.files}
    return {"county": predict.load_bundle(directory / "county"), "fire": predict.load_bundle(directory / "fire"),
            "adapter": json.loads((directory / "hrrr_adapter.json").read_text(encoding="utf-8")), "fuel_grid": fuel_grid}


def county_cells(lat: np.ndarray, lon: np.ndarray) -> tuple[list[str], np.ndarray]:
    """(sorted FIPS list, index into it of the county containing each cell; -1 outside Missouri), from the
    county boundaries every map here is drawn with. Computed for the grid in hand, so any HRRR crop works."""
    import geopandas as gpd

    path = API_ROOT / "maps" / "shapefiles" / "MO_County_Boundaries" / "MO_County_Boundaries.shp"
    counties = gpd.read_file(path).to_crs("EPSG:4326")
    counties["fips"] = counties["COUNTYFIPS"].astype(str).str.zfill(3).radd("29")
    fips = sorted(counties["fips"])
    position = {code: i for i, code in enumerate(fips)}
    points = gpd.GeoDataFrame({"cell": np.arange(lat.size)}, geometry=gpd.points_from_xy(lon.ravel(), lat.ravel()), crs="EPSG:4326")
    joined = gpd.sjoin(points, counties[["fips", "geometry"]], predicate="within").drop_duplicates("cell")
    index = np.full(lat.size, -1, dtype=int)
    index[joined["cell"].to_numpy()] = joined["fips"].map(position).to_numpy()
    return fips, index.reshape(lat.shape)


def pfdi_peak_flame_ft(hrrr: dict, sky_transmission: np.ndarray, day: date) -> np.ndarray:
    """The day's highest Physical Fire Danger Index flame length (ft) per cell, hour by hour from HRRR itself.

    The index is physics, not a fitted model, so it takes HRRR's own hourly temperature, humidity, wind and gust
    rather than the gridMET-like features the trained models need. The cached file has no cloud field: each
    hour's sun is the clear-sky value for that hour times `sky_transmission`, the day's share of clear-sky
    irradiance (from the adapter's daily irradiance)."""
    from services import derived_met

    herb, woody = physical_danger.live_moisture_pct(datetime(day.year, day.month, day.day))
    best = np.full(hrrr["lat"].shape, np.nan)
    for hour in range(hrrr["t2m"].shape[0]):
        when = hrrr["valid_time"][hour].astype("datetime64[s]").astype(datetime).replace(tzinfo=timezone.utc)
        solar = physical_danger.clear_sky_irradiance_wm2(derived_met.solar_elevation_deg(hrrr["lat"], hrrr["lon"], when)) * sky_transmission
        out = physical_danger.physical_fire_danger(
            hrrr["t2m"][hour].astype(float) - 273.15, hrrr["r2"][hour], np.hypot(hrrr["u10"][hour], hrrr["v10"][hour]), solar,
            gust_10m_ms=None if hrrr["gust"] is None else hrrr["gust"][hour], live_herb_pct=herb, live_woody_pct=woody)
        best = np.fmax(best, out["flame_length_ft"])
    return best


def compute(hrrr: dict, state: dict, release: dict, day: date, fips: list[str], cell_county: np.ndarray) -> dict:
    """Everything for one day.

    hrrr          hrrr_day.read_cache_file's dict (hourly t2m, r2, u10, v10, tp, gust; 2-D lat, lon)
    state         gridmet.latest_state's dict (30 days of rain, slow fuel moisture, on gridMET's grid)
    release       load_release's dict
    fips, cell_county   county_cells' result for this grid

    Returns `grid` (2-D arrays: PFDI flame length / level / index and the fire model's two probabilities, NaN
    outside Missouri), `county` (per FIPS: the county model's two probabilities, and the county means of the
    grid products), and `features` (the county feature rows, kept in the record so a forecast can be audited)."""
    lat, lon = hrrr["lat"], hrrr["lon"]
    doy = np.full(lat.shape, day.timetuple().tm_yday)
    day_values = hrrr_day.aggregate(hrrr["t2m"], hrrr["r2"], hrrr["u10"], hrrr["v10"], hrrr["tp"], hrrr["gust"])
    day_values["clear_sky_wm2"] = ft.clear_sky_wm2(lat, doy)[0]
    no_rain_field = hrrr["tp"] is None
    if no_rain_field:                                               # keep the other features: adapt them as for a dry day ...
        day_values["precip_mm"] = np.zeros(lat.shape)
    weather = hrrr_day.adapt(release["adapter"], day_values)
    if no_rain_field:                                               # ... but the day's rain itself is unknown, not zero
        weather["precip_day_mm"] = np.full(lat.shape, np.nan)
    rain_before = np.moveaxis(gridmet.to_cells(state["rain"], state["lat"], state["lon"], lat, lon), 0, -1)   # (y, x, 30)
    slow = {name: gridmet.to_cells(state[name], state["lat"], state["lon"], lat, lon) for name in gridmet.STATE_VARS}
    fuel = release["fuel_grid"]
    fuels = {name: gridmet.to_cells(fuel[name], fuel["lat"], fuel["lon"], lat, lon)
             for name in ("frac_grass", "frac_timber", "frac_nonburnable", "canopy_cover_pct")}

    def assemble(values: dict, rain, slow_values: dict, latitude, day_of_year) -> dict:
        window = np.concatenate([rain, np.asarray(values["precip_day_mm"])[..., None]], axis=-1)
        return ft.assemble(tmax_c=values["tmax_c"], rh_min=values["rh_min"], rh_max=values["rh_max"], wind_ms=values["wind_ms"],
                           vpd_kpa=values["vpd_kpa"], srad_wm2=values["srad_wm2"], precip_window_mm=window,
                           lat_deg=latitude, day_of_year=day_of_year, stn_wind_peak_ms=values["stn_wind_peak_ms"],
                           stn_gust_peak_ms=values["stn_gust_peak_ms"], stn_rh_min=values["stn_rh_min"], **slow_values)

    # --- per cell: the physical index and the fire model ---
    cell = assemble(weather, rain_before, slow, lat, doy)
    cell.update(fuels)
    inside = cell_county >= 0
    transmission = np.clip(np.divide(weather["srad_wm2"], day_values["clear_sky_wm2"], out=np.ones(lat.shape),
                                     where=day_values["clear_sky_wm2"] > 0), 0.25, 1.0)      # 0.25 is full overcast (Kasten & Czeplak)
    flame_ft = np.where(inside, pfdi_peak_flame_ft(hrrr, transmission, day), np.nan)
    flame_m = flame_ft / physical_danger.M_TO_FT
    fire = predict.predict(release["fire"], pd.DataFrame({name: values.ravel() for name, values in cell.items()}))
    grid = {"pfdi_flame_ft": flame_ft, "pfdi_level": risk.flame_length_categories(flame_m),
            "pfdi_index": physical_danger.index_from_flame_length(flame_m),
            "fire_p10": np.where(inside, fire["escape_10ac"].reshape(lat.shape), np.nan),
            "fire_p100": np.where(inside, fire["escape_100ac"].reshape(lat.shape), np.nan)}

    # --- per county: the county model, on county means of its inputs as it was trained ---
    n = len(fips)
    mean = lambda values: hrrr_day.county_mean(values, cell_county, n)
    county_weather = {name: mean(values) for name, values in weather.items()}
    county_rain = np.stack([mean(rain_before[..., i]) for i in range(rain_before.shape[-1])], axis=-1)
    county_slow = {name: mean(values) for name, values in slow.items()}
    rows = assemble(county_weather, county_rain, county_slow, mean(lat), np.full(n, day.timetuple().tm_yday))
    rows.update({name: mean(values) for name, values in fuels.items()})
    frame = pd.DataFrame(rows)
    probability = predict.predict(release["county"], frame)
    usable = frame[release["county"]["features"]].notna().all(axis=1).to_numpy()
    county = {"p_10ac": np.where(usable, probability["escape_10ac"], np.nan),
              "p_100ac": np.where(usable, probability["escape_100ac"], np.nan),
              **{name: mean(values) for name, values in grid.items() if name != "pfdi_level"},
              "pfdi_level_max": np.array([np.max(levels) if (levels := grid["pfdi_level"][cell_county == i]).size
                                          and np.isfinite(levels).all() else np.nan for i in range(n)])}
    return {"grid": grid, "county": county, "features": frame, "lat": lat, "lon": lon}


def find_cache_file(day: date, directory: Path, cycle_hour: int = 12) -> Optional[Path]:
    """The daily forecast's cached HRRR file for `day` (the cycle hour, or one hour late as that forecast allows)."""
    found = [path for path in sorted(Path(directory).glob(f"hrrr_{day:%Y%m%d}_*z_f*.nc"))
             if (match := CACHE_NAME.search(path.name)) and int(match.group(2)) in (cycle_hour, cycle_hour + 1)]
    return found[-1] if found else None


def publish_to_testbed(images: dict, result: dict, record: dict) -> None:
    """Copy the day's maps into the Testbed images under fixed names and list them in the Testbed manifest
    (services/beta_products.py). The Testbed page shows them behind its not-for-operations acknowledgement."""
    import shutil

    from services.beta_products import BETA_ROOT, load_manifest, save_manifest
    from services.fire_escape import render

    folder = BETA_ROOT / "images"
    folder.mkdir(parents=True, exist_ok=True)
    previews = {}
    for kind, source in images.items():
        target = folder / f"fire_escape_{kind}_latest.png"
        shutil.copyfile(source, target)
        previews[kind] = f"images/{target.name}"
    manifest = load_manifest()
    manifest.setdefault("products", {})["fire_escape"] = {
        "kind": "raster-preview", "not_for_operations": True, "previews": previews,
        "summary": {kind: render.summary_line(kind, result) for kind in render.MAPS},
        "valid_local_date": record["valid_local_date"], "generated_at": record["recorded_at"], "run_id": record["run_id"],
        "gridmet_lag_days": record["gridmet_lag_days"], "model_version": record["bundle"],
    }
    save_manifest(manifest)


def _rounded(values, digits: int = 5) -> list:
    return [None if not np.isfinite(v) else round(float(v), digits) for v in np.asarray(values, dtype=float)]


def run(day: date, hrrr_path: Optional[Path] = None, release_dir: Optional[Path] = None, out_dir: Optional[Path] = None,
        state: Optional[dict] = None, render_maps: bool = True, publish: bool = False) -> dict:
    """Compute, record and draw one day; with `publish`, also put the maps on the Testbed.
    Returns {"status": "ok", ...} or {"status": "skipped", "reason": ...}."""
    try:
        release_dir = Path(release_dir or fire_escape.bundle_dir())
        release = load_release(release_dir)
        if release is None:
            return {"status": "skipped", "reason": f"no bundle in {release_dir} (install one: scripts/install_fire_escape.py)"}
        path = Path(hrrr_path) if hrrr_path else find_cache_file(day, fire_escape.hrrr_cache_dir())
        if path is None or not path.is_file():
            return {"status": "skipped", "reason": f"no cached HRRR file for {day}"}
        hrrr = hrrr_day.read_cache_file(path)
        if any(hrrr[name] is None for name in ("t2m", "r2", "u10", "v10")) or hrrr["t2m"].shape[0] < MIN_HOURS:
            return {"status": "skipped", "reason": f"{path.name} does not hold the burning period"}
        state = state or gridmet.latest_state(day, fire_escape.gridmet_cache_dir())
        if state is None:
            return {"status": "skipped", "reason": f"gridMET has nothing within {gridmet.MAX_LAG_DAYS} days of {day}"}
        fips, cell_county = county_cells(hrrr["lat"], hrrr["lon"])
        result = compute(hrrr, state, release, day, fips, cell_county)

        out_dir = Path(out_dir or fire_escape.evidence_dir())
        out_dir.mkdir(parents=True, exist_ok=True)
        run_id = f"{day:%Y%m%d}_{path.stem.split('_')[2]}"
        county = result["county"]
        record = {
            "model": fire_escape.MODEL, "run_id": run_id, "valid_local_date": day.isoformat(),
            "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "not_for_operations": True, "hrrr_file": path.name, "gridmet_latest_day": state["latest_day"],
            "gridmet_lag_days": state["lag_days"], "rain_days_known": state["rain_days_known"],
            "bundle": {"county": release["county"]["version"], "fire": release["fire"]["version"],
                       "dataset_sha256": release["county"].get("dataset_sha256"),
                       "release_tag": (install.installed(release_dir) or {}).get("tag")},
            "county_fips": fips, **{name: _rounded(values) for name, values in county.items()},
            "features": {name: _rounded(result["features"][name], 3) for name in release["county"]["features"]},
        }
        record_path = out_dir / f"{run_id}.fire_escape.json"
        record_path.write_text(json.dumps(record), encoding="utf-8")           # a rerun of the same run replaces its record
        np.savez_compressed(out_dir / f"{run_id}.fire_escape.npz", lat=result["lat"].astype("float32"),
                            lon=result["lon"].astype("float32"), **{k: v.astype("float32") for k, v in result["grid"].items()})
        images = {}
        if render_maps:
            try:
                from services.fire_escape import render
                images = render.render_all(result, fips, day, out_dir / "maps", state["lag_days"])
                if publish:
                    publish_to_testbed(images, result, record)
            except Exception as exc:                                            # a drawing failure never loses the record
                logger.warning("fire_escape maps failed (non-fatal): %s", exc, exc_info=True)
        return {"status": "ok", "run_id": run_id, "record": str(record_path), "images": {k: str(v) for k, v in images.items()},
                "gridmet_lag_days": state["lag_days"],
                "county_p10_max": float(np.nanmax(county["p_10ac"])), "county_p100_max": float(np.nanmax(county["p_100ac"]))}
    except Exception as exc:
        logger.warning("fire_escape run for %s failed (non-fatal): %s", day, exc, exc_info=True)
        return {"status": "skipped", "reason": f"error: {exc}"}
