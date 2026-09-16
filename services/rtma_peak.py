"""Build an additive, RTMA-driven daily peak fire-danger surface.

Uses the same 10:00–21:00 CT window as the peak forecast and end-of-day
verification, and renders a branded Missouri map to match the forecast and
realtime analysis products.
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import xarray as xr
from zoneinfo import ZoneInfo

from core.config import ARCHIVE_RAW_DATA_DIR, GIS_DIR, IMAGES_DIR
from core.executors import get_rtma_job_lock, run_in_process_pool_async
from core.fire_danger import calculate_fire_danger
from core.beta_fire_danger import score_fire_danger
from forecast.export_fire_danger_gis import export_geotiff
from services.rtma_capture import fetch_rtma
from services.mrms_capture import fetch_mrms, mrms_enabled
from services.verification_rainfall import CONTRACT_VERSION, adjust_grid, load_fuel_model_raster, load_mrms_grid

logger = logging.getLogger(__name__)

CHICAGO_TZ = ZoneInfo("America/Chicago")
PROJECT_DIR = Path(__file__).resolve().parent.parent
MAPS_DIR = PROJECT_DIR / "maps"
ASSETS_DIR = PROJECT_DIR / "assets"

RTMA_PEAK_DIR = Path(GIS_DIR) / "rtma_peak"
RTMA_PEAK_ARCHIVE_DIR = RTMA_PEAK_DIR / "archive"
RTMA_PEAK_TODAY_TIF = Path(GIS_DIR) / "rtma_peak_today.tif"
RTMA_PEAK_IMAGE_DIR = Path(IMAGES_DIR) / "rtma_peak"
RTMA_PEAK_IMAGE_ARCHIVE_DIR = RTMA_PEAK_IMAGE_DIR / "archive"
RTMA_PEAK_TODAY_PNG = Path(IMAGES_DIR) / "mo-rtma-observedpeakfiredanger.png"
RTMA_ADJUSTED_PEAK_DIR = Path(GIS_DIR) / "rtma_peak_rainfall_adjusted"

RTMA_FUEL_MOISTURE_DIR = Path(GIS_DIR) / "rtma_fuel_moisture"
RTMA_FUEL_MOISTURE_ARCHIVE_DIR = RTMA_FUEL_MOISTURE_DIR / "archive"
RTMA_FUEL_MOISTURE_IMAGE_DIR = Path(IMAGES_DIR) / "rtma_fuel_moisture"
RTMA_FUEL_MOISTURE_IMAGE_ARCHIVE_DIR = RTMA_FUEL_MOISTURE_IMAGE_DIR / "archive"
RTMA_FUEL_MOISTURE_TODAY_TIF = Path(GIS_DIR) / "rtma_fuel_moisture_today.tif"
RTMA_FUEL_MOISTURE_TODAY_PNG = Path(IMAGES_DIR) / "mo-rtma-fuelmoisture.png"
FUEL_MOISTURE_RANGE = (1.0, 40.0)  # matches _calibrate_fuel_moisture's clip bounds

RTMA_REDUCTION_ARCHIVE_DIR = Path(GIS_DIR) / "rtma_rainfall_reduction" / "archive"
RTMA_IMPACT_DIR = Path(GIS_DIR) / "rtma_rainfall_impact"
RTMA_IMPACT_ARCHIVE_DIR = RTMA_IMPACT_DIR / "archive"
RTMA_IMPACT_IMAGE_DIR = Path(IMAGES_DIR) / "rtma_rainfall_impact"
RTMA_IMPACT_IMAGE_ARCHIVE_DIR = RTMA_IMPACT_IMAGE_DIR / "archive"
RTMA_IMPACT_TODAY_TIF = Path(GIS_DIR) / "rtma_rainfall_impact_today.tif"
RTMA_IMPACT_TODAY_PNG = Path(IMAGES_DIR) / "mo-rtma-rainfallimpact.png"
RTMA_IMPACT_DEFAULT_DAYS = 7

# Same fire-weather window as DailyForecast peak maps and endOfDayReport.
PEAK_WINDOW_START_HOUR = 10
PEAK_WINDOW_HOURS = 12
MINIMUM_FUEL_MOISTURE_STATIONS = 3
FUEL_MOISTURE_MAX_AGE = timedelta(minutes=75)


def _hours_for_local_date(target_date: date):
    start = datetime.combine(target_date, datetime.min.time(), tzinfo=CHICAGO_TZ) + timedelta(
        hours=PEAK_WINDOW_START_HOUR
    )
    return [start.astimezone(timezone.utc) + timedelta(hours=i) for i in range(PEAK_WINDOW_HOURS)]


def _lon180(lon: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon, dtype=float)
    return np.where(lon > 180.0, lon - 360.0, lon)


def _squeeze2d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    while values.ndim > 2:
        values = np.take(values, 0, axis=0)
    return values


def _lon_lat_meshes(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon = _lon180(_squeeze2d(lon))
    lat = _squeeze2d(lat)
    if lon.ndim == 1 and lat.ndim == 1:
        return np.meshgrid(lon, lat)
    return lon, lat


def _parse_local_date(target_date: str | date | None) -> date:
    if target_date is None:
        return datetime.now(CHICAGO_TZ).date()
    if isinstance(target_date, date):
        return target_date
    try:
        return datetime.strptime(target_date, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"date must be YYYY-MM-DD, got: {target_date}") from exc


def _observation_series(observations: dict, *names: str) -> list:
    for name in names:
        values = observations.get(name)
        if isinstance(values, dict):
            values = values.get("value")
        if values is not None:
            return values if isinstance(values, list) else [values]
    return []


def _load_fuel_moisture_archive(target_date: date, archive_dir: Path | None = None) -> dict | None:
    path = Path(archive_dir or ARCHIVE_RAW_DATA_DIR) / f"raw_data_{target_date:%Y%m%d}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.info("No archived RAWS fuel moisture for %s; using the RTMA RH estimate", target_date)
        return None
    except (OSError, json.JSONDecodeError):
        logger.exception("Unable to read archived RAWS fuel moisture from %s", path)
        return None
    return payload if isinstance(payload, dict) else None


def _fuel_moisture_observations(payload: dict | None, valid_time: datetime) -> list[dict]:
    """Return each station's closest measured fuel moisture for an RTMA hour."""
    if not payload:
        return []
    selected = []
    target = valid_time.astimezone(timezone.utc)
    for station in payload.get("STATION", []):
        observations = station.get("OBSERVATIONS", {})
        times = _observation_series(observations, "date_time")
        values = _observation_series(
            observations,
            "fuel_moisture_value_1",
            "fuel_moisture_set_1",
            "fuel_moisture",
        )
        candidates = []
        for index, raw_time in enumerate(times):
            if index >= len(values) or values[index] is None:
                continue
            try:
                observed_time = datetime.fromisoformat(str(raw_time).replace("Z", "+00:00"))
                if observed_time.tzinfo is None:
                    observed_time = observed_time.replace(tzinfo=timezone.utc)
                age = abs(target - observed_time.astimezone(timezone.utc))
                value = float(values[index])
            except (TypeError, ValueError):
                continue
            if age <= FUEL_MOISTURE_MAX_AGE and np.isfinite(value) and 0 < value <= 60:
                candidates.append((age, value, observed_time))
        if not candidates:
            continue
        try:
            longitude = float(station["LONGITUDE"])
            latitude = float(station["LATITUDE"])
        except (KeyError, TypeError, ValueError):
            continue
        _, value, observed_time = min(candidates, key=lambda candidate: candidate[0])
        selected.append({
            "station": station.get("STID") or station.get("ID"),
            "longitude": longitude,
            "latitude": latitude,
            "fuel_moisture": value,
            "observation_time": observed_time.isoformat(),
        })
    return selected


def _calibrate_fuel_moisture(
    estimate: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    observations: list[dict],
) -> np.ndarray:
    """Bias-correct the RH estimate with inverse-distance RAWS residuals."""
    if len(observations) < MINIMUM_FUEL_MOISTURE_STATIONS:
        return estimate

    weighted_residual = np.zeros(estimate.shape, dtype=float)
    weight_sum = np.zeros(estimate.shape, dtype=float)
    latitude_scale = np.cos(np.deg2rad(float(np.nanmean(lat))))
    for observation in observations:
        station_lon = observation["longitude"]
        station_lat = observation["latitude"]
        nearest = np.nanargmin(
            np.square((lon - station_lon) * latitude_scale) + np.square(lat - station_lat)
        )
        estimated_at_station = float(estimate.ravel()[nearest])
        residual = np.clip(observation["fuel_moisture"] - estimated_at_station, -15.0, 15.0)
        distance_squared = (
            np.square((lon - station_lon) * latitude_scale)
            + np.square(lat - station_lat)
        )
        weights = 1.0 / (distance_squared + 0.05 ** 2)
        weighted_residual += weights * residual
        weight_sum += weights

    correction = np.divide(
        weighted_residual,
        weight_sum,
        out=np.zeros_like(weighted_residual),
        where=weight_sum > 0,
    )
    return np.clip(estimate + correction, 1.0, 40.0)


def _classify_grid(
    ds: xr.Dataset,
    scorer=calculate_fire_danger,
    score_key: str | None = None,
    fuel_moisture_observations: list[dict] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return class grid, -180/180 lon/lat meshes, and the calibrated fuel-moisture grid for one RTMA hour."""
    lon, lat = _lon_lat_meshes(ds["longitude"].values, ds["latitude"].values)
    rh = _squeeze2d(np.asarray(ds["r2"].values, dtype=float))
    wind = np.hypot(
        _squeeze2d(np.asarray(ds["u10"].values, dtype=float)),
        _squeeze2d(np.asarray(ds["v10"].values, dtype=float)),
    ) * 1.9438444924406

    # RTMA has no fuel-moisture field, so RH supplies continuous coverage.
    # Where archived RAWS measurements exist, their residual from this estimate
    # is interpolated across the grid to anchor the field to observed fuels.
    fuel_moisture = _calibrate_fuel_moisture(
        3.0 + 0.25 * rh,
        lon,
        lat,
        fuel_moisture_observations or [],
    )
    if rh.shape != lon.shape:
        raise ValueError(f"RTMA field shape {rh.shape} does not match lon/lat {lon.shape}")
    valid = np.isfinite(fuel_moisture) & np.isfinite(rh) & np.isfinite(wind)
    result = np.full(rh.shape, np.nan, dtype=float)
    if score_key:
        classify = np.vectorize(lambda fm, rh, wind: scorer(fm, rh, wind)[score_key], otypes=[float])
    else:
        classify = np.vectorize(scorer, otypes=[float])
    result[valid] = classify(fuel_moisture[valid], rh[valid], wind[valid])
    return result, lon, lat, fuel_moisture


def _nearest_grid_lookup(source_values, source_lon, source_lat, target_lon, target_lat):
    """Nearest-neighbour resample of a source grid onto the RTMA grid."""
    from scipy.spatial import cKDTree

    source_lon = np.asarray(source_lon, dtype=float)
    source_lat = np.asarray(source_lat, dtype=float)
    values = np.asarray(source_values)
    if source_lon.ndim == 1 and source_lat.ndim == 1:
        source_lon, source_lat = np.meshgrid(source_lon, source_lat)
    if values.shape != source_lon.shape:
        raise ValueError("source values and coordinates have incompatible shapes")
    source_lon = source_lon.ravel()
    source_lat = source_lat.ravel()
    values = values.ravel()
    valid = np.isfinite(source_lon) & np.isfinite(source_lat) & np.isfinite(values)
    if not valid.any():
        raise ValueError("source grid contains no valid cells")
    tree = cKDTree(np.column_stack((source_lat[valid], source_lon[valid])))
    indices = tree.query(
        np.column_stack((np.asarray(target_lat).ravel(), np.asarray(target_lon).ravel()))
    )[1]
    return values[valid][indices].reshape(np.asarray(target_lat).shape)


def _missouri_mask(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    import geopandas as gpd
    from shapely.geometry import Point
    from shapely.prepared import prep

    border_path = MAPS_DIR / "shapefiles" / "MO_State_Boundary" / "MO_State_Boundary.shp"
    missouriborder = gpd.read_file(border_path)
    if missouriborder.crs and str(missouriborder.crs) != "EPSG:4326":
        missouriborder = missouriborder.to_crs("EPSG:4326")
    missouri_geom = missouriborder.geometry.iloc[0].buffer(0.01)
    prepared_geom = prep(missouri_geom)
    lon_mesh, lat_mesh = _lon_lat_meshes(lon, lat)
    points_flat = np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()])
    mask_flat = np.array([prepared_geom.contains(Point(pt)) for pt in points_flat])
    return mask_flat.reshape(lon_mesh.shape), lon_mesh, lat_mesh


def _render_png(
    grid: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    out_path: Path,
    target_date: date,
    fuel_moisture_note: str,
) -> Path:
    """Render the branded 2048x1152 fire-danger map used by forecast/realtime products."""
    import cartopy.crs as ccrs
    import geopandas as gpd
    import matplotlib.font_manager as font_manager
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    out_path.parent.mkdir(parents=True, exist_ok=True)

    mask, lon_mesh, lat_mesh = _missouri_mask(lon, lat)
    if mask.shape != grid.shape:
        raise ValueError(f"Missouri mask shape {mask.shape} does not match peak grid {grid.shape}")
    masked = np.where(mask, grid, np.nan)
    if not np.isfinite(masked).any():
        logger.warning("Missouri mask dropped every RTMA cell; drawing the unmasked peak grid")
        masked = grid

    pixelw, pixelh, mapdpi = 2048, 1152, 144
    extent = (-95.8, -89.1, 35.8, 40.8)
    data_crs = ccrs.PlateCarree()
    map_crs = ccrs.LambertConformal(central_longitude=-92.45, central_latitude=38.3)

    colors = ["#90EE90", "#FFED4E", "#FFA500", "#FF0000", "#8B0000"]
    labels = ["Low", "Moderate", "Elevated", "Critical", "Extreme"]
    bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bins, len(colors))

    fig = plt.figure(figsize=(pixelw / mapdpi, pixelh / mapdpi), dpi=mapdpi, facecolor="#E8E8E8")
    ax = plt.axes([0, 0, 1, 1], projection=map_crs)
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_extent(extent, crs=data_crs)

    cs = ax.contourf(
        lon_mesh, lat_mesh, masked, transform=data_crs,
        levels=bins, cmap=cmap, norm=norm, alpha=0.7, zorder=7, antialiased=True,
    )
    ax.contour(
        lon_mesh, lat_mesh, masked, transform=data_crs,
        levels=bins[1:-1], colors="black", linewidths=0.3, alpha=0.2, zorder=8,
    )

    counties = gpd.read_file(MAPS_DIR / "shapefiles" / "MO_County_Boundaries" / "MO_County_Boundaries.shp")
    if counties.crs and counties.crs != data_crs.proj4_init:
        counties = counties.to_crs(data_crs.proj4_init)
    ax.add_geometries(counties.geometry, crs=data_crs, edgecolor="#B6B6B6", facecolor="none", linewidth=1, zorder=5)

    missouriborder = gpd.read_file(MAPS_DIR / "shapefiles" / "MO_State_Boundary" / "MO_State_Boundary.shp")
    if missouriborder.crs and missouriborder.crs != data_crs.proj4_init:
        missouriborder = missouriborder.to_crs(data_crs.proj4_init)
    ax.add_geometries(missouriborder.geometry, crs=data_crs, edgecolor="#000000", facecolor="none", linewidth=1.5, zorder=6)

    cax = fig.add_axes([0.02, 0.08, 0.02, 0.6])
    cbar = plt.colorbar(cs, cax=cax, label="Fire Danger Level")
    cbar.set_ticks([0, 1, 2, 3, 4])
    cbar.set_ticklabels(labels)
    ax.set_anchor("W")
    fig.subplots_adjust(left=0.05)

    for font_path in (
        ASSETS_DIR / "Montserrat/static/Montserrat-Regular.ttf",
        ASSETS_DIR / "Plus_Jakarta_Sans/static/PlusJakartaSans-Regular.ttf",
        ASSETS_DIR / "Plus_Jakarta_Sans/static/PlusJakartaSans-Bold.ttf",
    ):
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
    plt.rcParams["font.family"] = "Montserrat"

    fig.text(0.99, 0.97, "Missouri Peak Fire Danger (RTMA)", fontsize=26, fontweight="bold", ha="right", va="top", fontname="Plus Jakarta Sans")
    fig.text(
        0.99, 0.90,
        f"RTMA Analysis Peak | Valid: {target_date.isoformat()} 10:00–21:00 CT",
        fontsize=16, ha="right", va="top", fontname="Montserrat",
    )
    fig.text(
        0.99, 0.62,
        "Peak Fire Danger from hourly RTMA analyses (10:00–21:00 CT)\n\n"
        "Fire Danger Criteria:\n"
        "Low: FM ≥ 15% (fuels too wet to spread significantly)\n\n"
        "Moderate: FM < 15% AND (RH < 45% OR Wind ≥ 10 kts)\n\n"
        "Elevated: FM < 9% AND\n"
        "  (RH < 35% & Wind ≥ 12 kts) OR (RH < 25% & Wind ≥ 5 kts)\n\n"
        "Critical: FM < 9% AND (RH < 25% & Wind ≥ 15 kts)\n\n"
        "Extreme: FM < 7% AND (RH < 20% & Wind ≥ 25 kts)\n\n"
        "Data Source: NOAA RTMA (t2m/r2/u10/v10)\n"
        f"Fuel moisture: {fuel_moisture_note}\n"
        "For More Info, Visit ShowMeFire.org",
        fontsize=10, ha="right", va="top", linespacing=1.6, fontname="Montserrat",
    )
    fig.text(0.02, 0.01, "ShowMeFire.org", fontsize=20, fontweight="bold", ha="left", va="bottom", fontname="Montserrat")

    svg_path = ASSETS_DIR / "LightBackGroundLogo.svg"
    try:
        import cairosvg
        png_bytes = cairosvg.svg2png(url=str(svg_path))
        logo = mpimg.imread(BytesIO(png_bytes), format="png")
        ax.add_artist(AnnotationBbox(OffsetImage(logo, zoom=0.03), (0.99, 0.01), frameon=False, xycoords="figure fraction", box_alignment=(1, 0)))
    except Exception:
        pass

    fig.savefig(out_path, dpi=mapdpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return out_path


def _load_boundary_layers(proj4_init: str):
    """County/state outlines reprojected to the map's CRS, shared by every branded render."""
    import geopandas as gpd

    counties = gpd.read_file(MAPS_DIR / "shapefiles" / "MO_County_Boundaries" / "MO_County_Boundaries.shp")
    if counties.crs and counties.crs != proj4_init:
        counties = counties.to_crs(proj4_init)
    missouriborder = gpd.read_file(MAPS_DIR / "shapefiles" / "MO_State_Boundary" / "MO_State_Boundary.shp")
    if missouriborder.crs and missouriborder.crs != proj4_init:
        missouriborder = missouriborder.to_crs(proj4_init)
    return counties, missouriborder


def _apply_branding_fonts():
    import matplotlib.font_manager as font_manager
    import matplotlib.pyplot as plt

    for font_path in (
        ASSETS_DIR / "Montserrat/static/Montserrat-Regular.ttf",
        ASSETS_DIR / "Plus_Jakarta_Sans/static/PlusJakartaSans-Regular.ttf",
        ASSETS_DIR / "Plus_Jakarta_Sans/static/PlusJakartaSans-Bold.ttf",
    ):
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
    plt.rcParams["font.family"] = "Montserrat"


def _draw_logo(ax):
    import matplotlib.image as mpimg
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    svg_path = ASSETS_DIR / "LightBackGroundLogo.svg"
    try:
        import cairosvg
        png_bytes = cairosvg.svg2png(url=str(svg_path))
        logo = mpimg.imread(BytesIO(png_bytes), format="png")
        ax.add_artist(AnnotationBbox(OffsetImage(logo, zoom=0.03), (0.99, 0.01), frameon=False, xycoords="figure fraction", box_alignment=(1, 0)))
    except Exception:
        pass


def _export_generic_geotiff(
    grid: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    out_path: Path,
    *,
    categorical: bool,
    dtype: str,
    nodata: float,
    band_description: str,
    run_date: datetime,
) -> None:
    """Write a single-band GeoTIFF on the canonical EPSG:32615 Missouri grid.

    Unlike export_fire_danger_gis.export_geotiff, this does not bin values
    into the 0-4 danger-category scale - it's for auxiliary continuous (fuel
    moisture) or small-integer (rainfall reduction) products.
    """
    import rasterio
    from services.gis_publisher import canonical_grid, regrid_lonlat

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    projected = regrid_lonlat(grid, lon, lat, categorical=categorical)
    regridded = np.where(np.isfinite(projected), projected, nodata).astype(dtype)
    grid_spec = canonical_grid()
    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=grid_spec["height"],
        width=grid_spec["width"],
        count=1,
        dtype=dtype,
        crs=grid_spec["crs"],
        transform=grid_spec["transform"],
        nodata=nodata,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(regridded, 1)
        dst.update_tags(BAND_1=band_description, GENERATED=run_date.isoformat())


def _raster_lon_lat_mesh(transform, crs, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Derive lon/lat meshes (EPSG:4326) from a regular projected raster's transform."""
    import rasterio
    from pyproj import Transformer

    rows, cols = np.indices(shape)
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(np.asarray(xs), np.asarray(ys))
    return np.asarray(lon).reshape(shape), np.asarray(lat).reshape(shape)


def _render_fuel_moisture_png(
    grid: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    out_path: Path,
    target_date: date,
    fuel_moisture_note: str,
) -> Path:
    """Render the day's driest calibrated fuel-moisture estimate.

    "Merged" in the sense that it's the RTMA RH-based estimate bias-corrected
    against RAWS station fuel-moisture observations where available - see
    _calibrate_fuel_moisture. Shows the driest (most fire-prone) hour of the
    day, the natural fuel-moisture counterpart to the danger peak product.
    """
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask, lon_mesh, lat_mesh = _missouri_mask(lon, lat)
    masked = np.where(mask, grid, np.nan) if mask.shape == grid.shape else grid
    if not np.isfinite(masked).any():
        logger.warning("Missouri mask dropped every RTMA cell; drawing the unmasked fuel-moisture grid")
        masked = grid

    pixelw, pixelh, mapdpi = 2048, 1152, 144
    extent = (-95.8, -89.1, 35.8, 40.8)
    data_crs = ccrs.PlateCarree()
    map_crs = ccrs.LambertConformal(central_longitude=-92.45, central_latitude=38.3)

    vmin, vmax = FUEL_MOISTURE_RANGE
    cmap = plt.get_cmap("BrBG")

    fig = plt.figure(figsize=(pixelw / mapdpi, pixelh / mapdpi), dpi=mapdpi, facecolor="#E8E8E8")
    ax = plt.axes([0, 0, 1, 1], projection=map_crs)
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_extent(extent, crs=data_crs)

    cs = ax.contourf(
        lon_mesh, lat_mesh, masked, transform=data_crs,
        levels=np.linspace(vmin, vmax, 21), cmap=cmap, vmin=vmin, vmax=vmax,
        alpha=0.75, zorder=7, antialiased=True, extend="both",
    )

    counties, missouriborder = _load_boundary_layers(data_crs.proj4_init)
    ax.add_geometries(counties.geometry, crs=data_crs, edgecolor="#B6B6B6", facecolor="none", linewidth=1, zorder=5)
    ax.add_geometries(missouriborder.geometry, crs=data_crs, edgecolor="#000000", facecolor="none", linewidth=1.5, zorder=6)

    cax = fig.add_axes([0.02, 0.08, 0.02, 0.6])
    plt.colorbar(cs, cax=cax, label="Fuel Moisture (%)")
    ax.set_anchor("W")
    fig.subplots_adjust(left=0.05)

    _apply_branding_fonts()

    fig.text(0.99, 0.97, "Missouri Fuel Moisture (RTMA + RAWS)", fontsize=26, fontweight="bold", ha="right", va="top", fontname="Plus Jakarta Sans")
    fig.text(
        0.99, 0.90,
        f"Driest calibrated estimate | Valid: {target_date.isoformat()} 10:00–21:00 CT",
        fontsize=16, ha="right", va="top", fontname="Montserrat",
    )
    fig.text(
        0.99, 0.75,
        "RTMA RH-based fuel moisture, bias-corrected against nearby\n"
        "RAWS station observations where available. Shows the driest\n"
        "(most fire-prone) hour of the 10:00–21:00 CT window.\n\n"
        f"Fuel moisture: {fuel_moisture_note}\n"
        "Data Source: NOAA RTMA (t2m/r2) + RAWS stations\n"
        "For More Info, Visit ShowMeFire.org",
        fontsize=10, ha="right", va="top", linespacing=1.6, fontname="Montserrat",
    )
    fig.text(0.02, 0.01, "ShowMeFire.org", fontsize=20, fontweight="bold", ha="left", va="bottom", fontname="Montserrat")

    _draw_logo(ax)
    fig.savefig(out_path, dpi=mapdpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return out_path


def _render_rainfall_impact_png(
    grid: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    out_path: Path,
    end_date: date,
    window_days: int,
    days_available: int,
) -> Path:
    """Render the worst rainfall-driven fire-danger suppression seen in the trailing window."""
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask, lon_mesh, lat_mesh = _missouri_mask(lon, lat)
    masked = np.where(mask, grid, np.nan) if mask.shape == grid.shape else grid
    if not np.isfinite(masked).any():
        logger.warning("Missouri mask dropped every RTMA cell; drawing the unmasked rainfall-impact grid")
        masked = grid

    pixelw, pixelh, mapdpi = 2048, 1152, 144
    extent = (-95.8, -89.1, 35.8, 40.8)
    data_crs = ccrs.PlateCarree()
    map_crs = ccrs.LambertConformal(central_longitude=-92.45, central_latitude=38.3)

    colors = ["#F0F0F0", "#9EC9E2", "#2166AC"]
    labels = ["No suppression", "-1 category", "-2 categories"]
    bins = [-0.5, 0.5, 1.5, 2.5]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bins, len(colors))

    fig = plt.figure(figsize=(pixelw / mapdpi, pixelh / mapdpi), dpi=mapdpi, facecolor="#E8E8E8")
    ax = plt.axes([0, 0, 1, 1], projection=map_crs)
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_extent(extent, crs=data_crs)

    cs = ax.contourf(
        lon_mesh, lat_mesh, masked, transform=data_crs,
        levels=bins, cmap=cmap, norm=norm, alpha=0.75, zorder=7, antialiased=True,
    )
    ax.contour(
        lon_mesh, lat_mesh, masked, transform=data_crs,
        levels=bins[1:-1], colors="black", linewidths=0.3, alpha=0.2, zorder=8,
    )

    counties, missouriborder = _load_boundary_layers(data_crs.proj4_init)
    ax.add_geometries(counties.geometry, crs=data_crs, edgecolor="#B6B6B6", facecolor="none", linewidth=1, zorder=5)
    ax.add_geometries(missouriborder.geometry, crs=data_crs, edgecolor="#000000", facecolor="none", linewidth=1.5, zorder=6)

    cax = fig.add_axes([0.02, 0.08, 0.02, 0.6])
    cbar = plt.colorbar(cs, cax=cax, label="Rainfall-Driven Danger Suppression")
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(labels)
    ax.set_anchor("W")
    fig.subplots_adjust(left=0.05)

    _apply_branding_fonts()

    fig.text(0.99, 0.97, "Missouri Rainfall Impact on Fire Danger", fontsize=26, fontweight="bold", ha="right", va="top", fontname="Plus Jakarta Sans")
    fig.text(
        0.99, 0.90,
        f"Trailing {window_days}-day window | Through {end_date.isoformat()} ({days_available}/{window_days} days available)",
        fontsize=16, ha="right", va="top", fontname="Montserrat",
    )
    fig.text(
        0.99, 0.75,
        "Worst same-day fire-danger category reduction that realized\n"
        "rainfall, fuel type, RH, and wind produced at each pixel over\n"
        "the trailing window.\n\n"
        "Data Source: NOAA RTMA + ShowMeFire fuel-moisture analysis\n"
        "For More Info, Visit ShowMeFire.org",
        fontsize=10, ha="right", va="top", linespacing=1.6, fontname="Montserrat",
    )
    fig.text(0.02, 0.01, "ShowMeFire.org", fontsize=20, fontweight="bold", ha="left", va="bottom", fontname="Montserrat")

    _draw_logo(ax)
    fig.savefig(out_path, dpi=mapdpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return out_path


def generate_rtma_peak(
    target_date: str | date | None = None,
    *,
    output_root: Path | None = None,
    experimental: bool = False,
    fuel_moisture_archive_dir: Path | None = None,
) -> dict:
    """Generate and archive the RTMA peak for a local date.

    Missing individual RTMA hours are skipped.  The run fails only when no
    usable hours exist, which makes historical retries safe and resumable.
    """
    local_date = _parse_local_date(target_date)
    fuel_archive = _load_fuel_moisture_archive(local_date, fuel_moisture_archive_dir)

    peak = None
    adjusted_peak = None
    lon = lat = None
    cumulative_precip = None
    fuel_source = None
    adjusted_hours = 0
    mrms_hours = 0
    used_hours = []
    measured_hours = 0
    measured_station_observations = 0
    fuel_moisture_min = None
    peak_reduction = None
    for hour in _hours_for_local_date(local_date):
        try:
            path = fetch_rtma(hour)
            with xr.open_dataset(path) as ds:
                fuel_observations = _fuel_moisture_observations(fuel_archive, hour)
                use_measurements = len(fuel_observations) >= MINIMUM_FUEL_MOISTURE_STATIONS
                current, current_lon, current_lat, hour_fuel_moisture = _classify_grid(
                    ds,
                    score_fire_danger if experimental else calculate_fire_danger,
                    "score" if experimental else None,
                    fuel_observations if use_measurements else None,
                )
                adjusted_current = current
                hour_reduction = None
                if fuel_source is None:
                    try:
                        fuel_source = load_fuel_model_raster()
                    except Exception as exc:
                        logger.info("Rainfall-adjusted RTMA map unavailable: %s", exc)
                        fuel_source = False
                if fuel_source:
                    try:
                        fuel_grid = _nearest_grid_lookup(
                            fuel_source[0], fuel_source[1], fuel_source[2],
                            current_lon, current_lat,
                        )
                        mrms_used = False
                        try:
                            mrms = load_mrms_grid(hour.replace(tzinfo=None))
                            precip = _nearest_grid_lookup(
                                mrms[0], mrms[1], mrms[2], current_lon, current_lat
                            )
                            mrms_hours += 1
                            mrms_used = True
                        except (FileNotFoundError, ValueError):
                            if mrms_enabled():
                                try:
                                    fetch_mrms(hour)
                                    mrms = load_mrms_grid(hour.replace(tzinfo=None))
                                    precip = _nearest_grid_lookup(
                                        mrms[0], mrms[1], mrms[2], current_lon, current_lat
                                    )
                                    mrms_hours += 1
                                    mrms_used = True
                                except Exception:
                                    logger.info("MRMS unavailable for %s; using RTMA APCP", hour)
                        if not mrms_used:
                            apcp = _squeeze2d(np.asarray(ds["apcp"].values, dtype=float))
                            if apcp.shape != current.shape:
                                raise ValueError("RTMA APCP shape does not match danger grid")
                            precip = apcp
                        if precip.shape != current.shape:
                            raise ValueError("realized precipitation shape does not match danger grid")
                        cumulative_precip = (
                            precip if cumulative_precip is None else cumulative_precip + precip
                        )
                        precip = cumulative_precip
                        rh_grid = _squeeze2d(np.asarray(ds["r2"].values, dtype=float))
                        wind_grid = np.hypot(
                            _squeeze2d(np.asarray(ds["u10"].values, dtype=float)),
                            _squeeze2d(np.asarray(ds["v10"].values, dtype=float)),
                        ) * 1.9438444924406
                        adjusted_current, hour_reduction = adjust_grid(
                            current,
                            precip,
                            fuel_grid,
                            relative_humidity=rh_grid,
                            wind_kts=wind_grid,
                        )
                        adjusted_hours += 1
                    except Exception:
                        logger.exception("Unable to apply rainfall suppression for RTMA hour %s", hour)
                if peak is None:
                    peak = current
                    adjusted_peak = adjusted_current
                    lon, lat = current_lon, current_lat
                elif current.shape == peak.shape and np.allclose(current_lon, lon) and np.allclose(current_lat, lat):
                    peak = np.fmax(peak, current)
                    adjusted_peak = np.fmax(adjusted_peak, adjusted_current)
                else:
                    logger.warning("Skipping RTMA hour %s because its grid does not match the first hour", hour)
                    continue
                fuel_moisture_min = (
                    hour_fuel_moisture if fuel_moisture_min is None
                    else np.fmin(fuel_moisture_min, hour_fuel_moisture)
                )
                if hour_reduction is not None:
                    peak_reduction = (
                        hour_reduction.astype(float) if peak_reduction is None
                        else np.fmax(peak_reduction, hour_reduction)
                    )
                used_hours.append(hour.isoformat())
                if use_measurements:
                    measured_hours += 1
                    measured_station_observations += len(fuel_observations)
        except Exception:
            logger.exception("Unable to process RTMA hour %s for %s", hour, local_date)

    if peak is None or not np.isfinite(peak).any():
        raise RuntimeError(f"No usable RTMA analyses found for {local_date}")
    if adjusted_peak is None:
        adjusted_peak = peak

    if measured_hours:
        fuel_moisture_mode = "rh_estimate_calibrated_with_raws"
        fuel_moisture_note = f"RTMA RH estimate calibrated with RAWS ({measured_hours}/{len(used_hours)} hours)"
    else:
        fuel_moisture_mode = "rh_estimate_only"
        fuel_moisture_note = "estimated from RTMA RH (archived RAWS measurements unavailable)"

    if output_root is None:
        gis_dir = Path(GIS_DIR)
        image_dir = Path(IMAGES_DIR)
        today_tif = RTMA_PEAK_TODAY_TIF
        today_png = RTMA_PEAK_TODAY_PNG
    else:
        gis_dir = Path(output_root) / "gis"
        image_dir = Path(output_root) / "images"
        today_tif = gis_dir / "rtma_peak_today.tif"
        today_png = image_dir / "rtma_peak_today.png"
    tif_dir = gis_dir / "rtma_peak" / "archive"
    png_dir = image_dir / "rtma_peak" / "archive"
    tif_path = tif_dir / f"{local_date.isoformat()}.tif"
    adjusted_tif_path = gis_dir / "rtma_peak_rainfall_adjusted" / "archive" / f"{local_date.isoformat()}.tif"
    png_path = png_dir / f"{local_date.isoformat()}.png"
    metadata_path = tif_dir / f"{local_date.isoformat()}.json"
    today_tif.parent.mkdir(parents=True, exist_ok=True)
    if not export_geotiff(peak, lon, lat, tif_path, run_date=datetime.combine(local_date, datetime.min.time(), tzinfo=CHICAGO_TZ)):
        raise RuntimeError(f"Failed to write RTMA peak GeoTIFF for {local_date}")
    adjusted_tif_path.parent.mkdir(parents=True, exist_ok=True)
    if not export_geotiff(
        adjusted_peak,
        lon,
        lat,
        adjusted_tif_path,
        run_date=datetime.combine(local_date, datetime.min.time(), tzinfo=CHICAGO_TZ),
    ):
        logger.warning("Failed to write rainfall-adjusted RTMA GeoTIFF for %s", local_date)
    import rasterio
    with rasterio.open(tif_path, "r+") as destination:
        destination.update_tags(
            SOURCE="NOAA RTMA + ShowMeFire fuel-moisture analysis",
            FUEL_MOISTURE_MODE=fuel_moisture_mode,
            RAWS_MEASURED_HOURS=str(measured_hours),
            RTMA_HOURS_USED=str(len(used_hours)),
            RAINFALL_ADJUSTMENT_CONTRACT=CONTRACT_VERSION,
        )
    shutil.copy2(tif_path, today_tif)
    _render_png(peak, lon, lat, png_path, local_date, fuel_moisture_note)
    today_png.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(png_path, today_png)

    run_datetime = datetime.combine(local_date, datetime.min.time(), tzinfo=CHICAGO_TZ)

    fuel_moisture_tif_dir = gis_dir / "rtma_fuel_moisture" / "archive"
    fuel_moisture_png_dir = image_dir / "rtma_fuel_moisture" / "archive"
    fuel_moisture_today_tif = (
        RTMA_FUEL_MOISTURE_TODAY_TIF if output_root is None else gis_dir / "rtma_fuel_moisture_today.tif"
    )
    fuel_moisture_today_png = (
        RTMA_FUEL_MOISTURE_TODAY_PNG if output_root is None else image_dir / "rtma_fuel_moisture_today.png"
    )
    fuel_moisture_map = None
    try:
        fuel_moisture_tif_path = fuel_moisture_tif_dir / f"{local_date.isoformat()}.tif"
        fuel_moisture_png_path = fuel_moisture_png_dir / f"{local_date.isoformat()}.png"
        _export_generic_geotiff(
            fuel_moisture_min, lon, lat, fuel_moisture_tif_path,
            categorical=False, dtype="float32", nodata=-9999.0,
            band_description="RTMA RH-based fuel moisture (%), RAWS-calibrated where available; daily minimum",
            run_date=run_datetime,
        )
        fuel_moisture_today_tif.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fuel_moisture_tif_path, fuel_moisture_today_tif)
        _render_fuel_moisture_png(fuel_moisture_min, lon, lat, fuel_moisture_png_path, local_date, fuel_moisture_note)
        fuel_moisture_today_png.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fuel_moisture_png_path, fuel_moisture_today_png)
        fuel_moisture_map = {
            "tif": f"rtma_fuel_moisture/archive/{local_date.isoformat()}.tif",
            "png": f"rtma_fuel_moisture/archive/{local_date.isoformat()}.png",
        }
    except Exception:
        logger.exception("Fuel moisture map export failed for %s", local_date)

    rainfall_reduction_tif = None
    if peak_reduction is not None:
        try:
            reduction_archive_dir = gis_dir / "rtma_rainfall_reduction" / "archive"
            reduction_tif_path = reduction_archive_dir / f"{local_date.isoformat()}.tif"
            _export_generic_geotiff(
                peak_reduction, lon, lat, reduction_tif_path,
                categorical=True, dtype="uint8", nodata=255,
                band_description="Worst same-day rainfall-driven danger category reduction: 0-2",
                run_date=run_datetime,
            )
            rainfall_reduction_tif = f"rtma_rainfall_reduction/archive/{local_date.isoformat()}.tif"
        except Exception:
            logger.exception("Rainfall reduction export failed for %s", local_date)
    rainfall_impact_map = None
    try:
        rainfall_impact_map = generate_rainfall_impact_map(local_date, output_root=output_root)
    except Exception:
        logger.exception("Rainfall impact map generation failed for %s", local_date)

    result = {
        "date": local_date.isoformat(),
        "hours_used": len(used_hours),
        "window": "10:00-21:00 CT",
        "peak_class": int(np.nanmax(peak)),
        "tif": f"rtma_peak/archive/{local_date.isoformat()}.tif",
        "png": f"rtma_peak/archive/{local_date.isoformat()}.png",
        "rainfall_adjusted_tif": (
            f"rtma_peak_rainfall_adjusted/archive/{local_date.isoformat()}.tif"
            if adjusted_hours else None
        ),
        "rainfall_adjustment": {
            "contract_version": CONTRACT_VERSION,
            "hours_applied": adjusted_hours,
            "fuel_source": fuel_source[3] if fuel_source else None,
            "provider": (
                "mrms"
                if mrms_hours == len(used_hours) and mrms_hours
                else "mrms_then_rtma" if mrms_hours
                else "rtma"
            ),
            "mrms_hours": mrms_hours,
            "rtma_fallback_hours": max(0, adjusted_hours - mrms_hours),
        },
        "experimental": experimental,
        "fuel_moisture": {
            "mode": fuel_moisture_mode,
            "measured_hours": measured_hours,
            "total_hours": len(used_hours),
            "station_observations_used": measured_station_observations,
            "minimum_stations_per_hour": MINIMUM_FUEL_MOISTURE_STATIONS,
        },
        "fuel_moisture_map": fuel_moisture_map,
        "rainfall_reduction_tif": rainfall_reduction_tif,
        "rainfall_impact_map": rainfall_impact_map,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_metadata_path = metadata_path.with_suffix(".json.tmp")
    temporary_metadata_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary_metadata_path.replace(metadata_path)
    return result


def generate_rainfall_impact_map(
    end_date: str | date | None = None,
    days: int = RTMA_IMPACT_DEFAULT_DAYS,
    output_root: Path | None = None,
) -> dict:
    """Combine the trailing `days` of daily rainfall-suppression grids into one map.

    Each day's grid (written by generate_rtma_peak as rtma_rainfall_reduction)
    already holds the worst same-day category reduction rainfall produced;
    this takes the elementwise max across the window so a pixel that got
    relief on any recent day still shows it, even after that day stops being
    "today". Missing days (job didn't run, or predate this feature) are
    skipped - resumable/best-effort like generate_rtma_peak itself.
    """
    import rasterio

    local_date = _parse_local_date(end_date)
    if output_root is None:
        gis_dir = Path(GIS_DIR)
        image_dir = Path(IMAGES_DIR)
        today_tif = RTMA_IMPACT_TODAY_TIF
        today_png = RTMA_IMPACT_TODAY_PNG
    else:
        gis_dir = Path(output_root) / "gis"
        image_dir = Path(output_root) / "images"
        today_tif = gis_dir / "rtma_rainfall_impact_today.tif"
        today_png = image_dir / "rtma_rainfall_impact_today.png"
    reduction_archive_dir = gis_dir / "rtma_rainfall_reduction" / "archive"

    combined = None
    transform = crs = None
    dates_used = []
    for offset in range(days):
        day = local_date - timedelta(days=offset)
        path = reduction_archive_dir / f"{day.isoformat()}.tif"
        if not path.is_file():
            continue
        try:
            with rasterio.open(path) as src:
                band = src.read(1).astype(float)
                band[band == src.nodata] = np.nan
                if combined is None:
                    combined = band
                    transform, crs = src.transform, src.crs
                elif band.shape == combined.shape:
                    combined = np.fmax(combined, band)
                else:
                    logger.warning("Skipping %s in rainfall impact window: grid shape mismatch", path)
                    continue
            dates_used.append(day.isoformat())
        except Exception:
            logger.exception("Unable to read rainfall reduction grid %s", path)

    if combined is None:
        raise RuntimeError(
            f"No rainfall-reduction grids available in the trailing {days} days through {local_date}"
        )

    tif_dir = gis_dir / "rtma_rainfall_impact" / "archive"
    png_dir = image_dir / "rtma_rainfall_impact" / "archive"
    tif_path = tif_dir / f"{local_date.isoformat()}.tif"
    png_path = png_dir / f"{local_date.isoformat()}.png"
    tif_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    regridded = np.where(np.isfinite(combined), combined, 255).astype("uint8")
    with rasterio.open(
        tif_path, "w",
        driver="GTiff",
        height=regridded.shape[0],
        width=regridded.shape[1],
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform,
        nodata=255,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(regridded, 1)
        dst.update_tags(
            BAND_1="Worst rainfall-driven fire-danger category reduction over trailing window: 0-2",
            WINDOW_DAYS=str(days),
            DAYS_AVAILABLE=str(len(dates_used)),
            THROUGH_DATE=local_date.isoformat(),
        )
    today_tif.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tif_path, today_tif)

    lon_mesh, lat_mesh = _raster_lon_lat_mesh(transform, crs, regridded.shape)
    _render_rainfall_impact_png(combined, lon_mesh, lat_mesh, png_path, local_date, days, len(dates_used))
    today_png.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(png_path, today_png)

    return {
        "date": local_date.isoformat(),
        "window_days": days,
        "days_available": len(dates_used),
        "dates_used": dates_used,
        "tif": f"rtma_rainfall_impact/archive/{local_date.isoformat()}.tif",
        "png": f"rtma_rainfall_impact/archive/{local_date.isoformat()}.png",
    }


def generate_rtma_peak_for_verification(target_date: str | date | None = None) -> dict | None:
    """Best-effort RTMA peak for the verification date; never raise to the caller."""
    try:
        result = generate_rtma_peak(target_date)
        logger.info("RTMA peak generated for verification: %s", result)
        return result
    except Exception:
        logger.exception("RTMA peak generation failed for verification date %s", target_date)
        return None


async def run_rtma_peak_job():
    try:
        async with get_rtma_job_lock():
            result = await run_in_process_pool_async(generate_rtma_peak)
        logger.info("RTMA peak generated: %s", result)
    except Exception:
        logger.exception("Scheduled RTMA peak generation failed")
