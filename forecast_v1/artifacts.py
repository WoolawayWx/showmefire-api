from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr
from rasterio.enums import Resampling
from rasterio.shutil import copy as raster_copy
from rasterio.transform import from_origin
import rasterio

from .contracts import ARCHIVE_ENCODINGS, PUBLIC_GRID, VARIABLE_UNITS, utc_rfc3339


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _netcdf_encoding(dataset: xr.Dataset) -> dict:
    encoding = {}
    for name in dataset.data_vars:
        spec = ARCHIVE_ENCODINGS.get(name)
        if not spec:
            encoding[name] = {"zlib": True, "complevel": 6, "shuffle": True}
            continue
        item = {"dtype": spec.dtype, "_FillValue": spec.fill_value, "zlib": True, "complevel": 6, "shuffle": True}
        if spec.scale_factor is not None:
            item["scale_factor"] = spec.scale_factor
        encoding[name] = item
    return encoding


def write_netcdf(dataset: xr.Dataset, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    dataset.to_netcdf(temp, engine="netcdf4", encoding=_netcdf_encoding(dataset))
    with xr.open_dataset(temp) as reopened:
        if dict(reopened.sizes) != dict(dataset.sizes):
            raise ValueError(f"NetCDF round-trip dimension mismatch for {target}")
    os.replace(temp, target)
    return target


def write_cog(data: xr.DataArray, path: str | Path, *, variable: str, band_times: Iterable[str], categorical: bool = False, byte_data: bool = False) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(data.values)
    if values.ndim == 2:
        values = values[np.newaxis, ...]
    if values.shape[-2:] != (PUBLIC_GRID.height, PUBLIC_GRID.width):
        raise ValueError(f"public COG must be {PUBLIC_GRID.height}x{PUBLIC_GRID.width}, got {values.shape[-2:]}")
    dtype = "uint8" if categorical or byte_data else "float32"
    nodata = 255 if categorical or byte_data else -9999.0
    encoded = np.where(np.isfinite(values), values, nodata).astype(dtype)
    temp = target.with_suffix(".working.tif")
    with rasterio.open(
        temp, "w", driver="GTiff", width=PUBLIC_GRID.width, height=PUBLIC_GRID.height,
        count=encoded.shape[0], dtype=dtype, crs=PUBLIC_GRID.crs,
        transform=from_origin(PUBLIC_GRID.west, PUBLIC_GRID.north, PUBLIC_GRID.resolution_m, PUBLIC_GRID.resolution_m),
        tiled=True, blockxsize=256, blockysize=256, interleave="band", compress="DEFLATE",
        predictor=1 if dtype == "uint8" else 3, nodata=nodata,
    ) as dst:
        dst.write(encoded)
        timestamps = list(band_times)
        for band in range(1, encoded.shape[0] + 1):
            valid_time = timestamps[band - 1] if band - 1 < len(timestamps) else ""
            dst.set_band_description(band, f"{variable};lead={band - 1};valid={valid_time}")
            dst.update_tags(band, lead_hour=str(band - 1), valid_time=valid_time, units=VARIABLE_UNITS.get(variable, str(data.attrs.get("units", ""))))
        factors = [factor for factor in (2, 4, 8, 16) if min(PUBLIC_GRID.width, PUBLIC_GRID.height) // factor >= 16]
        if factors:
            dst.build_overviews(factors, Resampling.nearest if categorical else Resampling.average)
            dst.update_tags(ns="rio_overview", resampling="nearest" if categorical else "average")
    raster_copy(temp, target, driver="COG", compress="DEFLATE", blocksize=256, overview_resampling="nearest" if categorical else "average")
    temp.unlink(missing_ok=True)
    with rasterio.open(target) as check:
        if check.count != encoded.shape[0] or not check.is_tiled:
            raise ValueError(f"COG validation failed for {target}")
    return target


def write_points_parquet(rows: list[dict], path: str | Path) -> Path:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Parquet publication requires the pinned pyarrow dependency") from exc
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    sort_keys = [(name, "ascending") for name in ("station_id", "model", "member", "valid_time_utc") if name in table.column_names]
    if sort_keys:
        table = table.sort_by(sort_keys)
    pq.write_table(table, target, compression="zstd", use_dictionary=True, row_group_size=10_000)
    return target


def write_manifest(payload: dict, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, target)
    return target


# Same house style every other map on the site uses (forecast/DailyForecast.py
# et al: create_base_map/add_boundaries/add_title_and_branding). Ported here
# rather than imported since those live in standalone forecast scripts, not a
# shared module - kept in sync by eye, not by reference.
_APP_ROOT = Path("/app") if Path("/app").exists() else Path(__file__).resolve().parent.parent
_COUNTY_SHAPEFILE = _APP_ROOT / "maps/shapefiles/MO_County_Boundaries/MO_County_Boundaries.shp"
_STATE_SHAPEFILE = _APP_ROOT / "maps/shapefiles/MO_State_Boundary/MO_State_Boundary.shp"
_LOGO_SVG = _APP_ROOT / "assets/LightBackGroundLogo.svg"
_FONT_PATHS = (
    _APP_ROOT / "assets/Montserrat/static/Montserrat-Regular.ttf",
    _APP_ROOT / "assets/Plus_Jakarta_Sans/static/PlusJakartaSans-Regular.ttf",
    _APP_ROOT / "assets/Plus_Jakarta_Sans/static/PlusJakartaSans-Bold.ttf",
)
_MAP_EXTENT = (-95.8, -89.1, 35.8, 40.8)
_MAP_PIXELS = (2048, 1152)
_MAP_DPI = 144
_FIRE_DANGER_CRITERIA = (
    "Fire Danger Criteria:\n"
    "Low:  FM ≥ 15% (fuels too wet to spread significantly)\n\n"
    "Moderate:  FM < 15% AND (RH < 45% OR Wind ≥ 10 kts)\n\n"
    "Elevated:  FM < 9% WITH (RH < 35% and Wind >= 12) or (RH < 25% and Wind >= 5)\n"
    "Critical:  FM < 9% WITH (RH < 25% AND Wind >= 15 kts)\n\n"
    "Extreme:  FM < 7% WITH (RH < 20% AND Wind >= 25 kts)"
)


def _grid_lonlat():
    """Pixel-center lon/lat for the public UTM grid, for cartopy plotting."""
    from pyproj import Transformer

    cols = PUBLIC_GRID.west + (np.arange(PUBLIC_GRID.width) + 0.5) * PUBLIC_GRID.resolution_m
    rows = PUBLIC_GRID.north - (np.arange(PUBLIC_GRID.height) + 0.5) * PUBLIC_GRID.resolution_m
    easting, northing = np.meshgrid(cols, rows)
    transformer = Transformer.from_crs(PUBLIC_GRID.crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(easting, northing)
    return lon, lat


def _mask_to_state(values: np.ndarray, lon: np.ndarray, lat: np.ndarray, state_geometry) -> np.ndarray:
    """NaN out grid cells outside Missouri - the public grid extends a degree
    past the state as a research buffer, but every other map on the site
    only ever shows data inside the state line."""
    import shapely.vectorized

    inside = shapely.vectorized.contains(state_geometry, lon, lat)
    masked = values.copy()
    masked[~inside] = np.nan
    return masked


def write_static_graphic(
    data: xr.DataArray,
    png_path: str | Path,
    webp_path: str | Path,
    *,
    title: str,
    valid_period: str,
    run_time: str,
    status: str,
    categorical: bool = False,
    boundary_geojson: str | Path | None = None,
) -> tuple[Path, Path]:
    """Render one archival PNG/WebP in the same branded style (projection,
    county/state boundaries, title block, logo) as every other map the site
    publishes.

    `boundary_geojson` is accepted for backward compatibility but unused -
    it pointed at GIS_DIR ("/app/gis"), which in production is a separate
    volume mount that shadows the repo's own gis/ directory at that same
    container path, so the file was never actually found and masking/the
    boundary line silently never ran. State/county boundaries now come from
    maps/shapefiles/, which is plain repo content and always present.
    """
    import cartopy.crs as ccrs
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib import font_manager
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from PIL import Image as PILImage

    png = Path(png_path)
    webp = Path(webp_path)
    png.parent.mkdir(parents=True, exist_ok=True)
    webp.parent.mkdir(parents=True, exist_ok=True)

    values = np.asarray(data.values, dtype=float)
    if categorical:
        values[values == 255] = np.nan
    lon, lat = _grid_lonlat()
    have_boundaries = _COUNTY_SHAPEFILE.is_file() and _STATE_SHAPEFILE.is_file()
    state = gpd.read_file(_STATE_SHAPEFILE).to_crs(epsg=4326) if have_boundaries else None
    if state is not None:
        from shapely.ops import unary_union

        values = _mask_to_state(values, lon, lat, unary_union(state.geometry))

    data_crs = ccrs.PlateCarree()
    map_crs = ccrs.LambertConformal(central_longitude=-92.45, central_latitude=38.3)
    figure = plt.figure(figsize=(_MAP_PIXELS[0] / _MAP_DPI, _MAP_PIXELS[1] / _MAP_DPI), dpi=_MAP_DPI, facecolor="#E8E8E8")
    axis = plt.axes((0, 0, 1, 1), projection=map_crs)
    axis.set_frame_on(False)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_extent(_MAP_EXTENT, crs=data_crs)

    if categorical:
        colors = ["#90EE90", "#FFED4E", "#FFA500", "#FF0000", "#8B0000"]
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(bins, len(colors))
        image = axis.contourf(lon, lat, values, transform=data_crs, levels=bins, cmap=cmap, norm=norm, alpha=0.85, zorder=7)
        axis.contour(lon, lat, values, transform=data_crs, levels=bins[1:-1], colors="black", linewidths=0.3, alpha=0.2, zorder=8)
    else:
        image = axis.pcolormesh(lon, lat, values, transform=data_crs, cmap="viridis", shading="nearest", alpha=0.85, zorder=7)

    if have_boundaries:
        counties = gpd.read_file(_COUNTY_SHAPEFILE).to_crs(epsg=4326)
        axis.add_geometries(counties.geometry, crs=data_crs, edgecolor="#B6B6B6", facecolor="none", linewidth=1, zorder=5)
        axis.add_geometries(state.geometry, crs=data_crs, edgecolor="#000000", facecolor="none", linewidth=1.5, zorder=9)

    cax = figure.add_axes((0.02, 0.08, 0.02, 0.6))
    if categorical:
        colorbar = figure.colorbar(image, cax=cax)
        colorbar.set_ticks([0, 1, 2, 3, 4])
        colorbar.set_ticklabels(["Low", "Moderate \nHigh", "Elevated \nHigh", "Critical \n Very High", "Extreme"])
    else:
        figure.colorbar(image, cax=cax, label=str(data.attrs.get("units", "")))
    axis.set_anchor("W")
    plt.subplots_adjust(left=0.05)

    for font_path in _FONT_PATHS:
        if font_path.is_file():
            font_manager.fontManager.addfont(str(font_path))
    plt.rcParams["font.family"] = "Montserrat"
    valid_date = valid_period.split(" ", 1)[0]
    try:
        run_label = datetime.fromisoformat(run_time.replace("Z", "+00:00")).strftime("%Y-%m-%d %HZ")
    except ValueError:
        run_label = run_time
    subtitle = f"Model Run: {run_label} | Valid: {valid_date}"
    description = (
        (_FIRE_DANGER_CRITERIA + "\n\n") if categorical else ""
    ) + f"{status}\nData Source: HRRR/RRFS/GEFS Blend | ShowMeFire Forecast-V1\nFor More Info, Visit ShowMeFire.org"
    figure.text(0.99, 0.97, title, fontsize=26, fontweight="bold", ha="right", va="top", fontname="Plus Jakarta Sans")
    figure.text(0.99, 0.90, subtitle, fontsize=16, ha="right", va="top", fontname="Montserrat")
    figure.text(0.99, 0.62, description, fontsize=10, ha="right", va="top", linespacing=1.6, fontname="Montserrat")
    figure.text(0.02, 0.01, "ShowMeFire.org", fontsize=20, fontweight="bold", ha="left", va="bottom", fontname="Montserrat")
    if _LOGO_SVG.is_file():
        try:
            import cairosvg
            from io import BytesIO

            png_bytes = cairosvg.svg2png(url=str(_LOGO_SVG))
            image_data = mpimg.imread(BytesIO(png_bytes), format="png")
            imagebox = OffsetImage(image_data, zoom=0.03)
            annotation = AnnotationBbox(imagebox, (0.99, 0.01), frameon=False, xycoords="figure fraction", box_alignment=(1, 0))
            axis.add_artist(annotation)
        except (ImportError, FileNotFoundError, OSError):
            pass

    figure.savefig(png, format="png", dpi=_MAP_DPI, metadata={"Title": title, "Description": f"{valid_period}; {status}"})
    plt.close(figure)
    with PILImage.open(png) as source:
        source.save(webp, "WEBP", quality=86, method=6)
    return png, webp


def artifact_record(path: Path, *, run_id: str, kind: str, variable: str = "", aggregation: str = "", object_key: str | None = None, dtype: str = "binary", unit: str | None = None, valid_start: str | None = None, valid_end: str | None = None) -> dict:
    return {
        "run_id": run_id, "kind": kind, "variable": variable, "aggregation": aggregation,
        "valid_start_utc": valid_start, "valid_end_utc": valid_end, "unit": unit,
        "storage_data_type": dtype, "crs": PUBLIC_GRID.crs if kind == "raster" else None,
        "grid_id": PUBLIC_GRID.id if kind in {"raster", "cube"} else None,
        "object_key": object_key, "local_path": str(path.resolve()), "checksum": sha256_file(path),
        "byte_size": path.stat().st_size, "status": "ready",
    }


def promote_directory(staging: Path, final: Path) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        raise FileExistsError(f"immutable run directory already exists: {final}")
    os.replace(staging, final)
