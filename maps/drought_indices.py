"""Missouri drought index maps: EDDI, VegDRI, QuickDRI and ForDRI.

Same look as drought_monitor.py / the station maps (standard Missouri basemap,
right-side title block, fill masked to the state). None of these change more
than once a day (EDDI daily, the three NDMC indices weekly), but maps.sh runs
every script here every 15 minutes, so each product first makes a cheap HEAD
request and only downloads and renders when its source has actually changed
(the signature of the last render is kept next to its status JSON).

Sources:
  EDDI      NOAA PSL daily CONUS grids (1/8 deg ASCII), several timescales
  VegDRI    NDMC/USGS 1 km GeoTIFF, mirrored by NOAA NCEI for drought.gov
  QuickDRI  NDMC/USGS 1 km GeoTIFF, mirrored by NOAA NCEI for drought.gov
  ForDRI    NDMC weekly 0.045 deg GeoTIFF from fordri.unl.edu
"""
import io
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio
import requests
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import from_bounds, from_origin
from rasterio.warp import Resampling, reproject

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_DIR))

from maps import raster_maps as rm
from maps import station_plot as sp
from maps import station_products as spx

HEADERS = {"User-Agent": "ShowMeFire/1.0 graphics@showmefire.org"}

# Lon/lat grid every source is resampled onto before drawing: a little larger
# than the map frame, ~500 m cells so the 1 km products keep their detail.
GRID_BOUNDS = (-96.2, 35.4, -88.7, 41.2)
GRID_RES = 0.005
GRID_W = round((GRID_BOUNDS[2] - GRID_BOUNDS[0]) / GRID_RES)
GRID_H = round((GRID_BOUNDS[3] - GRID_BOUNDS[1]) / GRID_RES)
GRID_TRANSFORM = from_bounds(*GRID_BOUNDS, GRID_W, GRID_H)

# The NCEI copies of the NDMC GeoTIFFs are tagged EPSG:4326 but are really in
# the US National Atlas equal-area projection (EPSG:2163, 1 km cells).
NDMC_CRS = CRS.from_proj4("+proj=laea +lat_0=45 +lon_0=-100 +x_0=0 +y_0=0 +a=6370997 +b=6370997 +units=m +no_defs")

EDDI_URL = "https://downloads.psl.noaa.gov/Projects/EDDI/CONUS_archive/data/{year}/EDDI_ETrs_{scale}_{day}.asc"
EDDI_SCALES = [
    ("01wk", "1wk", "1-Week", "week"),
    ("02wk", "2wk", "2-Week", "2 weeks"),
    ("01mn", "1mo", "1-Month", "month"),
    ("03mn", "3mo", "3-Month", "3 months"),
]
# EDDI is a standardized (inverse-normal) percentile, so the official
# percentile categories (ED0 = 70-80th ... ED4 = 98th+) map to these values.
EDDI_BREAKS = [-np.inf, -2.054, -1.645, -1.282, -0.842, -0.524, 0.524, 0.842, 1.282, 1.645, 2.054, np.inf]
# Official drought.gov EDDI colors, wettest to driest; None = near normal (unshaded).
EDDI_CLASSES = [
    ("EW4", "Exceptional Wet", "#0000AA"),
    ("EW3", "Extreme Wet", "#0000FF"),
    ("EW2", "Severe Wet", "#00AAFF"),
    ("EW1", "Moderate Wet", "#01FFFF"),
    ("EW0", "Abnormally Wet", "#AAFF55"),
    (None, "Near Normal", None),
    ("ED0", "Abnormally Dry", "#FFFF00"),
    ("ED1", "Moderate Drought", "#FCD37F"),
    ("ED2", "Severe Drought", "#FFAA00"),
    ("ED3", "Extreme Drought", "#E60000"),
    ("ED4", "Exceptional Drought", "#730000"),
]

VEGDRI_URL = "https://www.ncei.noaa.gov/pub/data/nidis/geotiff/ndmc-vegdri-1km.tif"
# VegDRI is stored as 128 + 16 * index; codes above 252 are land-cover flags.
# Category edges and colors match the drought.gov VegDRI legend.
VEGDRI_CLASSES = [
    (1, 64, "Extreme Drought", "#6b0000"),
    (65, 80, "Severe Drought", "#c81e32"),
    (81, 96, "Moderate Drought", "#e66900"),
    (97, 112, "Pre-Drought Stress", "#ffe500"),
    (113, 168, "Near Normal", "#ffffff"),
    (169, 184, "Unusually Moist", "#87fac2"),
    (185, 207, "Very Moist", "#53c289"),
    (208, 252, "Extremely Moist", "#288233"),
    (253, 253, "Water", "#0070fd"),
    (254, 254, "Out of Season", "#b2b2b2"),
    (255, 255, "Water", "#0070fd"),
]

QUICKDRI_URL = "https://www.ncei.noaa.gov/pub/data/nidis/geotiff/ndmc-quickdri-1km.tif"
QUICKDRI_SCALE = 100.0  # stored as SPEI-scaled index x 100
QUICKDRI_FLAGS = {-32765: ("Urban", "#808080"), -32764: ("Out of Season", "#cccccc"), -32763: ("Water", "#005ce6")}
# The official QuickDRI map is a gradient, not categories (quickdri.unl.edu FAQ).
QUICKDRI_CMAP = LinearSegmentedColormap.from_list(
    "quickdri", ["#a83800", "#f57c00", "#ffd24d", "#fff6c2", "#ffffff", "#c9f0ff", "#5ec8f0", "#0a7fb0"],
)

FORDRI_URL = "https://fordri.unl.edu/data/ForDRI_CONUS/tif/fordri_{day}.tif"
FORDRI_CMAP = LinearSegmentedColormap.from_list(
    "fordri", ["#730000", "#e60000", "#f99b3d", "#ffff00", "#f7f7f7", "#3ce60e", "#1ea08c", "#0a328c"],
)


# --- shared helpers -----------------------------------------------------------

def _signature_path(product: str) -> Path:
    return spx.STATUS_DIR / f"{product}.signature"


def _unchanged(product: str, png_name: str, signature: str) -> bool:
    try:
        return (spx.IMAGES_DIR / png_name).is_file() and _signature_path(product).read_text() == signature
    except OSError:
        return False


def _remember(product: str, signature: str) -> None:
    spx.STATUS_DIR.mkdir(parents=True, exist_ok=True)
    _signature_path(product).write_text(signature)


def _head(url: str) -> requests.Response:
    return requests.head(url, headers=HEADERS, timeout=30, allow_redirects=True)


def _http_signature(response: requests.Response) -> str:
    return f"{response.headers.get('ETag', '')}|{response.headers.get('Last-Modified', '')}"


def _get(url: str, timeout: int = 180) -> bytes:
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.content


def _state_mask() -> np.ndarray:
    state = sp._missouri_geometry()
    if state is None:
        return np.ones((GRID_H, GRID_W), dtype=bool)
    return rasterize([(state, 1)], out_shape=(GRID_H, GRID_W), transform=GRID_TRANSFORM, fill=0, dtype="uint8").astype(bool)


def _to_grid(source: np.ndarray, src_transform, src_crs, *, nodata) -> np.ndarray:
    """Resample a source raster onto the Missouri lon/lat grid (NaN = no data).

    Nearest-neighbor throughout: these are coarse or categorical grids, and
    showing their real cells is more honest than smoothing them."""
    destination = np.full((GRID_H, GRID_W), np.nan, dtype="float64")
    reproject(
        source.astype("float64"), destination, src_transform=src_transform, src_crs=src_crs, src_nodata=nodata,
        dst_transform=GRID_TRANSFORM, dst_crs="EPSG:4326", dst_nodata=np.nan,
        resampling=Resampling.nearest,
    )
    return destination


def _read_window(content: bytes, crs=None) -> np.ndarray:
    """Read just the Missouri part of a CONUS GeoTIFF and put it on the grid."""
    with rasterio.MemoryFile(content) as memfile, memfile.open() as src:
        src_crs = crs or src.crs
        from rasterio.warp import transform_bounds
        from rasterio.windows import from_bounds as window_from_bounds

        window = window_from_bounds(*transform_bounds("EPSG:4326", src_crs, *GRID_BOUNDS), src.transform).round_offsets().round_lengths()
        data = src.read(1, window=window, boundless=True, fill_value=src.nodata if src.nodata is not None else 0)
        return data, src.window_transform(window), src_crs, src.nodata, src.tags()


def _draw_rgba(ax, data_crs, rgba: np.ndarray) -> None:
    rgba = rgba.copy()
    rgba[~_state_mask(), 3] = 0.0
    west, south, east, north = GRID_BOUNDS
    ax.imshow(rgba, origin="upper", extent=(west, east, south, north), transform=data_crs,
              interpolation="nearest", regrid_shape=1600, zorder=7)


def _class_rgba(class_index: np.ndarray, colors: list) -> np.ndarray:
    """RGBA image from an integer class grid (-1 / None color = transparent)."""
    from matplotlib.colors import to_rgba

    table = np.array([to_rgba(color) if color else (0, 0, 0, 0) for color in colors] + [(0, 0, 0, 0)])
    return table[np.where(class_index < 0, len(colors), class_index)]


def _continuous_fill(fig, ax, data_crs, values: np.ndarray, *, cmap, vmin: float, vmax: float, label: str,
                     ticks: list) -> None:
    """Continuous fill with the station maps' colorbar (left-side wedge) and fill blend."""
    blended = rm.blend_cmap(cmap)
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    rgba = blended(norm(np.nan_to_num(values, nan=0.0)))
    rgba[~np.isfinite(values), 3] = 0.0
    _draw_rgba(ax, data_crs, rgba)
    rm.left_colorbar(fig, blended, norm, label, ticks=ticks)


def _sunday_on_or_before(day: date) -> date:
    return day - timedelta(days=(day.weekday() + 1) % 7)


def _tiff_valid_date(tags: dict) -> str:
    """NDMC weeks end on Sunday; the GeoTIFF is stamped when it was processed."""
    try:
        processed = datetime.strptime(tags["TIFFTAG_DATETIME"], "%Y:%m:%d %H:%M:%S").date()
    except (KeyError, ValueError):
        return "Latest weekly release"
    return "Week ending " + _sunday_on_or_before(processed).strftime("%b %d, %Y")


def _save(fig, ax, *, product: str, png_name: str, title: str, subtitle: str, description: str, started: float, **extra) -> None:
    sp.add_branding(fig, ax, title=title, subtitle=subtitle, description=description)
    status = {
        "product": product, "status": "updated", "valid": subtitle.removeprefix("Valid: "),
        "last_update": spx.valid_time_label(), "runtime_sec": round(time.time() - started, 2), **extra,
    }
    sp.save_and_record_status(fig, spx.IMAGES_DIR / png_name, spx.STATUS_DIR / f"{product}.json", status)
    print(f"{title} updated, {subtitle} ({status['runtime_sec']:.2f}s)")


def _area_share(classes: np.ndarray, labels: list) -> dict:
    """Percent of Missouri's valid pixels in each class, for the status JSON."""
    inside = classes[_state_mask()]
    inside = inside[inside >= 0]
    if not inside.size:
        return {}
    counts = np.bincount(inside, minlength=len(labels))
    return {labels[i]: round(100 * counts[i] / inside.size, 1) for i in range(len(labels)) if counts[i]}


# --- EDDI ---------------------------------------------------------------------

def _latest_eddi_day() -> tuple[date, requests.Response]:
    """Newest day with a published 2-week EDDI grid (PSL posts them a few days late)."""
    today = spx.now_central().date()
    for back in range(0, 15):
        day = today - timedelta(days=back)
        response = _head(EDDI_URL.format(year=day.year, scale="02wk", day=f"{day:%Y%m%d}"))
        if response.status_code == 200:
            return day, response
    raise RuntimeError("no EDDI grid published in the last 15 days")


def _read_eddi(content: bytes) -> np.ndarray:
    lines = content.decode("ascii").splitlines()
    header = {line.split()[0].lower(): float(line.split()[1]) for line in lines[:6]}
    values = np.loadtxt(io.StringIO("\n".join(lines[6:])), dtype="float64")
    transform = from_origin(header["xllcorner"], header["yllcorner"] + header["nrows"] * header["cellsize"],
                            header["cellsize"], header["cellsize"])
    return _to_grid(values, transform, "EPSG:4326", nodata=header["nodata_value"])


def render_eddi(scale: str, slug: str, label: str, period: str, day: date) -> None:
    started = time.time()
    values = _read_eddi(_get(EDDI_URL.format(year=day.year, scale=scale, day=f"{day:%Y%m%d}")))
    classes = np.where(np.isfinite(values), np.digitize(np.nan_to_num(values), EDDI_BREAKS[1:-1]), -1)
    colors = [color for _code, _name, color in EDDI_CLASSES]

    fig, ax, data_crs, _map_crs, _dpi = sp.generate_basemap()
    _draw_rgba(ax, data_crs, _class_rgba(classes, colors))
    handles = [Patch(facecolor=color or "none", edgecolor="#555555", linewidth=0.4,
                     label=f"{code}  {name}" if code else f"{name} (unshaded)")
               for code, name, color in reversed(EDDI_CLASSES)]
    rm.category_legend(fig, handles, "Evaporative Demand")
    _save(
        fig, ax, product=f"eddi_{slug}", png_name=f"mo-eddi-{slug}.png",
        title=f"Missouri {label} EDDI", subtitle=f"Valid: {day:%b %d, %Y}",
        description=(
            "Data Source: NOAA PSL Evaporative Demand\nDrought Index (EDDI), 1/8-degree grid\n\n"
            f"How unusual the atmosphere's thirst for water has\nbeen over the past {period} (warm, dry, windy,\n"
            "sunny). High demand dries fuels and soils fast\nand is an early warning of flash drought."
        ),
        started=started, timescale=label,
        area_percent=_area_share(classes, [code or "Normal" for code, _name, _color in EDDI_CLASSES]),
    )


def run_eddi() -> list:
    day, response = _latest_eddi_day()
    signature = f"{day:%Y%m%d}|{_http_signature(response)}"
    products = []
    for scale, slug, label, period in EDDI_SCALES:
        product = f"eddi_{slug}"
        if _unchanged(product, f"mo-eddi-{slug}.png", signature):
            print(f"EDDI {label} unchanged since {day}; skipping.")
            continue

        def build(scale=scale, slug=slug, label=label, period=period, product=product):
            render_eddi(scale, slug, label, period, day)
            _remember(product, signature)

        products.append((f"EDDI {label}", build))
    return products


# --- VegDRI -------------------------------------------------------------------

def render_vegdri() -> None:
    started = time.time()
    data, transform, crs, nodata, tags = _read_window(_get(VEGDRI_URL), NDMC_CRS)
    values = _to_grid(data, transform, crs, nodata=0)
    classes = np.full(values.shape, -1, dtype=int)
    for index, (low, high, _name, _color) in enumerate(VEGDRI_CLASSES):
        classes[(values >= low) & (values <= high)] = index
    colors = [color for _low, _high, _name, color in VEGDRI_CLASSES]

    fig, ax, data_crs, _map_crs, _dpi = sp.generate_basemap()
    _draw_rgba(ax, data_crs, _class_rgba(classes, colors))
    seen, handles = set(), []
    for _low, _high, name, color in VEGDRI_CLASSES:
        if name not in seen:
            seen.add(name)
            handles.append(Patch(facecolor=color, edgecolor="#555555", linewidth=0.4, label=name))
    rm.category_legend(fig, handles, "Vegetation Conditions")
    names = [name for _low, _high, name, _color in VEGDRI_CLASSES]
    _save(
        fig, ax, product="vegdri", png_name="mo-vegdri.png", title="Missouri VegDRI",
        subtitle=f"Valid: {_tiff_valid_date(tags)}",
        description=(
            "Data Source: Vegetation Drought Response Index\n(NDMC, USGS EROS, HPRCC) via NOAA NCEI\n\n"
            "Satellite vegetation greenness combined with\nclimate data to show drought stress on\n"
            "vegetation at 1 km. Updated weekly."
        ),
        started=started, area_percent=_area_share(classes, names),
    )


# --- QuickDRI -----------------------------------------------------------------

def render_quickdri() -> None:
    started = time.time()
    data, transform, crs, _nodata, tags = _read_window(_get(QUICKDRI_URL), NDMC_CRS)
    values = _to_grid(data, transform, crs, nodata=-32767)
    flags = {code: np.isclose(values, code) for code in QUICKDRI_FLAGS}
    index = np.where(values > -32000, values / QUICKDRI_SCALE, np.nan)

    fig, ax, data_crs, _map_crs, _dpi = sp.generate_basemap()
    _continuous_fill(fig, ax, data_crs, index, cmap=QUICKDRI_CMAP, vmin=-2.5, vmax=2.5,
                     label="Drier  \u2190  QuickDRI  \u2192  Wetter", ticks=[-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5])
    flag_class = np.full(values.shape, -1, dtype=int)
    for position, code in enumerate(QUICKDRI_FLAGS):
        flag_class[flags[code]] = position
    _draw_rgba(ax, data_crs, _class_rgba(flag_class, [color for _name, color in QUICKDRI_FLAGS.values()]))
    rm.category_legend(fig, [Patch(facecolor=color, edgecolor="#555555", linewidth=0.4, label=name)
                           for name, color in QUICKDRI_FLAGS.values()], "Other", y=0.30)
    inside = index[_state_mask() & np.isfinite(index)]
    _save(
        fig, ax, product="quickdri", png_name="mo-quickdri.png", title="Missouri QuickDRI",
        subtitle=f"Valid: {_tiff_valid_date(tags)}",
        description=(
            "Data Source: Quick Drought Response Index\n(NDMC, USGS, NASA) via NOAA NCEI\n\n"
            "Short-term (about 4 weeks) drying or improving\nconditions from soil moisture, ET, vegetation\n"
            "and SPEI. Orange shows rapid drying that can\nsignal emerging (flash) drought. Weekly."
        ),
        started=started,
        missouri_mean=round(float(inside.mean()), 2) if inside.size else None,
        missouri_pct_drier_than_minus1=round(100 * float((inside <= -1).mean()), 1) if inside.size else None,
    )


# --- ForDRI -------------------------------------------------------------------

def _latest_fordri_day() -> tuple[date, requests.Response]:
    """ForDRI is issued weekly for Mondays; look back over the last few."""
    today = spx.now_central().date()
    monday = today - timedelta(days=today.weekday())
    for week in range(0, 5):
        day = monday - timedelta(weeks=week)
        response = _head(FORDRI_URL.format(day=f"{day:%Y-%m-%d}"))
        if response.status_code == 200 and "tif" in response.headers.get("Content-Type", "tif"):
            return day, response
    raise RuntimeError("no ForDRI GeoTIFF in the last 5 weeks")


def render_fordri(day: date) -> None:
    started = time.time()
    data, transform, crs, nodata, _tags = _read_window(_get(FORDRI_URL.format(day=f"{day:%Y-%m-%d}")))
    data = np.where(np.isfinite(data) & (data > -9000), data, np.nan)
    values = _to_grid(data, transform, crs, nodata=np.nan)

    fig, ax, data_crs, _map_crs, _dpi = sp.generate_basemap()
    _continuous_fill(fig, ax, data_crs, values, cmap=FORDRI_CMAP, vmin=-2.0, vmax=2.0,
                     label="Drier  \u2190  ForDRI  \u2192  Wetter", ticks=[-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
    inside = values[_state_mask() & np.isfinite(values)]
    _save(
        fig, ax, product="fordri", png_name="mo-fordri.png", title="Missouri ForDRI",
        subtitle=f"Valid: {day:%b %d, %Y}",
        description=(
            "Data Source: Forest Drought Response Index\n(NDMC, USDA Forest Service)\n\n"
            "Drought stress in forests from satellite\nvegetation, soil moisture, ET and climate\n"
            "indices. Non-forest land is unshaded.\nUpdated weekly on Mondays."
        ),
        started=started,
        missouri_mean=round(float(inside.mean()), 2) if inside.size else None,
        missouri_pct_below_minus1=round(100 * float((inside <= -1).mean()), 1) if inside.size else None,
    )


# --- driver -------------------------------------------------------------------

def _weekly(product: str, png_name: str, url: str, render) -> list:
    response = _head(url)
    response.raise_for_status()
    signature = _http_signature(response)
    if _unchanged(product, png_name, signature):
        print(f"{product} unchanged since last render; skipping.")
        return []

    def build():
        render()
        _remember(product, signature)

    return [(product, build)]


def _fordri() -> list:
    day, response = _latest_fordri_day()
    signature = f"{day}|{_http_signature(response)}"
    if _unchanged("fordri", "mo-fordri.png", signature):
        print(f"ForDRI unchanged since {day}; skipping.")
        return []

    def build():
        render_fordri(day)
        _remember("fordri", signature)

    return [("ForDRI", build)]


# None of these sources publish more than once a day, so once a source has
# been checked and found unchanged it is not asked again for this long.
RECHECK_MINUTES = 60


def _checked_recently(name: str) -> bool:
    try:
        stamp = spx.STATUS_DIR / f"drought_{name.lower()}.checked"
        return time.time() - stamp.stat().st_mtime < RECHECK_MINUTES * 60
    except OSError:
        return False


def _mark_checked(name: str) -> None:
    spx.STATUS_DIR.mkdir(parents=True, exist_ok=True)
    (spx.STATUS_DIR / f"drought_{name.lower()}.checked").touch()


def main() -> None:
    products = []
    # A source being down must not stop the others from being checked.
    for name, collect in [
        ("EDDI", run_eddi),
        ("VegDRI", lambda: _weekly("vegdri", "mo-vegdri.png", VEGDRI_URL, render_vegdri)),
        ("QuickDRI", lambda: _weekly("quickdri", "mo-quickdri.png", QUICKDRI_URL, render_quickdri)),
        ("ForDRI", _fordri),
    ]:
        if _checked_recently(name):
            continue
        try:
            found = collect()
        except requests.RequestException as exc:
            # The last map stays valid; just try again next cycle.
            print(f"{name}: source unreachable ({type(exc).__name__}); will retry next cycle.")
            continue
        except Exception as exc:  # noqa: BLE001 - reported via run_products below
            def failed(exc=exc):
                raise exc
            products.append((f"{name} check", failed))
            continue
        if found:
            products.extend(found)  # not marked: a failed render is retried next cycle
        else:
            _mark_checked(name)
    if not products:
        print("Drought indices: nothing new to render.")
        return
    spx.run_products(products, label="Drought index maps")


if __name__ == "__main__":
    main()
