"""Render current NOAA RFC QPE precipitation products for Missouri."""

from __future__ import annotations

import base64
import logging
import re
import textwrap
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import cartopy.crs as ccrs
import matplotlib.font_manager as font_manager
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import requests
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Rectangle
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from scipy.ndimage import gaussian_filter

from maps import station_plot as sp

PROJECT_DIR = Path(__file__).resolve().parents[1]
SERVICE_URL = "https://mapservices.weather.noaa.gov/raster/rest/services/obs/rfc_qpe/MapServer"
CENTRAL = ZoneInfo("America/Chicago")
EXTENT = (-95.8, -89.1, 35.8, 40.8)
def _accumulation(stem: str, period: str, title_period: str):
    return (stem, f"Last {period} Observed (inches)", f"{title_period} Precipitation", "inches")


def _departure(stem: str, layer_period: str, title_period: str):
    """Observed minus normal, in inches (NOAA's "Departure from Normal" layers)."""
    return (stem, f"{layer_period} Departure from Normal (inches)", f"{title_period} Precipitation Departure from Normal", "departure")


PRODUCTS = (
    _accumulation("precip-1hr", "1 Hours", "1-Hour"),
    _accumulation("precip-3hr", "3 Hours", "3-Hour"),
    _accumulation("precip-6hr", "6 Hours", "6-Hour"),
    _accumulation("precip-12hr", "12 Hours", "12-Hour"),
    ("precip-24hr", "Last 24 Hours Observed (inches)", "24-Hour Precipitation", "inches"),
    _accumulation("precip-2day", "2 Days", "2-Day"),
    _accumulation("precip-3day", "3 Days", "3-Day"),
    _accumulation("precip-4day", "4 Days", "4-Day"),
    _accumulation("precip-5day", "5 Days", "5-Day"),
    ("precip-7day", "Last 7 Days Observed (inches)", "7-Day Precipitation", "inches"),
    _accumulation("precip-10day", "10 Days", "10-Day"),
    _accumulation("precip-14day", "14 Days", "14-Day"),
    ("precip-30day", "Last 30 Days Observed (inches)", "30-Day Precipitation", "inches"),
    ("precip-60day", "Last 60 Days Observed (inches)", "60-Day Precipitation", "inches"),
    _accumulation("precip-90day", "90 Days", "90-Day"),
    _accumulation("precip-120day", "120 Days", "120-Day"),
    _accumulation("precip-180day", "180 Days", "180-Day"),
    _accumulation("precip-365day", "365 Days", "365-Day"),
    ("precip-30day-percent-normal", "Last 30 Days Percent of Normal (%)", "30-Day Precipitation Percent of Normal", "percent"),
    ("precip-60day-percent-normal", "Last 60 Days Percent of Normal (%)", "60-Day Precipitation Percent of Normal", "percent"),
    _departure("precip-7day-departure", "Last 7 Days", "7-Day"),
    _departure("precip-14day-departure", "Last 14 Days", "14-Day"),
    _departure("precip-30day-departure", "Last 30 Days", "30-Day"),
    _departure("precip-60day-departure", "Last 60 Days", "60-Day"),
    _departure("precip-90day-departure", "Last 90 Days", "90-Day"),
    _departure("precip-180day-departure", "Last 180 Days", "180-Day"),
    _departure("precip-365day-departure", "Last 365 Days", "365-Day"),
    _departure("precip-mtd-departure", "Month to Date", "Month-to-Date"),
    _departure("precip-ytd-departure", "Year to Date", "Year-to-Date"),
)
# NOAA refreshes the short accumulations hourly, so these are republished every
# hour (core/scheduler.py); the multi-day totals change slowly and only go out
# with the twice-daily run.
HOURLY_STEMS = frozenset({"precip-1hr", "precip-3hr", "precip-6hr", "precip-12hr", "precip-24hr"})
LOG = logging.getLogger("precipitation_graphics")


def update_schedule_for(stem: str) -> str:
    """The schedule printed on the graphic; matches the core/scheduler.py jobs."""
    return "Updates hourly" if stem in HOURLY_STEMS else "Updates twice daily (6 AM & 6 PM CT)"


def _get_json(url: str, *, session=requests, timeout: int = 45) -> dict:
    response = session.get(url, params={"f": "json"}, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise RuntimeError(f"ArcGIS service error: {payload['error']}")
    return payload


def resolve_layer_ids(service: str = SERVICE_URL, *, session=requests) -> dict[str, int]:
    """Resolve named leaf layers dynamically, failing clearly if NOAA changes them."""
    metadata = _get_json(service, session=session)
    layers = metadata.get("layers", [])
    if not layers:
        raise RuntimeError("NOAA QPE service did not return its layer catalog")
    named_products = {}
    for layer in layers:
        name = str(layer.get("name", "")).strip().casefold()
        for _, wanted, _, _ in PRODUCTS:
            if name == wanted.casefold():
                named_products[wanted] = layer
    missing = [wanted for _, wanted, _, _ in PRODUCTS if wanted not in named_products]
    if missing:
        raise RuntimeError("NOAA QPE layers not found: " + ", ".join(missing))
    # Each named product is a group; its Image child is the raster. Select the
    # child directly to omit NOAA's service footprint/boundary overlays.
    resolved = {}
    for wanted, parent in named_products.items():
        parent_id = int(parent["id"])
        image_layer = next((layer for layer in layers if layer.get("parentLayerId") == parent_id and str(layer.get("name", "")).strip().casefold() == "image"), None)
        if image_layer is None:
            raise RuntimeError(f"NOAA QPE image sublayer not found for {wanted}")
        resolved[wanted] = int(image_layer["id"])
    return resolved


def fetch_layer_png(layer_id: int, *, service: str = SERVICE_URL, session=requests) -> bytes:
    lon_min, lon_max, lat_min, lat_max = EXTENT
    params = {
        # ArcGIS export expects bbox as xmin,ymin,xmax,ymax; EXTENT is in
        # matplotlib's (lon_min, lon_max, lat_min, lat_max) order.
        "bbox": f"{lon_min},{lat_min},{lon_max},{lat_max}",
        "bboxSR": "4326", "imageSR": "4326", "size": "1500,1000",
        "format": "png32", "transparent": "true", "layers": f"show:{layer_id}",
        "f": "image",
    }
    response = session.get(f"{service}/export", params=params, timeout=90)
    response.raise_for_status()
    if not response.content.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError(f"NOAA QPE layer {layer_id} did not return a PNG image")
    return response.content


def fetch_legend_entries(layer_id: int, *, service: str = SERVICE_URL, session=requests) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Return official ArcGIS legend labels and their swatch colors."""
    payload = _get_json(f"{service}/legend", session=session)
    match = next((item for item in payload.get("layers", []) if int(item.get("layerId", -1)) == layer_id), None)
    if match is None:
        raise RuntimeError(f"NOAA QPE legend is missing layer {layer_id}")
    entries = []
    for item in match.get("legend", []):
        label = str(item.get("label", "")).strip()
        image_data = item.get("imageData")
        if not label or not image_data:
            continue
        swatch = mpimg.imread(BytesIO(base64.b64decode(image_data)), format="png")
        pixel = swatch[swatch.shape[0] // 2, swatch.shape[1] // 2]
        if len(pixel) == 3:
            pixel = (*pixel, 1.0)
        entries.append((label, tuple(float(channel) for channel in pixel)))
    if not entries:
        raise RuntimeError(f"NOAA QPE legend for layer {layer_id} has no labeled swatches")
    return entries


def _shorten_legend_label(label: str) -> str:
    """Condense NOAA's verbose legend sentences into compact range text.

    Units are already stated once in the graphic's description line, so the
    legend itself only needs the numbers, e.g. "Greater than or equal to 10"
    -> "≥ 10", "10 to 15" -> "10–15".
    """
    label = label.strip()
    lower = label.lower()
    if lower.startswith("greater than or equal to"):
        return f"≥ {label.rsplit(' ', 1)[-1]}"
    if lower.startswith("less than"):
        return f"< {label.rsplit(' ', 1)[-1]}"
    if lower in ("missing data", "missing_data"):
        return "No Data"
    match = re.match(r"^(-?[\d.]+)\s+to\s+(-?[\d.]+)$", label, re.IGNORECASE)
    if match:
        low, high = match.groups()
        # An en dash next to a minus sign is unreadable ("-1–-0.5"); keep "to" for negatives.
        return f"{low} to {high}" if "-" in low + high else f"{low}–{high}"
    return label


def _smooth_and_mask_to_state(raster: np.ndarray, state_geometry, extent: tuple[float, float, float, float], *, sigma: float = 1.6) -> np.ndarray:
    """Soften NOAA's blocky color bands and clip the raster to Missouri's outline.

    Smoothing runs on alpha-premultiplied color so blurring never pulls in
    fully-transparent black from outside NOAA's own data footprint. The state
    mask is applied after smoothing so the outline itself stays crisp instead
    of bleeding into neighboring states, matching the look of the RBF/gaussian
    forecast graphics (fuelmoisturemap.py, rhmap-fil.py, windmap-fil.py).
    """
    lon_min, lon_max, lat_min, lat_max = extent
    height, width = raster.shape[:2]
    if raster.shape[2] == 3:
        raster = np.dstack([raster, np.ones((height, width), dtype=raster.dtype)])

    alpha = raster[..., 3]
    premultiplied_rgb = raster[..., :3] * alpha[..., None]
    blurred_rgb = np.dstack([gaussian_filter(premultiplied_rgb[..., channel], sigma=sigma) for channel in range(3)])
    blurred_alpha = gaussian_filter(alpha, sigma=sigma)
    safe_alpha = np.clip(blurred_alpha, 1e-6, None)
    smoothed_rgb = np.where(blurred_alpha[..., None] > 1e-6, blurred_rgb / safe_alpha[..., None], 0.0)

    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
    state_mask = rasterize([(state_geometry, 1)], out_shape=(height, width), transform=transform, fill=0, dtype="uint8").astype(bool)

    result = np.dstack([smoothed_rgb, blurred_alpha])
    result[~state_mask, 3] = 0.0
    return np.clip(result, 0.0, 1.0)


def _draw_map(image_bytes: bytes, legend: list[tuple[str, tuple[float, float, float, float]]], title: str, units: str, timestamp: datetime, output: Path, update_schedule: str) -> None:
    data_crs = ccrs.PlateCarree()
    map_crs = ccrs.LambertConformal(central_longitude=-92.45, central_latitude=38.3)
    # A bit lighter than the other graphics' #E8E8E8: NOAA's own "greater than
    # or equal to X" swatch on the accumulation (inches) products is a very
    # close gray (~#DCDCDC), which reads as background at a glance otherwise.
    fig = plt.figure(figsize=(2048 / 144, 1152 / 144), dpi=144, facecolor="#F5F5F3")
    # Layout matches the other realtime graphics (fuelmoisturemap.py,
    # rhmap-fil.py, windmap-fil.py, realtimefiredanger.py): a full-bleed axes
    # anchored to the west after set_extent, which leaves a right-hand gutter
    # for the text block and a left-hand gutter for the stacked legend.
    ax = plt.axes([0, 0, 1, 1], projection=map_crs)
    ax.set_extent(EXTENT, crs=data_crs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    state_geometry = sp._missouri_geometry()

    raster = mpimg.imread(BytesIO(image_bytes), format="png")
    raster = _smooth_and_mask_to_state(raster, state_geometry, EXTENT)
    ax.imshow(raster, extent=EXTENT, transform=data_crs, origin="upper", interpolation="bilinear", zorder=1)

    # Pre-projected into the map CRS (cached): letting cartopy re-project the
    # lon/lat outlines on every draw was ~70% of this function's runtime.
    for path, color, width, zorder in ((sp.COUNTY_SHAPEFILE, "#B6B6B6", 0.65, 5), (sp.STATE_SHAPEFILE, "#202020", 1.6, 8)):
        ax.add_geometries(
            sp._projected_boundary(str(path), map_crs.proj4_init), crs=map_crs,
            edgecolor=color, facecolor="none", linewidth=width, zorder=zorder,
        )

    ax.set_anchor("W")
    plt.subplots_adjust(left=0.05)

    for relative in (
        "assets/Montserrat/static/Montserrat-Regular.ttf",
        "assets/Plus_Jakarta_Sans/static/PlusJakartaSans-Regular.ttf",
        "assets/Plus_Jakarta_Sans/static/PlusJakartaSans-Bold.ttf",
    ):
        font_path = PROJECT_DIR / relative
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
    # Same wrapping as station_plot.wrap_title: past 40 characters the title would run into the map.
    title_lines = textwrap.wrap(title, 40) or [title]
    drop = (len(title_lines) - 1) * 26 * 1.2 / 72 * 144 / 1152
    fig.text(0.99, 0.97, "\n".join(title_lines), fontsize=26, fontweight="bold", ha="right", va="top", fontname="Plus Jakarta Sans", color="#202020")
    fig.text(0.99, 0.90 - drop, f"NOAA RFC QPE | Valid Time: {timestamp.strftime('%Y-%m-%d %H:%M CT')}", fontsize=16, ha="right", va="top", fontname="Montserrat", color="#333333")
    # Same line as station_plot.add_update_schedule (not imported: this runs in the API process).
    fig.text(0.99, 0.855 - drop, update_schedule, fontsize=12, color="#555555", ha="right", va="top", fontname="Montserrat")
    description = {
        "inches": f"Accumulated precipitation ({units})",
        "departure": "Observed minus normal precipitation (inches);\npositive is wetter than normal, negative is drier",
    }.get(units, "Precipitation relative to normal (%)")
    fig.text(
        0.99, 0.62,
        f"{description}\n\n"
        "Data Source: NOAA River Forecast Centers\n"
        "Quantitative Precipitation Estimate (QPE)\n\n"
        "For More Info, Visit ShowMeFire.org",
        fontsize=10, ha="right", va="top", linespacing=1.6, fontname="Montserrat", color="#333333",
    )
    fig.text(0.02, 0.01, "ShowMeFire.org", fontsize=20, fontweight="bold", ha="left", va="bottom", fontname="Montserrat", color="#202020")

    # Custom compact legend, stacked on the left in the same spot as the
    # other realtime graphics' colorbar (fuelmoisturemap.py, rhmap-fil.py,
    # windmap-fil.py, realtimefiredanger.py). Shortened labels (see
    # _shorten_legend_label) keep it narrow enough to sit there, unlike
    # NOAA's original verbose sentences, which needed a backdrop panel to
    # stay legible over the map at this position.
    legend_top, legend_bottom = 0.68, 0.08
    row_height = (legend_top - legend_bottom) / len(legend)
    swatch_w = 0.018
    swatch_h = min(row_height * 0.7, swatch_w)
    title_x, swatch_x, label_x = 0.022, 0.037, 0.062
    legend_title = {
        "inches": "Accumulated Precipitation (in)",
        "departure": "Departure from Normal (in)",
    }.get(units, "Precipitation vs. Normal (%)")
    fig.text(
        title_x, (legend_top + legend_bottom) / 2, legend_title,
        transform=fig.transFigure, rotation=90, fontsize=11, ha="center", va="center",
        fontname="Montserrat", color="#222222",
    )
    for index, (label, color) in enumerate(legend):
        row_top = legend_top - index * row_height
        fig.add_artist(Rectangle(
            (swatch_x, row_top - swatch_h), swatch_w, swatch_h,
            transform=fig.transFigure, facecolor=color, edgecolor="#777777", linewidth=0.4,
        ))
        fig.text(
            label_x, row_top - swatch_h / 2, _shorten_legend_label(label),
            transform=fig.transFigure, fontsize=8.5, ha="left", va="center",
            fontname="Montserrat", color="#222222",
        )

    logo_path = PROJECT_DIR / "assets/LightBackGroundLogo.svg"
    if logo_path.exists():
        try:
            import cairosvg
            logo = mpimg.imread(BytesIO(cairosvg.svg2png(url=str(logo_path))), format="png")
            ax.add_artist(AnnotationBbox(OffsetImage(logo, zoom=0.03), (0.99, 0.01), frameon=False, xycoords="figure fraction", box_alignment=(1, 0)))
        except Exception as error:
            LOG.warning("Could not render logo: %s", error)

    # The NOAA map service applies the official product color ramp. Its legend
    # endpoint is queried separately so the graphic's key uses the same ramp.
    fig.savefig(output, dpi=144, bbox_inches=None, pad_inches=0, facecolor=fig.get_facecolor())
    plt.close(fig)


def generate_graphics(
    *, output_dir: Path | None = None, service: str = SERVICE_URL, session=requests, now: datetime | None = None,
    stems: frozenset | set | None = None,
) -> list[Path]:
    """Render the NOAA QPE graphics (all of PRODUCTS, or only those whose stem
    is in `stems`). Network fetches run in parallel; drawing stays serial
    because Matplotlib is not thread-safe."""
    output_dir = Path(output_dir or PROJECT_DIR / "images")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = (now or datetime.now(CENTRAL)).astimezone(CENTRAL)
    layer_ids = resolve_layer_ids(service, session=session)
    products = [product for product in PRODUCTS if stems is None or product[0] in stems]

    def fetch(product):
        layer_id = layer_ids[product[1]]
        return (
            fetch_layer_png(layer_id, service=service, session=session),
            fetch_legend_entries(layer_id, service=service, session=session),
        )

    with ThreadPoolExecutor(max_workers=6) as pool:
        fetched = list(pool.map(fetch, products))

    outputs = []
    for (stem, _layer_name, title, units), (image, legend) in zip(products, fetched):
        path = output_dir / f"mo-{stem}.png"
        temporary = path.with_suffix(".tmp.png")
        _draw_map(image, legend, title, units, timestamp, temporary, update_schedule_for(stem))
        temporary.replace(path)
        outputs.append(path)
        LOG.info("Rendered %s", path)
    return outputs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    generate_graphics()
