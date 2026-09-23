"""Render current NOAA RFC QPE precipitation products for Missouri."""

from __future__ import annotations

import base64
import logging
from datetime import datetime
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.font_manager as font_manager
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import requests
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.ticker import MaxNLocator

PROJECT_DIR = Path(__file__).resolve().parents[1]
SERVICE_URL = "https://mapservices.weather.noaa.gov/raster/rest/services/obs/rfc_qpe/MapServer"
CENTRAL = ZoneInfo("America/Chicago")
EXTENT = (-95.8, -89.1, 35.8, 40.8)
PRODUCTS = (
    ("precip-24hr", "Last 24 Hours Observed (inches)", "24-Hour Precipitation", "inches"),
    ("precip-7day", "Last 7 Days Observed (inches)", "7-Day Precipitation", "inches"),
    ("precip-30day", "Last 30 Days Observed (inches)", "30-Day Precipitation", "inches"),
    ("precip-60day", "Last 60 Days Observed (inches)", "60-Day Precipitation", "inches"),
    ("precip-30day-percent-normal", "Last 30 Days Percent of Normal (%)", "30-Day Precipitation Percent of Normal", "percent"),
    ("precip-60day-percent-normal", "Last 60 Days Percent of Normal (%)", "60-Day Precipitation Percent of Normal", "percent"),
)
LOG = logging.getLogger("precipitation_graphics")


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
    params = {
        "bbox": ",".join(str(value) for value in EXTENT),
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


def _draw_map(image_bytes: bytes, legend: list[tuple[str, tuple[float, float, float, float]]], title: str, units: str, timestamp: datetime, output: Path) -> None:
    data_crs = ccrs.PlateCarree()
    map_crs = ccrs.LambertConformal(central_longitude=-92.45, central_latitude=38.3)
    fig = plt.figure(figsize=(2048 / 144, 1152 / 144), dpi=144, facecolor="#E8E8E8")
    ax = fig.add_axes([0.05, 0.04, 0.90, 0.82], projection=map_crs)
    ax.set_extent(EXTENT, crs=data_crs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    raster = mpimg.imread(BytesIO(image_bytes), format="png")
    ax.imshow(raster, extent=EXTENT, transform=data_crs, origin="upper", interpolation="nearest", zorder=1)

    counties = gpd.read_file(PROJECT_DIR / "maps/shapefiles/MO_County_Boundaries/MO_County_Boundaries.shp")
    boundary = gpd.read_file(PROJECT_DIR / "maps/shapefiles/MO_State_Boundary/MO_State_Boundary.shp")
    for frame, color, width, zorder in ((counties, "#B6B6B6", 0.65, 5), (boundary, "#202020", 1.6, 8)):
        if frame.crs is None:
            raise RuntimeError("Missouri map boundary data has no CRS")
        frame = frame.to_crs("EPSG:4326")
        ax.add_geometries(frame.geometry, crs=data_crs, edgecolor=color, facecolor="none", linewidth=width, zorder=zorder)

    for relative in (
        "assets/Montserrat/static/Montserrat-Regular.ttf",
        "assets/Plus_Jakarta_Sans/static/PlusJakartaSans-Regular.ttf",
        "assets/Plus_Jakarta_Sans/static/PlusJakartaSans-Bold.ttf",
    ):
        font_path = PROJECT_DIR / relative
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
    fig.text(0.98, 0.955, title, fontsize=24, fontweight="bold", ha="right", va="top", fontname="Plus Jakarta Sans", color="#202020")
    fig.text(0.98, 0.905, f"NOAA RFC QPE | {timestamp.strftime('%Y-%m-%d %H:%M CT')}", fontsize=14, ha="right", va="top", fontname="Montserrat", color="#333333")
    fig.text(0.055, 0.885, f"Accumulated precipitation ({units})" if units == "inches" else "Precipitation relative to normal (%)", fontsize=12, ha="left", va="top", fontname="Montserrat", color="#333333")
    fig.text(0.02, 0.012, "ShowMeFire.org", fontsize=18, fontweight="bold", ha="left", va="bottom", fontname="Montserrat", color="#202020")

    # Use NOAA's own labeled swatches so thresholds and colors always match
    # the layer being displayed, including percent-of-normal products.
    rows = (len(legend) + 1) // 2
    panel = fig.add_axes([0.052, 0.10, 0.19, min(0.58, 0.025 + rows * 0.022)])
    panel.set_facecolor((1, 1, 1, 0.88))
    panel.set_xticks([])
    panel.set_yticks([])
    for spine in panel.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.5)
    for index, (label, color) in enumerate(legend):
        col = index // rows
        row = index % rows
        x = 0.025 + col * 0.5
        y = 0.98 - (row + 1) / rows
        panel.add_patch(Rectangle((x, y + 0.015), 0.075, 0.065, transform=panel.transAxes, facecolor=color, edgecolor="#666666", linewidth=0.25))
        panel.text(x + 0.09, y + 0.047, label, transform=panel.transAxes, ha="left", va="center", fontsize=5.8, color="#222222")
    panel.set_xlim(0, 1)
    panel.set_ylim(0, 1)

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


def generate_graphics(*, output_dir: Path | None = None, service: str = SERVICE_URL, session=requests, now: datetime | None = None) -> list[Path]:
    output_dir = Path(output_dir or PROJECT_DIR / "images")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = (now or datetime.now(CENTRAL)).astimezone(CENTRAL)
    layer_ids = resolve_layer_ids(service, session=session)
    outputs = []
    for stem, layer_name, title, units in PRODUCTS:
        image = fetch_layer_png(layer_ids[layer_name], service=service, session=session)
        legend = fetch_legend_entries(layer_ids[layer_name], service=service, session=session)
        path = output_dir / f"mo-{stem}.png"
        temporary = path.with_suffix(".tmp.png")
        _draw_map(image, legend, title, units, timestamp, temporary)
        temporary.replace(path)
        outputs.append(path)
        LOG.info("Rendered %s", path)
    return outputs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    generate_graphics()
