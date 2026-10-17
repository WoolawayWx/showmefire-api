"""GOES-East (GOES-19) satellite maps for Missouri: single ABI channels, the
standard fire/microphysics RGB composites, and a fire-detection map.

All imagery comes from NOAA's public GOES archive on AWS (no key needed):
  ABI-L2-MCMIPC  all 16 bands at 2 km, CONUS sector every 5 minutes (~60 MB)
  ABI-L2-CMIPC   band 2 (red visible) at 0.5 km, daytime only (~70 MB)
  ABI-L2-FDCC    Fire Detection and Characterization, every 5 minutes (~0.5 MB)

One download feeds every product, drawn on the standard Missouri basemap with
the same title/branding block as the station maps, and masked to the state
outline like the station and drought fills. Day-only products (visible, true color...)
are skipped at night and night-only ones by day; their last image stays up
with its own valid time. A scan already rendered is not redrawn.
"""
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import requests
import shapely
import xarray as xr
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from maps import raster_maps as rm
from maps import station_plot as sp
from maps import station_products as spx
from services.derived_met import solar_elevation_deg

BUCKET = "https://noaa-goes19.s3.amazonaws.com"
SATELLITE = "GOES-19 (GOES-East)"
HEADERS = {"User-Agent": "ShowMeFire/1.0 graphics@showmefire.org"}
# Fire pixels outside this lon/lat box (well beyond the map frame) are ignored.
READ_BOX = (-98.5, 33.8, -86.4, 42.6)
CENTER = (38.3, -92.45)
DAY_SUN_DEG = 5.0     # sun this high over central Missouri: daytime products
NIGHT_SUN_DEG = -3.0  # sun this low: night-only products
FIRE_WINDOW_MIN = 60
SIGNATURE_PATH = spx.STATUS_DIR / "satellite_maps.signature"


# --- data access --------------------------------------------------------------

def _list_keys(product: str, hours_back: int = 2) -> list[str]:
    """Object keys for `product` from the last few hourly folders, oldest first."""
    now = datetime.now(timezone.utc)
    keys = []
    for back in range(hours_back, -1, -1):
        folder = now - timedelta(hours=back)
        response = requests.get(BUCKET, params={"list-type": "2", "prefix": f"{product}/{folder:%Y/%j/%H}/"},
                                headers=HEADERS, timeout=30)
        response.raise_for_status()
        keys += [part.split("</Key>")[0] for part in response.text.split("<Key>")[1:]]
    return sorted(keys, key=_scan_start)


def _scan_start(key: str) -> datetime:
    stamp = key.rsplit("_s", 1)[1][:13]  # sYYYYJJJHHMMSS (tenths dropped)
    return datetime.strptime(stamp, "%Y%j%H%M%S").replace(tzinfo=timezone.utc)


def _download(key: str, folder: Path) -> Path:
    path = folder / key.rsplit("/", 1)[1]
    with requests.get(f"{BUCKET}/{key}", headers=HEADERS, timeout=180, stream=True) as response:
        response.raise_for_status()
        with open(path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
    return path


def _geos_crs(dataset: xr.Dataset) -> ccrs.Geostationary:
    projection = dataset["goes_imager_projection"].attrs
    return ccrs.Geostationary(
        central_longitude=float(projection["longitude_of_projection_origin"]),
        satellite_height=float(projection["perspective_point_height"]),
        sweep_axis=str(projection["sweep_angle_axis"]),
        globe=ccrs.Globe(semimajor_axis=float(projection["semi_major_axis"]),
                         semiminor_axis=float(projection["semi_minor_axis"])),
    )


@lru_cache(maxsize=1)
def _map_grid() -> tuple:
    """(map CRS, extent, x mesh, y mesh) for one output pixel per cell of the
    standard Missouri basemap's axes, in that map's projected coordinates."""
    fig, ax, _data_crs, map_crs, dpi = sp.generate_basemap()
    ax.apply_aspect()
    box = ax.get_position()
    width, height = round(box.width * sp.PIXEL_WIDTH), round(box.height * sp.PIXEL_HEIGHT)
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    plt.close(fig)
    xs = x0 + (np.arange(width) + 0.5) * (x1 - x0) / width
    ys = y1 - (np.arange(height) + 0.5) * (y1 - y0) / height  # top row first
    x_mesh, y_mesh = np.meshgrid(xs, ys)
    return map_crs, (x0, x1, y0, y1), x_mesh, y_mesh


class Scene:
    """One ABI file resampled (nearest pixel) onto the Missouri map grid.

    Letting cartopy warp each image from the geostationary projection cost ~3s
    of every ~4s map, and it repeated the same warp for every product. Here the
    map-pixel -> satellite-pixel lookup is computed once per file and every
    band is just indexed through it."""

    def __init__(self, path: Path):
        self.dataset = xr.open_dataset(path, engine="h5netcdf")
        self.crs = _geos_crs(self.dataset)
        self.start = datetime.fromisoformat(self.dataset.attrs["time_coverage_start"].replace("Z", "+00:00"))
        height = float(self.dataset["goes_imager_projection"].attrs["perspective_point_height"])
        x = self.dataset["x"].values * height
        y = self.dataset["y"].values * height
        map_crs, self.extent, x_mesh, y_mesh = _map_grid()
        source = self.crs.transform_points(map_crs, x_mesh, y_mesh)
        columns = np.rint((source[..., 0] - x[0]) / (x[1] - x[0]))
        rows = np.rint((source[..., 1] - y[0]) / (y[1] - y[0]))
        self.valid = (np.isfinite(columns) & np.isfinite(rows) & (columns >= 0) & (columns < x.size)
                      & (rows >= 0) & (rows < y.size))
        columns, rows = np.where(self.valid, columns, 0).astype(int), np.where(self.valid, rows, 0).astype(int)
        # Only the satellite rows/columns the map touches are ever read.
        self.window = (slice(rows[self.valid].min(), rows[self.valid].max() + 1),
                       slice(columns[self.valid].min(), columns[self.valid].max() + 1))
        self.rows = rows - self.window[0].start
        self.columns = columns - self.window[1].start
        self._bands: dict = {}

    def band(self, name: str) -> np.ndarray:
        """Band `name` on the map grid (NaN where the satellite has no data)."""
        if name not in self._bands:
            window = self.dataset[name].isel(y=self.window[0], x=self.window[1]).values.astype("float64")
            sampled = window[np.clip(self.rows, 0, window.shape[0] - 1), np.clip(self.columns, 0, window.shape[1] - 1)]
            self._bands[name] = np.where(self.valid, sampled, np.nan)
        return self._bands[name]

    def close(self) -> None:
        self.dataset.close()


# --- RGB and colormap recipes -------------------------------------------------

def _stretch(values: np.ndarray, low: float, high: float, gamma: float = 1.0) -> np.ndarray:
    """Linear stretch low->high to 0-1 (high < low inverts), then the usual
    RGB-recipe gamma (output = scaled ** (1 / gamma))."""
    scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
    return scaled ** (1.0 / gamma)


def _rgb(red, green, blue) -> np.ndarray:
    image = np.dstack([red, green, blue])
    alpha = np.all(np.isfinite(image), axis=2).astype("float64")
    return np.dstack([np.nan_to_num(image), alpha])


def true_color(scene: Scene) -> np.ndarray:
    # ABI has no green band: the CIMSS "natural true color" blend synthesizes one.
    red, blue, veggie = scene.band("CMI_C02"), scene.band("CMI_C01"), scene.band("CMI_C03")
    green = 0.45 * red + 0.10 * veggie + 0.45 * blue
    return _rgb(*(_stretch(band, 0.0, 1.0, 2.2) for band in (red, green, blue)))


def fire_temperature(scene: Scene) -> np.ndarray:
    # CIRA Fire Temperature RGB.
    return _rgb(_stretch(scene.band("CMI_C07"), 273.15, 333.15, 0.4),
                _stretch(scene.band("CMI_C06"), 0.0, 0.50),
                _stretch(scene.band("CMI_C05"), 0.0, 0.50))


def day_land_cloud_fire(scene: Scene) -> np.ndarray:
    # CIRA Day Land Cloud/Fire RGB (a.k.a. natural color with 2.2 um).
    return _rgb(_stretch(scene.band("CMI_C06"), 0.0, 1.0),
                _stretch(scene.band("CMI_C03"), 0.0, 1.0),
                _stretch(scene.band("CMI_C02"), 0.0, 1.0))


def night_microphysics(scene: Scene) -> np.ndarray:
    # CIRA Nighttime Microphysics RGB.
    c13 = scene.band("CMI_C13")
    return _rgb(_stretch(scene.band("CMI_C15") - c13, -6.7, 2.6),
                _stretch(c13 - scene.band("CMI_C07"), -3.1, 5.2),
                _stretch(c13, 243.55, 292.65))


def day_cloud_phase(scene: Scene) -> np.ndarray:
    # CIRA/JMA Day Cloud Phase Distinction RGB.
    return _rgb(_stretch(scene.band("CMI_C13"), 280.65, 219.65),
                _stretch(scene.band("CMI_C02"), 0.0, 0.78),
                _stretch(scene.band("CMI_C05"), 0.01, 0.59))


# Band 7 (3.9 um) carries reflected sunlight as well as emitted heat by day.
# The reflected part (Rosenfeld & Lensky 1998, BAMS 79, 2457; the method behind
# the EUMETSAT/CIRA Day Microphysics RGB) is
#   r = (L7 - B7(T13)) / (S cos(sza) / pi - B7(T13))
# using band 13 as the emitted temperature. Planck constants are the GOES-R ABI
# band 7 values (L1b planck_fk1/fk2/bc1/bc2, radiance in mW/(m2 sr cm-1)).
# The CO2 correction is omitted, which matters little this far from the limb.
B7_FK1, B7_FK2, B7_BC1, B7_BC2 = 2.02263e5, 3698.19, 0.43361, 0.99939
B7_SOLAR = 16.7  # TOA solar irradiance at 3.9 um, mW/(m2 cm-1) (~11 W/m2/um)


def _band7_radiance(kelvin: np.ndarray) -> np.ndarray:
    return B7_FK1 / np.expm1(B7_FK2 / (B7_BC1 + B7_BC2 * kelvin))


def band7_reflectance(t7: np.ndarray, t13: np.ndarray, sun_elevation_deg) -> np.ndarray:
    """Solar-reflected fraction of the 3.9 um signal (0-1); NaN where the sun is too low."""
    emitted = _band7_radiance(t13)
    cos_sza = np.sin(np.radians(sun_elevation_deg))
    solar = np.where(cos_sza > 0.05, B7_SOLAR * cos_sza / np.pi, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip((_band7_radiance(t7) - emitted) / (solar - emitted), 0.0, 1.0)


@lru_cache(maxsize=1)
def _map_lonlat() -> tuple:
    map_crs, _extent, x_mesh, y_mesh = _map_grid()
    points = ccrs.PlateCarree().transform_points(map_crs, x_mesh, y_mesh)
    return points[..., 0], points[..., 1]


def day_microphysics(scene: Scene) -> np.ndarray:
    # CIRA/EUMETSAT Day Microphysics RGB (bands 3, 7 reflected part, 14).
    lon, lat = _map_lonlat()
    r39 = band7_reflectance(scene.band("CMI_C07"), scene.band("CMI_C13"), solar_elevation_deg(lat, lon, scene.start))
    return _rgb(_stretch(scene.band("CMI_C03"), 0.0, 1.0),
                _stretch(r39, 0.0, 0.6, 2.5),
                _stretch(scene.band("CMI_C14"), 203.0, 323.0))


def airmass(scene: Scene) -> np.ndarray:
    # EUMETSAT/CIRA Air Mass RGB (bands 8-10, 12-13, 8).
    return _rgb(_stretch(scene.band("CMI_C08") - scene.band("CMI_C10"), -26.2, 0.6),
                _stretch(scene.band("CMI_C12") - scene.band("CMI_C13"), -43.2, 6.7),
                _stretch(scene.band("CMI_C08"), 243.9, 208.5))


def dust(scene: Scene) -> np.ndarray:
    # EUMETSAT/CIRA Dust RGB (bands 15-13, 14-11, 13).
    return _rgb(_stretch(scene.band("CMI_C15") - scene.band("CMI_C13"), -6.7, 2.6),
                _stretch(scene.band("CMI_C14") - scene.band("CMI_C11"), -0.5, 20.0, 2.5),
                _stretch(scene.band("CMI_C13"), 261.2, 288.7))


def day_convection(scene: Scene) -> np.ndarray:
    # EUMETSAT/CIRA Day Convection RGB (bands 8-10, 7-13, 5-2).
    return _rgb(_stretch(scene.band("CMI_C08") - scene.band("CMI_C10"), -35.0, 5.0),
                _stretch(scene.band("CMI_C07") - scene.band("CMI_C13"), -5.0, 60.0, 0.5),
                _stretch(scene.band("CMI_C05") - scene.band("CMI_C02"), -0.75, 0.25))


SANDWICH_COLD_C = -30.0


def sandwich(scene: Scene) -> np.ndarray:
    # CIRA Sandwich: band 2 visible texture multiplied by band 13 IR colors on cold tops.
    visible = _stretch(scene.band("CMI_C02"), 0.0, 1.0, 1.8)
    ir_c = scene.band("CMI_C13") - C
    colors = IR_CMAP(np.clip((ir_c - IR_RANGE[0]) / (IR_RANGE[1] - IR_RANGE[0]), 0, 1))[..., :3]
    colors = np.where((ir_c < SANDWICH_COLD_C)[..., None], colors, 1.0)
    image = colors * visible[..., None]
    return _rgb(image[..., 0], image[..., 1], np.where(np.isfinite(ir_c), image[..., 2], np.nan))


def _cmap_from_stops(name: str, stops: list[tuple[float, str]], low: float, high: float) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, [((value - low) / (high - low), color) for value, color in stops])


C = 273.15
IR_RANGE = (-90.0, 50.0)
IR_CMAP = _cmap_from_stops("ir_enhanced", [
    (-90, "#ffffff"), (-80, "#ff66ff"), (-70, "#8b008b"), (-62, "#ff0000"), (-52, "#ffff00"),
    (-42, "#00c800"), (-32, "#0050ff"), (-25, "#00e5ff"), (-24.9, "#d8d8d8"), (50, "#000000"),
], *IR_RANGE)
WV_RANGE = (-75.0, 0.0)
WV_CMAP = _cmap_from_stops("water_vapor", [
    (-75, "#ffffff"), (-65, "#00e5e5"), (-55, "#0050c8"), (-45, "#20a040"), (-38, "#f0f0f0"),
    (-28, "#808080"), (-18, "#303030"), (-10, "#b5651d"), (0, "#ffa500"),
], *WV_RANGE)
SWIR_RANGE = (-50.0, 75.0)
SWIR_CMAP = _cmap_from_stops("swir_fire", [
    (-50, "#ffffff"), (40, "#000000"), (45, "#ffff00"), (55, "#ff8000"), (65, "#ff0000"), (75, "#7f0000"),
], *SWIR_RANGE)


SPLIT_RANGE = (-2.0, 8.0)
SPLIT_CMAP = _cmap_from_stops("split_window", [
    (-2, "#8c510a"), (0, "#f6e8c3"), (1, "#f5f5f5"), (3, "#80cdc1"), (5, "#01665e"), (8, "#003c30"),
], *SPLIT_RANGE)


def _split_window(_band: str, scene: Scene) -> np.ndarray:
    return scene.band("CMI_C13") - scene.band("CMI_C15")


def _celsius(band: str, scene: Scene) -> np.ndarray:
    return scene.band(band) - C


def _reflectance(band: str, scene: Scene) -> np.ndarray:
    return np.clip(scene.band(band), 0.0, 1.0) ** 0.5  # square-root stretch brings out low-contrast land


# Each product: png, title, which file ("multi" 2 km, "visible" 0.5 km), when
# it is useful, how it is drawn, and its description/legend.
CHANNELS = [
    dict(key="visible", png="mo-sat-visible.png", title="Missouri Visible Satellite", file="visible", when="day",
         band=("CMI", _reflectance), cmap="gray", norm=(0, 1), units=None, reflectance_label="Band 2 reflectance (%)",
         blurb="Band 2 red visible (0.64 µm), 0.5 km.\nSunlight reflected by clouds, smoke plumes\nand the ground. Daytime only."),
    dict(key="veggie", png="mo-sat-veggie.png", title="Missouri Veggie Band", file="multi", when="day",
         band=("CMI_C03", _reflectance), cmap="gray", norm=(0, 1), units=None, reflectance_label="Band 3 reflectance (%)",
         blurb="Band 3 near-infrared (0.86 µm), 2 km.\nHealthy vegetation is bright; water, burn\nscars and stressed vegetation are darker.\nDaytime only."),
    dict(key="swir", png="mo-sat-swir.png", title="Missouri Shortwave Infrared", file="multi", when="any",
         band=("CMI_C07", _celsius), cmap=SWIR_CMAP, norm=SWIR_RANGE, units="Band 7 brightness temperature (°C)",
         blurb="Band 7 shortwave IR (3.9 µm), 2 km.\nThe fire channel: active fires show as hot\nspots (yellow-red). Also shows low cloud\nand fog at night."),
    dict(key="wv_upper", png="mo-sat-wv-upper.png", title="Missouri Water Vapor (Upper)", file="multi", when="any",
         band=("CMI_C08", _celsius), cmap=WV_CMAP, norm=WV_RANGE, units="Band 8 brightness temperature (°C)",
         blurb="Band 8 upper-level water vapor (6.2 µm).\nMoisture roughly 350-500 mb; dark/orange is\ndry air aloft, which can mix down and drop\nsurface humidity on fire days."),
    dict(key="wv_mid", png="mo-sat-wv-mid.png", title="Missouri Water Vapor (Mid)", file="multi", when="any",
         band=("CMI_C09", _celsius), cmap=WV_CMAP, norm=WV_RANGE, units="Band 9 brightness temperature (°C)",
         blurb="Band 9 mid-level water vapor (6.9 µm).\nMoisture roughly 500-750 mb; dark/orange\nshows dry slots and sinking air."),
    dict(key="wv_low", png="mo-sat-wv-low.png", title="Missouri Water Vapor (Low)", file="multi", when="any",
         band=("CMI_C10", _celsius), cmap=WV_CMAP, norm=WV_RANGE, units="Band 10 brightness temperature (°C)",
         blurb="Band 10 low-level water vapor (7.3 µm).\nMoisture roughly 750-900 mb; orange marks\ndry lower-level air that favors low RH."),
    dict(key="ir", png="mo-sat-ir.png", title="Missouri Infrared Satellite", file="multi", when="any",
         band=("CMI_C13", _celsius), cmap=IR_CMAP, norm=IR_RANGE, units="Band 13 brightness temperature (°C)",
         blurb="Band 13 clean longwave IR (10.3 µm), 2 km.\nCloud-top and ground temperature, day and\nnight. Colors mark cold, high cloud tops\n(storms)."),
    dict(key="splitwindow", png="mo-sat-splitwindow.png", title="Missouri Split Window Difference", file="multi",
         when="any", band=("split", _split_window), cmap=SPLIT_CMAP, norm=SPLIT_RANGE,
         units="Band 13 − Band 15 (°C)",
         blurb="Bands 13 minus 15 (10.3 − 12.3 µm), 2 km.\nIn clear air, larger values (green) mean\nmore low-level moisture; brown marks dry\nair or dust. Watch for sharp dry lines."),
]

RGBS = [
    dict(key="truecolor", png="mo-sat-truecolor.png", title="Missouri True Color", when="day", recipe=true_color,
         blurb="Bands 1, 2, 3 with a synthetic green, 2 km.\nRoughly what the eye would see: clouds\nwhite, smoke gray-tan, vegetation green.",
         key_colors=[("Cloud", "#f2f2f2"), ("Smoke / haze", "#b9a98f"), ("Vegetation", "#3f6b2a"), ("Bare / dormant land", "#8f7a5a")]),
    dict(key="firetemp", png="mo-sat-firetemp.png", title="Missouri Fire Temperature", when="any", recipe=fire_temperature,
         blurb="CIRA Fire Temperature RGB (bands 7, 6, 5).\nFire hot spots go red, then orange, yellow\nand white as they intensify. Works day and\nnight; clouds are blue/cyan in daytime.",
         key_colors=[("Warm fire", "#ff2000"), ("Hotter fire", "#ff9a00"), ("Very hot fire", "#ffff60"), ("Ice cloud", "#2a8cd0"), ("Water cloud", "#7fd8e0")]),
    dict(key="daylandcloudfire", png="mo-sat-daylandcloudfire.png", title="Missouri Day Land Cloud Fire", when="day",
         recipe=day_land_cloud_fire,
         blurb="CIRA Day Land Cloud/Fire RGB (bands 6, 3, 2).\nVegetation green, burn scars brown-red,\nactive fires bright red/pink, water cloud\nwhite, ice cloud cyan.",
         key_colors=[("Vegetation", "#2f9a2a"), ("Burn scar / bare", "#8a4a30"), ("Active fire", "#ff3060"), ("Water cloud", "#f0f0f0"), ("Ice cloud", "#40d0e8")]),
    dict(key="daycloudphase", png="mo-sat-daycloudphase.png", title="Missouri Day Cloud Phase RGB", when="day",
         recipe=day_cloud_phase,
         blurb="CIRA Day Cloud Phase Distinction RGB\n(bands 13, 2, 5). Separates low water\nclouds from glaciating and ice clouds, and\nshows snow and land.",
         key_colors=[("Low water cloud", "#80e0ff"), ("Ice cloud", "#e01060"), ("Glaciating", "#ffff40"), ("Land", "#4060ff"), ("Snow", "#40ff80")]),
    dict(key="nightmicro", png="mo-sat-nightmicro.png", title="Missouri Night Microphysics", when="night",
         recipe=night_microphysics,
         blurb="CIRA Nighttime Microphysics RGB\n(bands 15, 13, 7). Picks out fog and low\nstratus (aqua) from high cloud; low cloud\nand fog limit overnight RH recovery\nforecasts. Night only.",
         key_colors=[("Fog / low stratus", "#60e0c8"), ("Thick high cloud", "#a01818"), ("Thin cirrus", "#202060"), ("Clear land", "#d090a0")]),
    dict(key="daymicro", png="mo-sat-daymicro.png", title="Missouri Day Microphysics", when="day",
         recipe=day_microphysics,
         blurb="CIRA/EUMETSAT Day Microphysics RGB\n(bands 3, 7 reflected, 14). Small-droplet\nfog and new cloud look pale pink/white,\nlarge drops pink, ice red/orange. Daytime\ncounterpart to Night Microphysics.",
         key_colors=[("Fog / small droplets", "#f4d8ec"), ("Water cloud, large drops", "#e080b8"), ("Ice cloud", "#c02820"), ("Small ice / new storm", "#f09030"), ("Clear land", "#8a6f8f"), ("Green vegetation", "#40a040")]),
    dict(key="airmass", png="mo-sat-airmass.png", title="Missouri Air Mass RGB", when="any", recipe=airmass,
         blurb="EUMETSAT/CIRA Air Mass RGB (bands 8, 10,\n12, 13). Red-orange marks dry, sinking\nstratospheric air behind troughs that can\nmix down and crash afternoon RH.",
         key_colors=[("Dry sinking air (jet / PV)", "#c04020"), ("Warm, moist air mass", "#2f8f2f"), ("Cold air mass", "#5040a0"), ("Thick high cloud", "#f0f0f0"), ("Mid-level cloud", "#c8a878")]),
    dict(key="dust", png="mo-sat-dust.png", title="Missouri Dust RGB", when="any", recipe=dust,
         blurb="EUMETSAT/CIRA Dust RGB (bands 11, 13,\n14, 15). Blowing dust turns magenta-pink;\nuseful on dry, windy days with high\nspread potential.",
         key_colors=[("Blowing dust", "#ff40c0"), ("Thick ice cloud", "#8a1a1a"), ("Thin cirrus", "#151530"), ("Low water cloud", "#8070f0"), ("Clear land, warm day", "#c07040")]),
    dict(key="dayconvection", png="mo-sat-dayconvection.png", title="Missouri Day Convection RGB", when="day",
         recipe=day_convection,
         blurb="EUMETSAT/CIRA Day Convection RGB (bands\n2, 5, 7, 8, 10, 13). Bright yellow marks\nstrong updrafts with small ice: storms\nthat may bring lightning and outflow.",
         key_colors=[("Strong updraft, small ice", "#ffe040"), ("Mature storm, large ice", "#e03030"), ("Low water cloud", "#3060d0"), ("Thin cirrus", "#c040c0")]),
    dict(key="sandwich", png="mo-sat-sandwich.png", title="Missouri Sandwich RGB", when="day", recipe=sandwich,
         blurb="CIRA Sandwich (band 2 visible x band 13 IR).\nVisible cloud texture with IR colors on\ntops colder than -30 °C: overshooting\ntops and storm structure.",
         key_colors=[("Top -30 to -40 °C", "#1050ff"), ("Top -40 to -50 °C", "#00c800"), ("Top -50 to -60 °C", "#ffff00"), ("Top -60 to -70 °C", "#ff0000"), ("Top colder than -70 °C", "#8b008b")]),
]

# Fire Detection and Characterization mask codes (filtered = 30-35).
FIRE_CLASSES = [
    ("High confidence", "#d7191c", {10, 11, 30, 31}),
    ("Medium confidence", "#fd8d3c", {12, 13, 32, 33}),
    ("Low confidence", "#fed976", {14, 15, 34, 35}),
]


# --- drawing ------------------------------------------------------------------

def _draw_image(ax, scene: Scene, image: np.ndarray, **kwargs) -> None:
    # Already on the map grid, so cartopy places it without re-projecting.
    drawn = ax.imshow(image, origin="upper", extent=scene.extent, transform=ax.projection, interpolation="nearest",
                      zorder=7, **kwargs)
    rm.clip_to_state(ax, drawn)


def _colorbar(fig, cmap, norm_range: tuple, label: str, ticks=None, tick_labels=None) -> None:
    """Vertical colorbar in the open wedge west of the state, same spot as the station fills."""
    colorbar = fig.colorbar(ScalarMappable(norm=Normalize(*norm_range), cmap=cmap),
                            cax=fig.add_axes([0.025, 0.15, 0.018, 0.5]), label=label, ticks=ticks)
    if tick_labels is not None:
        colorbar.ax.set_yticklabels(tick_labels)


def _key_legend(fig, items: list, title: str = "What the Colors Show") -> None:
    """Color key in the same style as the drought category legends."""
    handles = [Patch(facecolor=color, edgecolor="#555555", linewidth=0.4, label=name) for name, color in items]
    legend = fig.legend(
        handles=handles, title=title, loc="upper right", bbox_to_anchor=(0.99, 0.38), bbox_transform=fig.transFigure,
        fontsize=12, title_fontsize=13, frameon=False, labelspacing=0.9, handlelength=1.6, handleheight=1.3,
    )
    legend.set_zorder(20)


def _finish(fig, ax, *, product: str, png: str, title: str, scene_time: datetime, blurb: str, started: float, **extra) -> dict:
    valid = spx.valid_time_label(scene_time.astimezone(spx.CENTRAL))
    sp.add_branding(fig, ax, title=title, subtitle=f"Valid Time: {valid}",
                    description=f"Data Source: NOAA {SATELLITE} ABI\n\n{blurb}")
    status = {"product": f"satellite_{product}", "status": "updated", "valid_time": valid,
              "last_update": spx.valid_time_label(), "runtime_sec": round(time.time() - started, 2), **extra}
    sp.save_and_record_status(fig, spx.IMAGES_DIR / png, spx.STATUS_DIR / f"satellite_{product}.json", status)
    print(f"{title} updated, scan {valid} ({status['runtime_sec']:.2f}s)")
    return status


def render_channel(spec: dict, scene: Scene) -> dict:
    started = time.time()
    band, transform = spec["band"]
    values = transform(band, scene)
    cmap = matplotlib.colormaps[spec["cmap"]] if isinstance(spec["cmap"], str) else spec["cmap"]
    fig, ax, _data_crs, _map_crs, _dpi = sp.generate_basemap()
    _draw_image(ax, scene, np.ma.masked_invalid(values), cmap=cmap, vmin=spec["norm"][0], vmax=spec["norm"][1])
    if spec["units"]:
        _colorbar(fig, cmap, spec["norm"], spec["units"])
    else:  # reflectance bands are drawn square-root stretched; label ticks in true reflectance
        percents = [0, 5, 10, 20, 40, 60, 80, 100]
        _colorbar(fig, cmap, spec["norm"], spec["reflectance_label"], ticks=np.sqrt(np.array(percents) / 100),
                  tick_labels=[str(value) for value in percents])
    return _finish(fig, ax, product=spec["key"], png=spec["png"], title=spec["title"], scene_time=scene.start,
                   blurb=spec["blurb"], started=started)


def render_rgb(spec: dict, scene: Scene) -> dict:
    started = time.time()
    fig, ax, _data_crs, _map_crs, _dpi = sp.generate_basemap()
    _draw_image(ax, scene, spec["recipe"](scene))
    _key_legend(fig, spec["key_colors"])
    return _finish(fig, ax, product=spec["key"], png=spec["png"], title=spec["title"], scene_time=scene.start,
                   blurb=spec["blurb"], started=started)


def _fire_pixels(path: Path) -> list[dict]:
    """Fire pixels inside the read box from one FDC file."""
    with xr.open_dataset(path, engine="h5netcdf") as dataset:
        mask = dataset["Mask"].values
        codes = {code for _name, _color, group in FIRE_CLASSES for code in group}
        rows, columns = np.nonzero(np.isin(mask, list(codes)))
        if not rows.size:
            return []
        height = float(dataset["goes_imager_projection"].attrs["perspective_point_height"])
        x = dataset["x"].values[columns] * height
        y = dataset["y"].values[rows] * height
        lonlat = ccrs.PlateCarree().transform_points(_geos_crs(dataset), x, y)
        power = dataset["Power"].values[rows, columns]
        temperature = dataset["Temp"].values[rows, columns]
        pixels = []
        for index in range(rows.size):
            lon, lat = float(lonlat[index, 0]), float(lonlat[index, 1])
            if READ_BOX[0] <= lon <= READ_BOX[2] and READ_BOX[1] <= lat <= READ_BOX[3]:
                pixels.append({
                    "pixel": (int(rows[index]), int(columns[index])), "lon": round(lon, 4), "lat": round(lat, 4),
                    "code": int(mask[rows[index], columns[index]]),
                    "frp_mw": None if not np.isfinite(power[index]) else round(float(power[index]), 1),
                    "temp_k": None if not np.isfinite(temperature[index]) else round(float(temperature[index]), 1),
                })
        return pixels


def _confidence(code: int) -> int:
    return next(index for index, (_name, _color, group) in enumerate(FIRE_CLASSES) if code in group)


def render_fire_detections(fdc_paths: list[Path], scan_time: datetime) -> dict:
    started = time.time()
    merged: dict = {}
    for path in fdc_paths:
        for pixel in _fire_pixels(path):
            seen = merged.setdefault(pixel["pixel"], {**pixel, "scans": 0})
            seen["scans"] += 1
            if _confidence(pixel["code"]) < _confidence(seen["code"]):
                seen["code"] = pixel["code"]
            if pixel["frp_mw"] is not None and (seen["frp_mw"] is None or pixel["frp_mw"] > seen["frp_mw"]):
                seen["frp_mw"] = pixel["frp_mw"]
    fig, ax, data_crs, map_crs, _dpi = sp.generate_basemap()
    # Keep only pixels inside the drawn frame (the read box is larger).
    candidates = list(merged.values())
    if candidates:
        points = map_crs.transform_points(data_crs, np.array([item["lon"] for item in candidates]),
                                          np.array([item["lat"] for item in candidates]))
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        candidates = [item for item, (x, y, _z) in zip(candidates, points) if x0 <= x <= x1 and y0 <= y <= y1]
    detections = sorted(candidates, key=lambda item: (_confidence(item["code"]), -(item["frp_mw"] or 0)))
    for level in reversed(range(len(FIRE_CLASSES))):  # high confidence drawn on top
        group = [item for item in detections if _confidence(item["code"]) == level]
        if group:
            sizes = [40 + 6 * min(item["frp_mw"] or 0, 150) ** 0.75 for item in group]
            ax.scatter([item["lon"] for item in group], [item["lat"] for item in group], s=sizes, transform=data_crs,
                       color=FIRE_CLASSES[level][1], edgecolor="black", linewidth=0.8, zorder=10 + (2 - level) * 0.1)
    handles = [Line2D([], [], marker="o", linestyle="none", markersize=11, markerfacecolor=color,
                      markeredgecolor="black", label=name) for name, color, _group in FIRE_CLASSES]
    legend = fig.legend(handles=handles, title="Fire Detections", loc="upper right", bbox_to_anchor=(0.99, 0.30),
                        bbox_transform=fig.transFigure, fontsize=12, title_fontsize=13, frameon=False, labelspacing=0.9)
    legend.set_zorder(20)

    state = sp._missouri_geometry()
    in_state = [item for item in detections
                if state is None or shapely.contains_xy(state, item["lon"], item["lat"])]
    strongest = max((item["frp_mw"] or 0 for item in in_state), default=0)
    summary = (f"{len(in_state)} fire pixel{'s' if len(in_state) != 1 else ''} in Missouri"
               + (f", strongest {strongest:.0f} MW" if strongest else "") + f"\n({len(detections)} on the map)")
    blurb = (f"Fire Detection and Characterization (FDC),\n2 km pixels detected in the last {FIRE_WINDOW_MIN} min.\n"
             "Dot size grows with fire radiative power.\n" + summary
             + "\n\nNot every detection is a wildfire (flares,\nindustry, hot surfaces); small or brief fires\nand fires under cloud are often missed.")
    return _finish(fig, ax, product="fire_detections", png="mo-sat-fire-detections.png",
                   title="Missouri Satellite Fire Detections", scene_time=scan_time, blurb=blurb, started=started,
                   window_minutes=FIRE_WINDOW_MIN, scans=len(fdc_paths), missouri_count=len(in_state),
                   map_count=len(detections),
                   detections=[{key: value for key, value in item.items() if key != "pixel"} for item in in_state[:200]])


# --- driver -------------------------------------------------------------------

def _wanted(when: str, sun: float) -> bool:
    return when == "any" or (when == "day" and sun >= DAY_SUN_DEG) or (when == "night" and sun <= NIGHT_SUN_DEG)


def main() -> None:
    multi_key = _list_keys("ABI-L2-MCMIPC")[-1]
    try:
        if SIGNATURE_PATH.read_text() == multi_key:
            print(f"Satellite scan {multi_key.rsplit('/', 1)[1]} already rendered; skipping.")
            return
    except OSError:
        pass
    scan_time = _scan_start(multi_key)
    sun = float(solar_elevation_deg(*CENTER, scan_time))
    print(f"Satellite scan {scan_time:%Y-%m-%d %H:%M} UTC, sun {sun:.1f} deg over central Missouri")

    channels = [spec for spec in CHANNELS if _wanted(spec["when"], sun)]
    rgbs = [spec for spec in RGBS if _wanted(spec["when"], sun)]
    with tempfile.TemporaryDirectory(prefix="smf-goes-") as folder, ThreadPoolExecutor(max_workers=6) as pool:
        folder = Path(folder)
        multi = pool.submit(_download, multi_key, folder)
        visible = None
        if any(spec["file"] == "visible" for spec in channels):
            visible_keys = [key for key in _list_keys("ABI-L2-CMIPC") if "-M6C02_" in key or "-M3C02_" in key]
            visible = pool.submit(_download, visible_keys[-1], folder) if visible_keys else None
        cutoff = scan_time - timedelta(minutes=FIRE_WINDOW_MIN)
        fdc_keys = [key for key in _list_keys("ABI-L2-FDCC") if cutoff < _scan_start(key) <= scan_time + timedelta(minutes=5)]
        fdc = [pool.submit(_download, key, folder) for key in fdc_keys]

        scenes = {}

        def scene(kind: str) -> Scene:
            if kind not in scenes:
                scenes[kind] = Scene((multi if kind == "multi" else visible).result())
            return scenes[kind]

        products = [(spec["title"], lambda spec=spec: render_channel(spec, scene(spec["file"])))
                    for spec in channels if spec["file"] == "multi" or visible is not None]
        products += [(spec["title"], lambda spec=spec: render_rgb(spec, scene("multi"))) for spec in rgbs]
        products.append(("Fire detections", lambda: render_fire_detections([job.result() for job in fdc], scan_time)))
        try:
            spx.run_products(products, label="Satellite maps")
        finally:
            for opened in scenes.values():
                opened.close()
    spx.STATUS_DIR.mkdir(parents=True, exist_ok=True)
    SIGNATURE_PATH.write_text(multi_key)


if __name__ == "__main__":
    main()
