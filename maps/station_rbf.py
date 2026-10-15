"""Helpers shared by the station-based map scripts (imported, never run: see scripts/maps.sh).

The Missouri-only `/stations` feed leaves the interpolation with no points past the border, so
values near it are extrapolated. `buffer_points` adds neighbor-state stations from
`/stations/buffer`; a map script still draws markers, labels and masks to Missouri only.

`build_fill_products` turns a station feed (`/stations/precip`, `/stations/et`) into the RGBA
GeoTIFF fills served by routers/conditions.py. It grids with inverse-distance weighting rather
than the multiquadric RBF the other maps use: an RBF overshoots and goes negative around the
many true zeros in rainfall, while IDW stays within the observed range.
"""
import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import requests
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

try:  # run as a script from maps/ (scripts/maps.sh)
    from realtime_geotiff import export_continuous_rgba_geotiff
except ImportError:  # imported as maps.station_rbf (tests)
    from maps.realtime_geotiff import export_continuous_rgba_geotiff

MAPS_DIR = Path(__file__).resolve().parent
REALTIME_DIR = MAPS_DIR.parent / 'gis/realtime'
STATE_BOUNDARY = MAPS_DIR / 'shapefiles/MO_State_Boundary/MO_State_Boundary.shp'

# Same grid the other station maps use.
EXTENT = (-95.8, -89.1, 35.8, 40.8)  # lon_min, lon_max, lat_min, lat_max
GRID_SIZE = 400
KM_PER_DEGREE = 111.2
MID_LATITUDE = 38.5

# Same filter the map scripts apply to /stations.
VALID_NETWORKS = {'RAWS', 'ASOS/AWOS', 'MOCOMAGNET', 'Missouri Mesonet'}


def _port():
    return os.getenv('PORT', '8000')


def buffer_stations():
    """Neighbor-state station dicts from ``/stations/buffer`` limited to the map networks.

    Returns [] if the endpoint is unavailable, so a map still renders from Missouri stations alone.
    """
    try:
        response = requests.get(f'http://localhost:{_port()}/stations/buffer', timeout=60)
        response.raise_for_status()
        stations = response.json().get('stations') or []
    except Exception as error:
        print(f"Warning: buffer stations unavailable, interpolating from Missouri stations only ({error})")
        return []
    return [station for station in stations if station.get('network') in VALID_NETWORKS]


def buffer_points(variable, cap=None):
    """(lon, lat, value) for neighbor-state stations that report ``variable``.

    Returns [] if the buffer endpoint is unavailable, so a map still renders (as before) from
    Missouri stations alone.
    """
    try:
        response = requests.get(f'http://localhost:{_port()}/stations/buffer', timeout=60)
        response.raise_for_status()
        stations = response.json().get('stations') or []
    except Exception as error:
        print(f"Warning: buffer stations unavailable, interpolating from Missouri stations only ({error})")
        return []

    points = []
    for station in stations:
        if station.get('network') not in VALID_NETWORKS:
            continue
        value = (station.get('observations') or {}).get(variable, {}).get('value')
        if value is None:
            continue
        points.append((station['longitude'], station['latitude'], min(value, cap) if cap is not None else value))
    return points


def fetch_records(path):
    """Station records from a local API feed such as ``/stations/precip``."""
    response = requests.get(f'http://localhost:{_port()}{path}', timeout=120)
    response.raise_for_status()
    return response.json().get('stations') or []


def make_grid():
    """(lon_mesh, lat_mesh), rows south to north, as the GeoTIFF exporter expects."""
    lon_min, lon_max, lat_min, lat_max = EXTENT
    return np.meshgrid(np.linspace(lon_min, lon_max, GRID_SIZE), np.linspace(lat_min, lat_max, GRID_SIZE))


def _to_km(lon, lat):
    return np.column_stack([np.asarray(lon) * KM_PER_DEGREE * math.cos(math.radians(MID_LATITUDE)),
                            np.asarray(lat) * KM_PER_DEGREE])


def idw_grid(points, lon_mesh, lat_mesh, k=8, power=2.0, max_km=80.0):
    """Inverse-distance-weighted grid of ``(lon, lat, value)`` points.

    Uses the ``k`` nearest stations within ``max_km``; cells with none are NaN (no data, not
    zero). Output always lies within the range of the input values, and a station's own
    value is reproduced at its location.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    grid = np.full(lon_mesh.shape, np.nan)
    if len(points) == 0:
        return grid
    tree = cKDTree(_to_km(points[:, 0], points[:, 1]))
    distances, indexes = tree.query(_to_km(lon_mesh.ravel(), lat_mesh.ravel()), k=min(k, len(points)),
                                    distance_upper_bound=max_km)
    distances = distances.reshape(-1, min(k, len(points)))
    indexes = indexes.reshape(distances.shape)
    found = np.isfinite(distances)
    weights = np.where(found, 1.0 / np.maximum(distances, 1.0) ** power, 0.0)
    values = np.where(found, points[:, 2][np.minimum(indexes, len(points) - 1)], 0.0)
    total = weights.sum(axis=1)
    result = np.full(total.shape, np.nan)
    ok = total > 0
    result[ok] = (weights * values).sum(axis=1)[ok] / total[ok]
    return result.reshape(lon_mesh.shape)


def smooth_nan(grid, sigma):
    """Gaussian smoothing that ignores NaN cells (so edges don't fade toward zero)."""
    if sigma <= 0:
        return grid
    valid = np.isfinite(grid)
    weight = gaussian_filter(valid.astype(float), sigma)
    smoothed = gaussian_filter(np.where(valid, grid, 0.0), sigma)
    out = np.full(grid.shape, np.nan)
    ok = valid & (weight > 1e-6)
    out[ok] = smoothed[ok] / weight[ok]
    return out


@lru_cache(maxsize=1)
def _missouri_boundary():
    """Missouri outline in lon/lat, read once per process."""
    import geopandas as gpd

    return gpd.read_file(STATE_BOUNDARY).to_crs(4326).geometry.union_all()


def mask_to_missouri(grid, lon_mesh, lat_mesh):
    """NaN everywhere outside the Missouri state boundary."""
    import shapely

    boundary = _missouri_boundary()
    inside = shapely.contains_xy(boundary, lon_mesh, lat_mesh)
    return np.where(inside, grid, np.nan)


def grid_for_spec(records, spec, lon_mesh, lat_mesh, sigma=3.0, grids=None):
    """Missouri-masked grid for one spec, or None when no station has its field.

    Stations whose field is None are left out (unknown is not zero). ``grids`` is an optional
    dict that memoizes the result by spec name, so the GeoTIFF and PNG passes grid each field once.
    """
    if grids is not None and spec['name'] in grids:
        return grids[spec['name']]
    result = _grid_for_spec(records, spec, lon_mesh, lat_mesh, sigma)
    if grids is not None:
        grids[spec['name']] = result
    return result


def _grid_for_spec(records, spec, lon_mesh, lat_mesh, sigma):
    points = [(r['longitude'], r['latitude'], r[spec['field']])
              for r in records if r.get(spec['field']) is not None]
    if not points:
        return None, 0
    grid = idw_grid(points, lon_mesh, lat_mesh, max_km=spec['max_km'])
    return mask_to_missouri(smooth_nan(grid, sigma), lon_mesh, lat_mesh), len(points)


def build_fill_products(records, specs, out_dir=None, sigma=3.0, grids=None):
    """Grid each spec's field from the station ``records`` and write its GeoTIFF fill.

    A spec is a dict: ``name`` (file stem), ``field``, ``cmap``, ``vmin``, ``vmax``,
    ``max_km``, ``description``, ``legend``. Returns the names written; a spec with no
    stations writes nothing.
    """
    out_dir = Path(out_dir) if out_dir else REALTIME_DIR
    lon_mesh, lat_mesh = make_grid()
    written = []
    for spec in specs:
        grid, count = grid_for_spec(records, spec, lon_mesh, lat_mesh, sigma, grids)
        if grid is None:
            print(f"No stations with {spec['field']}; leaving {spec['name']}.tif unchanged")
            continue
        ok = export_continuous_rgba_geotiff(
            grid_values=grid, lon_mesh=lon_mesh, lat_mesh=lat_mesh, out_path=out_dir / f"{spec['name']}.tif",
            cmap_name=spec['cmap'], vmin=spec['vmin'], vmax=spec['vmax'],
            description=spec['description'], source='Synoptic observations + ShowMeFire interpolation',
            legend=spec.get('legend'))
        if ok:
            written.append(spec['name'])
            print(f"Wrote {spec['name']}.tif from {count} stations")
        else:
            print(f"Warning: failed to export {spec['name']}.tif")
    return written
