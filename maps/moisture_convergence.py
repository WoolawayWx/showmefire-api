"""Moisture Convergence: highest-complexity, most-expendable item in the
current-conditions catalog (see the implementation plan's open risks). Unlike
every other product here, no existing maps/*.py script computes a spatial
divergence - this interpolates wind (u, v) and specific humidity onto the
same 400x400 grid rhmap-fil.py/windmap-fil.py already use, then computes
horizontal moisture flux divergence (-nabla . (q*V)) via numpy.gradient with
degree-to-meter conversion at Missouri's latitude.

This is a diagnostic/relative-units product, not a calibrated physical
moisture-tendency forecast - treat the sign (converging vs. diverging) as the
useful signal, not the absolute magnitude.
"""
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import Rbf
from scipy.ndimage import gaussian_filter
from shapely.geometry import Point

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_DIR))

from maps import station_plot as sp
from services import derived_met

STATUS_PATH = PROJECT_DIR / "data" / "current_conditions" / "status" / "moistureconvergence.json"
PNG_PATH = PROJECT_DIR / "images" / "mo-moistureconvergence.png"

GRID_SIZE = 400
METERS_PER_DEGREE_LAT = 111_320.0


def compute_moisture_convergence(u_ms: np.ndarray, v_ms: np.ndarray, q_kg_per_kg: np.ndarray,
                                  lon_mesh: np.ndarray, lat_mesh: np.ndarray) -> np.ndarray:
    """-div(q * V) on a regular lon/lat mesh, converting degree spacing to
    meters at each row's latitude. Positive = moisture converging."""
    mean_lat_rad = np.radians(np.nanmean(lat_mesh))
    meters_per_degree_lon = METERS_PER_DEGREE_LAT * np.cos(mean_lat_rad)

    qu = q_kg_per_kg * u_ms
    qv = q_kg_per_kg * v_ms

    dlon = lon_mesh[0, 1] - lon_mesh[0, 0]
    dlat = lat_mesh[1, 0] - lat_mesh[0, 0]
    dx_m = dlon * meters_per_degree_lon
    dy_m = dlat * METERS_PER_DEGREE_LAT

    d_qu_dx = np.gradient(qu, dx_m, axis=1)
    d_qv_dy = np.gradient(qv, dy_m, axis=0)
    return -(d_qu_dx + d_qv_dy)


if __name__ == "__main__":
    start_time = time.time()
    stations = sp.fetch_stations()

    points_u, points_v, points_q, points_lon, points_lat = [], [], [], [], []
    for station in stations:
        obs = station.get("observations", {})
        speed = obs.get("wind_speed", {}).get("value")
        direction = obs.get("wind_direction", {}).get("value")
        dewpoint = obs.get("dew_point_temperature", {}).get("value")
        elevation_ft = station.get("elevation")
        if None in (speed, direction, dewpoint, elevation_ft):
            continue
        pressure_hpa = derived_met.standard_atmosphere_pressure_hpa(elevation_ft * 0.3048)
        q_g_per_kg = derived_met.specific_humidity_g_per_kg(dewpoint, pressure_hpa)
        points_u.append(derived_met.mph_to_ms(-speed * np.sin(np.radians(direction))))
        points_v.append(derived_met.mph_to_ms(-speed * np.cos(np.radians(direction))))
        points_q.append(q_g_per_kg / 1000.0)  # kg/kg for the flux calculation
        points_lon.append(station["longitude"])
        points_lat.append(station["latitude"])

    fig, ax, data_crs, map_crs, mapdpi = sp.generate_basemap()
    have_data = len(points_lon) >= 4  # Rbf needs enough points for a stable fit
    plotted_cells = 0

    if have_data:
        grid_lon = np.linspace(sp.MO_EXTENT[0], sp.MO_EXTENT[1], GRID_SIZE)
        grid_lat = np.linspace(sp.MO_EXTENT[2], sp.MO_EXTENT[3], GRID_SIZE)
        grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)

        u_grid = gaussian_filter(Rbf(points_lon, points_lat, points_u, function="multiquadric", smooth=0.5)(grid_lon_mesh, grid_lat_mesh), sigma=2)
        v_grid = gaussian_filter(Rbf(points_lon, points_lat, points_v, function="multiquadric", smooth=0.5)(grid_lon_mesh, grid_lat_mesh), sigma=2)
        q_grid = gaussian_filter(Rbf(points_lon, points_lat, points_q, function="multiquadric", smooth=0.5)(grid_lon_mesh, grid_lat_mesh), sigma=2)

        convergence = compute_moisture_convergence(u_grid, v_grid, q_grid, grid_lon_mesh, grid_lat_mesh)
        # Scale to a display-friendly magnitude (1e-8 /s is a typical
        # synoptic-scale moisture-convergence order of magnitude).
        convergence_scaled = convergence / 1e-8

        state_path = sp.STATE_SHAPEFILE
        if state_path.is_file():
            state = gpd.read_file(state_path)
            if state.crs != data_crs.proj4_init:
                state = state.to_crs(data_crs.proj4_init)
            if not state.empty:
                missouri_geom = state.geometry.iloc[0]
                grid_points = [Point(lon, lat) for lon, lat in zip(grid_lon_mesh.ravel(), grid_lat_mesh.ravel())]
                within_mask = gpd.GeoSeries(grid_points).within(missouri_geom).values.reshape(grid_lon_mesh.shape)
                convergence_scaled[~within_mask] = np.nan

        finite = convergence_scaled[np.isfinite(convergence_scaled)]
        vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
        vmax = vmax or 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        mesh = ax.contourf(
            grid_lon_mesh, grid_lat_mesh, convergence_scaled, transform=data_crs,
            levels=np.linspace(-vmax, vmax, 41), cmap="BrBG", norm=norm, alpha=0.8, zorder=7,
        )
        cax = fig.add_axes([0.02, 0.08, 0.02, 0.6])
        fig.colorbar(mesh, cax=cax, label="Moisture Convergence (relative units)")
        plotted_cells = int(np.isfinite(convergence_scaled).sum())

    sp.redraw_boundaries(ax, data_crs)

    status_note = "" if have_data else " - insufficient station coverage for this cycle"
    sp.add_branding(
        fig, ax,
        title="Estimated Moisture Convergence",
        subtitle=f"Valid Time: {pd.Timestamp.now(tz='America/Chicago').strftime('%Y-%m-%d %H:%M CT')}",
        description=(
            "Data Source: Official Sites Only from Synoptic Weather\n"
            "AWOS, RAWS, Missouri Mesonet\n\n"
            "ESTIMATED moisture flux convergence from interpolated wind and\n"
            "specific humidity (brown = diverging/drying, green = converging/\n"
            "moistening). Diagnostic/relative units, not a calibrated\n"
            "forecast quantity." + status_note
        ),
    )

    runtime_sec = time.time() - start_time
    sp.save_and_record_status(
        fig, PNG_PATH, STATUS_PATH,
        {
            "product": "moisture_convergence_estimated",
            "status": "updated" if have_data else "insufficient_data",
            "estimated": True,
            "last_update": pd.Timestamp.now(tz="America/Chicago").strftime("%Y-%m-%d %H:%M CT"),
            "runtime_sec": round(runtime_sec, 2),
            "grid_cells_plotted": plotted_cells,
            "station_count": len(points_lon),
        },
    )
    print(f"Moisture convergence map updated ({len(points_lon)} stations, {runtime_sec:.2f}s)")
