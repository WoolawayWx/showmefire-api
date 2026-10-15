"""Titled PNG maps (same look as mo-rh.png) for the station precipitation and ET fills.

Imported by stationprecipmap.py / etmap.py, never run directly (see scripts/maps.sh). Each spec
that has a ``png`` key is drawn from the same Missouri-masked IDW grid as its GeoTIFF fill.
"""
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import cartopy.crs as ccrs
import matplotlib.font_manager as font_manager
import matplotlib.image as mpimg
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

try:
    import station_plot as sp
    from station_rbf import EXTENT, MAPS_DIR, grid_for_spec, make_grid
except ImportError:  # imported as maps.station_png (tests)
    from maps import station_plot as sp
    from maps.station_rbf import EXTENT, MAPS_DIR, grid_for_spec, make_grid

PROJECT_DIR = MAPS_DIR.parent
IMAGES_DIR = PROJECT_DIR / 'images'
PIXEL_W, PIXEL_H, DPI = 2048, 1152, 144
LABEL_MIN_DISTANCE_DEG = 0.12


def _load_fonts():
    for name in ('Montserrat/static/Montserrat-Regular.ttf', 'Plus_Jakarta_Sans/static/PlusJakartaSans-Regular.ttf',
                 'Plus_Jakarta_Sans/static/PlusJakartaSans-Bold.ttf'):
        path = PROJECT_DIR / 'assets' / name
        if path.exists():
            font_manager.fontManager.addfont(str(path))
    plt.rcParams['font.family'] = 'Montserrat'


def _basemap():
    data_crs = ccrs.PlateCarree()
    fig = plt.figure(figsize=(PIXEL_W / DPI, PIXEL_H / DPI), dpi=DPI, facecolor='#E8E8E8')
    ax = plt.axes([0, 0, 1, 1], projection=ccrs.LambertConformal(central_longitude=-92.45, central_latitude=38.3))
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_extent(EXTENT, crs=data_crs)
    ax.set_autoscale_on(False)
    # Cached, pre-projected outlines drawn above the fill (zorder 8/9), as in the other maps.
    sp.redraw_boundaries(ax, data_crs)
    return fig, ax, data_crs


def render_station_png(grid, spec, labels, out_path):
    """Draw ``grid`` (south-to-north rows, NaN outside Missouri) with ``spec``'s title and text.

    ``labels`` is (lon, lat, value) for the Missouri stations to mark.
    """
    _load_fonts()
    fig, ax, data_crs = _basemap()
    # Contour in map-projected coordinates (cached per process) instead of having cartopy re-project
    # every polygon; this is the same 400x400 lon/lat grid, so the picture is unchanged.
    _, _, _, x_mesh, y_mesh = sp._fill_grid(400, ax.projection, data_crs)
    vmin, vmax = spec['vmin'], spec['vmax']
    # Keep true zeros (dry) in the lowest colour band instead of dropping them as "below range".
    shown = np.where(np.isfinite(grid), np.maximum(grid, vmin + 1e-6 * (vmax - vmin)), np.nan)
    contours = ax.contourf(x_mesh, y_mesh, shown, levels=np.linspace(vmin, vmax, 161),
                           cmap=sp.opaque_blend_cmap(spec['cmap'], 0.75), zorder=7, antialiased=False, extend='max')
    plt.colorbar(contours, cax=fig.add_axes([0.02, 0.08, 0.02, 0.6]), label=spec['units_label'],
                 ticks=spec.get('ticks', np.linspace(vmin, vmax, 7)))

    placed = []
    for lon, lat, value in labels:
        position = np.array([lon, lat])
        if all(np.linalg.norm(position - other) > LABEL_MIN_DISTANCE_DEG for other in placed):
            ax.scatter(lon, lat, transform=data_crs, color='black', s=20, zorder=10, edgecolor='white', linewidth=1)
            ax.text(lon, lat + 0.03, spec['label_fmt'].format(value), transform=data_crs, fontsize=9, color='black',
                    zorder=11, ha='center', va='bottom',
                    path_effects=[path_effects.withStroke(linewidth=2, foreground='white')])
            placed.append(position)

    ax.set_anchor('W')
    plt.subplots_adjust(left=0.05)
    fig.text(0.99, 0.97, spec['title'], fontsize=26, fontweight='bold', ha='right', va='top', fontname='Plus Jakarta Sans')
    fig.text(0.99, 0.90, 'Valid Time: ' + pd.Timestamp.now(tz='America/Chicago').strftime('%Y-%m-%d %H:%M CT'),
             fontsize=16, ha='right', va='top', fontname='Montserrat')
    fig.text(0.99, 0.6, spec['blurb'], fontsize=12, ha='right', va='top', fontname='Montserrat')
    fig.text(0.02, 0.01, 'ShowMeFire.org', fontsize=20, fontweight='bold', ha='left', va='bottom', fontname='Montserrat')
    try:
        import cairosvg
        logo = mpimg.imread(BytesIO(cairosvg.svg2png(url=str(PROJECT_DIR / 'assets/LightBackGroundLogo.svg'))), format='png')
        ax.add_artist(AnnotationBbox(OffsetImage(logo, zoom=0.03), (0.99, 0.01), frameon=False,
                                     xycoords='figure fraction', box_alignment=(1, 0)))
    except Exception:  # the logo is decoration: never lose the map over it
        pass

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def render_png_products(records, specs, images_dir=None, grids=None):
    """Render a PNG for every spec with a ``png`` key. Returns the paths written."""
    images_dir = Path(images_dir) if images_dir else IMAGES_DIR
    lon_mesh, lat_mesh = make_grid()
    written = []
    for spec in specs:
        if 'png' not in spec:
            continue
        grid, _ = grid_for_spec(records, spec, lon_mesh, lat_mesh, grids=grids)
        if grid is None:
            print(f"No stations with {spec['field']}; leaving {spec['png']} unchanged")
            continue
        labels = [(r['longitude'], r['latitude'], r[spec['field']]) for r in records
                  if r.get('in_state') and r.get(spec['field']) is not None]
        render_station_png(grid, spec, labels, images_dir / spec['png'])
        written.append(images_dir / spec['png'])
        print(f"Wrote {spec['png']}")
    return written
