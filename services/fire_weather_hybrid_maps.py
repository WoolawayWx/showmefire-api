"""Experimental daily graphics using the current mesonet basemap/branding."""
from __future__ import annotations
from pathlib import Path
import numpy as np


def region_bounds(path):
    """SEMA files are NAD83 UTM, while map extents require longitude/latitude."""
    import pyproj
    import cartopy.io.shapereader as shpreader
    geometries=list(shpreader.Reader(str(path)).geometries())
    if not geometries:return None
    bounds=np.asarray([geometry.bounds for geometry in geometries])
    source=(bounds[:,0].min(),bounds[:,1].min(),bounds[:,2].max(),bounds[:,3].max())
    crs=pyproj.CRS.from_wkt(Path(path).with_suffix(".prj").read_text())
    west,south,east,north=pyproj.Transformer.from_crs(crs,"EPSG:4326",always_xy=True).transform_bounds(*source,densify_pts=21)
    return west,east,south,north


def render(directory):
    import netCDF4
    import cartopy.crs as ccrs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    from maps import station_plot as sp
    from services.fire_weather_hybrid.contracts import LABELS, COLORS
    from services.fire_weather_hybrid_store import read_json
    directory = Path(directory)
    manifest = read_json(directory/"manifest.json")
    with netCDF4.Dataset(directory/"forecast.nc") as ds:
        lat,lon = ds["latitude"][:], ds["longitude"][:]
        values = np.ma.filled(ds["danger_category"][:],np.nan)
        # A peak uses coincident hourly danger, not separately maximized inputs.
        from services.fire_weather_hybrid.contracts import utc, LOCAL_TZ
        issued=utc(manifest["issued_at_utc"])
        future = np.array([kind == "forecast" and utc(valid)>=issued
                           for kind,valid in zip(manifest["step_kinds"],manifest["valid_times_utc"])])
        peak = np.full(values.shape[1:],np.nan)
        for index in np.flatnonzero(future):
            peak = np.fmax(peak,values[index])
    out = directory/"images"
    out.mkdir(exist_ok=True)
    regions = [("mo",None)]
    shapes = Path(__file__).resolve().parents[1]/"maps/shapefiles/SEMA Regions"
    for path in sorted(shapes.glob("Region*.shp")):
        bounds=region_bounds(path)
        if bounds:regions.append((path.stem.lower().replace("region","reg").replace(" ",""),bounds))
    for name,bounds in regions:
        fig,ax,data_crs,_,dpi = sp.generate_basemap()
        try:
            if bounds:
                west,east,south,north = bounds
                pad = max(east-west,north-south)*.12
                ax.set_extent((west-pad,east+pad,south-pad,north+pad),crs=data_crs)
                ax.set_position([.01,.04,.63,.92])
            ax.pcolormesh(lon,lat,np.ma.masked_invalid(peak),transform=ccrs.PlateCarree(),shading="nearest",
                          cmap=ListedColormap(COLORS),norm=BoundaryNorm(np.arange(-.5,5.5),5),zorder=7)
            legend=fig.legend(handles=[Patch(facecolor=color,label=label) for color,label in zip(COLORS,LABELS)],
                      loc="upper right",bbox_to_anchor=(.99,.37),bbox_transform=fig.transFigure,
                      title="Missouri-relative danger",fontsize=12,frameon=False)
            legend.set_zorder(100)
            sp.add_branding(fig,ax,title="Experimental Fire Danger",subtitle="Day 1 · remaining forecast hours",
                description=f"STAFF PREVIEW\nIssued: {manifest['issued_at_utc']}\nModel: {manifest.get('model_version','unknown')}\n"
                "Conditional surface-fire potential\n1 km output spacing\nExperimental relative severity scale",
                update_schedule="Experimental hourly updates")
            for text in fig.texts:text.set_zorder(100)
            fig.savefig(out/f"{name}-danger-day1.png",dpi=dpi)
        finally:
            plt.close(fig)
    return sorted(path.name for path in out.glob("*.png"))

