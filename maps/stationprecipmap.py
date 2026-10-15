"""Station precipitation maps (rolling 1 h and 24 h) for the current-conditions map and Mesonet page.

Uses /stations/precip: ASOS/AWOS and RAWS across Missouri and the neighbor states, with dry
stations as real zeros. Writes gis/realtime/station_precip_{1h,24h}.tif (map fills) and
images/mo-stationprecip-{1hr,24hr}.png (uploaded to the CDN by scripts/upload_cdn.py).
"""
import numpy as np

from station_png import render_png_products
from station_rbf import build_fill_products, fetch_records

BLURB = (
    "Data Source: Official Sites Only from Synoptic Weather\n"
    "ASOS/AWOS and RAWS rain gauges, including stations just outside\n"
    "Missouri so the state edges are anchored.\n\n"
    "Stations reporting no rain are plotted as 0.00 in.\n"
    "Values interpolated from available stations; local variations may occur."
)

SPECS = [
    dict(name='station_precip_1h', field='precip_1h_in', cmap='Blues', vmin=0, vmax=1.0, max_km=80,
         description='Missouri station precipitation, last hour (RGBA)',
         legend='Precipitation in the last hour (in) mapped with Blues, transparent outside Missouri',
         png='mo-stationprecip-1hr.png', title='Missouri 1-Hour Station Precipitation',
         units_label='Precipitation, last hour (in)', ticks=np.arange(0, 1.01, 0.1), label_fmt='{:.2f}', blurb=BLURB),
    dict(name='station_precip_24h', field='precip_24h_in', cmap='Blues', vmin=0, vmax=3.0, max_km=80,
         description='Missouri station precipitation, rolling 24 hours (RGBA)',
         legend='Precipitation in the last 24 hours (in) mapped with Blues, transparent outside Missouri',
         png='mo-stationprecip-24hr.png', title='Missouri 24-Hour Station Precipitation',
         units_label='Precipitation, last 24 hours (in)', ticks=np.arange(0, 3.01, 0.5), label_fmt='{:.2f}', blurb=BLURB),
]


def main():
    records = fetch_records('/stations/precip')
    grids = {}  # each field is gridded once and shared by the GeoTIFF and PNG passes
    build_fill_products(records, SPECS, grids=grids)
    render_png_products(records, SPECS, grids=grids)


if __name__ == '__main__':
    main()
