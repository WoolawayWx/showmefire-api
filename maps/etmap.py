"""Station reference-ET maps (current, rolling 24 h, peak and low hourly rate).

Uses /stations/et: hourly FAO-56 Penman-Monteith at RAWS stations (the only network with solar
radiation) across Missouri and the neighbor states. Writes gis/realtime/station_et_*.tif (map
fills) and images/mo-et-*.png (uploaded to the CDN by scripts/upload_cdn.py).
"""
import numpy as np

from station_png import render_png_products
from station_rbf import build_fill_products, fetch_records

BLURB = (
    "Data Source: RAWS Stations from Synoptic Weather\n"
    "(the only official network reporting solar radiation)\n\n"
    "Grass reference evapotranspiration, FAO-56 Penman-Monteith,\n"
    "calculated hourly from temperature, humidity, wind and solar radiation.\n"
    "Values interpolated from available stations (RAWS are sparse, so areas far from a\n"
    "station are estimates); local variations may occur."
)

SPECS = [
    dict(name='station_et_current', field='et_current_in_hr', cmap='YlOrBr', vmin=0, vmax=0.04, max_km=150,
         description='Missouri reference ET, current hourly rate (RGBA)',
         legend='Reference ET, latest hour (in/hr) mapped with YlOrBr, transparent outside Missouri',
         png='mo-et-current.png', title='Missouri Current Evapotranspiration',
         units_label='Reference ET, latest hour (in/hr)', ticks=np.arange(0, 0.0401, 0.005), label_fmt='{:.3f}', blurb=BLURB),
    dict(name='station_et_24h', field='et_24h_in', cmap='YlOrBr', vmin=0, vmax=0.30, max_km=150,
         description='Missouri reference ET, rolling 24 hours (RGBA)',
         legend='Reference ET, last 24 hours (in) mapped with YlOrBr, transparent outside Missouri',
         png='mo-et-24hr.png', title='Missouri 24-Hour Evapotranspiration',
         units_label='Reference ET, last 24 hours (in)', ticks=np.arange(0, 0.301, 0.05), label_fmt='{:.3f}', blurb=BLURB),
    dict(name='station_et_max', field='et_max_in_hr', cmap='YlOrBr', vmin=0, vmax=0.04, max_km=150,
         description='Missouri reference ET, peak hourly rate in the last 24 hours (RGBA)',
         legend='Peak hourly reference ET, last 24 hours (in/hr) mapped with YlOrBr, transparent outside Missouri',
         png='mo-et-peak-24hr.png', title='Missouri Peak Evapotranspiration Rate',
         units_label='Peak hourly ET, last 24 hours (in/hr)', ticks=np.arange(0, 0.0401, 0.005), label_fmt='{:.3f}', blurb=BLURB),
    dict(name='station_et_min', field='et_min_in_hr', cmap='YlOrBr', vmin=0, vmax=0.01, max_km=150,
         description='Missouri reference ET, lowest hourly rate in the last 24 hours (RGBA)',
         legend='Lowest hourly reference ET, last 24 hours (in/hr) mapped with YlOrBr, transparent outside Missouri',
         png='mo-et-low-24hr.png', title='Missouri Lowest Evapotranspiration Rate',
         units_label='Lowest hourly ET, last 24 hours (in/hr)', ticks=np.arange(0, 0.0101, 0.001), label_fmt='{:.3f}', blurb=BLURB),
]


def main():
    records = fetch_records('/stations/et')
    build_fill_products(records, SPECS)
    render_png_products(records, SPECS)


if __name__ == '__main__':
    main()
