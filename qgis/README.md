# Show Me Fire QGIS Server project

`showmefire.qgz` is the QGIS Server project for the published GIS volume. It
intentionally contains only the operational Day 1 forecast raster, the Day 1
forecast polygons, and the all-county burn-ban layer.

The project expects the server data volume at `/data`:

```text
/data/qgis/showmefire.qgz
/data/latest/forecast_peak_fire_danger.tif
/data/peak_fire_danger_polygons.geojson
/data/burn_bans.gpkg
```

The forecast raster is intentionally referenced through the stable
`/data/latest/forecast_peak_fire_danger.tif` path. The forecast publisher can
replace that file on each run without changing the QGIS project.

The polygon layer is categorized by `danger_level` and the burn-ban layer is
categorized by `status` (`active` / `inactive`). The forecast polygon
publisher buffers cell regions by 250 meters and clips them to the Missouri
state boundary before writing GeoJSON.

For a server deployment, add the QGIS service beside the API service in the
same compose file, using the API's published GIS volume. Keep these limits on
the GIS services so forecast compute has priority:

```yaml
  qgis-server:
    image: qgis/qgis-server:4.0-trixie
    cpus: "0.50"
    mem_limit: 1g
    memswap_limit: 1g
    cpu_shares: 128
    volumes:
      - /mnt/adrive/showmefire-data/gis:/data:ro
      - ./showmefire-api/qgis/showmefire.qgz:/project/showmefire.qgz:ro
```
