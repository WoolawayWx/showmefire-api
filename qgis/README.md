# Show Me Fire QGIS Server project

`showmefire.qgz` is the QGIS Server project for the published GIS volume.

The project expects the server data volume at `/data`:

```text
/data/qgis/showmefire.qgz
/data/latest/forecast_peak_fire_danger.tif
/data/state_missouri.geojson
/data/vectors/fire_detections.gpkg
/data/vectors/weather_stations.gpkg
```

The forecast raster is intentionally referenced through the stable
`/data/latest/forecast_peak_fire_danger.tif` path. The forecast publisher can
replace that file on each run without changing the QGIS project.

The project was styled using the existing `api/gis/peak_fire_danger.tif` as a
sample. The sample is not copied into this directory; production must publish
the forecast file at the path above before serving the project.
