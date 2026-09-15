# Verification data acquisition

## MRMS: operational cache

MRMS is optional and can be enabled without changing API startup:

```bash
VERIFICATION_MRMS_ENABLED=true
VERIFICATION_MRMS_ROOT=cache/mrms
VERIFICATION_MRMS_PRODUCT=MultiSensor_QPE_01H_Pass2
```

The scheduler downloads the previous complete UTC hour every 15 minutes from
NOAA's public MRMS HTTP directory, decodes the compressed GRIB2, clips it to
the buffered Missouri domain, and writes:

```text
cache/mrms/mrms_YYYYMMDD_HHz.nc
```

The normalized NetCDF contains `precipitation` in millimetres, latitude and
longitude coordinates, and provider/product/valid-time metadata. Files older
than `MRMS_RETENTION_DAYS` (default seven) are removed. A failed download is
logged and leaves the previous cache intact; it is never represented as zero
rain.

For alternate NOAA mirrors or products:

```bash
VERIFICATION_MRMS_ROOT_URL=https://mrms.ncep.noaa.gov/2D/MultiSensor_QPE_01H_Pass1
```

The product filename convention must remain:

```text
MRMS_<product>_00.00_YYYYMMDD-HH0000.grib2.gz
```

## NLCD: infrequent operator download

NLCD is static geography and is intentionally not downloaded by the API
scheduler or during application startup. Run this command when a new annual
release is needed:

```bash
cd api
python scripts/download_nlcd.py \
  --year 2023 \
  --output data/static/nlcd_class.tif
```

The default source is the USGS Annual NLCD Collection 1.0 CONUS mosaic. If
USGS/MRLC changes the collection path, provide the official URL explicitly:

```bash
python scripts/download_nlcd.py \
  --url "https://official-source.example/nlcd.tif" \
  --output data/static/nlcd_class.tif
```

The script:

1. Resumes a partial download when possible.
2. Accepts a GeoTIFF or a ZIP containing exactly one GeoTIFF.
3. Clips the source to the buffered Missouri bounding box.
4. Preserves the source CRS and categorical values.
5. Writes a SHA-256 manifest beside the output as `nlcd_class.json`.

The API automatically checks the downloader's default output path:

```text
api/data/static/nlcd_class.tif
```

No environment variable is needed for that standard location. For a custom
production mount, override it with:

```bash
VERIFICATION_NLCD_RASTER=/app/data/static/nlcd_class.tif
```

The API only reads the raster and validates that it has one band, a CRS, and
valid coordinates. Keep the raster and manifest together when promoting or
backing up a deployment.
