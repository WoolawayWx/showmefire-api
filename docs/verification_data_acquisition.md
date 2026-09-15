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

## LANDFIRE FBFM40: infrequent operator download

The fuel model raster is static geography and is intentionally not downloaded
by the API scheduler or during application startup. Run this command when a
new release is needed:

```bash
cd api
python scripts/download_landfire.py \
  --output data/static/fbfm40_class.tif
```

The source is USGS/USFS LANDFIRE's 40 Scott & Burgan Fire Behavior Fuel
Models (FBFM40) product, retrieved through LFPS (the LANDFIRE Product
Service). The script submits an async clip-to-AOI job for the buffered
Missouri bounding box, polls it to completion, and downloads the resulting
GeoTIFF/ZIP over plain HTTPS — no browser, captcha, or AWS credentials
required.

> USGS ScienceBase's Annual NLCD bulk-download flow was previously used for
> this raster, but every year's archive is now S3-backed and its
> captcha-gated `requestDownload` endpoint returns a server-side 500. LFPS was
> adopted instead because it is genuinely scriptable and because FBFM40's
> fire-behavior fuel classes are a better fit for this app's fuel-regime
> policy than generic land-cover classes ever were.

`--layer` selects the LANDFIRE edition (default `LF2023_FBFM40`; also
available: `LF2024_FBFM40`). Avoid `LF2025_FBFM40` — as of this writing it
errors for this AOI ("no raster statistics... AOI falls outside input data").
`--email` is a required LFPS form field that is only format-validated, not
verified; override it with `VERIFICATION_LANDFIRE_EMAIL` if you want job
records attributable to a real address.

The script:

1. Submits the LFPS job and polls `--poll-seconds` apart for up to
   `--timeout-minutes` (defaults: 10s / 30m).
2. Accepts the resulting GeoTIFF or a ZIP containing exactly one GeoTIFF.
3. Clips the source to the buffered Missouri bounding box.
4. Preserves the source CRS and categorical values.
5. Writes a SHA-256 manifest beside the output as `fbfm40_class.json`.

The API automatically checks the downloader's default output path:

```text
api/data/static/fbfm40_class.tif
```

No environment variable is needed for that standard location. For a custom
production mount, override it with:

```bash
VERIFICATION_FUEL_RASTER=/app/data/static/fbfm40_class.tif
```

The API only reads the raster and validates that it has one band, a CRS, and
valid coordinates. Keep the raster and manifest together when promoting or
backing up a deployment.
