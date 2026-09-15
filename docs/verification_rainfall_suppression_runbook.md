# Rainfall-conditioned verification maps

## Purpose

The verification system keeps the original station-observed and RTMA peak maps
unchanged. It additionally produces a rainfall-conditioned RTMA map and, when
both inputs exist, a combined station/RTMA map. The adjusted products are
diagnostic verification layers; they do not replace the operational forecast
or observed products.

## Data flow

1. RAWS/Synoptic observations are archived by `services.synoptic`.
2. `forecast/endOfDayReport.py` extracts station precipitation and converts
   Synoptic's English-unit accumulation from inches to millimetres.
3. RTMA cache files contain hourly `apcp` in millimetres. The RTMA peak job
   accumulates these values through the 10:00–21:00 Central verification
   window.
4. The LANDFIRE FBFM40 fuel model raster is loaded only from a local,
   administrator-configured raster or NetCDF static bundle. The API does not
   download or rebuild geography.
5. `services.verification_rainfall` maps FBFM40 fuel model codes to fuel
   regimes, applies the rainfall policy, and reduces the ordinal category by
   zero, one, or two levels.
6. `services.verification_artifacts` aligns the adjusted RTMA raster with the
   station raster and writes the combined GeoTIFF.

The provider precedence is:

```text
MRMS -> RTMA APCP -> station precipitation
```

MRMS is read from the configured root when a matching NetCDF or GeoTIFF exists.
NetCDF files must contain `precipitation`, `precip_mm`, or `apcp`, plus
latitude/longitude coordinates; values are millimetres. Until a matching MRMS
file exists, RTMA is the spatial provider and stations are retained for local
validation/provenance. A missing provider never becomes zero rain.

## Configuration

The following settings are optional:

```bash
# Either a categorical FBFM40 GeoTIFF...
VERIFICATION_FUEL_RASTER=/data/static/fbfm40_class.tif

# ...or a NetCDF bundle containing fuel_model, latitude, longitude.
VERIFICATION_STATIC_BUNDLE=/data/static/static_bundle.nc

# Optional directory of administrator-provided MRMS NetCDF/GeoTIFF files.
VERIFICATION_MRMS_ROOT=/data/mrms
```

If neither fuel-raster setting is configured, the raw maps continue to work
and the rainfall-adjusted RTMA/combined artifacts are reported as
unavailable.

## FBFM40 regimes and thresholds

The current contract is `verification-rainfall-v2`. Values are millimetres of
accumulated rain. Codes are LANDFIRE's 40 Scott & Burgan Fire Behavior Fuel
Models (FBFM40); non-burnable urban/snow-ice/water/barren codes (91, 92, 98,
99) are intentionally unmapped and never produce a regime:

| Regime | FBFM40 codes | Threshold | Relief e-folding period |
| --- | --- | ---: | ---: |
| Grass/pasture | 101–109 (GR1–GR9) | 2.5 mm | 18 hours |
| Agriculture | 93 (NB3) | 5.0 mm | 48 hours |
| Shrubland | 121–124 (GS1–GS4), 141–149 (SH1–SH9) | 6.3 mm | 72 hours |
| Open woodland | 161–165 (TU1–TU5) | 12.7 mm | 120 hours |
| Dense forest | 181–189 (TL1–TL9), 201–204 (SB1–SB4) | 38.1 mm | 336 hours |

The dense-forest threshold is the midpoint of the requested 25–50+ mm range.
It is intentionally a documented policy value, not a claim that all timber
fuels respond identically.

## Degradation formula

For each pixel:

```text
rainfall_fraction = clamp(accumulation_mm / threshold_mm, 0, 1)
time_decay = exp(-hours_since_rain / relief_hours)
weather_factor =
  clamp(0.65 + 0.35 * RH/60 - 0.25 * wind_kts/25, 0.35, 1.0)
category_reduction =
  round(2 * rainfall_fraction * time_decay * weather_factor)
adjusted_category = max(0, raw_category - category_reduction)
```

The adjustment is bounded to two ordinal levels. Rain alone cannot create a
Low observation from an unknown or invalid input, and a missing fuel model or
rainfall value returns zero reduction with an explicit reason. High wind and low humidity
reduce the effective relief. The current implementation does not model
rainfall intensity, runoff, soil infiltration, or green-up rebound; those
require additional validated inputs.

## Generated artifacts

For a completed date `YYYY-MM-DD`, the API may expose:

- `rtma_peak/archive/YYYY-MM-DD.tif`: original RTMA peak.
- `observed_peak/archive/YYYY-MM-DD.tif`: original station-observed peak.
- `rtma_peak_rainfall_adjusted/archive/YYYY-MM-DD.tif`: rainfall-adjusted
  RTMA peak, when the fuel model raster and RTMA APCP were usable.
- `station_peak_rainfall_adjusted/archive/YYYY-MM-DD.tif`: station peak after
  transferring the spatial rainfall reduction diagnosed from raw versus
  adjusted RTMA. The station danger remains the source category.
- `verification_combined/archive/YYYY-MM-DD.tif`: rounded pixel mean of the
  aligned rainfall-adjusted station and rainfall-adjusted RTMA categories.
- `verification_combined/archive/YYYY-MM-DD.json`: input coverage, contract
  version, and fallback metadata.

All categorical GeoTIFFs use `0..4` for Low through Extreme and `255` for
NoData. The public verification report includes the adjusted/combined paths
and a `rainfall_suppression` object containing configuration, provenance, and
fallback diagnostics.

## Reruns and troubleshooting

The RTMA peak generation is performed before a verification rerun. Re-run the
date after placing the required fuel model raster/bundle in the configured
path. Check:

```text
GET /verification/report/YYYY-MM-DD
```

Inspect `rainfall_suppression.artifacts.fallback_reason` and
`rainfall_suppression.report_metadata`. Common reasons are:

- `fuel_configured` is false: configure one of the fuel raster settings.
- `station_or_adjusted_rtma_raster_unavailable`: one raw input or adjusted
  RTMA artifact has not been generated.
- `artifact_generation_failed:*`: inspect API logs for raster CRS/shape errors.

Historical raw reports remain readable when new artifacts are absent. The
rainfall contract version is stored in generated report metadata so policy
changes can be distinguished from changes in forecast skill.
