# Plan: a finer ground-cover and fuel product (`ground_cover_static`)

Written 2026-11-13. This is a plan for the desktop (where there is storage and compute for the 30 m rasters). Most of the
work happens in `ShowMeFire-Models/`; a smaller part lands in `api/` once the product is published. Nothing in this plan
has been built yet. Every fact below marked "checked" was read from the code or measured on live data on that date.

## 1. Why we are doing this

What we found, in order:

1. **The fuel data we serve is too coarse for an incident.** The production `fire_behavior_static` bundle (version 2026.4)
   is a 256 x 256 grid at about 3.1 km x 3.2 km (checked on the server: x spacing 3117 m, y spacing -3187 m). It was built to
   match the HRRR grid for the spread-rate maps. One cell is about 10 km^2, the same as one NGFS detection pixel
   (measured: median 2.88 x 2.98 km, 8-10 km^2, over 500 live footprints). So an incident's "fuel mix" is one value, and the
   panel read "100% grass" on every one of 25 incidents checked.
2. **The only fine land-cover data we have comes from NOAA's NGFS feed** (`land_cover` string such as
   `Trees:91,Developed:4,Grass/Herbs:4,Cropland:1`). Of 500 live incidents, 295 (all GOES-19) have it; the other 205 (VIIRS, MODIS
   and a few unknown) have none.
3. **The live confidence score depends on that string.** `api/services/incident_confidence.py` (about lines 184-190) takes
   cropland and water fractions from `land_cover` and applies `CROPLAND_PENALTY` / `WATER_PENALTY`. With no string the
   fractions are 0.0, so those penalties never apply to roughly 41% of incidents.
4. **The shadow detection model uses fuel and canopy features that production never fills.** `api/services/detection_shadow.py`
   reads `fire_events.fuel_model_fbfm40` and `canopy_cover_pct`; they are NULL in production because
   `api/services/static_context.py` needs raw rasters (`data/static/source/fbfm40.tif`, `canopy_cover.tif`) that the server does
   not have (project rule: production never downloads or builds geography).

Already shipped to make the display honest (no data change): the incident panel and app show NOAA's land-cover mix as
"Ground cover" and show the modeled fuel as a single line with a "coarse grid" note unless the API reports
`land_cover_context.fuel_cells >= 5` (then the bar appears automatically). This plan produces the data that makes that bar real.

## 2. Goal and non-goals

**Goal.** A versioned, checksummed static product at roughly 480 m that stores, for each cell, the share of each land-cover
class (NLCD) and each fuel group (LANDFIRE FBFM40) plus canopy. The API reads it to give every incident (any source) a real
ground-cover and fuel mix, sampled over the detection footprint instead of one point or a fixed circle.

**Non-goals for v1.**
- No change to the live confidence score until the shadow comparison (section 8) has been reviewed.
- No writing fuel or canopy into `fire_events` (it would shift the shadow model's inputs away from what it trained on).
- No retraining. (Section 9 lists what retraining would need, if you decide you want fuel as a model feature.)
- No new raw-data downloads on the production server.

## 3. Decisions to confirm before building (defaults in bold)

| # | Decision | Options | Why it matters |
|---|---|---|---|
| 1 | Cell size | **480 m (16 x 16 of 30 m pixels)**, 960 m, 240 m | 480 m is about 6 x 6 cells under one NGFS pixel and 1-3 cells for a VIIRS pixel. 240 m quadruples the size for little gain. Must be a multiple of 30 so blocks nest exactly. |
| 2 | Domain | **Missouri plus the one-degree buffer used elsewhere: west -96.8, south 34.8, east -88.1, north 41.8** | Same domain as `static_features/build_fire_behavior_bundle.py` (`BBOX`). Keeps border incidents covered. |
| 3 | Grid CRS | **EPSG:5070 (NAD83 / Conus Albers)** | NLCD and LANDFIRE are natively 5070 at 30 m, so aggregation is exact with no resampling. |
| 4 | NLCD year | **Latest complete Annual NLCD** (as in `spatial_fuel_moisture_runbook.md` section 6) | Record the release in the manifest. |
| 5 | LANDFIRE release | **Same release as the current `fire_behavior_static` bundle** if still available, otherwise the newest (record it) | Keeps fuel codes consistent with the spread-rate maps. |
| 6 | Hazard class | **Skip in v1** | A per-fuel "spread potential" class is useful for a "most hazardous fuel present" line, but it needs a defined lookup. Decide after v1 (section 10). |
| 7 | Also ship a 30 m COG? | **No** | Exact footprint sampling at 30 m is possible later but adds large files and reader code. |

## 4. What already exists (reuse it)

All paths are relative to the repo named.

**ShowMeFire-Models/**
- `static_features/download_fire_behavior_assets.py`: downloads LANDFIRE FBFM40, canopy cover, canopy height (LFPS) and the 3DEP
  DEM into `$SMF_DATA_ROOT/static/source/` and writes `source_manifest.json`. Needs `--email` and an explicit
  `--landfire-release`. Default domain is the buffered Missouri box; `--output-resolution` is optional (leave it out to keep 30 m).
- `static_features/download_sources.py`: acquires the six rasters used by the spatial model, including `nlcd-class` (Annual NLCD)
  and `fbfm40`, from a local file or official URL. See `docs/spatial_fuel_moisture_runbook.md` section 6. If you already ran
  this for the spatial model, the NLCD and FBFM40 rasters may already be on the desktop: check
  `$SMF_DATA_ROOT/static/source/` before downloading anything.
- `static_features/build_fire_behavior_bundle.py` and `fire_behavior_schema.py`: the pattern to copy (immutable versioned
  NetCDF, JSON manifest with sha256, grid fingerprint, validation gates `crs_validated`, `units_validated`, `nodata_validated`).
- `pipelines/register_fire_behavior_static.py`, `pipelines/publish_release.py`, `models/model_types.py`: registry and release
  plumbing. The registry in `ShowMeFire-Models` and in `api/` are independent copies; neither shares files with the other.
- `docs/observed_spread_rate_runbook.md` sections 5-7: the register, publish, import and promote commands for an asset-only
  model type. This plan follows the same flow with a new type name.

**api/**
- `models/versioning.py` (type-specific beta checks around line 257), `pipelines/import_model.py` (known types near line 40,
  asset verification near line 78), `pipelines/promote_model.py` (the `--model` choices near line 27).
- `services/incident_fuel.py`: today's reader of the coarse 3 km bundle; it is where the finer product plugs in.
- `services/incident_confidence.py`, `services/detection_shadow.py`: the consumers described in section 1.
- Libraries already in `requirements.txt`: numpy, xarray, netCDF4, rasterio, pyproj, shapely, scipy. The reader needs no new dependency.

## 5. Product specification: `ground_cover_static` v1

One NetCDF4 file, `ground_cover_static_<version>.nc`, plus `ground_cover_static_<version>.json` manifest.

**Grid.** Regular grid in EPSG:5070, cell size 480 m, origin snapped to a multiple of 480 m so every 16 x 16 block of 30 m
pixels nests exactly. Coordinates `x`, `y` (cell centres, metres). 2-D `latitude` and `longitude` are optional; the reader will
project instead. Expected size roughly 1,800 x 1,800 cells (about 3 million cells); compressed with zlib and shuffle it should be
tens of MB, not hundreds. If it comes out over about 100 MB, stop and reconsider the cell size.

**Variables** (dimensions `y`, `x`; stored as integers, percent of the cell's valid 30 m pixels unless noted):

NLCD land cover (shares sum to 100, plus `other`):
`nlcd_water_pct` (11), `nlcd_developed_pct` (21-24), `nlcd_barren_pct` (31), `nlcd_forest_pct` (41-43),
`nlcd_shrub_pct` (52), `nlcd_grass_pct` (71), `nlcd_pasture_pct` (81), `nlcd_crops_pct` (82),
`nlcd_wetlands_pct` (90, 95), `nlcd_other_pct` (everything else, including 12, 51, 73, 74).

LANDFIRE FBFM40 fuel groups (Scott and Burgan families; shares sum to 100):
the five non-burnable classes kept separate, `fbfm_nb_developed_pct` (91), `fbfm_nb_snow_pct` (92),
`fbfm_nb_cropland_pct` (93), `fbfm_nb_water_pct` (98), `fbfm_nb_bare_pct` (99), then `fbfm_grass_pct` (101-109),
`fbfm_grass_shrub_pct` (121-124), `fbfm_shrub_pct` (141-149), `fbfm_timber_understory_pct` (161-165),
`fbfm_timber_litter_pct` (181-189), `fbfm_slash_pct` (201-204), and `fbfm_other_pct` (anything else, including nodata inside
the domain).
`fbfm40_mode` (int16): the most common FBFM40 code in the cell.

Continuous: `canopy_cover_mean_pct` (uint8), `canopy_height_mean_m` (float32 or scaled uint8), `valid_pct` (share of the
cell's 30 m pixels that had valid data in all inputs).

The group names must match the labels the API already uses in `api/services/incident_fuel.py` (`_RANGES`): Developed, Snow and
ice, Cropland, Water, Bare ground, Grass, Grass and shrub, Shrub, Timber with understory, Timber litter, Slash and blowdown.
That is why the non-burnable classes are stored one by one.

**Manifest** (JSON) must include: `schema_version`, `bundle_version`, `sha256`, `grid_fingerprint` (hash of CRS, origin, cell size,
shape), `crs_wkt`, `cell_size_m`, `bbox`, `source_manifest` (file names, sizes, checksums, release names and years for NLCD and
LANDFIRE), `validation` with `crs_validated`, `units_validated`, `nodata_validated` all true, and `synthetic: false`.

## 6. Desktop procedure

All commands from `ShowMeFire-Models/` with `SMF_DATA_ROOT` set, as in `observed_spread_rate_runbook.md` section 1.

### Step 1. Gather the source rasters

1. Check `$SMF_DATA_ROOT/static/source/` for `fbfm40.tif`, `canopy_cover.tif`, `canopy_height.tif` and an NLCD class raster.
2. Download whatever is missing:
   ```powershell
   python static_features/download_fire_behavior_assets.py --email "<your email>" --landfire-release LF2024
   ```
   For NLCD, follow `docs/spatial_fuel_moisture_runbook.md` section 6 (`static_features/download_sources.py --nlcd-class-file ...`).
   LFPS jobs for the full 30 m domain can take more than 15 minutes per the existing runbook.
3. Confirm every raster is single-band, 30 m, and either already EPSG:5070 or that you are happy to warp it (the builder will
   warp with nearest-neighbour onto the master grid). Record release names and years.

### Step 2. Write the builder

New file `static_features/build_ground_cover_bundle.py`, modelled on `build_fire_behavior_bundle.py`.

Algorithm (memory-bounded; the 30 m inputs are about 27,000 x 27,000 pixels per layer, so never load a whole layer):
1. Define the 480 m output grid over the 5070 extent of `BBOX`, snapped to 480 m. The matching 30 m master grid is the same
   extent at 30 m.
2. For each source, open a `rasterio.vrt.WarpedVRT` onto the master 30 m grid (crs EPSG:5070, nearest for categorical layers,
   average for canopy cover and height). Sources that are already on the master grid read straight through.
3. Process in tiles of 2,048 x 2,048 input pixels (128 x 128 output cells), reading a tile from each source:
   - Map codes to group ids with a lookup array (one `LUT[code]` per layer) so grouping is a single vectorised step.
   - Reshape each tile to `(128, 16, 128, 16)`; for each group k, `(ids == k).sum(axis=(1, 3))` gives that group's pixel count per
     cell. Convert to percent of valid pixels, rounded, then fix rounding so each set sums to exactly 100 (largest-remainder).
   - `fbfm40_mode`: for each FBFM40 code present in the tile (about 40), count per cell and keep the argmax.
   - Canopy means over valid pixels only; `valid_pct` from the validity masks.
4. Write the NetCDF with compression, set attributes, compute `grid_fingerprint` and the file `sha256`, write the manifest, and
   run `validate_bundle`-style checks (below) before declaring success.
5. Bundles are immutable. Any change to inputs or builder logic means a new `bundle_version`.

Expect the run to be I/O bound. A single pass over three 30 m layers should take tens of minutes on a desktop, not hours; this
is an estimate, not a measurement. Keep tile size and worker count as flags so you can tune them.

### Step 3. Validate before registering

The builder must refuse to write a manifest unless all of these pass.

1. **Structure:** CRS is EPSG:5070; cell size 480 m; origin is a multiple of 480 m; no all-nodata rows or columns inside the domain.
2. **Sums:** for every cell with `valid_pct >= 50`, the NLCD shares sum to 100 and the FBFM shares sum to 100 (tolerance 1).
3. **Spot checks** (hard-coded lat/lon with expected values, written as a test): a cell in Mark Twain National Forest has
   `nlcd_forest_pct >= 70`; downtown Kansas City has `nlcd_developed_pct >= 60`; a cell on Lake of the Ozarks has
   `nlcd_water_pct >= 50`; a Bootheel cell has `nlcd_crops_pct >= 50`. Pick the exact coordinates when you build and record them in
   the test. Add one fuel check (a known grassland or cedar-glade area) once you have looked at the data.
4. **Agreement with NOAA's own numbers** (this is the most useful gate):
   - Fetch `https://api.showmefire.org/api/fires/incidents.geojson?limit=500` and keep the GOES-19 incidents that have a
     `LAND_COVER` string; fetch the matching pixel polygons from
     `https://api.showmefire.org/fires/detections-footprints.geojson?limit=500`.
   - For each, average our shares over the pixel polygon and compare with NOAA's shares class by class.
   - NOAA classes are Trees, Grass/Herbs, Shrubs, Cropland, Developed, Water, Barren. Do not assume which NLCD classes make up
     "Cropland" or "Grass/Herbs" (pasture and hay may sit in either). Fit non-negative least squares from our NLCD shares to each
     NOAA class and report the weights; that result defines the mapping the API will use.
   - Report mean and median absolute error per class and a scatter plot. Target: median absolute error at most about 10-15
     percentage points for each major class. If it is far worse, investigate (CRS offset, NLCD year, pixel parallax) before
     going further; do not tune the mapping to hide a real problem.
5. **Unit tests** in `tests/test_ground_cover_bundle.py` using tiny synthetic 30 m rasters (for example 32 x 32 pixels with known
   class blocks) that check exact aggregation, the sum-to-100 fix-up, the mode, and nodata handling.

### Step 4. Register, publish, import, promote

This mirrors `observed_spread_rate_runbook.md` sections 5-7 with a new model type. First add the type in both repos (they are
independent copies):

**ShowMeFire-Models**
- `models/model_types.py`: add `"ground_cover_static"` to `KNOWN_MODEL_TYPES`.
- `pipelines/promote_model.py`: add it to the `--model` choices.
- New `pipelines/register_ground_cover_static.py`: copy of `register_fire_behavior_static.py` with the new type name; it
  registers `static_bundle` and `static_manifest` as assets with `shadow_required: False`.
- `pipelines/publish_release.py`: add the hint message for the new type next to the `fire_behavior_static` one.

**api**
- `models/versioning.py`: add the same asset checks as `fire_behavior_static` (needs `static_bundle` and `static_manifest`).
- `pipelines/import_model.py`: add the type to the known set and a `_verify_ground_cover_assets` that checks checksums, rejects
  `synthetic`, requires the three validation gates, checks the expected variables and grid fingerprint, and checks the grid is
  EPSG:5070 at 480 m.
- `pipelines/promote_model.py`: add the type to the `--model` choices.
- Add tests next to the existing import and promote tests (`tests/test_import_model.py`, `tests/test_promote_model.py`).

Then on the desktop:
```bash
python pipelines/register_ground_cover_static.py \
  --bundle "$SMF_DATA_ROOT/static/bundles/ground_cover_static_${GC_BUNDLE_VERSION}.nc"
python pipelines/publish_release.py --model ground_cover_static --version 0.0.1-beta.1 --repo "$SMF_GITHUB_REPO"
```
On the API server, from the `api/` directory (these are the same commands as section 7 of `observed_spread_rate_runbook.md`;
`api/docs/beta_operations_runbook.md` covers the server-side beta workflow):
```bash
python pipelines/import_model.py --model ground_cover_static --tag ground_cover_static-v0.0.1-beta.1 --repo "$SMF_GITHUB_REPO"
python pipelines/promote_model.py --model ground_cover_static
```
Never copy or swap the files by hand; promotion switches the registry entry atomically. Keep the GitHub release as a prerelease
until the server import and the API checks in section 7 succeed.

## 7. API work (after the product is promoted)

Do this in `api/` after the bundle is live, so the code can be tested against the real product.

1. **New `services/ground_cover.py`.** Load the active `ground_cover_static` assets through
   `models.versioning.load_active_assets("ground_cover_static", "stable")`, with the same checksum and mtime caching approach as
   `services/fire_behavior_static.py`. Project lat/lon to EPSG:5070 with `pyproj.Transformer` (create it once) and index by
   `floor((x - x0) / 480)`.
2. **Sample the right area for each incident:**
   - NGFS incidents: the union of the members' `footprint_geojson` pixel polygons (already stored per event).
   - Incidents with an outline: `shape_geojson`.
   - Point-only sources (VIIRS, MODIS, reports, dispatch): the 3 x 3 block of cells around the point (about 1.4 km square).
   Weight cells by how much of each is inside the polygon. Return the shares, the dominant FBFM40 code, mean canopy, `cells`
   (the number of cells actually used) and `resolution_km: 0.48`.
3. **Plug it into `services/incident_fuel.py`.** Prefer the new product; fall back to the 3 km bundle; fall back to `None`.
   Fill `land_cover_context.fuel_groups` from the FBFM shares (same group names as today), and add `ground_cover` (NLCD shares)
   plus `fuel_cells`. The website panel and the app already show a real fuel bar when `fuel_cells >= 5` and the "ground cover" bar
   from `land_cover`; add the NLCD-based shares as a fallback for incidents with no NOAA string.
4. **Tests:** a synthetic 480 m bundle in a temp file (like `tests/test_incident_fuel.py`), covering polygon sampling, point
   sampling, an incident near the domain edge, and the fallback chain.
5. **Performance budget:** a cached lookup under 20 ms and a cold one under 100 ms for a footprint union of up to about 10
   cells. The file is read once and kept in memory (a few tens of MB).

## 8. Shadow scoring: fix the missing land-cover input without changing live scores

Goal: see what the confidence score would be if every incident had cropland and water fractions, before anyone relies on it.

1. In `api/`, add a scheduled job (API group; remember every new `add_job` id must be listed in `core/job_groups.py` and a test
   checks it) that runs every 30 minutes over active incidents that have no NOAA `land_cover` string.
2. For each, build a stand-in `land_cover` string from our NLCD shares (class mapping from the validation in step 3.4) and call
   the existing scorer on a copy of the members. Store, in a new table, the incident id, time, live score and label, shadow score
   and label, and the shares used. Never write to the live score.
3. After at least two weeks (and at least about 100 incidents), review: the distribution of score changes, how many labels
   would flip (for example moderate to low), and the 20 largest changes by hand against satellite imagery and what staff decided.
4. Only if that review looks sensible: add an environment flag (for example `SMF_GROUND_COVER_SCORING=1`) that makes the live
   scorer use the stand-in string for incidents with no NOAA string. Keep the flag off by default and document it in `CLAUDE.md`.
5. Note the open question the shadow run answers: the cropland penalty was designed around NGFS's coarse pixels. Whether the
   same penalty is right for 375 m VIIRS pixels is not known from the code, which is why this is a shadow comparison first.

**Status (2026-11-13): steps 1-3 are built, step 4 is deliberately not.**

- Job `ground_cover_shadow` (api group, every 30 minutes, listed in `core/job_groups.py`) runs
  `services/ground_cover_shadow.py`. It scores incidents active in the last 24 h that have no NOAA `land_cover` string,
  once as stored and once with a stand-in string from our NLCD mix, and appends both to `incident_confidence_shadow`
  (a row is only added when the live score, shadow score or stand-in changes). It reads the live scorer's code but never
  writes a live score, and it does nothing when the ground cover product is not loaded.
- Both scores come from the v2 scorer so the only difference is land cover; `live_is_v2` records whether v2 is what the
  site shows. If v1 is live, "live" here is what v2 would say, not what users see.
- Switch off with `SMF_GROUND_COVER_SHADOW=0`. Roll back by removing the job or leaving it off; nothing else changes.
- Review: `python scripts/review_ground_cover_shadow.py` (after at least 14 days and about 100 incidents; the report says
  whether it is ready). It lists the score-change distribution, label flips (for example moderate to low) and the 20
  largest changes to check by hand against satellite imagery and what staff decided.
- Step 4 (an `SMF_GROUND_COVER_SCORING=1` flag that feeds the stand-in string to the live scorer) is NOT implemented. Add
  it only after that review, off by default. There is no `api/CLAUDE.md`, so document the flag here when it exists.

## 9. If you later want fuel as a model feature (not part of v1)

The shadow detection model reads `fuel_model_fbfm40` and `canopy_cover_pct` (`api/detection-confidence-model/features.py`). They
are missing in production today, so a model trained on production-like data effectively ignores them. To use them properly:
backfill them for historical training rows offline (the product is static, so any event's lat/lon gives the same answer) in
`api/scripts/export_detection_training_data.py`, retrain and validate in `ShowMeFire-Models`, publish through the usual
release path, and only then start filling the columns at ingest. Do not fill the columns at ingest first.

## 10. Possible follow-ups

- A "most hazardous fuel present" line (decision 6): define a lookup from FBFM40 code to a 0-4 spread-potential class (for
  example from fixed reference Rothermel runs), store the share of each cell at class 3 or higher, and show it in the panel.
- A 30 m COG for exact footprint sampling (decision 7).
- Use the same product for the fire-weather and spread-rate work at the incident scale.

## 11. Rollback

- The API keeps the 3 km reader as a fallback, so removing or rolling back the new product restores today's behaviour
  (honest "coarse grid" display, no fuel bar).
- Promotion is atomic at the registry-entry level; to roll back, promote the previous version or remove the stable entry.
- The shadow job only writes to its own table; disabling the job or leaving the scoring flag off changes no live score.

## 12. Checklist

- [ ] Confirm the decisions in section 3.
- [ ] Source rasters present, releases recorded.
- [ ] `build_ground_cover_bundle.py` written with unit tests passing.
- [ ] Bundle built; all validation gates in section 6 step 3 pass, including the NOAA agreement report (save it with the release notes).
- [ ] New model type added in both repos, with tests.
- [ ] Registered, published as a prerelease, imported and promoted on the server.
- [ ] `services/ground_cover.py` and the `incident_fuel.py` change merged, with tests; the panel and app show a fuel bar.
- [x] Shadow job written and tested (deploy it, then let it run).
- [ ] Review done after two weeks; decision recorded on the live-score flag.

## 13. Prompt to paste into a Claude Code session on the desktop

> Read `api/docs/ground_cover_product_plan.md`, `ShowMeFire-Models/docs/observed_spread_rate_runbook.md` and
> `ShowMeFire-Models/static_features/build_fire_behavior_bundle.py`. Then implement section 6 steps 2 and 3 of the plan:
> `static_features/build_ground_cover_bundle.py` with `tests/test_ground_cover_bundle.py`, using my existing source rasters in
> `$SMF_DATA_ROOT/static/source/` (check what is there first). Use the defaults in section 3 unless I say otherwise. Do not run
> any download until you have told me what it will fetch and its size. Do not touch the `api/` repo yet.
