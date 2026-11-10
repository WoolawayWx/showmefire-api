# Staff-only Missouri fire-weather hybrid

## Current implementation

This is a new experimental family, `fire_weather_hybrid`, separate from the
production fuel-moisture and rule-based danger products. It includes real
XGBoost task training, out-of-fold native-HRRR spatial examples, a residual
U-Net, GPU training, ONNX parity checks, checksum-bound candidate bundles,
1 km grid contracts, private API delivery, and the existing data viewer.

The first local candidate is **diagnostic**, not operationally verified.
There is no automatic public registration, promotion, CDN upload, or scheduler
enablement. No production pointer or legacy moisture formula is modified.

## Model inputs and outputs

The shared contract is `model-training/fire_weather_hybrid/contracts.py`.
Forecast inputs are issuance-time temperature, dewpoint/RH, earth-relative
winds, gusts, precipitation increments, radiation, fuel states, landscape,
lead/time features and optional RAP-derived HDW/Haines. Missing optional values
remain NaN. Station residual tasks correct temperature, RH and speed when observations have recorded units and usable wind-height metadata. All four fuel residual tasks and the red-flag classifier are supported, but are deferred when measured percent-moisture labels or both event classes are missing. The historical archive mixes Celsius/Fahrenheit and metric/knots records; ambiguous values are excluded. This is stricter than the legacy loader.

Two boosted regressors predict potential surface spread (m/min) and intensity
(kW/m). The U-Net corrects their log1p outputs using native-weather and
landscape patches. Reference ranks of spread/intensity feed frozen relative
danger cutpoints (40/70/85/95 percentiles): Low, Moderate, Elevated, Critical,
Extreme. Flame length is derived consistently from intensity.

These behavior targets are Rothermel model outputs, not observed fire growth.
The migrated archive has 18 station locations, inherited physical labels,
assumed receipt timestamps and incomplete fuel classes. Its scale is labeled
`station-diagnostic`; it must not be mistaken for a statewide area/time
climatology. Newly recorded source receipt times and prospective independent
verification are required before any operational skill claim.

## Data and grid

The canonical grid is EPSG:32615, 801 x 792, 1000 m spacing, west 150000,
north 4647000. This exactly subdivides the existing forecast's buffered extent.
Each cell has a stable row/column identifier. Data delivery is clipped using an
explicit Missouri mask. Source resolution is independent of output spacing.

`fire_weather_hybrid_inputs.normalized_hour` accepts the existing framework's
`forecast_v1.adapters.SourceCube` for HRRR/RRFS/GEFS and reprojects one hour at
a time. It requires explicit wind reference metadata, rotates components before
regridding, and uses area-overlap averaging for precipitation. Ensemble means
receive one source weight, rather than one vote per member. This adapter does
not change existing acquisition schedules or blend weights. WeatherNext now has a real authenticated Google Cloud downloader and native-cube adapter. Project `weatherdata-513919` completed a live 48-hour statistics-feed download on 2026-11-06, including temperature/dewpoint, winds, rain and radiation. The 2026-11-06 13Z cycle covers valid hours 14Z November 6 through 13Z November 8. The regional NetCDF is 7,058,075 bytes, checksum 1d981236107bbe150155cf3d42b99b5d1530a8aabc9171c89e08f03bcf138399. Actual fields also passed the native-to-1km adapter across the complete canonical grid. The statistics feed is an ensemble mean, not individual members.

`services/fire_weather_hybrid_acquisition.py` now reuses NOAA acquisition, RTMA capture, MRMS precipitation and the RAP profile/index implementation. It builds staged inputs through persisted official NFDRS4 fuel states, optional station anchoring and the native-source landscape. Source failures, actual receipts, analysis/forecast roles, initialization and readiness are recorded. No missing rain is silently treated as dry.

WeatherNext statistics chunks cover the globe. A small Missouri NetCDF can require gigabytes of source-chunk reads. The downloader reports an uncompressed source-byte estimate separately from output bytes, enforces both budgets, uses bounded concurrency and resumes private hourly caches. Its statistics means are not individual members. Raw ensembles require explicitly configured requester-pays access and a suitable source budget.

Use the dedicated reader environment so the API keeps its pinned Zarr v2 stack:

```powershell
# Run from model-training after installing requirements-weathernext.txt there.
.venv\Scripts\python.exe -m fire_weather_hybrid.weathernext doctor --config ..\api\config\weathernext.json
.venv\Scripts\python.exe -m fire_weather_hybrid.weathernext fetch --config ..\api\config\weathernext.json --hours 48 --output data\fire_weather_hybrid\weathernext-preflight
```

Authenticate using `gcloud auth application-default login --project=weatherdata-513919`. ADC is already configured on this Windows machine; a server needs its own approved workload identity/ADC. Never put credentials into JSON configuration or the repository. `scripts/fetch_weathernext.py` exposes the same doctor/fetch CLI from the API checkout.

## Train and retrain

Run in `model-training`'s dedicated environment, with outputs inside a writable
data root (the current `.env` points at a separate, read-only training archive).

```powershell
.venv\Scripts\python.exe -m fire_weather_hybrid.retrain `
  --data-root D:\_Development\ShowMeFire\training-data `
  --output data\fire_weather_hybrid\candidate-YYYYMMDD `
  --landscape data\fire_weather_hybrid\landscape-1km-v2.nc `
  --spatial-hours 48 --rounds 250 --epochs 30
```

`--panel` accepts a modern Parquet/CSV panel with the exact shared feature
columns, timezone-aware `issued_at`, `valid_time`, `available_at`, and
`target_<task>` labels. `--spatial` accepts checked out-of-fold spatial shards.
Pass `--risk-scale <previous>/risk_scale.json` on subsequent retrains to preserve
tier meaning. `--sync` explicitly invokes the existing archive synchronization
entrypoint; it is off by default. Bulk outputs are ignored by Git.

The trainer splits five-day episodes chronologically, purges forecast overlap,
reserves calibration/test partitions, and writes native JSON models, OOF
Parquet, spatial checkpoint/ONNX, evaluation and risk-scale assets. Without
trained spatial data it writes `awaiting_spatial_data`, not a complete bundle.
Training now uses nested chronological correction stacks and a later spatial validation set with early stopping. Evaluation includes tree-only/hybrid ablation, raw-weather baselines and lead-binned marginal error radii. Prospective coverage and independent fire behavior verification remain open gates. Inspect the metrics: fitting a correction is not evidence that it improves the raw guidance.

## Private serving

Configure:

```text
SMF_FIRE_WEATHER_HYBRID_ENABLED=false
SMF_FIRE_WEATHER_HYBRID_DIR=data/fire-weather-hybrid
SMF_FIRE_WEATHER_HYBRID_BUNDLE=<checksum-verified candidate directory>
SMF_FIRE_WEATHER_HYBRID_ACQUIRE_CONFIG=config/fire-weather-hybrid.local.json
# Or supply SMF_FIRE_WEATHER_HYBRID_INPUT=<canonical staged NetCDF>
SMF_FIRE_WEATHER_HYBRID_MAPS=true
```

Staged input requires canonical x/y axes and grid fingerprint, timezone-aware
`issued_at`, `time` covering the complete Chicago calendar day, Missouri and
burnable masks, weather, all four moisture variables (unknown cells may remain
NaN), and JSON `provenance` with `sources[{name,cycle_time,available_at}]`.
Add `fuel_physics` describing the actual backend; never label a lag approximation
as the full Nelson model. Optional `office`, `wind_20ft_ms` and `gfdi` support
office-specific red flag screening. Analyses are explicitly labeled by step.

After configuring an enabled worker, run:

```text
python scripts/run_fire_weather_hybrid.py --acquire-config config/fire-weather-hybrid.local.json
```

`--days 2` exercises the compatible two-local-day contract. No scheduled task is
installed automatically. The command takes an exclusive worker lock, writes
private progress to `status.json`, and commits the latest pointer only after
complete artifacts. A failed run leaves the last complete run intact. Inspect
any `.building` directory and stale lock before manually removing them.

Staff page: `/admin/fire-weather-hybrid`. All routes, including Zarr chunks,
are under `/api/admin/fire-weather-hybrid` and require admin authentication.
Routes expose status, runs, manifests, points and `files/forecast.nc`,
`files/rasters/*.tif`, `files/images/*.png`. The viewer uses credentials,
disables CDN delivery, and keeps private caches local to the component.

Exports are standard consolidated Zarr v2/zlib, compressed NetCDF, COGs and
PNG maps using the mesonet basemap/branding and SEMA region boundaries.
Category overviews use nearest-neighbor treatment. Steps are local-day indexes;
actual forecast leads and analysis/forecast roles are separate metadata.

## Fuel initialization and deployment

Build the official pinned NFDRS binding using `model-training/fire_weather_hybrid/build_nfdrs.py --source <NFDRS4-checkout> --output <private-native-directory>`. The source checkout must be at commit `84223b95fd395a62ea344257f80a9949cee71e6e`. Install `pybind11`/`setuptools` in the build environment and preserve upstream source/license notices. Rebuild on the worker's Python ABI and operating system: the Windows cp313 library cannot be copied into the Linux Python 3.11 API container.

The built Windows library is under `model-training/data/fire_weather_hybrid/native`. The native-source landscape is `landscape-1km-v2.nc` in the same data root. The v2 file converts LANDFIRE CH from meters-times-ten to meters; the legacy capture manifest incorrectly labels the native CH values as meters. Configure these real paths in a copy of `config/fire-weather-hybrid.example.json`, along with the dedicated WeatherNext Python and private cache/state directories.

The worker fails closed without a causal fuel checkpoint. An explicitly requested `--allow-cold-start` replay initializes each class and records convergence flags; a one-week replay does not establish a conditioned 1000-h state. The conservative cold-start readiness threshold is five time constants. Long history replay takes substantially longer than a normal tick and saves analyzed checkpoints for restart. Forecast states never advance the observed checkpoint pointer. The worker logs to private rotating `logs/pipeline.log` and updates private status JSON; its pipeline lock spans acquisition and output publication.

Full-Missouri native-engine benchmark on this machine: 179,281 active cells, about 23 seconds per simulated hour and 1.65 GB peak resident memory, using uniform synthetic forcing. This is a runtime check, not meteorological validation; it does not establish the full worker's 15-minute target including network, inference and exports.

## Remaining scientific and operational gates

- Condition and independently verify long-lag fuels; audit station sensor heights, QC, units and actual receipt times.
- Obtain verified measured fuel-percent labels and positive qualifying-event labels. Missing tasks remain absent rather than using ambiguous `gm` values or fabricating classifier examples.
- Require weather corrections to beat raw-guidance baselines. Current diagnostic candidates can fail this gate and must remain staff research products.
- Expand the station-diagnostic risk reference into a statewide area/time reference; run geographic holdouts, independent behavior verification and prospective shadow verification for at least 30 days.
- Learn/evaluate source weights when aligned multi-source archives exist. Current weights are configured. Marginal station error radii do not establish calibrated joint trajectory uncertainty.
- Benchmark the entire real-data worker and confirm Linux deployment, complete initial history and all acquisition credentials before scheduling it.

The engineering foundation is implemented, but these gates are material. The candidate is experimental, public_eligible=false, and is not the completed operational forecast.

## Validation

Training tests: `python -m pytest tests/test_fire_weather_hybrid.py -q`.
API tests: the same command in `api`. On restricted Windows environments,
FastAPI TestClient needs permission for its local event-loop sockets.
API tests verify private asset access, atomic publication, format agreement,
point values and the byte-identical runtime mirror. Runtime modules are
exported with `python -m fire_weather_hybrid.export_runtime --api-root ../api`.

Station verification can be run with
`python scripts/verify_fire_weather_hybrid.py --run <id> --observations <qc.csv>`.
The CSV requires station_id, valid_time, lat, lon, qc_pass and measured variables
in contract units. The verifier matches within 30 minutes, excludes analysis
steps and future/unverified hours, and marks missing fuel classes deferred.
Reports are private at `/api/admin/fire-weather-hybrid/verification`.

Earlier diagnostic experiment (2026-11-06): 68,244 archived station/forecast rows;
five boosted tasks; 128 sparse-supervision patches from eight native HRRR hours;
three CUDA training epochs. The complete CPU prediction path passes its smoke
check. This short run validates execution, not model convergence or superiority.
Candidate: `model-training/data/fire_weather_hybrid/candidate-20261106`.

The newer units-checked experiment is `model-training/data/fire_weather_hybrid/candidate-units-verified-v2-20261106`. It trained 16 CUDA epochs, retaining epoch 11, using 382 training and 130 validation patches from 32 native HRRR hours. Sparse held-out log1p SmoothL1 loss improved from 0.6023 (tree-only) to 0.07895 (hybrid); this compares physical-reference targets, not observed fire growth. ONNX maximum absolute parity error was 9.54e-7. Only 1,133 rows supported the complete causal correction stack. Temperature and RH failed the raw-weather calibration baseline gate; wind improved. Fuel-residual models and the classifier were deferred for missing trustworthy labels. `readiness.json` records these limitations beside the checked bundle.

Final focused checks: 21 model tests, 25 private API/acquisition/forecast regression tests, 32 existing fire-weather indices/history tests, and 21 website tests passed. A separate bounded 48-hour test combined the actual native fuel engine, trained trees, CPU ONNX, private exports and point lookup using synthetic weather on a real landscape patch. It is an engineering integration check. Statewide and nine regional map layouts were rendered and visually inspected using a labeled layout fixture, including geographic conversion of the SEMA UTM boundaries.

The detailed model rationale and input/output plan is `model-training/docs/fire_weather_hybrid_plan.md`. A machine-specific, Git-ignored `config/fire-weather-hybrid.local.json` references the verified Windows reader, native library, landscape and cache. It does not enable a worker or a scheduler. Linux needs rebuilt native bindings and its own approved credentials.
