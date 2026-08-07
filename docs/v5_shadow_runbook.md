# V5 offline beta and adaptive shadow

V5 is not a replacement for the stable `fuel_moisture` model. Its registry type is
`fuel_moisture_station_summer_guarded`, and an offline beta remains
`production_eligible: false` until prospective evidence and three canaries pass.

## Offline evidence (training workstation)

```powershell
python spatial/evaluate_v5_offline_v2.py
python spatial/register_v5_beta.py
```

The evaluator refuses to overwrite its paired rows or report. Archive both files
with the candidate bundle. Registration validates their checksums and never edits
the stable API model pointer.

## Install and enable shadow (API server)

```bash
python scripts/install_v5_shadow_bundle.py /path/to/v5-bundle.tar.gz --sha256 ARCHIVE_SHA256
```

Set these values on the backend container and restart it:

```text
SMF_V5_SHADOW_BUNDLE=/app/data/model-bundles/v5/ARCHIVE_SHA256
SMF_V5_EVIDENCE_ROOT=/app/data/model-shadow/v5
V5_SHADOW_ENABLED=true
V5_SHADOW_MAX_FAILURES=3
```

Run `python scripts/predeploy_check.py` before enabling. The normal HRRR job writes
immutable `*.prediction.json` files. The scheduler attaches same-station FM, RH,
and wind observations to separate `*.observation.json` files every three hours.
Three consecutive V5 failures disable only V5; stable forecast generation remains
authoritative.

## Predeclared checkpoints

Run exactly one day-14 evaluation. If it fails or lacks support, continue to day 30.

```powershell
python spatial/evaluate_v5_shadow.py --evidence-root PATH --checkpoint day14
python spatial/evaluate_v5_shadow.py --evidence-root PATH --checkpoint day30
```

Each checkpoint report is immutable. Day 14 requires 95% paired-bootstrap
probability; day 30 requires 90%. Both retain the 1% non-inferiority and category
safety limits.

After a passing prospective checkpoint, record three distinct non-public canaries:

```bash
python scripts/v5_canary.py RUN_ID
python scripts/authorize_v5_promotion.py REPORT.json --bundle "$SMF_V5_SHADOW_BUNDLE"
```

Authorization only writes `promotion-authorization.json`; it does not switch public
serving. A future explicit activation must preserve stable `1.0.0` as the rollback
target and begin seven-day monitoring.
