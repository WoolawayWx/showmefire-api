"""Convert legacy station forecast JSON archives to versioned Parquet objects."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from forecast_v1.artifacts import artifact_record, sha256_file, write_points_parquet
from forecast_v1.contracts import PUBLIC_GRID, utc_rfc3339
from forecast_v1.r2_store import ForecastR2Store
from forecast_v1.repository import ensure_schema, json_text, replace_assets, transaction, upsert_run


def convert(path: Path, output_root: Path, *, relational: bool, r2: ForecastR2Store) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    run_time = datetime.fromisoformat(str(payload["run_date"]).replace("Z", "+00:00")).astimezone(timezone.utc)
    run_id = f"legacy-{run_time:%Y%m%dT%H%MZ}-{sha256_file(path)[:8]}"
    rows = []
    max_lead = 0
    stations = payload.get("stations", {})
    iterable = stations.items() if isinstance(stations, dict) else ((station.get("id"), station) for station in stations)
    for station_id, station in iterable:
        for lead, forecast in enumerate(station.get("forecasts", [])):
            max_lead = max(max_lead, lead)
            rows.append({
                "run_id": run_id, "station_id": station_id, "model": "legacy-public", "member": "deterministic",
                "valid_time_utc": utc_rfc3339(datetime.fromisoformat(forecast["time"].replace("Z", "+00:00"))), "lead_hour": lead,
                "temperature_2m": forecast.get("temp_c"), "relative_humidity_2m": forecast.get("rh"),
                "wind_speed_10m": forecast.get("wind_speed_ms"), "precipitation_increment": forecast.get("precip_interval_mm", forecast.get("precip_mm")),
                "fuel_moisture_p50": forecast.get("fuel_moisture"), "fire_danger": forecast.get("fire_danger"),
            })
    relative = Path("legacy") / f"{run_time:%Y/%m/%d}" / run_id / "points" / "public.parquet"
    target = write_points_parquet(rows, output_root / relative)
    object_key = "forecast-v1/" + relative.as_posix()
    checksum = sha256_file(target)
    if r2.configured:
        r2.upload_immutable(target, object_key, checksum)
    if relational:
        ensure_schema()
        now = utc_rfc3339(datetime.now(timezone.utc))
        asset = artifact_record(target, run_id=run_id, kind="points", variable="legacy-public", object_key=object_key, dtype="Parquet/Zstd")
        with transaction() as connection:
            upsert_run(connection, {
                "run_id": run_id, "cycle_time_utc": utc_rfc3339(run_time), "issued_at_utc": utc_rfc3339(run_time), "completed_at_utc": now,
                "status": "complete", "is_public": 0, "grid_id": PUBLIC_GRID.id, "horizon_hours": max(1, min(240, max_lead)),
                "schema_version": "forecast-v1-legacy-import", "config_version": "legacy", "model_version": "legacy",
                "source_cycles_json": "{}", "manifest_key": None, "manifest_checksum": None,
                "warnings_json": json_text(["legacy_import"]), "superseded_run_id": None, "created_at_utc": now,
            })
            replace_assets(connection, run_id, [asset])
    return {"runId": run_id, "rows": len(rows), "objectKey": object_key, "sha256": checksum}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("data/forecast-v1"))
    parser.add_argument("--relational", action="store_true", help="Create synthetic legacy run rows for verification lookups")
    args = parser.parse_args()
    r2 = ForecastR2Store()
    for path in args.paths:
        print(json.dumps(convert(path, args.output_root, relational=args.relational, r2=r2)))


if __name__ == "__main__":
    main()
