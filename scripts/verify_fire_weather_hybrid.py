"""Verify archived forecast hours against explicitly QC-passed station rows.

CSV columns: station_id, valid_time, lat, lon, qc_pass, plus measured weather /
fuel variables in the hybrid's units. This never feeds later truth back into a
forecast and excludes elapsed analysis steps from forecast skill metrics.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from services import fire_weather_hybrid_store as store
from services.fire_weather_hybrid.verification import continuous


def verify(root, run_id, observations, *, now=None):
    root = Path(root)
    directory = store.resolve_run(root,run_id)
    manifest = store.read_json(directory/"manifest.json")
    frame = pd.read_csv(observations)
    required = {"station_id","valid_time","lat","lon","qc_pass"}
    if not required <= set(frame):
        raise ValueError(f"missing observation columns: {sorted(required-set(frame))}")
    frame = frame[frame.qc_pass.astype(str).str.lower().isin(["true","1","1.0"])].copy()
    frame["valid_time"] = pd.to_datetime(frame.valid_time,utc=True)
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    frame = frame[frame.valid_time <= now].drop_duplicates(["station_id","valid_time"],keep="last")
    times = pd.to_datetime(manifest["valid_times_utc"],utc=True)
    pairs = {}
    bounds = {"temp_c":(-60,60),"rh_pct":(0,100),"wind_speed_ms":(0,70),"fm1_pct":(0,150),
              "fm10_pct":(0,150),"fm100_pct":(0,150),"fm1000_pct":(0,150)}
    stations = set()
    for station,group in frame.groupby("station_id"):
        group = group.sort_values("valid_time")
        last = group.iloc[-1]
        try:
            point = store.point(directory,float(last.lat),float(last.lon))
        except ValueError:
            continue
        for index,valid in enumerate(times):
            if manifest["step_kinds"][index] != "forecast" or valid > now or valid < pd.Timestamp(manifest["issued_at_utc"]):
                continue
            nearest = (group.valid_time-valid).abs().idxmin()
            observed = group.loc[nearest]
            if abs(observed.valid_time-valid) > pd.Timedelta(minutes=30):
                continue
            for name,(low,high) in bounds.items():
                if name not in group or name not in point["series"]:
                    continue
                value,predicted = observed[name],point["series"][name][index]
                if pd.notna(value) and predicted is not None and low <= float(value) <= high:
                    pairs.setdefault(name,[]).append((float(value),float(predicted)))
                    stations.add(station)
    metrics = {name:continuous(*np.asarray(values).T) for name,values in pairs.items()}
    for name in bounds:
        metrics.setdefault(name,{"status":"deferred","samples":0})
    report = {"cycle":manifest["cycle"],"model_version":manifest.get("model_version"),"verified_at":now.isoformat(),
              "station_count":len(stations),"metrics":metrics,"scope":"QC-passed station observations; forecast steps only",
              "independent_spatial_truth":False}
    store.atomic_json(root/"verification"/f"{manifest['cycle']}.json",report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=store.root_dir())
    parser.add_argument("--run",required=True)
    parser.add_argument("--observations",type=Path,required=True)
    args = parser.parse_args()
    print(verify(args.root,args.run,args.observations))

