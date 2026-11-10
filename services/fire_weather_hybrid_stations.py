"""Causal, QC-controlled station snapshots and localized fuel anchoring."""
from __future__ import annotations
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import numpy as np
import pyproj
from scipy.spatial import cKDTree
from services.fire_weather_hybrid.contracts import GRID,utc,iso
from services import fire_weather_hybrid_store as store


def fetch_snapshot(directory,*,snapshot=None):
    if snapshot:
        payload=json.loads(Path(snapshot).read_text())
        if "received_at" not in payload:
            raise ValueError("station snapshots require receipt timestamps")
        return payload
    import requests
    # core.config does not export the token; services/synoptic.py reads the environment.
    token=os.getenv("SYNOPTIC_API_TOKEN")
    if not token:
        from dotenv import dotenv_values
        token=dotenv_values(Path(__file__).resolve().parents[1]/".env").get("SYNOPTIC_API_TOKEN")
    if not token: return {"stations":[],"received_at":iso(datetime.now(timezone.utc)),"status":"unavailable"}
    response=requests.get("https://api.synopticdata.com/v2/stations/latest",params={"token":token,"state":"MO","units":"english,fuel_moisture|%",
        "within":"180","status":"active","network":"1,2,156,65"},timeout=30)
    if response.status_code!=200: raise RuntimeError("station observation service unavailable")
    from services.synoptic import flatten_station_data
    stations=[flatten_station_data(station,None) for station in response.json().get("STATION",[])]
    payload={"stations":[station for station in stations if station],"received_at":iso(datetime.now(timezone.utc)),"units":"english"}
    target=Path(directory)/f"stations_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
    store.atomic_json(target,payload)
    return payload


def points(payload,name,valid,as_of,*,maximum_age_hours=3):
    valid,as_of=utc(valid),utc(as_of)
    if utc(payload["received_at"])>as_of: return []
    result=[]
    for station in payload.get("stations",[]):
        if station.get("qc_flagged"): continue
        value=(station.get("observations") or {}).get(name) or {}
        if value.get("value") is None or not value.get("time"): continue
        stamp=utc(value["time"])
        age=(valid-stamp).total_seconds()/3600
        if stamp>as_of or not 0<=age<=maximum_age_hours: continue
        number=float(value["value"])
        if name=="air_temp": number=(number-32)*5/9
        if not np.isfinite(number): continue
        if name=="fuel_moisture" and not 0<=number<=150: continue
        if name=="air_temp" and not -60<=number<=60: continue
        result.append({"station_id":station.get("stid"),"lat":float(station["latitude"]),"lon":float(station["longitude"]),
                       "value":number,"valid_time":stamp,"age_hours":age})
    return result


def interpolate(points,shape,*,maximum_distance_km=150):
    if not points: return np.full(shape,np.nan,"float32"),np.zeros(shape,"float32")
    sx,sy=pyproj.Transformer.from_crs("EPSG:4326",GRID.crs,always_xy=True).transform([p["lon"] for p in points],[p["lat"] for p in points])
    xx,yy=np.meshgrid(GRID.x,GRID.y)
    distance,index=cKDTree(np.column_stack((sx,sy))).query(np.column_stack((xx.ravel(),yy.ravel())),k=min(4,len(points)))
    if distance.ndim==1: distance,index=distance[:,None],index[:,None]
    weight=np.exp(-np.square(distance/60000))
    weight[distance>maximum_distance_km*1000]=0
    values=np.asarray([p["value"] for p in points])[index]
    total=weight.sum(axis=1)
    field=np.divide((values*weight).sum(axis=1),total,out=np.full(len(total),np.nan),where=total>0)
    gain=np.where(total>0,.6*np.exp(-np.square(distance[:,0]/70000)),0)
    return field.reshape(shape).astype("float32"),gain.reshape(shape).astype("float32")
