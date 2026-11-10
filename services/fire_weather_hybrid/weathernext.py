"""Bounded WeatherNext 3 acquisition, usable in a separate Zarr-v3 environment.

Reference: https://developers.google.com/weathernext/guides/gcs
Credentials use Google Application Default Credentials, never JSON config tokens.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import xarray as xr

from .contracts import utc, iso

ROOTS = {
    "statistics": "gs://weathernext3_statistics_spatial/weathernext_3_0_0_statistics/zarr",
    "ensemble": "gs://weathernext3_spatial/weathernext_3_0_0/zarr",
}
# Names and resolutions are explicit: surface statistics are not ensemble members.
VARIABLES = {
    "temperature_2m": ("temperature_2m", "degC", "temperature"),
    "dewpoint_temperature_2m": ("dewpoint_2m", "degC", "temperature"),
    "u_component_of_wind_10m": ("wind_u_10m", "m s-1", "wind"),
    "v_component_of_wind_10m": ("wind_v_10m", "m s-1", "wind"),
    "total_precipitation_1hr": ("precipitation_increment", "mm", "precip"),
    "surface_solar_radiation_downwards_1hr": ("shortwave_down", "W m-2", "solar"),
}


@dataclass(frozen=True)
class Config:
    mode: str = "statistics"
    project: str | None = None
    root: str | None = None
    store: str | None = None
    latitude: str = "lat_0p1"
    longitude: str = "lon_0p1"
    maximum_age_hours: int = 24
    maximum_download_bytes: int = 512*1024**2
    maximum_source_bytes: int = 8*1024**3
    download_concurrency: int = 3
    # west,east,south,north; one-degree Missouri research buffer plus grid edges.
    bbox: tuple = (-97.5, -87.5, 34.5, 42.)

    def __post_init__(self):
        if self.mode not in ROOTS:
            raise ValueError("WeatherNext mode must be statistics or ensemble")
        if self.mode == "ensemble" and not self.project:
            raise ValueError("raw WeatherNext ensembles require an explicit requester-pays project")
        if not (1 <= self.maximum_age_hours <= 48) or self.maximum_download_bytes <= 0:
            raise ValueError("invalid WeatherNext age or download budget")
        if self.maximum_source_bytes<=0 or not 1<=self.download_concurrency<=4:
            raise ValueError("invalid WeatherNext source budget/concurrency")
        for path in (self.root,self.store):
            if path and (not path.startswith("gs://") or "?" in path or "@" in path):
                raise ValueError("WeatherNext paths must be plain gs:// URIs without credentials")

    def apply_credentials(self):
        """Use GOOGLE_APPLICATION_CREDENTIALS from the process environment or the API's .env.

        Both google.auth and obstore honor that variable. Only the key file's path
        is handled here, never its contents. If unset, standard ADC applies.
        """
        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            return
        for env_file in (Path.cwd()/".env", Path(__file__).resolve().parents[2]/".env"):
            if not env_file.is_file():
                continue
            for line in env_file.read_text(encoding="utf-8").splitlines():
                key, separator, value = line.strip().partition("=")
                if separator and key.strip() == "GOOGLE_APPLICATION_CREDENTIALS":
                    value = value.strip().strip("'\"")
                    if not value:
                        continue
                    path = Path(value).expanduser()
                    if not path.is_absolute():
                        path = env_file.parent/path
                    if not path.is_file():
                        raise FileNotFoundError(f"GOOGLE_APPLICATION_CREDENTIALS in {env_file.name} not found: {path}")
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(path.resolve())
                    return

    @classmethod
    def read(cls,path=None):
        data = json.loads(Path(path).read_text()) if path else {}
        unknown = set(data)-set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown WeatherNext configuration fields: {sorted(unknown)}")
        return cls(**data)


def _atomic(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_name(path.name+".tmp")
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temp.replace(path)


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024**2),b""):
            digest.update(block)
    return digest.hexdigest()


def doctor(config: Config):
    config.apply_credentials()
    missing=[]
    for name in ("zarr","obstore","google.auth"):
        try:found=importlib.util.find_spec(name)
        except ModuleNotFoundError:found=None
        if found is None:missing.append(name)
    result = {"status":"ready","mode":config.mode,"project":config.project,"missing_dependencies":missing,
              "requester_pays":config.mode == "ensemble"}
    if missing:
        return {**result,"status":"missing_dependencies","action":"Install requirements-weathernext.txt in the reader environment."}
    from importlib.metadata import version
    if int(version("zarr").split(".")[0])!=3:
        return {**result,"status":"incompatible_zarr","action":"Use the dedicated WeatherNext/Zarr-v3 reader environment; keep the API's v2 environment separate."}
    import google.auth
    from google.auth.exceptions import DefaultCredentialsError
    try:
        _,project = google.auth.default(scopes=["https://www.googleapis.com/auth/devstorage.read_only"],quota_project_id=config.project)
        result["credential_project"] = project
    except DefaultCredentialsError:
        return {**result,"status":"credentials_missing","action":"Run gcloud auth application-default login using the WeatherNext-approved account, or configure GOOGLE_APPLICATION_CREDENTIALS/workload identity on the worker."}
    return result


def _session(config):
    import google.auth
    from google.auth.transport.requests import AuthorizedSession
    credentials,_ = google.auth.default(scopes=["https://www.googleapis.com/auth/devstorage.read_only"],quota_project_id=config.project)
    return AuthorizedSession(credentials)


def discover(config: Config, issued_at, *, cycle=None, session=None):
    """List only bounded per-hour prefixes; never enumerate the global archive."""
    if config.store:
        if cycle is None:
            raise ValueError("an explicit WeatherNext store requires --cycle")
        return config.store,utc(cycle)
    root = config.root or ROOTS[config.mode]
    bucket,prefix = root.removeprefix("gs://").split("/",1)
    session = session or _session(config)
    latest = utc(cycle or issued_at).replace(minute=0,second=0,microsecond=0)
    count = 1 if cycle else config.maximum_age_hours+1
    for back in range(count):
        candidate = latest-timedelta(hours=back)
        if candidate.year < 2026:
            return f"{root}/{candidate.year}/{'6hr' if candidate.hour % 6 == 0 else 'interim'}_inits.zarr",candidate
        part = f"{prefix}/2026_to_present/{candidate:%Y%m%d_%H}hr_"
        response = session.get(f"https://storage.googleapis.com/storage/v1/b/{quote(bucket,safe='')}/o",
                               params={"prefix":part,"delimiter":"/","maxResults":1000},timeout=30)
        if response.status_code in {401,403}:
            raise PermissionError("WeatherNext access denied: authenticate an allowlisted account and verify project/bucket permissions")
        response.raise_for_status()
        choices = sorted(p for p in response.json().get("prefixes",[]) if re.search(r"_\d+_preds/$",p))
        for choice in reversed(choices):
            object_name=quote(choice+"predictions.zarr/zarr.json",safe="")
            check=session.get(f"https://storage.googleapis.com/storage/v1/b/{quote(bucket,safe='')}/o/{object_name}",timeout=30)
            # Publishing a new folder can precede publishing the Zarr group.
            if check.status_code==404:continue
            if check.status_code in {401,403}:raise PermissionError("WeatherNext metadata access denied")
            check.raise_for_status()
            return f"gs://{bucket}/{choice}predictions.zarr",candidate
    raise FileNotFoundError("no available WeatherNext run within the configured age window")


def open_store(uri,config):
    import obstore
    import zarr
    bucket,prefix = uri.removeprefix("gs://").split("/",1)
    options = {"client_options":{"default_headers":{"x-goog-user-project":config.project}}} if config.mode == "ensemble" else {}
    backend = obstore.store.GCSStore(bucket=bucket,prefix=prefix,**options)
    return xr.open_zarr(zarr.storage.ObjectStore(backend),chunks=None,consolidated=False)


def _hourly(value,cycle,hours,start_hour=0):
    if "init_time" in value.dims:
        value = value.sel(init_time=np.datetime64(cycle.replace(tzinfo=None),"ns"),drop=True)
    if "lead_subtime" in value.dims:
        value = value.stack(hour=("lead_time","lead_subtime"),create_index=False)
        valid = np.datetime64(cycle.replace(tzinfo=None),"ns")+value.lead_time.values+value.lead_subtime.values
        value = value.drop_vars([name for name in ("lead_time","lead_subtime") if name in value.coords])
        value = value.assign_coords(hour=valid).rename(hour="time")
    elif "lead_time" in value.dims:
        lead = value.lead_time.values
        if not np.issubdtype(lead.dtype,np.timedelta64):
            raise ValueError("WeatherNext lead_time must contain decoded timedeltas")
        value = value.assign_coords(lead_time=np.datetime64(cycle.replace(tzinfo=None),"ns")+lead).rename(lead_time="time")
    elif "time" not in value.dims:
        raise ValueError("WeatherNext variable has no recognized forecast time axis")
    value = value.sortby("time")
    if len(np.unique(value.time.values)) != value.sizes["time"]:
        raise ValueError("WeatherNext hourly stacking produced duplicate valid times")
    start = np.datetime64(cycle.replace(tzinfo=None),"ns")
    return value.sel(time=slice(start+np.timedelta64(start_hour,"h"),start+np.timedelta64(hours,"h")))


def normalize(dataset,config: Config,cycle,hours=48,*,start_hour=0):
    """Crop/time-select every variable before reading any remote data chunks."""
    cycle = utc(cycle)
    if not 1 <= hours <= 48:
        raise ValueError("initial WeatherNext download horizon must be between 1 and 48 hours")
    lat,lon = config.latitude,config.longitude
    if lat not in dataset.coords or lon not in dataset.coords or dataset[lat].ndim != 1 or dataset[lon].ndim != 1:
        raise ValueError("configure WeatherNext surface latitude/longitude coordinate names")
    west,east,south,north = config.bbox
    # Index coordinates only, preserving lazy data indexing. Avoid a global
    # longitude sort/rechunk of the remote model fields.
    latitude = np.asarray(dataset[lat].values)
    longitude = ((np.asarray(dataset[lon].values)+180)%360)-180
    iy,ix = np.flatnonzero((latitude >= south)&(latitude <= north)),np.flatnonzero((longitude >= west)&(longitude <= east))
    if min(len(ix),len(iy)) < 2:
        raise ValueError("WeatherNext coordinates do not cover the Missouri buffer")
    cropped = dataset.isel({lat:iy,lon:ix})
    cropped = cropped.assign_coords({lon:longitude[ix]}).sortby(lat).sortby(lon)
    output = {}
    suffix = "_mean" if config.mode == "statistics" else ""
    for source,(target,units,kind) in VARIABLES.items():
        name = source+suffix
        if name not in cropped:
            continue
        value = _hourly(cropped[name],cycle,hours,start_hour)
        allowed = {"time",lat,lon,"sample"}
        if set(value.dims)-allowed:
            raise ValueError(f"unselected WeatherNext dimensions for {name}: {value.dims}")
        incoming = str(value.attrs.get("units","")).lower().replace(" ","").replace("**","^")
        if kind == "temperature":
            if incoming in {"k","kelvin"}: value = value-273.15
            elif incoming not in {"c","degc","celsius","degree_celsius","degrees_celsius"}: raise ValueError(f"unknown temperature units for {name}")
        elif kind == "precip":
            if incoming in {"m","metre","meter"}: value = value*1000
            elif incoming not in {"mm","kgm^-2","kg/m^2"}: raise ValueError(f"unknown precipitation units for {name}")
        elif kind == "solar":
            if incoming in {"jm^-2","j/m^2","jm-2"}: value = value/3600
            elif incoming not in {"wm^-2","w/m^2","wm-2"}: raise ValueError(f"unknown radiation units for {name}")
        elif incoming not in {"ms^-1","m/s","ms-1"}:
            raise ValueError(f"unknown wind units for {name}")
        value = value.rename({lat:"y",lon:"x",**({"sample":"member"} if "sample" in value.dims else {})}).astype("float32")
        value.attrs = {"units":units,"source_variable":name}
        if target == "precipitation_increment": value.attrs["accumulation_semantics"] = "incremental"
        output[target] = value.transpose(*(["member"] if "member" in value.dims else []),"time","y","x")
    required = {"temperature_2m","dewpoint_2m","wind_u_10m","wind_v_10m","precipitation_increment"}
    if not required <= output.keys():
        raise ValueError(f"WeatherNext store missing required surface fields: {sorted(required-output.keys())}")
    result = xr.Dataset(output,attrs={"crs":"EPSG:4326","model":"weathernext3","mode":config.mode,"cycle_time":iso(cycle),
                                    "native_spacing_degrees":.1,"wind_reference":"earth"})
    if "init_time" in result.coords:
        result = result.drop_vars("init_time")
    t,td = result.temperature_2m,result.dewpoint_2m
    result["relative_humidity_2m"] = (100*np.exp(17.625*td/(243.04+td)-17.625*t/(243.04+t))).clip(0,100)
    result.relative_humidity_2m.attrs["units"] = "%"
    if result.nbytes > config.maximum_download_bytes:
        raise ValueError(f"WeatherNext regional selection exceeds download budget ({result.nbytes} bytes)")
    if not result.sizes.get("time"):
        raise ValueError("WeatherNext selection contains no forecast hours")
    return result


def fetch(config: Config,output,*,issued_at=None,cycle=None,hours=48,opener=None,discoverer=None,progress=None):
    config.apply_credentials()
    issued = utc(issued_at or datetime.now(timezone.utc))
    uri,cycle = (discoverer or discover)(config,issued,cycle=cycle)
    if cycle > issued or (issued-cycle).total_seconds()/3600 > config.maximum_age_hours:
        raise ValueError("WeatherNext cycle is future or stale")
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    # Include the request horizon: a successful six-hour access probe must not
    # block a subsequent complete forecast download for the same cycle.
    target = output/f"weathernext_{cycle:%Y%m%d_%H}z_{config.mode}_f{hours:02d}.nc"
    manifest_path = target.with_suffix(".json")
    selection={"bbox":list(config.bbox),"latitude":config.latitude,"longitude":config.longitude}
    if target.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        # Earlier probes supported only the default Missouri selection.
        previous_selection=manifest.get("selection",{"bbox":list(Config().bbox),"latitude":"lat_0p1","longitude":"lon_0p1"})
        if previous_selection==selection and manifest.get("source_store") == uri and manifest.get("sha256") == _hash(target) and manifest.get("horizon_hours",0) >= hours:
            _atomic(output/"latest.json",{"file":target.name,"manifest":manifest_path.name})
            return target,manifest
        raise ValueError("existing WeatherNext cache differs or is corrupt; use a fresh output directory")
    dataset = (opener or open_store)(uri,config)
    temporary = target.with_suffix(".nc.tmp")
    try:
        suffix="_mean" if config.mode=="statistics" else ""
        source_estimate=0
        for source in VARIABLES:
            name=source+suffix
            if name not in dataset:continue
            selected=_hourly(dataset[name],cycle,hours)
            # The statistics bucket uses full-global spatial chunks. A small
            # regional output does not imply a similarly small GCS transfer.
            chunks=dataset[name].encoding.get("chunks")
            if chunks:
                sizes={dim:size for dim,size in zip(dataset[name].dims,chunks)}
                spatial=np.prod([sizes.get(dim,selected.sizes[dim]) for dim in (config.latitude,config.longitude)])
                other=np.prod([selected.sizes[dim] for dim in selected.dims if dim not in {config.latitude,config.longitude}])
                source_estimate+=int(spatial*other*dataset[name].dtype.itemsize)
            else:source_estimate+=selected.nbytes
        if source_estimate>config.maximum_source_bytes:
            raise ValueError("WeatherNext source chunks exceed the explicit uncompressed source-byte budget")
        # Resume cropped hour caches instead of repeating every global chunk
        # when a later variable/hour fails. Native chunks are read with bounded
        # concurrency; NetCDF writes stay on the main thread.
        first=next(dataset[source+suffix] for source in VARIABLES if source+suffix in dataset)
        selected_times=_hourly(first,cycle,hours).time.values
        if not len(selected_times) or selected_times[-1]<np.datetime64(cycle.replace(tzinfo=None),"ns")+np.timedelta64(hours,"h"):
            raise ValueError("WeatherNext run does not reach the requested forecast horizon")
        offsets=[int((stamp-np.datetime64(cycle.replace(tzinfo=None),"ns"))/np.timedelta64(1,"h")) for stamp in selected_times]
        if len(offsets)>1 and np.any(np.diff(offsets)!=1):raise ValueError("WeatherNext forecast has missing hourly intervals")
        parts=output/(target.stem+".parts");parts.mkdir(exist_ok=True)
        contract={"store":uri,"config":asdict(config),"variables":list(VARIABLES)}
        contract_path=parts/"contract.json"
        if contract_path.exists() and json.loads(contract_path.read_text())!=json.loads(json.dumps(contract)):
            raise ValueError("WeatherNext partial-cache contract mismatch")
        _atomic(contract_path,contract)
        files=[];missing=[]
        for hour in offsets:
            part=parts/f"f{hour:03d}.nc";digest=part.with_suffix(".json")
            files.append(part)
            if not part.exists() or not digest.exists() or _hash(part)!=json.loads(digest.read_text())["sha256"]:missing.append(hour)
        def read_hour(hour):return hour,normalize(dataset,config,cycle,hour or 1,start_hour=hour).isel(time=slice(0,1)).load()
        # Map prefetches at most this bounded batch, preventing a large queue of
        # decoded global chunks from growing behind sequential disk writes.
        with ThreadPoolExecutor(max_workers=config.download_concurrency) as pool:
            for start in range(0,len(missing),config.download_concurrency):
                for hour,regional_hour in pool.map(read_hour,missing[start:start+config.download_concurrency]):
                    part=parts/f"f{hour:03d}.nc";temp_part=part.with_suffix(".tmp")
                    regional_hour.to_netcdf(temp_part,engine="netcdf4")
                    temp_part.replace(part);_atomic(part.with_suffix(".json"),{"sha256":_hash(part)})
                    if progress:progress({"stage":"weathernext","forecast_hour":hour,"hours_requested":hours})
        from contextlib import ExitStack
        with ExitStack() as stack:
            regional=xr.concat([stack.enter_context(xr.open_dataset(path)).load() for path in files],dim="time")
        if regional.nbytes>config.maximum_download_bytes:raise ValueError("WeatherNext regional output exceeds download budget")
        # Crop is already bounded. NetCDF stores preserve actual members, not
        # synthetic members reconstructed from marginal percentiles.
        regional.to_netcdf(temporary,engine="netcdf4",encoding={name:{"zlib":True,"complevel":4} for name in regional.data_vars})
        with xr.open_dataset(temporary) as check:
            if dict(check.sizes) != dict(regional.sizes): raise ValueError("WeatherNext cache round-trip mismatch")
        temporary.replace(target)
        manifest = {"schema":"smf-weathernext-v1","source_store":uri,"cycle_time":iso(cycle),
                    "available_at":iso(datetime.now(timezone.utc)),"mode":config.mode,"horizon_hours":hours,
                    "native_resolution_degrees":.1,"wind_reference":"earth","dimensions":dict(regional.sizes),
                    "variables":list(regional.data_vars),"sha256":_hash(target),"bytes":target.stat().st_size,
                    "selection":selection,
                    "spread_kind":"real_members" if config.mode == "ensemble" else "ensemble_mean_only"}
        manifest["source_chunk_uncompressed_bytes_estimate"]=source_estimate
        manifest["transfer_note"]="GCS global chunks; compressed network bytes differ from regional output size"
        _atomic(manifest_path,manifest)
        _atomic(output/"latest.json",{"file":target.name,"manifest":manifest_path.name})
        return target,manifest
    finally:
        dataset.close()
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("doctor","fetch"))
    parser.add_argument("--config",type=Path)
    parser.add_argument("--cycle")
    parser.add_argument("--output",type=Path,default=Path("data/fire-weather-hybrid/cache/weathernext"))
    parser.add_argument("--hours",type=int,default=48)
    args = parser.parse_args()
    config = Config.read(args.config)
    if args.command == "doctor":
        report = doctor(config)
        print(json.dumps(report))
        return 0 if report["status"] == "ready" else 2
    try:
        path,manifest = fetch(config,args.output,cycle=args.cycle,hours=args.hours,progress=lambda value:print(json.dumps(value),file=sys.stderr,flush=True))
        print(json.dumps({"file":str(path),"manifest":manifest}))
        return 0
    except Exception as error:
        # Remote exceptions can contain request authorization details. Keep the
        # user-facing failure actionable without printing raw transport errors.
        print(json.dumps({"status":"failed","error_type":type(error).__name__,
                          "action":"Run doctor; confirm the approved account, store schema and requested cycle."}),file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
