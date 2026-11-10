"""Complete native-source acquisition/conditioning for staff hybrid forecasts.

The existing forecast-v1 downloader, RTMA/MRMS capture and atmospheric-index
implementations remain the source of truth. New caches and state are private.
"""
from __future__ import annotations
from dataclasses import dataclass,fields
from datetime import datetime,timedelta,timezone
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pyproj
import xarray as xr

from forecast_v1.acquisition import (AcquisitionSpec,acquire_source,SURFACE_SEARCHES,GEFS_SURFACE_SEARCHES,
                                   UPPER_AIR_SEARCH,PBL_SEARCH,_clip_and_project_axes)
from forecast_v1.adapters import SourceCube
from services.fire_weather_hybrid.contracts import GRID,FUELS,LOCAL_TZ,day_hours,utc,iso
from services.fire_weather_hybrid.fuel_state import FuelState
from services.fire_weather_hybrid.sources import blend
from services.fire_weather_hybrid.trees import sha256
from services.fire_weather_hybrid_inputs import normalized_hour
from services import fire_weather_hybrid_store as store

logger=logging.getLogger(__name__)
SHAPE=(GRID.height,GRID.width)


@dataclass
class Config:
    landscape: str
    cache: str="data/fire-weather-hybrid/cache"
    states: str="data/fire-weather-hybrid/fuel-state"
    native_library: str | None=None
    weather_python: str | None=None
    weathernext_config: str | None=None
    weathernext_cache: str | None=None
    weathernext_enabled: bool=False
    allow_cold_start: bool=False
    bootstrap_hours: int=72
    # Daily product: a rolling window beginning at this UTC cycle hour. None keeps
    # the legacy America/Chicago calendar-day window.
    anchor_hour_utc: int | None=12
    # Whole-source retries for transient network failures. Downloaded GRIB subsets
    # stay cached, so a retry resumes rather than restarting.
    source_attempts: int=3
    source_retry_wait_seconds: float=30.
    bootstrap_moisture_pct: float=20.
    sources: tuple=("hrrr","rrfs","gefs")
    weights: dict | None=None
    rtma_archive: str | None=None
    station_snapshot: str | None=None
    acquire_rap: bool=True

    @classmethod
    def read(cls,path):
        data=json.loads(Path(path).read_text())
        unknown=set(data)-{item.name for item in fields(cls)}
        if unknown: raise ValueError(f"unknown acquisition configuration: {sorted(unknown)}")
        result=cls(**data)
        if not 0<=result.bootstrap_moisture_pct<=150 or result.bootstrap_hours<1:
            raise ValueError("invalid explicit fuel initialization")
        if result.source_attempts<1 or result.source_retry_wait_seconds<0:
            raise ValueError("invalid source retry settings")
        if result.anchor_hour_utc is not None and not 0<=result.anchor_hour_utc<=23:
            raise ValueError("anchor_hour_utc must be 0-23 or null")
        if not set(result.sources)<={"hrrr","rrfs","gefs"}:
            raise ValueError("unsupported NOAA source")
        return result


def native_cube(ds,cycle,name,aliases,*,kelvin=True):
    projected=_clip_and_project_axes(ds)
    data={}
    for original,target in aliases.items():
        if original in projected:
            value=projected[original].squeeze(drop=True)
            if value.ndim!=2: continue
            data[target]=value-273.15 if kelvin and target in {"temperature_2m","dewpoint_2m"} else value
    result=xr.Dataset(data,attrs=dict(projected.attrs)).expand_dims(time=[np.datetime64(cycle.replace(tzinfo=None),"ns")])
    return SourceCube(name,cycle,result,("deterministic",))


def read_analysis(path,valid):
    with xr.open_dataset(path) as ds:
        cube=native_cube(ds,valid,"rtma",{"t2m":"temperature_2m","d2m":"dewpoint_2m","r2":"relative_humidity_2m",
            "u10":"wind_u_10m","v10":"wind_v_10m","apcp":"precipitation_increment"})
        output=normalized_hour(cube,valid,wind_reference="grid")
        missing="apcp" not in ds or str(ds.apcp.attrs.get("missing","")).lower() in {"true","1"}
    if missing: output.pop("precip_mm",None)
    return output


def _record(path,cycle,name):
    receipt=Path(path).with_suffix(".receipt.json")
    if not receipt.exists():
        store.atomic_json(receipt,{"name":name,"cycle_time":iso(cycle),"available_at":iso(datetime.now(timezone.utc)),"file":Path(path).name})
    return store.read_json(receipt)


class Provider:
    def __init__(self,config: Config,*,as_of=None):
        self.config=config
        self.cache=Path(config.cache);self.cache.mkdir(parents=True,exist_ok=True)
        self.as_of=utc(as_of or datetime.now(timezone.utc))
        self.cubes={};self.records=[];self.warnings=[]
        self.history_records=[]

    def acquire(self,days=1,long_cycle=None):
        rounded=self.as_of.replace(minute=0,second=0,microsecond=0)
        # Latest hourly HRRR through f18 plus the extended cycle: the pinned
        # daily anchor when given, otherwise the latest six-hour cycle.
        short=rounded-timedelta(hours=2)
        long=long_cycle or short.replace(hour=(short.hour//6)*6)
        plans=[]
        if "hrrr" in self.config.sources:
            plans.extend([("hrrr",short,AcquisitionSpec("hrrr","hrrr","sfc",tuple(range(19)),(None,))),
                          ("hrrr_extended",long,AcquisitionSpec("hrrr","hrrr","sfc",tuple(range(49)),(None,)))])
        rrfs_cycle=(rounded-timedelta(hours=4)).replace(hour=((rounded-timedelta(hours=4)).hour//6)*6)
        if "rrfs" in self.config.sources:
            plans.append(("rrfs",rrfs_cycle,AcquisitionSpec("rrfs","rrfs",os.getenv("SMF_RRFS_PRODUCT","2dfld.13km"),tuple(range(73)),("control",),
                domain=os.getenv("SMF_RRFS_DOMAIN","conus"),extras=(("prslev.13km",UPPER_AIR_SEARCH),),upper_air_on_main=False)))
        if "gefs" in self.config.sources:
            plans.append(("gefs",rrfs_cycle,AcquisitionSpec("gefs","gefs","atmos.25",tuple(range(0,73,3)),("c00",)+tuple(f"p{i:02d}" for i in range(1,31)),
                searches=GEFS_SURFACE_SEARCHES,extras=(("atmos.5",UPPER_AIR_SEARCH),("atmos.5b",PBL_SEARCH)),upper_air_on_main=False)))
        for key,cycle,spec in plans:
            cache=self.cache/key/cycle.strftime("%Y%m%d_%Hz")
            marker=cache/"native.nc"
            try:
                if marker.exists():
                    ds=xr.open_dataset(marker)
                    cube=SourceCube(spec.public_name,cycle,ds,tuple(str(v) for v in ds.member.values))
                else:
                    cube=acquire_with_retry(spec,cycle,cache,attempts=self.config.source_attempts,
                                            wait_seconds=self.config.source_retry_wait_seconds)
                    marker.parent.mkdir(parents=True,exist_ok=True)
                    cube.dataset.to_netcdf(marker,engine="netcdf4",encoding={name:{"zlib":True,"complevel":4} for name in cube.dataset.data_vars})
                self.cubes[key]=cube
                self.records.append(_record(marker,cycle,spec.public_name))
                logger.info("hybrid source ready name=%s cycle=%s",key,iso(cycle))
            except Exception as error:
                self.warnings.append({"source":key,"status":"unavailable","error_type":type(error).__name__})
                logger.warning("hybrid source unavailable name=%s error_type=%s",key,type(error).__name__)
        if self.config.weathernext_enabled:
            self._weathernext()
        if self.config.acquire_rap:
            self._rap()
        try:
            from services.fire_weather_hybrid_stations import fetch_snapshot
            self.station_payload=fetch_snapshot(self.cache/"stations",snapshot=self.config.station_snapshot)
            self.records.append({"name":"stations","cycle_time":self.station_payload["received_at"],"available_at":self.station_payload["received_at"]})
        except Exception as error:
            self.station_payload={"stations":[],"received_at":iso(datetime.now(timezone.utc))}
            self.warnings.append({"source":"stations","status":"unavailable","error_type":type(error).__name__})
        if not self.cubes:
            raise RuntimeError("no usable weather forecast source was acquired")

    def _weathernext(self):
        if not self.config.weather_python or not self.config.weathernext_config:
            raise ValueError("WeatherNext requires a configured dedicated reader Python and JSON config")
        directory=Path(self.config.weathernext_cache) if self.config.weathernext_cache else self.cache/"weathernext"
        script=Path(__file__).resolve().parents[1]/"scripts/fetch_weathernext.py"
        try:
            result=subprocess.run([self.config.weather_python,str(script),"fetch","--config",self.config.weathernext_config,
                                   "--output",str(directory),"--hours","48"],capture_output=True,text=True,timeout=1800,check=False)
            if result.returncode:
                raise RuntimeError("WeatherNext reader failed; run fetch_weathernext.py doctor")
            pointer=store.read_json(directory/"latest.json")
            manifest=store.read_json(directory/pointer["manifest"])
            path=directory/pointer["file"]
            if sha256(path)!=manifest["sha256"]: raise ValueError("WeatherNext cache checksum mismatch")
            ds=xr.open_dataset(path)
            self.cubes["weathernext"]=SourceCube("weathernext",utc(manifest["cycle_time"]),ds,
                tuple(str(v) for v in ds.member.values) if "member" in ds.dims else ("ensemble_mean",))
            self.records.append({"name":"weathernext","cycle_time":manifest["cycle_time"],"available_at":manifest["available_at"],
                                 "mode":manifest["mode"],"spread_kind":manifest["spread_kind"]})
        except Exception as error:
            self.warnings.append({"source":"weathernext","status":"unavailable","error_type":type(error).__name__})
            logger.warning("WeatherNext unavailable; NOAA guidance remains usable")

    def forecast(self,valid):
        available={}
        for key,cube in self.cubes.items():
            source="hrrr" if key=="hrrr_extended" else key
            if source in available: continue
            result=normalized_hour(cube,valid,wind_reference="earth" if key=="weathernext" else "grid")
            if result: available[source]=result
        weights=self.config.weights or {"hrrr":.5,"rrfs":.3,"gefs":.1,"weathernext":.1}
        result,coverage=blend(available,weights)
        if not {"temp_c","rh_pct","wind_speed_ms","precip_mm","solar_wm2"}<=result.keys():
            raise RuntimeError(f"weather source coverage incomplete at {iso(valid)}")
        if hasattr(self,"rap_fields"):
            # These are current atmospheric indices, separate from future NWP
            # profiles. Their valid time and age are retained for verification.
            result.update(self.rap_fields)
        return result,{"sources":list(available),"weight_origin":"configured"}

    def _rap(self):
        from services.fire_weather import hrrr_profiles as profiles
        valid=self.as_of.replace(minute=0,second=0,microsecond=0)-timedelta(hours=1)
        try:
            fields=profiles.derive_fields(profiles.fetch_profiles(valid,workdir=self.cache/"rap-profiles",run=valid-timedelta(hours=1),fxx=1,
                model="rap",prs_product="awp130pgrb",sfc_product="awp130pgrb",include_80m=False))
            # Reuse the framework's curvilinear interpolation weights for indices;
            # precipitation never passes through this interpolation path.
            from services.ensemble_fire_danger.regrid import Regridder
            xx,yy=np.meshgrid(GRID.x,GRID.y)
            lon,lat=pyproj.Transformer.from_crs(GRID.crs,"EPSG:4326",always_xy=True).transform(xx,yy)
            mapping=Regridder(fields["lat"],fields["lon"],lat,lon,cache_dir=self.cache/"regrid")
            self.rap_fields={name:mapping(fields[name]) for name in ("hdw","haines")}
            self.records.append({"name":"rap","cycle_time":iso(valid-timedelta(hours=1)),"available_at":iso(datetime.now(timezone.utc)),"valid_time":iso(valid)})
        except Exception as error:
            self.warnings.append({"source":"rap","status":"unavailable","error_type":type(error).__name__})

    def analysis(self,valid):
        if valid>self.as_of: raise ValueError("cannot acquire future analyses")
        from services import rtma_capture
        local=self.cache/"rtma"/f"rtma_{valid:%Y%m%d_%H}z.nc"
        archived=Path(self.config.rtma_archive)/local.name if self.config.rtma_archive else local
        path=archived if archived.exists() else rtma_capture.fetch_rtma(valid,cache_dir=local.parent)
        output=read_analysis(path,valid)
        rain_origin="rtma"
        if "precip_mm" not in output or not np.isfinite(output["precip_mm"]).any():
            rain_origin="mrms"
            from services.fire_outlook import mrms
            from services.fire_weather_hybrid.regrid import remap
            from rasterio.transform import from_origin
            rain=mrms.fetch_hour(valid,self.cache/"mrms",(34.5,42.,-97.5,-87.5))
            if rain is None: raise RuntimeError(f"no RTMA/MRMS precipitation at {iso(valid)}")
            lat,lon=rain["lat"],rain["lon"]
            values=rain["mm"] if lat[0]>lat[-1] else rain["mm"][::-1]
            transform=from_origin(lon.min()-abs(lon[1]-lon[0])/2,lat.max()+abs(lat[1]-lat[0])/2,abs(lon[1]-lon[0]),abs(lat[1]-lat[0]))
            output["precip_mm"]=remap(values,transform,"EPSG:4326",kind="accumulation")
        # RTMA does not supply solar flux. Use an explicitly archived short-lead
        # radiation forecast for the same interval, not a zero-radiation default.
        output["solar_wm2"]=self.radiation(valid)
        self.history_records.append({"valid_time":iso(valid),"received_at":iso(datetime.now(timezone.utc)),
            "rtma_file":str(path),"rain_origin":rain_origin,
            "radiation_origin":"hrrr-short-lead"})
        return output

    def radiation(self,valid):
        try:
            weather,_=self.forecast(valid)
            if np.isfinite(weather["solar_wm2"]).any(): return weather["solar_wm2"]
        except RuntimeError:
            pass
        target=self.cache/"radiation"/f"{valid:%Y%m%d_%H}.nc"
        if not target.exists():
            from herbie import Herbie
            from core.domain import crop
            run=valid-timedelta(hours=1)
            result=Herbie(run.replace(tzinfo=None),model="hrrr",product="sfc",fxx=1,save_dir=str(self.cache/"hrrr-radiation"),verbose=False).xarray(":DSWRF:surface:",remove_grib=False)
            groups=result if isinstance(result,list) else [result]
            # cfgrib/ecCodes versions name surface downward shortwave either
            # "dswrf" or "sdswrf"; normalize so downstream aliasing is stable.
            ds=next(value for value in groups if "dswrf" in value or "sdswrf" in value)
            if "dswrf" not in ds: ds=ds.rename({"sdswrf":"dswrf"})
            ds=crop(ds).load()
            # Scalar GRIB time coordinates (e.g. timedelta "step") do not survive
            # a NetCDF round trip with current xarray; the hour is set by native_cube.
            ds=ds.drop_vars([n for n in ("step","time","valid_time","surface") if n in ds.variables])
            target.parent.mkdir(parents=True,exist_ok=True)
            ds.to_netcdf(target)
        with xr.open_dataset(target) as ds:
            cube=native_cube(ds,valid,"hrrr",{"dswrf":"shortwave_down"},kelvin=False)
            return normalized_hour(cube,valid,wind_reference="grid")["solar_wm2"]

    def close(self):
        for cube in self.cubes.values(): cube.dataset.close()


def install_dns_cache(ttl_seconds=300.):
    """Cache hostname lookups for this process, serving a stale answer if a fresh one fails.

    Herbie re-requests an index file per query and per forecast hour, so a full
    acquisition sends thousands of rapid requests; without a cache, bursts of
    resolver failures (seen on Windows) discard otherwise-available NOAA data.
    """
    import socket,time,threading
    if getattr(socket.getaddrinfo,"_smf_cached",False): return
    original=socket.getaddrinfo;store={};lock=threading.Lock()
    def cached(host,port,*args,**kwargs):
        key=(host,port,args,tuple(sorted(kwargs.items())))
        now=time.monotonic()
        with lock: hit=store.get(key)
        if hit and now-hit[0]<ttl_seconds: return hit[1]
        try:
            result=original(host,port,*args,**kwargs)
        except OSError:
            if hit: return hit[1]
            raise
        with lock: store[key]=(now,result)
        return result
    cached._smf_cached=True
    socket.getaddrinfo=cached


def acquire_with_retry(spec,cycle,cache,*,attempts=3,wait_seconds=30.,sleep=None):
    """Retry one whole source download; cached subsets make later attempts resume."""
    import time
    sleep=sleep or time.sleep
    for attempt in range(1,attempts+1):
        try:
            return acquire_source(spec,cycle,cache)
        except Exception as error:
            if attempt==attempts: raise
            logger.warning("hybrid source retry name=%s attempt=%d/%d error_type=%s",spec.public_name,attempt,attempts,type(error).__name__)
            sleep(wait_seconds*attempt)


def dominant_burnable_fuel(fractions,classes):
    """Largest-area burnable fuel model per cell (codes 91-99 are non-burnable); NaN when none.

    A mixed farm/forest cell whose overall dominant class is NB3 agriculture
    still burns in its forest or grass fraction.
    """
    classes=np.asarray(classes)
    burnable=(classes<91)|(classes>99)
    frac=np.where(burnable[:,None,None],np.nan_to_num(np.asarray(fractions,"float32")),0.)
    best=frac.argmax(axis=0)
    return np.where(frac.max(axis=0)>0,classes[best].astype("float32"),np.nan).astype("float32")


def latest_anchor(moment,hour):
    """Most recent UTC cycle at ``hour`` not after ``moment``."""
    moment=utc(moment);cycle=moment.replace(hour=hour,minute=0,second=0,microsecond=0)
    return cycle if cycle<=moment else cycle-timedelta(days=1)


def stage(config: Config,output,*,days=1,provider=None,allow_cold_start=None,bundle=None):
    provider=provider or Provider(config)
    try:return _stage(config,output,days=days,provider=provider,allow_cold_start=allow_cold_start,bundle=bundle)
    finally:provider.close()


def _stage(config: Config,output,*,days=1,provider=None,allow_cold_start=None,bundle=None):
    """Stream source hours through persisted fuel physics into canonical NetCDF."""
    import netCDF4
    provider=provider or Provider(config)
    issued=datetime.now(timezone.utc)
    anchor=latest_anchor(issued,config.anchor_hour_utc) if config.anchor_hour_utc is not None else None
    times=[anchor+timedelta(hours=i) for i in range(24*days)] if anchor else day_hours(issued.astimezone(LOCAL_TZ).date(),days)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with xr.open_dataset(config.landscape) as landscape:
        if landscape.attrs.get("grid_fingerprint")!=GRID.fingerprint: raise ValueError("landscape grid mismatch")
        static={name:np.asarray(value.values) for name,value in landscape.data_vars.items() if value.dims==("y","x")}
        if "fuel_fraction" in landscape:
            static["burnable_fuel_code"]=dominant_burnable_fuel(landscape["fuel_fraction"].values,landscape["fuel_class"].values)
    mask=(static["missouri_mask"]>0)&(static["burnable_fraction"]>0)&(static["static_coverage_fraction"]>.9)
    start=times[0]-timedelta(hours=1)
    checkpoint_files=[]
    for path in Path(config.states).glob("state_*.json"):
        doc=store.read_json(path)
        if utc(doc["valid_time"])<=start: checkpoint_files.append((utc(doc["valid_time"]),path))
    if checkpoint_files:
        state=FuelState.load(max(checkpoint_files)[1],mask,library=config.native_library,as_of=start)
    else:
        if not (config.allow_cold_start if allow_cold_start is None else allow_cold_start):
            raise RuntimeError("No causal fuel checkpoint: bootstrap history explicitly before starting the staff worker")
        initial=np.full((4,int(mask.sum())),config.bootstrap_moisture_pct)
        state=FuelState(mask,initial,start-timedelta(hours=config.bootstrap_hours),library=config.native_library,cold_start=True)
    while state.time<start:
        valid=state.time+timedelta(hours=1)
        state.advance(valid,provider.analysis(valid))
        if valid.hour%6==0 and not any(stamp==valid for stamp,_ in checkpoint_files):
            state.save(config.states)
        logger.info("fuel conditioning valid=%s hours=%s",iso(valid),state.hours)
    if not any(stamp==start for stamp,_ in checkpoint_files):
        state.save(config.states)
    # Initial conditioning can be long. Acquire fresh guidance afterward so it
    # does not become stale while an initial history replay runs.
    provider.as_of=datetime.now(timezone.utc)
    provider.acquire(days,long_cycle=anchor)
    changed=(latest_anchor(provider.as_of,config.anchor_hour_utc)!=anchor) if anchor else (
        provider.as_of.astimezone(LOCAL_TZ).date()!=issued.astimezone(LOCAL_TZ).date())
    if changed:
        raise RuntimeError("forecast window changed during initial conditioning; retry from the new checkpoint")
    roles=[];hour_sources=[]
    forecast_issued=datetime.now(timezone.utc)
    temperature_bias=None
    payload=getattr(provider,"station_payload",{"stations":[],"received_at":iso(forecast_issued)})
    if bundle is not None and "temp_c_residual" not in bundle.models:
        from services.fire_weather_hybrid_stations import points,interpolate
        differences=[]
        temperature_sources={}
        for point in points(payload,"air_temp",forecast_issued,forecast_issued):
            try:
                stamp=point["valid_time"].replace(minute=0,second=0,microsecond=0)
                if stamp not in temperature_sources:
                    baseline,_=provider.forecast(stamp)
                    temperature_sources[stamp]=baseline["temp_c"]
                row,col=GRID.cell(point["lat"],point["lon"])
                value=point["value"]-temperature_sources[stamp][row,col]
                if np.isfinite(value): differences.append({**point,"value":float(np.clip(value,-8,8))})
            except (ValueError,RuntimeError): continue
        if len(differences)>=3:
            field,gain=interpolate(differences,mask.shape)
            temperature_bias=np.nan_to_num(field)*gain
    temporary=output.with_suffix(".nc.tmp")
    try:
        with netCDF4.Dataset(temporary,"w") as nc:
            nc.createDimension("time",len(times));nc.createDimension("y",GRID.height);nc.createDimension("x",GRID.width)
            nc.createVariable("x","f8",("x",))[:]=GRID.x;nc.createVariable("y","f8",("y",))[:]=GRID.y
            time_axis=nc.createVariable("time","f8",("time",));time_axis.units="seconds since 1970-01-01 00:00:00 UTC";time_axis[:]=[v.timestamp() for v in times]
            for name,value in static.items():
                if name=="office": nc.createVariable(name,str,("y","x"))[:]=value.astype(object)
                else: nc.createVariable(name,"f4",("y","x"),fill_value=np.nan,zlib=True)[:]=value
            variables={};analysis_branch=True;used_anchors=set()
            for index,valid in enumerate(times):
                role="forecast"
                if valid<=provider.as_of-timedelta(hours=2):
                    weather=provider.analysis(valid);role="analysis";sources={"sources":["rtma","mrms","hrrr-radiation"]}
                    # Observed analyses must never receive forecast-bias corrections.
                    weather["weather_bias_applied"]=np.ones(mask.shape,"float32")
                else:
                    if analysis_branch:
                        # Persist analyzed state only. Forecast trajectories must
                        # never become the next cycle's observed fuel history.
                        try: state.save(config.states)
                        except ValueError: pass  # already saved at this exact analysis timestamp
                        analysis_branch=False
                    weather,sources=provider.forecast(valid)
                    raw={"raw_"+name:np.asarray(weather[name],"float32").copy() for name in ("temp_c","rh_pct","wind_speed_ms")}
                    if bundle is not None:
                        from services.fire_weather_hybrid.contracts import feature_matrix
                        from services.fire_weather_hybrid.corrections import apply
                        context={**static,**weather,**state.fields()}
                        tasks={name for name in bundle.models if name.endswith("_residual") and not name.startswith("fm")}
                        residuals=bundle.predict(feature_matrix(context,valid,forecast_issued,mask.shape),tasks=tasks)
                        weather=apply(weather,residuals,fuel=False)
                        weather["weather_bias_applied"]=np.ones(mask.shape,"float32")
                    if temperature_bias is not None:
                        from services.fire_weather_hybrid.runtime import dewpoint
                        weather["temp_c"]=weather["temp_c"]+temperature_bias
                        weather["dewpoint_c"]=dewpoint(weather["temp_c"],weather["rh_pct"])
                    # Explicit neutral open-wind height conversion, independent
                    # of canopy/midflame reductions inside surface-fire physics.
                    weather["wind_20ft_ms"]=weather["wind_speed_ms"]*np.log(6.096/.03)/np.log(10/.03)
                    weather.update(raw)
                values=state.advance(valid,weather)
                # Latest observations can condition only the current horizon,
                # never be repeatedly imposed through all future forecast hours.
                if valid<=forecast_issued.replace(minute=0,second=0,microsecond=0)+timedelta(hours=1):
                    from services.fire_weather_hybrid_stations import points,interpolate
                    anchors=points(payload,"fuel_moisture",valid,forecast_issued)
                    anchors=[point for point in anchors if (point["station_id"],point["valid_time"]) not in used_anchors]
                    if anchors:
                        target,gain=interpolate(anchors,mask.shape)
                        state.anchor10(target,gain)
                        values=state.fields()
                        used_anchors.update((point["station_id"],point["valid_time"]) for point in anchors)
                weather={**weather,**values}
                for name,value in weather.items():
                    if name not in variables:
                        variables[name]=nc.createVariable(name,"f4",("time","y","x"),zlib=True,complevel=4,fill_value=np.nan)
                    variables[name][index]=np.where(static["missouri_mask"]>0,value,np.nan)
                roles.append(role);hour_sources.append(sources)
            issued=datetime.now(timezone.utc)
            nc.grid_fingerprint=GRID.fingerprint;nc.issued_at=iso(issued)
            nc.window_start=iso(times[0]);nc.window_hours=len(times)
            if anchor: nc.anchor_cycle=iso(anchor)
            nc.model_reference_at=iso(forecast_issued)
            nc.fuel_physics="NFDRS4 Nelson-Carlson; pinned "+__import__('services.fire_weather_hybrid.fuel_state',fromlist=['COMMIT']).COMMIT
            nc.step_kinds=json.dumps(roles)
            nc.provenance=json.dumps({"sources":provider.records,"warnings":provider.warnings,"hour_sources":hour_sources,
                "fuel_history":getattr(provider,"history_records",[]),
                "fuel_initialization":{"cold_start":state.cold_start,"hours_conditioned_before_day":state.hours-len(times),
                    "class_ready":{str(fuel):not state.cold_start or state.hours-len(times)>=5*fuel for fuel in FUELS}},
                "landscape_sha256":sha256(config.landscape)})
        temporary.replace(output)
        return output
    finally:
        temporary.unlink(missing_ok=True)
