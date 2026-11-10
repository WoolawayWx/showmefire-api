from datetime import datetime,timedelta,timezone
from pathlib import Path
import json
import numpy as np
import pytest
import xarray as xr


def test_live_stage_uses_four_real_fuel_classes_and_saves_only_analyses(tmp_path,monkeypatch):
    from services import fire_weather_hybrid_acquisition as acquisition
    from services.fire_weather_hybrid.contracts import Grid,utc
    from services.fire_weather_hybrid.fuel_state import FuelState
    library=Path(__file__).resolve().parents[2]/"model-training/data/fire_weather_hybrid/native"
    if not list(library.glob("_smf_nfdrs*")):pytest.skip("build pinned native binding first")
    grid=Grid(id="test-stage",width=3,height=2,west=500000,north=4300000)
    monkeypatch.setattr(acquisition,"GRID",grid)
    mask=np.ones((2,3),bool)
    ds=xr.Dataset({name:(("y","x"),np.ones((2,3))) for name in ("missouri_mask","burnable_fraction","static_coverage_fraction")},
        coords={"x":grid.x,"y":grid.y},attrs={"grid_fingerprint":grid.fingerprint})
    landscape=tmp_path/"landscape.nc";ds.to_netcdf(landscape)
    now=datetime.now(timezone.utc);anchor=acquisition.latest_anchor(now,12);start=anchor-timedelta(hours=1)
    state=FuelState(mask,np.full((4,6),20),start,library=library,cold_start=False)
    states=tmp_path/"states";state.save(states)
    class Provider:
        as_of=now;records=[{"name":"hrrr","cycle_time":now.isoformat(),"available_at":now.isoformat()}];warnings=[]
        closed=False
        def acquire(self,days,long_cycle=None):self.long_cycle=long_cycle
        def analysis(self,time):return {"temp_c":np.full((2,3),25.),"rh_pct":np.full((2,3),30.),"solar_wm2":np.full((2,3),400.),"precip_mm":np.zeros((2,3)),"wind_speed_ms":np.ones((2,3))}
        def forecast(self,time):return self.analysis(time),{"sources":["hrrr"]}
        def close(self):self.closed=True
    provider=Provider();config=acquisition.Config(str(landscape),states=str(states),native_library=str(library))
    output=acquisition.stage(config,tmp_path/"input.nc",provider=provider)
    assert provider.closed
    with xr.open_dataset(output) as result:
        for fuel in (1,10,100,1000):assert np.isfinite(result[f"fm{fuel}_pct"]).all()
        assert "NFDRS4" in result.fuel_physics
        kinds=json.loads(result.step_kinds);assert "forecast" in kinds
        assert result.sizes["time"]==24
        assert utc(result.window_start)==anchor and utc(result.anchor_cycle)==anchor and result.window_hours==24
        assert provider.long_cycle==anchor
        assert "raw_temp_c" in result
    checkpoints=[json.loads(path.read_text()) for path in states.glob("state_*.json")]
    assert all(utc(row["valid_time"])<=now-timedelta(hours=1) for row in checkpoints)
    with pytest.raises(RuntimeError,match="checkpoint"):
        acquisition.stage(acquisition.Config(str(landscape),states=str(tmp_path/"missing")),tmp_path/"failed.nc",provider=provider)
    assert not (tmp_path/"failed.nc").exists()


def test_station_snapshot_cannot_leak_future_or_stale_readings():
    from services.fire_weather_hybrid_stations import points
    now=datetime(2026,11,6,12,tzinfo=timezone.utc)
    payload={"received_at":now.isoformat(),"stations":[{"stid":"A","latitude":38.,"longitude":-92.,
        "observations":{"fuel_moisture":{"value":8.,"time":(now-timedelta(hours=1)).isoformat()}}}]}
    assert len(points(payload,"fuel_moisture",now,now))==1
    assert not points(payload,"fuel_moisture",now-timedelta(hours=2),now)
    assert not points(payload,"fuel_moisture",now+timedelta(hours=4),now)
    assert not points(payload,"fuel_moisture",now,now-timedelta(minutes=1))


def test_sema_region_extents_are_projected_to_geographic_coordinates():
    from services.fire_weather_hybrid_maps import region_bounds
    directory=Path(__file__).resolve().parents[1]/"maps/shapefiles/SEMA Regions"
    for path in directory.glob("Region*.shp"):
        west,east,south,north=region_bounds(path)
        assert -97<west<east< -88
        assert 35<south<north<41


def test_weathernext_float32_coordinate_quantization_is_a_regular_grid():
    from forecast_v1.adapters import SourceCube
    from services.fire_weather_hybrid_inputs import normalized_hour
    stamp=datetime(2026,11,6,12,tzinfo=timezone.utc)
    lon=(np.arange(262.5,272.6,.1).astype('float32')+180)%360-180
    lat=np.arange(34.5,42.1,.1).astype('float32')
    data=xr.Dataset({'temperature_2m':(('time','y','x'),np.full((1,len(lat),len(lon)),20.,'float32'))},
        coords={'time':[np.datetime64('2026-11-06T12:00')],'x':lon,'y':lat},attrs={'crs':'EPSG:4326'})
    cube=SourceCube('weathernext',stamp,data,('ensemble_mean',))
    result=normalized_hour(cube,stamp,wind_reference='earth')
    assert np.isfinite(result['temp_c']).any()
    assert np.nanmean(result['temp_c'])==pytest.approx(20)
    data=data.assign_coords(x=data.x.values+np.where(np.arange(len(lon))==50,.02,0))
    with pytest.raises(ValueError,match='irregular'):
        normalized_hour(SourceCube('weathernext',stamp,data,('ensemble_mean',)),stamp,wind_reference='earth')


def test_latest_anchor_is_most_recent_12z_not_after_now():
    from services import fire_weather_hybrid_acquisition as acquisition
    assert acquisition.latest_anchor(datetime(2026,11,6,20,5,tzinfo=timezone.utc),12)==datetime(2026,11,6,12,tzinfo=timezone.utc)
    assert acquisition.latest_anchor(datetime(2026,11,6,12,tzinfo=timezone.utc),12)==datetime(2026,11,6,12,tzinfo=timezone.utc)
    assert acquisition.latest_anchor(datetime(2026,11,6,11,59,tzinfo=timezone.utc),12)==datetime(2026,11,5,12,tzinfo=timezone.utc)


def test_source_download_retries_transient_failures_then_succeeds_or_raises():
    from services import fire_weather_hybrid_acquisition as acquisition
    calls={"n":0};waits=[]
    class Spec:public_name="hrrr"
    def flaky(spec,cycle,cache):
        calls["n"]+=1
        if calls["n"]<3:raise RuntimeError("temporary DNS failure")
        return "cube"
    original=acquisition.acquire_source;acquisition.acquire_source=flaky
    try:
        assert acquisition.acquire_with_retry(Spec(),None,None,attempts=3,wait_seconds=5,sleep=waits.append)=="cube"
        assert calls["n"]==3 and waits==[5,10]
        calls["n"]=-10
        with pytest.raises(RuntimeError):
            acquisition.acquire_with_retry(Spec(),None,None,attempts=2,wait_seconds=0,sleep=waits.append)
    finally:acquisition.acquire_source=original


def test_dns_cache_reuses_answers_and_serves_stale_on_failure(monkeypatch):
    import socket
    from services import fire_weather_hybrid_acquisition as acquisition
    calls={"n":0};fail={"on":False}
    def fake(host,port,*args,**kwargs):
        calls["n"]+=1
        if fail["on"]:raise socket.gaierror(11001,"getaddrinfo failed")
        return [("answer",host,port)]
    monkeypatch.setattr(socket,"getaddrinfo",fake)
    acquisition.install_dns_cache(ttl_seconds=0.05)
    assert socket.getaddrinfo("example.test",443)==socket.getaddrinfo("example.test",443) and calls["n"]==1
    import time;time.sleep(0.08);fail["on"]=True
    assert socket.getaddrinfo("example.test",443)==[("answer","example.test",443)]  # stale beats failure
    with pytest.raises(OSError):socket.getaddrinfo("other.test",443)


def test_dominant_burnable_fuel_skips_nonburnable_classes():
    from services.fire_weather_hybrid_acquisition import dominant_burnable_fuel
    classes = np.array([93, 102, 189])
    fractions = np.zeros((3, 1, 3), "float32")
    fractions[:, 0, 0] = [.7, .2, .1]   # mostly farmland: burns as its grass
    fractions[:, 0, 1] = [0., .3, .7]   # forest litter
    fractions[0, 0, 2] = 1.             # all farmland: nothing burnable
    out = dominant_burnable_fuel(fractions, classes)[0]
    assert out[0] == 102 and out[1] == 189 and np.isnan(out[2])
