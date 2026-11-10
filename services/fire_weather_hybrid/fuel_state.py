"""Persistent, causal four-class fuel states backed by the actual NFDRS engine."""
from __future__ import annotations
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from .contracts import FUELS,GRID,utc,iso
from .trees import sha256

COMMIT="84223b95fd395a62ea344257f80a9949cee71e6e"


def native(directory=None):
    path=directory or os.getenv("SMF_NFDRS_LIBRARY_DIR")
    if path and str(Path(path).resolve()) not in sys.path:
        sys.path.insert(0,str(Path(path).resolve()))
    try:
        module=importlib.import_module("_smf_nfdrs")
    except ImportError as error:
        raise RuntimeError("Build the pinned NFDRS binding and configure SMF_NFDRS_LIBRARY_DIR") from error
    if module.upstream_commit != COMMIT:
        raise ValueError("NFDRS binding version mismatch")
    return module


class FuelState:
    def __init__(self,mask,initial_percent,valid_time,*,library=None,cold_start=True):
        self.mask=np.asarray(mask,bool)
        self.signature=hashlib.sha256((GRID.fingerprint+str(self.mask.shape)).encode()+self.mask.tobytes()).hexdigest()
        initial=np.asarray(initial_percent,float)
        if initial.shape != (4,int(self.mask.sum())):
            raise ValueError("initial fuel state must include all four active-cell moisture classes")
        self.engine=native(library).FuelGrid(initial)
        self.time=utc(valid_time)
        self.hours=0
        self.cold_start=bool(cold_start)
        self.observation_count=0

    def fields(self,values=None):
        values=self.engine.moisture() if values is None else values
        result={}
        for index,fuel in enumerate(FUELS):
            field=np.full(self.mask.shape,np.nan,"float32")
            field[self.mask]=values[index]
            result[f"fm{fuel}_pct"]=field
        return result

    def advance(self,time,weather):
        time=utc(time)
        elapsed=(time-self.time).total_seconds()/3600
        if not 0 < elapsed <= 1.000001:
            raise ValueError("fuel state requires consecutive hourly intervals; replay missing history first")
        arrays=[np.broadcast_to(np.asarray(weather[name],float),self.mask.shape)[self.mask] for name in ("temp_c","rh_pct","solar_wm2","precip_mm")]
        if not all(np.isfinite(value).all() for value in arrays):
            raise ValueError("fuel history contains missing weather or precipitation; state was not advanced")
        t,h,s,r=arrays
        if np.any((t< -60)|(t>60)) or np.any((h<.1)|(h>100)) or np.any((s<0)|(s>2000)) or np.any(r<0):
            raise ValueError("fuel history contains out-of-range weather; state was not advanced")
        values=self.engine.update(*arrays,elapsed)
        if not np.isfinite(values).all():
            raise ValueError("NFDRS rejected an hourly weather input")
        self.time=time
        self.hours+=elapsed
        return self.fields(values)

    def anchor10(self,target,gain):
        target=np.broadcast_to(target,self.mask.shape)[self.mask]
        gain=np.broadcast_to(gain,self.mask.shape)[self.mask]
        self.engine.anchor10(target,gain)
        self.observation_count+=int(np.sum(np.isfinite(target)&(gain>0)))

    def save(self,directory):
        directory=Path(directory)
        directory.mkdir(parents=True,exist_ok=True)
        stamp=self.time.strftime("%Y%m%d_%H%M%S")
        target=directory/f"state_{stamp}.bin"
        if target.exists():
            raise ValueError("fuel checkpoint is immutable")
        temp=target.with_suffix(".tmp")
        self.engine.save(str(temp))
        temp.replace(target)
        digest=sha256(target)
        manifest={"schema":"nfdrs-fuel-state-v1","commit":COMMIT,"valid_time":iso(self.time),"mask_hash":self.signature,
                  "hours_conditioned":self.hours,"cold_start":self.cold_start,"file":target.name,"sha256":digest,
                  "class_ready":{str(fuel):not self.cold_start or self.hours>=5*fuel for fuel in FUELS},
                  "anchored_cells":self.observation_count,"platform":sys.platform}
        path=target.with_suffix(".json")
        path.write_text(json.dumps(manifest,indent=2)+"\n")
        pointer=directory/"latest.json.tmp"
        pointer.write_text(json.dumps({"manifest":path.name})+"\n")
        pointer.replace(directory/"latest.json")
        return path

    @classmethod
    def load(cls,manifest_path,mask,*,library=None,as_of=None):
        manifest_path=Path(manifest_path)
        doc=json.loads(manifest_path.read_text())
        if doc.get("commit")!=COMMIT or doc.get("platform")!=sys.platform:
            raise ValueError("fuel checkpoint engine/platform mismatch")
        if as_of is not None and utc(doc["valid_time"])>utc(as_of):
            raise ValueError("future fuel checkpoint cannot initialize this forecast")
        path=(manifest_path.parent/doc["file"]).resolve()
        if path.parent!=manifest_path.parent.resolve() or sha256(path)!=doc["sha256"]:
            raise ValueError("fuel checkpoint path/checksum mismatch")
        result=cls(mask,np.full((4,int(np.asarray(mask,bool).sum())),20.),doc["valid_time"],library=library,cold_start=doc["cold_start"])
        if result.signature!=doc["mask_hash"]:
            raise ValueError("fuel checkpoint grid mask mismatch")
        result.engine.load(str(path))
        result.hours=doc["hours_conditioned"]
        result.observation_count=doc.get("anchored_cells",0)
        return result
