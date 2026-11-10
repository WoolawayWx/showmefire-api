"""The same bounded weather transformations in training, staging and serving."""
from __future__ import annotations
import numpy as np
from .contracts import FEATURES

LIMITS={"temp_c":8.,"rh_pct":25.,"wind_speed_ms":8.}


def apply(fields,residuals,*,fuel=True):
    output={name:np.asarray(value,"float32").copy() for name,value in fields.items()}
    original=output.get("wind_speed_ms")
    for name,limit in LIMITS.items():
        if name in output and f"{name}_residual" in residuals:
            output[name]+=np.clip(np.asarray(residuals[f"{name}_residual"]).reshape(output[name].shape),-limit,limit)
    output["rh_pct"]=np.clip(output["rh_pct"],0,100)
    output["wind_speed_ms"]=np.maximum(output["wind_speed_ms"],0)
    if original is not None:
        ratio=np.divide(output["wind_speed_ms"],original,out=np.ones_like(original),where=original>.05)
        for name in ("wind_u_ms","wind_v_ms","wind_20ft_ms"):
            if name in output: output[name]*=ratio
    if "gust_ms" in output: output["gust_ms"]=np.maximum(output["gust_ms"],output["wind_speed_ms"])
    gamma=np.log(np.clip(output["rh_pct"],.01,100)/100)+17.625*output["temp_c"]/(243.04+output["temp_c"])
    output["dewpoint_c"]=243.04*gamma/(17.625-gamma)
    if fuel:
        for size in (1,10,100,1000):
            name=f"fm{size}_pct"
            if name in output and f"{name}_residual" in residuals:
                output[name]=np.clip(output[name]+np.asarray(residuals[f"{name}_residual"]).reshape(output[name].shape),0,150)
    return output


def matrix(raw,residuals):
    raw=np.asarray(raw,"float32")
    fields={name:raw[:,index] for index,name in enumerate(FEATURES)}
    corrected=apply(fields,residuals)
    return np.column_stack([corrected[name] for name in FEATURES]).astype("float32")
