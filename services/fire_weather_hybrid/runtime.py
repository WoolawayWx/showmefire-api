"""Bounded CPU tree + ONNX inference, operating on one hour at a time."""
from __future__ import annotations
import numpy as np
from .contracts import FEATURES, feature_matrix
from .criteria import qualifying, PROFILES
from .risk import categories
from .corrections import apply as apply_corrections


def humidity(temp, dewpoint):
    t, td = np.asarray(temp, float), np.asarray(dewpoint, float)
    return np.clip(100*np.exp(17.625*td/(243.04+td)-17.625*t/(243.04+t)), 0, 100)


def dewpoint(temp, rh):
    gamma = np.log(np.clip(rh, .01, 100)/100)+17.625*np.asarray(temp)/(243.04+np.asarray(temp))
    return 243.04*gamma/(17.625-gamma)


def _applied(fields, name):
    """True when every finite cell carries the flag. Cells outside the domain are NaN
    by design, so a plain np.all(flag > 0) is False and re-applies a correction."""
    flag = fields.get(name)
    if flag is None:
        return False
    finite = np.asarray(flag, float)[np.isfinite(flag)]
    return bool(finite.size and np.all(finite > 0))


class Predictor:
    def __init__(self, bundle):
        import onnxruntime as ort
        self.bundle = bundle
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(bundle.root/"spatial.onnx"), sess_options=options,
                                           providers=["CPUExecutionProvider"])
        self.channels = bundle.manifest["spatial"]["channels"]
        self.mean = np.asarray(bundle.manifest["spatial"]["mean"], "float32")[:, None, None]
        self.std = np.asarray(bundle.manifest["spatial"]["std"], "float32")[:, None, None]
        if np.any(self.std <= 0) or len(self.channels) != len(self.mean):
            raise ValueError("invalid spatial normalizer")

    def spatial(self, fields, *, tile=128, halo=24):
        shape = np.asarray(fields["ros_m_min"]).shape
        output = np.zeros((2, *shape), "float32")
        for row in range(0, shape[0], tile):
            for col in range(0, shape[1], tile):
                r0, c0 = max(0, row-halo), max(0, col-halo)
                r1, c1 = min(shape[0], row+tile+halo), min(shape[1], col+tile+halo)
                inputs = np.stack([np.asarray(fields[name])[r0:r1, c0:c1] for name in self.channels])
                inputs = np.nan_to_num((inputs-self.mean)/self.std, nan=0., posinf=0., neginf=0.)
                pad = ((0, 0), (0, max(0, 8-inputs.shape[1])), (0, max(0, 8-inputs.shape[2])))
                inputs = np.pad(inputs, pad, mode="edge")
                residual = self.session.run(None, {"features": inputs[None].astype("float32")})[0][0]
                nr, nc = min(tile, shape[0]-row), min(tile, shape[1]-col)
                output[:, row:row+nr, col:col+nc] = residual[:, row-r0:row-r0+nr, col-c0:col-c0+nc]
        return output

    def predict_hour(self, fields, valid_time, issued_at, *, office=None):
        fields = {k: np.asarray(v, "float32").copy() for k, v in fields.items()}
        shape = fields["temp_c"].shape
        if "rh_pct" not in fields:
            fields["rh_pct"] = humidity(fields["temp_c"], fields["dewpoint_c"])
        if "wind_speed_ms" not in fields:
            fields["wind_speed_ms"] = np.hypot(fields["wind_u_ms"], fields["wind_v_ms"])
        if office is not None:
            offices = np.broadcast_to(np.asarray(office), shape)
            fields["office_is_dvn"] = np.where(np.isin(offices,list(PROFILES)),(offices == "DVN").astype(float),np.nan)
            fields["office_wind_threshold_ms"] = np.full(shape,np.nan,"float32")
            for key,profile in PROFILES.items():
                if "wind_mph" in profile:
                    fields["office_wind_threshold_ms"][offices == key] = profile["wind_mph"]/2.2369362921
        correction_tasks={name for name in self.bundle.models if name.endswith("_residual") and not name.startswith("fm")}
        # New acquired runs have already corrected weather before evolving fuels.
        # Fuel residuals remain post-physics and are applied exactly once here.
        if _applied(fields,"weather_bias_applied"):
            correction_tasks=set()
        residuals=self.bundle.predict(feature_matrix(fields,valid_time,issued_at,shape),tasks=correction_tasks)
        fields=apply_corrections(fields,residuals,fuel=False)
        fuel_tasks={name for name in self.bundle.models if name.startswith("fm") and name.endswith("_residual")}
        if _applied(fields,"fuel_bias_applied"):fuel_tasks=set()
        residuals=self.bundle.predict(feature_matrix(fields,valid_time,issued_at,shape),tasks=fuel_tasks)
        fields=apply_corrections(fields,residuals)
        # Re-score the behavior with corrected inputs. No actual future analyses
        # or fuel observations enter this matrix.
        tasks = self.bundle.predict(feature_matrix(fields, valid_time, issued_at, shape),tasks={"ros_m_min","intensity_kw_m","red_flag"})
        for name in ("ros_m_min", "intensity_kw_m"):
            fields[name] = tasks[name].reshape(shape)
        # Spatial corrections operate in log1p physical units.
        residual = self.spatial(fields)
        for index, name in enumerate(("ros_m_min", "intensity_kw_m")):
            fields[name] = np.maximum(np.expm1(np.log1p(fields[name])+np.clip(residual[index], -2., 2.)), 0)
        valid = (np.isfinite(fields["temp_c"]) & np.isfinite(fields["rh_pct"]) &
                 np.isfinite(fields["wind_speed_ms"]) & np.isfinite(fields["fm10_pct"]) &
                 (fields["burnable_fraction"] > 0) & (fields.get("missouri_mask", 1) > 0))
        for name in ("ros_m_min", "intensity_kw_m"):
            fields[name] = np.where(valid, fields[name], np.nan)
        fields["flame_length_m"] = .0775*np.power(fields["intensity_kw_m"], .46)
        fields["danger_category"] = categories(fields["ros_m_min"], fields["intensity_kw_m"], self.bundle.scale, fields["burnable_fraction"])
        # Keep explicit, staged height-corrected wind; never silently substitute 10m.
        if office is not None:
            fields["red_flag_criteria"] = qualifying(office, fields["rh_pct"], fields.get("wind_20ft_ms", np.full(shape, np.nan)),
                                                       fields["fm10_pct"], gfdi=fields.get("gfdi"))
        if "red_flag" in tasks:
            fields["red_flag_probability"] = np.where(valid, tasks["red_flag"].reshape(shape)*100, np.nan)
        for name in fields:
            fields[name] = np.broadcast_to(fields[name], shape).astype("float32")
        return fields



class PhysicsPredictor:
    """Raw forecast weather + NFDRS dead fuels -> Rothermel behavior -> absolute levels.

    No trained models: no weather corrections, behavior trees or spatial U-Net.
    Live fuel moisture uses the documented monthly Missouri defaults in behavior.py.
    """
    version = "physics-rothermel-v1"

    def __init__(self):
        from .risk import FLAME_LENGTH_SCALE
        self.scale = FLAME_LENGTH_SCALE

    def predict_hour(self, fields, valid_time, issued_at, *, office=None):
        from .behavior import surface_fire, live_moisture, wind_from_degrees, aspect_degrees
        from .contracts import LOCAL_TZ, utc
        from .risk import flame_length_categories
        fields = {k: np.asarray(v, "float32").copy() for k, v in fields.items()}
        shape = fields["temp_c"].shape
        if "rh_pct" not in fields:
            fields["rh_pct"] = humidity(fields["temp_c"], fields["dewpoint_c"])
        if "wind_speed_ms" not in fields:
            fields["wind_speed_ms"] = np.hypot(fields["wind_u_ms"], fields["wind_v_ms"])
        if "wind_20ft_ms" not in fields or not np.isfinite(fields["wind_20ft_ms"]).any():
            # Neutral open-wind conversion 10 m -> 20 ft, as staging does for forecast hours.
            fields["wind_20ft_ms"] = fields["wind_speed_ms"]*np.log(6.096/.03)/np.log(10/.03)
        code = fields.get("burnable_fuel_code")
        if code is None:
            code = np.where((fields["fuel_model_code"] < 91) | (fields["fuel_model_code"] > 99), fields["fuel_model_code"], np.nan)
        herb, woody = live_moisture(utc(valid_time).astimezone(LOCAL_TZ).month)
        wind_from = wind_from_degrees(fields["wind_u_ms"], fields["wind_v_ms"]) if "wind_u_ms" in fields else np.zeros(shape)
        behavior = surface_fire(code, fields["fm1_pct"], fields["fm10_pct"], fields["fm100_pct"], herb, woody,
                                fields["wind_20ft_ms"], wind_from, fields.get("slope_deg", 0.),
                                aspect_degrees(fields.get("aspect_sin", 0.), fields.get("aspect_cos", 1.)),
                                fields.get("canopy_cover_pct", 0.), fields.get("canopy_height_m", 0.))
        valid = (np.isfinite(fields["temp_c"]) & np.isfinite(fields["rh_pct"]) & np.isfinite(fields["wind_speed_ms"]) &
                 np.isfinite(fields["fm10_pct"]) & (fields["burnable_fraction"] > 0) & (fields.get("missouri_mask", 1) > 0))
        for name, values in behavior.items():
            fields[name] = np.where(valid, values, np.nan)
        fields["danger_category"] = flame_length_categories(fields["flame_length_m"], fields["burnable_fraction"], self.scale)
        if office is not None:
            offices = np.broadcast_to(np.asarray(office), shape)
            fields["office_is_dvn"] = np.where(np.isin(offices, list(PROFILES)), (offices == "DVN").astype(float), np.nan)
            fields["red_flag_criteria"] = qualifying(office, fields["rh_pct"], fields["wind_20ft_ms"], fields["fm10_pct"], gfdi=fields.get("gfdi"))
        for name in fields:
            fields[name] = np.broadcast_to(fields[name], shape).astype("float32")
        return fields
