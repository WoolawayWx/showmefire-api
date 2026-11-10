"""Source capabilities and strict temporal/ensemble semantics."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import timedelta
import numpy as np
from .contracts import utc


@dataclass(frozen=True)
class Capability:
    role: str
    max_age_hours: int
    native_spacing_km: float
    ensemble: bool = False


CAPABILITIES = {"hrrr": Capability("forecast", 12, 3), "rrfs": Capability("forecast", 18, 13),
                "gefs": Capability("forecast", 24, 25, True), "weathernext": Capability("forecast", 24, 10, True),
                "rap": Capability("profile-analysis", 3, 13), "rtma": Capability("analysis", 6, 2.5),
                "mrms": Capability("precip-analysis", 6, 1), "stations": Capability("observations", 3, 0)}


def validate_availability(source, cycle_time, available_at, issued_at):
    if source not in CAPABILITIES:
        raise ValueError(f"unsupported source: {source}")
    cycle, available, issued = map(utc, (cycle_time, available_at, issued_at))
    if cycle > available or available > issued or issued-cycle > timedelta(hours=CAPABILITIES[source].max_age_hours):
        raise ValueError(f"future, stale, or unavailable source: {source}")


def blend(fields_by_source, weights):
    """Each source gets one weight, regardless of its member count.

    Callers supply ensemble means here and retain members independently for
    uncertainty verification. Spread is not automatically a calibrated probability.
    """
    if any(not np.isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError("blend weights must be finite and nonnegative")
    output, coverage = {}, {}
    for name in set().union(*(set(value) for value in fields_by_source.values())):
        items = [(data[name], weights.get(source, 0)) for source, data in fields_by_source.items() if name in data and weights.get(source, 0) > 0]
        if not items:
            continue
        values = np.stack([item[0] for item in items]).astype(float)
        w = np.asarray([item[1] for item in items]).reshape((-1,)+(1,)*(values.ndim-1))
        valid = np.isfinite(values)
        denominator = (valid*w).sum(axis=0)
        output[name] = np.divide(np.where(valid, values*w, 0).sum(axis=0), denominator,
                                 out=np.full(values.shape[1:], np.nan), where=denominator > 0).astype("float32")
        coverage[name] = denominator.astype("float32")
    return output, coverage
