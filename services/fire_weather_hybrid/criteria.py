"""Qualifying conditions, never a prediction of official warning issuance."""
from __future__ import annotations

import numpy as np

VERSION = "mo-aop-dvn-2026-v1"
PROFILES = {
    "SGF": {"wind_mph": 20, "rh_max": 25, "fm10_lt": 9, "minimum_hours": 1},
    "EAX": {"wind_mph": 20, "rh_max": 25, "fm10_lt": 9, "minimum_hours": 1},
    "PAH": {"wind_mph": 15, "rh_max": 25, "fm10_lt": 9, "minimum_hours": 1},
    "LSX": {"wind_mph": 15, "rh_max": 25, "fm10_lt": 9, "minimum_hours": 1},
    "DVN": {"gfdi_gt": 40, "minimum_hours": 1},
}
SOURCES = {"missouri": "https://www.weather.gov/media/sgf/fire/aop.pdf",
           "DVN": "https://www.weather.gov/dvn/headlines"}


def qualifying(office, rh_pct, wind_20ft_ms, fm10_pct, *, gfdi=None):
    """Float mask: 1 meets numerical criteria, 0 does not, NaN unknown.

    Fuel readiness and collaborative exceptions are separate metadata; the
    numerical screen cannot reproduce a human office's warning decision.
    """
    rh, wind, fm = np.broadcast_arrays(rh_pct, wind_20ft_ms, fm10_pct)
    offices = np.broadcast_to(np.asarray(office), rh.shape)
    out = np.full(rh.shape, np.nan, dtype="float32")
    for key, profile in PROFILES.items():
        selected = offices == key
        if key == "DVN":
            if gfdi is not None:
                index = np.broadcast_to(np.asarray(gfdi, float), rh.shape)
                valid = selected & np.isfinite(index)
                out[valid] = (index[valid] > profile["gfdi_gt"]).astype(float)
        else:
            valid = selected & np.isfinite(rh) & np.isfinite(wind) & np.isfinite(fm)
            out[valid] = ((rh[valid] <= profile["rh_max"]) &
                          (wind[valid] * 2.2369362921 >= profile["wind_mph"]) &
                          (fm[valid] < profile["fm10_lt"])).astype(float)
    return out


def any_hour(values, axis=0):
    """Any coincident hourly event; an unobserved hour prevents a false negative."""
    values = np.asarray(values, float)
    return np.where(np.any(values == 1, axis=axis), 1.,
                    np.where(np.all(np.isfinite(values), axis=axis), 0., np.nan))

