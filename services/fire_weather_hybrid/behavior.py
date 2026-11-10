"""Vectorized Rothermel surface-fire behavior on the forecast grid.

A NumPy port of pyretechnics' surface-fire path (Rothermel 1972, Albini 1976,
Scott & Burgan 2005 dynamic fuel models, Andrews 2012 wind adjustment) so a
statewide 1 km grid runs in seconds without per-cell Python calls. Parity with
pyretechnics is tested cell by cell in model-training.

Units follow pyretechnics: wind at 20 ft in m/min, spread rate in m/min,
fireline intensity in kW/m, flame length in m. Moisture is a fraction (0.08 = 8%).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_TABLE = json.loads((Path(__file__).with_name("fuel_models.json")).read_text())["models"]
FT_TO_M = 0.30478512648582745
M_TO_FT = 3.281
BTU_FT_S_TO_KW_M = 3.46165186
RHO_P, S_T, S_E = 32.0, 0.0555, 0.01


def _size_class(sigma):
    return np.select([sigma >= 1200, sigma >= 192, sigma >= 96, sigma >= 48, sigma >= 16], [1., 2., 3., 4., 5.], 6.)


def _model(code):
    spec = _TABLE.get(str(int(code)))
    if spec is None or not spec["burnable"]:
        return None
    w = np.asarray(spec["w_o_lb_ft2"], float)
    sigma = np.asarray(spec["sigma_ft2_ft3"], float)
    dynamic = bool(spec["dynamic"])
    if dynamic:  # dead herbaceous takes the live herbaceous SAV (pyretechnics expand_compact_fuel_model)
        sigma = sigma.copy(); sigma[3] = sigma[4]
    with np.errstate(divide="ignore"):
        exp_a = np.where(sigma > 1e-12, np.exp(np.array([-138.]*4+[-500.]*2)/np.where(sigma > 0, sigma, 1.)), 0.)
    mx = spec["mx_dead"]
    return {"delta": spec["delta_ft"], "mx_dead": mx, "mx_dead_herb": mx if dynamic else 0., "h": spec["h_btu_lb"],
            "w": w, "sigma": sigma, "dynamic": dynamic, "exp_a": exp_a, "size": _size_class(sigma)}


def _one_model(m, moisture, wind_20ft_m_min, wind_from_deg, slope, aspect_deg, canopy_cover, canopy_height_ft):
    """All arrays shape (n,); moisture shape (6, n) as fractions."""
    n = moisture.shape[1]
    mf = moisture.copy()
    w = np.repeat(m["w"][:, None], n, axis=1)
    if m["dynamic"]:
        green = np.clip(mf[4]/0.9-1/3, 0., 1.)
        w[3] = m["w"][4]*(1-green); w[4] = m["w"][4]*green
        mf[3] = mf[0]
    sigma = m["sigma"][:, None]
    # Weighting factors
    a = sigma*w/RHO_P
    a0, a1 = a[:4].sum(0), a[4:].sum(0)
    f = np.zeros_like(a)
    f[:4] = np.divide(a[:4], a0, out=np.zeros_like(a[:4]), where=a0 > 1e-12)
    f[4:] = np.divide(a[4:], a1, out=np.zeros_like(a[4:]), where=a1 > 1e-12)
    at = a0+a1
    fi0 = np.divide(a0, at, out=np.zeros(n), where=at > 1e-12); fi1 = np.divide(a1, at, out=np.zeros(n), where=at > 1e-12)
    size = m["size"]
    g = np.zeros_like(f)
    for j in range(6):
        group = range(4) if j < 4 else range(4, 6)
        g[j] = sum(f[k] for k in group if size[k] == size[j])
    # Live moisture of extinction (Albini 1976)
    load = w*m["exp_a"][:, None]
    dead_load, live_load = load[:4].sum(0), load[4:].sum(0)
    dead_moist = (mf[:4]*load[:4]).sum(0)
    ok = (dead_load > 1e-12) & (live_load > 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        mx_live = np.where(ok, np.maximum(m["mx_dead"], 2.9*(dead_load/live_load)*(1-(dead_moist/dead_load)/m["mx_dead"])-0.226), m["mx_dead"])
    mx = np.stack([np.full(n, m["mx_dead"])]*3+[np.full(n, m["mx_dead_herb"]), mx_live, mx_live])
    # Rothermel no-wind, no-slope
    sig_p = fi0*(f[:4]*sigma[:4]).sum(0)+fi1*(f[4:]*sigma[4:]).sum(0)
    beta = (w/RHO_P).sum(0)/m["delta"]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        beta_op = np.where(sig_p > 0, 3.348*np.power(sig_p, -0.8189), 1.)
        def eta_s(fsum): return np.where(fsum > 0, 0.174*np.power(np.where(fsum > 0, S_E*fsum, 1.), -0.19), 1.)
        def eta_m(mf_i, mx_i):
            r = np.minimum(1., np.divide(mf_i, mx_i, out=np.zeros(n), where=mx_i > 0))
            return np.where(mx_i > 0, 1-2.59*r+5.11*r*r-3.52*r**3, 0.)
        fs0, fs1 = f[:4].sum(0), f[4:].sum(0)
        btus = ((g[:4]*w[:4]*(1-S_T)).sum(0)*m["h"]*fs0*eta_m((f[:4]*mf[:4]).sum(0), (f[:4]*mx[:4]).sum(0))*eta_s(fs0) +
                (g[4:]*w[4:]*(1-S_T)).sum(0)*m["h"]*fs1*eta_m((f[4:]*mf[4:]).sum(0), (f[4:]*mx[4:]).sum(0))*eta_s(fs1))
        A = np.where(sig_p > 0, 133.0*np.power(sig_p, -0.7913), 0.)
        Bp = np.power(sig_p, 1.5); C = beta/beta_op
        gamma = Bp/(495.0+0.0594*Bp)*np.power(C, A)*np.exp(A*(1-C))
        i_r = btus*gamma
        xi = np.exp((0.792+0.681*np.sqrt(sig_p))*(beta+0.1))/(192.0+0.2595*sig_p)
        rho_b = w.sum(0)/m["delta"]
        eps = np.where(sigma > 0, np.exp(-138.0/np.where(sigma > 0, sigma, 1.)), 0.)
        q_ig = np.where(mf > 0, 250.0+1116.0*mf, 0.)
        sink = rho_b*(fi0*(f[:4]*eps[:4]*q_ig[:4]).sum(0)+fi1*(f[4:]*eps[4:]*q_ig[4:]).sum(0))
        r0 = np.where(sink > 0, i_r*xi/sink, 0.)                       # ft/min
        t_res = np.where(sig_p > 0, 384.0/sig_p, 0.)
        i_s = i_r*r0*t_res/60.0                                          # Btu/ft/s
        u_max = 0.9*i_r*FT_TO_M                                          # m/min
        Bw = 0.02526*np.power(sig_p, 0.54); Cw = 7.47*np.exp(-0.133*np.power(sig_p, 0.55))
        Ew = 0.715*np.exp(-3.59*(sig_p*1e-4)); Fw = np.power(beta/beta_op, Ew)
        base_ros, base_i = r0*FT_TO_M, i_s*BTU_FT_S_TO_KW_M
        phis_g = np.where(beta > 0, 5.275*np.power(beta, -0.3), 0.)
        phiw_s = np.where(Fw > 0, (Cw/Fw)*np.power(M_TO_FT, Bw), 0.)
        ws_s = np.where(Bw > 0, FT_TO_M*np.power(Fw/Cw, 1/Bw), 0.)
        # Midflame wind (Andrews 2012)
        # Deviation from pyretechnics: 1 km cells average trace canopy (0.1% cover, a few cm
        # tall) into open ground, where Andrews' sheltered formula divides by ~0 and inflates
        # wind. As BehavePlus does, fuel counts as sheltered only above 5% cover, and
        # sheltering can only reduce wind below the unsheltered value.
        sheltered = (canopy_cover > 0.05) & (canopy_height_ft > 0)
        ch = np.where(sheltered, canopy_height_ft, 1.)
        waf_s = 0.555/(np.sqrt(np.where(sheltered, canopy_cover, 1.)/3.0*ch)*np.log((20.0+0.36*ch)/(0.13*ch)))
        d = m["delta"]
        waf_u = 1.83/np.log((20.0+0.36*d)/(0.13*d)) if d > 0 else 0.
        midflame = wind_20ft_m_min*np.where(sheltered, np.minimum(waf_s, waf_u), waf_u)
        # Wind and slope on the slope-tangential plane
        down, up = np.radians((wind_from_deg+180.) % 360.), np.radians((aspect_deg+180.) % 360.)
        wx, wy = midflame*np.sin(down), midflame*np.cos(down)
        sx, sy = slope*np.sin(up), slope*np.cos(up)
        w3 = np.stack([wx, wy, wx*sx+wy*sy]); s3 = np.stack([sx, sy, slope*slope])
        wmag, smag = np.linalg.norm(w3, axis=0), np.linalg.norm(s3, axis=0)
        phi_w = phiw_s*np.power(wmag, Bw)
        phi_s = phis_g*slope*slope
        unit_w = np.where((phi_w > 0) & (wmag > 0), w3/np.where(wmag > 0, wmag, 1.), w3)
        unit_s = np.where((phi_s > 0) & (smag > 0), s3/np.where(smag > 0, smag, 1.), s3)
        phi_e = np.linalg.norm(phi_w*unit_w+phi_s*unit_s, axis=0)
        eff = ws_s*np.power(phi_e, np.where(Bw > 0, 1/Bw, 0.))
        phi_e = np.where(eff > u_max, phiw_s*np.power(u_max, Bw), phi_e)
        ros = base_ros*(1+phi_e)
        intensity = base_i*(1+phi_e)
        flame = 0.07747042253266703*np.power(np.maximum(intensity, 0.), 0.46)
    spreads = base_ros > 1e-6  # m/min; float round-off above extinction moisture is not spread
    return (np.where(spreads, ros, 0.), np.where(spreads, intensity, 0.), np.where(spreads, flame, 0.))


def surface_fire(fuel_code, fm1, fm10, fm100, live_herb, live_woody, wind_20ft_ms, wind_from_deg,
                 slope_deg, aspect_deg, canopy_cover_pct, canopy_height_m):
    """Head-fire spread (m/min), intensity (kW/m) and flame length (m) per cell.

    Moisture inputs are percent. Non-burnable or unknown fuel codes return NaN;
    a burnable cell whose fuel is too wet to spread returns zero.
    """
    shape = np.shape(fuel_code)
    arr = lambda v: np.broadcast_to(np.asarray(v, float), shape).ravel()
    code = arr(fuel_code)
    out = [np.full(code.shape, np.nan) for _ in range(3)]
    moisture = np.stack([arr(fm1), arr(fm10), arr(fm100), arr(fm1), arr(live_herb), arr(live_woody)])/100.
    wind = np.maximum(arr(wind_20ft_ms), 0.)*60.
    slope = np.tan(np.radians(np.clip(arr(slope_deg), 0., 89.)))
    cc, ch = np.clip(arr(canopy_cover_pct)/100., 0., 1.), np.maximum(arr(canopy_height_m), 0.)*M_TO_FT
    usable = np.isfinite(code) & np.isfinite(moisture[:3]).all(0) & np.isfinite(wind)
    for value in np.unique(code[usable]):
        m = _model(value)
        if m is None:
            continue
        idx = np.flatnonzero(usable & (code == value))
        result = _one_model(m, moisture[:, idx], wind[idx], np.nan_to_num(arr(wind_from_deg)[idx]), np.nan_to_num(slope[idx]),
                            np.nan_to_num(arr(aspect_deg)[idx]), np.nan_to_num(cc[idx]), np.nan_to_num(ch[idx]))
        for target, values in zip(out, result):
            target[idx] = values
    return {"ros_m_min": out[0].reshape(shape), "intensity_kw_m": out[1].reshape(shape), "flame_length_m": out[2].reshape(shape)}


# Missouri live-fuel moisture by month (percent). Herbaceous 30% = fully cured
# (dynamic grass carries its load as dead fuel). These are documented
# assumptions, not measurements; replace with observed live fuel moisture when available.
LIVE_MOISTURE_BY_MONTH = {1: (30, 60), 2: (30, 60), 3: (30, 70), 4: (60, 100), 5: (120, 130), 6: (120, 130),
                          7: (90, 120), 8: (75, 110), 9: (60, 100), 10: (40, 80), 11: (30, 60), 12: (30, 60)}


def live_moisture(month):
    herb, woody = LIVE_MOISTURE_BY_MONTH[int(month)]
    return float(herb), float(woody)


def wind_from_degrees(u, v):
    return (np.degrees(np.arctan2(-np.asarray(u, float), -np.asarray(v, float)))+360.) % 360.


def aspect_degrees(aspect_sin, aspect_cos):
    return (np.degrees(np.arctan2(np.asarray(aspect_sin, float), np.asarray(aspect_cos, float)))+360.) % 360.
