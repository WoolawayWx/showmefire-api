"""Tests for services/ensemble_fire_danger (Day-1 ensemble fire danger beta)."""
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from core.fire_danger import (MAX_DEMOTION_FRACTION, RULE_SPEC, _green_factor, calculate_fire_danger,
                              seasonal_dampening_adjustment)
from services.ensemble_fire_danger import core, grib_idx, members
from services.ensemble_fire_danger.fm_features import build_fm_frame
from services.ensemble_fire_danger.regrid import Regridder, cell_size_km
from services.rule_uncertainty import MISSING_CATEGORY, category_vectorized

THRESHOLDS = RULE_SPEC["thresholds"]
API_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = API_ROOT.parent / "model-training"


def categorize(fm, rh, wind):
    return category_vectorized(fm, rh, wind, THRESHOLDS)


class SeasonalDampeningParityTests(unittest.TestCase):
    """The vectorized dampening must match core.fire_danger's scalar one cell for cell."""

    def test_matches_scalar_across_green_factors(self):
        rng = np.random.default_rng(3)
        fm = rng.uniform(1, 20, 4000)
        rh = rng.uniform(5, 60, 4000)
        wind = rng.uniform(0, 40, 4000)
        base = categorize(fm, rh, wind)
        for gdd in (None, 0.0, 150.0, 400.0, 800.0, 1100.0, 1500.0):
            green = _green_factor(gdd)
            vectorized = core.seasonal_dampening_vectorized(base.astype("int16"), fm, rh, wind, green, THRESHOLDS,
                                                            MAX_DEMOTION_FRACTION)
            for i in range(len(fm)):
                expected = seasonal_dampening_adjustment(int(base[i]), fm[i], rh[i], wind[i], gdd)
                self.assertEqual(int(vectorized[i]), expected, msg=f"gdd={gdd} i={i}")

    def test_missing_passes_through(self):
        cat = np.array([MISSING_CATEGORY, 3], dtype="int16")
        out = core.seasonal_dampening_vectorized(cat, np.array([np.nan, 5.0]), np.array([np.nan, 20.0]),
                                                 np.array([np.nan, 16.0]), 1.0, THRESHOLDS, MAX_DEMOTION_FRACTION)
        self.assertEqual(int(out[0]), MISSING_CATEGORY)


class PeakCategoryTests(unittest.TestCase):
    def test_matches_scalar_rule_hourly_max(self):
        rng = np.random.default_rng(5)
        fm = rng.uniform(3, 18, (12, 6, 7))
        rh = rng.uniform(10, 70, (12, 6, 7))
        wind = rng.uniform(0, 35, (12, 6, 7))
        peak = core.peak_category(fm, rh, wind, categorize)
        for y in range(6):
            for x in range(7):
                expected = max(calculate_fire_danger(fm[h, y, x], rh[h, y, x], wind[h, y, x]) for h in range(12))
                self.assertEqual(int(peak[y, x]), expected)

    def test_snow_forces_low(self):
        fm = np.full((2, 2, 2), 5.0)
        rh = np.full((2, 2, 2), 15.0)
        wind = np.full((2, 2, 2), 30.0)
        snow = np.array([[True, False], [False, False]])
        peak = core.peak_category(fm, rh, wind, categorize, snow_mask=snow)
        self.assertEqual(int(peak[0, 0]), 0)
        self.assertEqual(int(peak[1, 1]), 4)


class ProbabilityTests(unittest.TestCase):
    def test_all_members_same_category(self):
        peaks = np.full((5, 4, 4), 3, dtype="int8")
        for k, expected in ((1, 1.0), (2, 1.0), (3, 1.0), (4, 0.0)):
            p = core.exceedance_probability(peaks, [1] * 5, k)
            self.assertTrue(np.allclose(p, expected))

    def test_weights_and_missing_members(self):
        peaks = np.zeros((3, 1, 2), dtype="int8")
        peaks[0, 0, :] = 2
        peaks[2, 0, 1] = core.MISSING  # member 3 missing at cell 1
        p = core.exceedance_probability(peaks, [2.0, 1.0, 1.0], 2)
        self.assertAlmostEqual(float(p[0, 0]), 0.5)          # 2 / (2+1+1)
        self.assertAlmostEqual(float(p[0, 1]), 2.0 / 3.0)    # 2 / (2+1)

    def test_neighborhood_spreads_within_radius_only(self):
        peaks = np.zeros((2, 21, 21), dtype="int8")
        peaks[0, 10, 10] = 4
        p = core.exceedance_probability(peaks, [1, 1], 4, radius_cells=3)
        self.assertAlmostEqual(float(p[10, 13]), 0.5)
        self.assertAlmostEqual(float(p[10, 14]), 0.0)
        self.assertAlmostEqual(float(p[12, 12]), 0.5)  # sqrt(8) < 3

    def test_products_monotone_and_categorical(self):
        rng = np.random.default_rng(1)
        peaks = rng.integers(0, 5, (10, 30, 30)).astype("int8")
        out = core.probability_products(peaks, [1] * 10, radius_cells=2, smooth_sigma=1.5)
        for name in ("point", "neighborhood"):
            stack = out[name]
            self.assertTrue(np.all(np.diff(stack, axis=0) <= 1e-12), name)
            self.assertTrue(np.all((stack >= 0) & (stack <= 1)))
        self.assertTrue(np.all(out["neighborhood"] >= out["point"] - 1e-9))
        cat = out["categorical"]
        for k in core.CATEGORY_IDS:
            self.assertTrue(np.all(out["point"][k - 1][cat >= k] >= 0.5))

    def test_median_equivalence_of_default_threshold(self):
        peaks = np.array([0, 1, 2, 2, 3], dtype="int8").reshape(5, 1, 1)
        out = core.probability_products(peaks, [1] * 5, radius_cells=0, smooth_sigma=0)
        self.assertEqual(float(out["categorical"][0, 0]), 2.0)  # median member category

    def test_calibration_apply_and_identity(self):
        p = np.array([0.0, 0.25, 0.5, 1.0, np.nan])
        self.assertTrue(np.array_equal(core.apply_calibration(p, None), p, equal_nan=True))
        knots = {"x": [0.0, 0.5, 1.0], "y": [0.0, 0.2, 0.9]}
        out = core.apply_calibration(p, knots)
        self.assertAlmostEqual(float(out[1]), 0.1)
        self.assertTrue(np.isnan(out[4]))

    def test_enforce_monotone(self):
        stack = np.array([[0.3], [0.5], [0.1], [0.2]])
        self.assertTrue(np.allclose(core.enforce_monotone(stack).ravel(), [0.3, 0.3, 0.1, 0.1]))


class SyntheticDrawTests(unittest.TestCase):
    def test_recovers_mean_and_spread(self):
        shape = (2, 40, 40)
        t_mean, t_sprd = np.full(shape, 300.0), np.full(shape, 2.0)
        td_mean, td_sprd = np.full(shape, 285.0), np.full(shape, 1.5)
        w_mean, w_sprd = np.full(shape, 6.0), np.full(shape, 1.0)
        draws = list(core.synthetic_draws(t_mean, t_sprd, td_mean, td_sprd, w_mean, w_sprd, n_draws=400, seed=7,
                                          rho_t_td=-0.3, spatial_sigma=3.0))
        t = np.stack([d[0][0, 20, 20] for d in draws])
        self.assertAlmostEqual(float(t.mean()), 300.0, delta=0.4)
        self.assertAlmostEqual(float(t.std()), 2.0, delta=0.3)
        for t_draw, td_draw, w_draw in draws[:20]:
            self.assertTrue(np.all(td_draw <= t_draw))
            self.assertTrue(np.all(w_draw >= 0))

    def test_deterministic_for_seed(self):
        args = [np.full((1, 5, 5), v) for v in (300.0, 1.0, 290.0, 1.0, 5.0, 1.0)]
        a = next(core.synthetic_draws(*args, n_draws=1, seed=11, rho_t_td=0.0, spatial_sigma=1.0))
        b = next(core.synthetic_draws(*args, n_draws=1, seed=11, rho_t_td=0.0, spatial_sigma=1.0))
        self.assertTrue(np.array_equal(a[0], b[0]))

    def test_pooled_spread_includes_between_ensemble_variance(self):
        mean, spread = core.pooled_mean_spread([np.array([0.0]), np.array([2.0])],
                                               [np.array([1.0]), np.array([1.0])], [1, 1])
        self.assertAlmostEqual(float(mean[0]), 1.0)
        self.assertAlmostEqual(float(spread[0]), np.sqrt(2.0))

    def test_anchored_fm(self):
        member, control, prod = np.array([10.0]), np.array([8.0]), np.array([6.0])
        self.assertAlmostEqual(float(core.anchored_fm(member, control, prod, (1, 40))[0]), 8.0)
        self.assertAlmostEqual(float(core.anchored_fm(member, None, prod, (1, 40))[0]), 10.0)


class GribIdxTests(unittest.TestCase):
    NAM_IDX = "\n".join([
        "616:706210365:d=2026092112:TMP:2 m above ground:12 hour fcst:",
        "618:710339869:d=2026092112:DPT:2 m above ground:12 hour fcst:",
        "624.1:719894045:d=2026092112:UGRD:10 m above ground:12 hour fcst:",
        "624.2:719894045:d=2026092112:VGRD:10 m above ground:12 hour fcst:",
        "625:723608570:d=2026092112:TMP:surface:12 hour fcst:",
        "808:1046864594:d=2026092112:APCP:surface:9-12 hour acc fcst:",
    ])

    def test_submessages_share_one_range(self):
        entries = grib_idx.parse_idx(self.NAM_IDX)
        self.assertEqual(len(entries), 6)
        ranges = grib_idx.byte_ranges(entries)
        self.assertEqual(ranges[719894045], (719894045, 723608569))
        self.assertIsNone(ranges[1046864594][1])
        matches = grib_idx.select(entries, grib_idx.DEFAULT_SEARCHES)
        self.assertEqual([e.number for e in matches["u10"]], ["624.1"])
        self.assertEqual([e.number for e in matches["v10"]], ["624.2"])
        self.assertEqual(len(matches["t2m"]), 1)  # surface TMP excluded
        self.assertEqual(grib_idx._sub_index(matches["v10"][0]), 1)

    def test_ensprod_searches(self):
        idx = "\n".join([
            "40:1:d=2026092112:TMP:2 m above ground:18 hour fcst:wt ens mean",
            "42:2:d=2026092112:TMP:2 m above ground:18 hour fcst:ens spread",
            "172:3:d=2026092112:JFWPRB:10 m above ground:18 hour fcst:prob >=9 <20:prob fcst 0/14",
        ])
        entries = grib_idx.parse_idx(idx)
        self.assertEqual(len(grib_idx.select(entries, grib_idx.ENSPROD_SEARCHES["mean"])["t2m"]), 1)
        self.assertEqual(len(grib_idx.select(entries, grib_idx.ENSPROD_SEARCHES["sprd"])["t2m"]), 1)
        self.assertEqual(len(grib_idx.select(entries, grib_idx.ENSPROD_SEARCHES["prob"])["jfwprb"]), 1)

    def test_hourly_precip_layouts(self):
        one = np.ones((1, 1))
        # NAM 3-hour resetting buckets: 9-10, 9-11, 9-12
        nam = {10: {(9, 10): one * 1}, 11: {(9, 11): one * 3}, 12: {(9, 12): one * 4}}
        out = grib_idx.hourly_precip(nam, [10, 11, 12], (1, 1))
        self.assertEqual([float(out[f][0, 0]) for f in (10, 11, 12)], [1.0, 2.0, 1.0])
        # HRRR-style explicit 1-hour buckets win over run totals
        hrrr = {5: {(4, 5): one * 0.5, (0, 5): one * 9}}
        self.assertEqual(float(grib_idx.hourly_precip(hrrr, [5], (1, 1))[5][0, 0]), 0.5)
        # only a long window -> spread evenly
        only = {6: {(0, 6): one * 6}}
        self.assertEqual(float(grib_idx.hourly_precip(only, [6], (1, 1))[6][0, 0]), 1.0)
        # nothing published -> NaN (caller flags precip_missing)
        self.assertTrue(np.isnan(grib_idx.hourly_precip({}, [7], (1, 1))[7][0, 0]))

    def test_cycle_leads_and_url(self):
        anchor = datetime(2026, 9, 22, 12)
        self.assertEqual(grib_idx.cycle_leads(anchor, datetime(2026, 9, 22, 6), 4, 6), [10, 11, 12])
        url = grib_idx.render_url("x/{date}/t{hh}z.f{fxx:03d}", datetime(2026, 9, 22, 6), 7)
        self.assertEqual(url, "x/20260922/t06z.f007")


class ResolvePlanTests(unittest.TestCase):
    def setUp(self):
        self.config = members.load_config()
        self.anchor = datetime(2026, 9, 22, 12)

    def test_all_available_uses_target_cycles(self):
        plan = members.resolve_plan(self.config, self.anchor, lambda url: True)
        by_id = {p.member_id: p for p in plan}
        self.assertEqual(by_id["hrrr_0"].cycle, self.anchor)
        self.assertEqual(by_id["hrrr_m6"].cycle, self.anchor - timedelta(hours=6))
        self.assertEqual(by_id["hrrr_m12"].leads[0], 16)
        self.assertTrue(all(p.status == "resolved" for p in plan))

    def test_fallback_never_duplicates_a_claimed_cycle(self):
        # 12z NAM not published yet: nam_0 falls back to 06z, so nam_m6 must move to 00z.
        plan = members.resolve_plan(self.config, self.anchor, lambda url: "t12z" not in url or "nam" not in url)
        by_id = {p.member_id: p for p in plan}
        self.assertEqual(by_id["nam_0"].cycle.hour, 6)
        self.assertEqual(by_id["nam_m6"].cycle.hour, 0)

    def test_emulated_live_latency(self):
        emulate = self.anchor + timedelta(hours=float(self.config["live_run_offset_hours"]))
        plan = members.resolve_plan(self.config, self.anchor, lambda url: True, emulate_live_at=emulate)
        by_id = {p.member_id: p for p in plan}
        self.assertEqual(by_id["hrrr_0"].cycle.hour, 12)
        self.assertEqual(by_id["href"].cycle.hour, 0)  # HREF 12z (~3.1h) is not out by anchor+3h

    def test_archive_url_used_for_capture(self):
        seen = []
        members.resolve_plan(self.config, self.anchor, lambda url: seen.append(url) or True, use_archive=True,
                             kinds=("member",))
        self.assertTrue(any("noaa-nam-pds" in url for url in seen))
        self.assertFalse(any("nomads.ncep.noaa.gov/pub/data/nccf/com/nam" in url for url in seen))

    def test_unavailable_member_recorded(self):
        plan = members.resolve_plan(self.config, self.anchor, lambda url: "hiresw" not in url)
        arw = next(p for p in plan if p.member_id == "arw_0")
        self.assertEqual(arw.status, "unavailable")
        self.assertIn("not published", arw.reason)


class RegridTests(unittest.TestCase):
    def test_linear_field_is_exact_and_outside_is_nan(self):
        yy, xx = np.mgrid[0:30, 0:40]
        src_lat, src_lon = 36.0 + yy * 0.03, -95.0 + xx * 0.03 + yy * 0.002
        dst_lat, dst_lon = np.array([[36.3, 36.5], [37.0, 40.0]]), np.array([[-94.5, -94.2], [-94.0, -94.0]])
        with tempfile.TemporaryDirectory() as tmp:
            regridder = Regridder(src_lat, src_lon, dst_lat, dst_lon, cache_dir=Path(tmp))
            out = regridder(2.0 * src_lat - 3.0 * src_lon)
            cached = Regridder(src_lat, src_lon, dst_lat, dst_lon, cache_dir=Path(tmp))
            self.assertEqual(len(list(Path(tmp).glob("regrid_*.npz"))), 1)
            self.assertTrue(np.array_equal(cached.vertices, regridder.vertices))
        expected = 2.0 * dst_lat - 3.0 * dst_lon
        self.assertTrue(np.allclose(out[:1], expected[:1], atol=1e-8))
        self.assertTrue(np.isnan(out[1, 1]))  # lat 40 is outside the source hull

    def test_cell_size(self):
        yy, xx = np.mgrid[0:10, 0:10]
        lat, lon = 38.0 + yy * 0.027, -92.0 + xx * 0.0344
        self.assertAlmostEqual(cell_size_km(lat, lon), 3.0, delta=0.1)


class FmFeatureTests(unittest.TestCase):
    def test_frame_columns_and_precip_lag(self):
        shape = (3, 3)
        hist = [np.zeros(shape), np.full(shape, 2.0)]
        df = build_fm_frame(np.full(shape, 25.0), np.full(shape, 30.0), np.full(shape, 4.0), 15, 9,
                            [np.full(shape, 24.0)], [np.full(shape, 32.0)], hist, day_of_year=265)
        self.assertEqual(len(df), 9)
        self.assertTrue(np.allclose(df["precip_1h"], 2.0))      # latest history hour, not the current one
        self.assertTrue(np.allclose(df["hours_since_rain"], 0))
        self.assertTrue(np.allclose(df["temp_mean_3h"], 24.5))
        self.assertTrue(np.allclose(df["emc_baseline"], 6.0))

    def test_parity_with_daily_forecast_predict_fm_grid(self):
        try:
            import forecast.DailyForecast as daily
        except Exception as error:  # needs a registered stable fuel_moisture model
            self.skipTest(f"DailyForecast not importable here: {error}")
        from services.ensemble_fire_danger.fm_features import predict_fm_hourly
        import pandas as pd
        rng = np.random.default_rng(2)
        hours = 4
        temp = rng.uniform(10, 30, (hours, 5, 6))
        rh = rng.uniform(15, 90, (hours, 5, 6))
        ws = rng.uniform(0, 10, (hours, 5, 6))
        precip = rng.uniform(0, 1, (hours, 5, 6))
        valid = [pd.Timestamp("2026-09-22T16:00Z") + pd.Timedelta(hours=h) for h in range(hours)]
        ours = predict_fm_hourly(daily.FM_MODEL, daily.FEATURES, temp, rh, ws, precip, valid)
        t_hist, rh_hist, p_hist = [], [], []
        for i, v in enumerate(valid):
            fm, _, _ = daily.predict_fm_grid(temp[i], rh[i], ws[i], v.hour, v.month, t_hist, rh_hist, p_hist,
                                             day_of_year=v.dayofyear)
            self.assertTrue(np.allclose(ours[i], fm, atol=1e-5))
            t_hist.append(temp[i]); rh_hist.append(rh[i]); p_hist.append(precip[i])


class RenderSmokeTests(unittest.TestCase):
    def test_five_graphics_render_at_house_size(self):
        try:
            import cartopy  # noqa: F401
            from PIL import Image
            from services.ensemble_fire_danger.render import PIXEL_H, PIXEL_W, Renderer, descriptions
        except Exception as error:
            self.skipTest(f"plotting stack unavailable: {error}")
        yy, xx = np.mgrid[0:60, 0:70]
        lat, lon = 35.7 + yy * 0.087, -95.9 + xx * 0.1
        renderer = Renderer(lat, lon)
        text = descriptions(window_label="10:00-21:00 CT", members_text="10 members", calibrated_text="raw",
                            track_label="test")
        prob = np.clip((xx / 70.0) ** 2, 0, 1)
        with tempfile.TemporaryDirectory() as tmp:
            paths = [renderer.categorical(np.clip(xx // 14, 0, 4).astype(float), Path(tmp) / "cat.png",
                                          subtitle="s", description=text["categorical"], run_date=datetime.now())]
            for k in core.CATEGORY_IDS:
                paths.append(renderer.probability(k, prob * (1 - 0.2 * k), Path(tmp) / f"p{k}.png", subtitle="s",
                                                  description=text["probability"], run_date=datetime.now()))
            renderer.probability(4, np.zeros_like(prob), Path(tmp) / "empty.png", subtitle="s",
                                 description=text["probability"], run_date=datetime.now())
            for path in paths:
                with Image.open(path) as image:
                    self.assertEqual(image.size, (PIXEL_W, PIXEL_H))
                self.assertTrue(path.with_suffix(".webp").exists())


class ContractMirrorTests(unittest.TestCase):
    MIRRORED = ("core.py", "grib_idx.py", "members.py", "regrid.py", "tracks.py", "fm_features.py",
                "member_config.json")

    def test_training_copies_are_byte_identical(self):
        training = TRAINING_ROOT / "ensemble_fire_danger"
        if not training.exists():
            self.skipTest("model-training checkout not alongside api/")
        ours = API_ROOT / "services" / "ensemble_fire_danger"
        for name in self.MIRRORED:
            self.assertEqual((ours / name).read_bytes().replace(b"\r\n", b"\n"),
                             (training / name).read_bytes().replace(b"\r\n", b"\n"), msg=name)

    def test_registered_in_contract_mirrors(self):
        mirrors = json.loads((API_ROOT / "core" / "contract_mirrors.json").read_text())
        paths = {pair["api_path"] for pair in mirrors["pairs"]}
        for name in self.MIRRORED:
            self.assertIn(f"services/ensemble_fire_danger/{name}", paths)


class GridContractTests(unittest.TestCase):
    def test_api_bounds_crop_matches_county_cells_grid(self):
        """The ensemble's fallback target grid (HRRR buffered crop -> mo_bounds)
        must be the county_cells.json grid, or county summaries misassign."""
        from core.risk_fusion_county_reference import county_cells
        cached = sorted((API_ROOT.parent / "data" / "cache" / "ensemble" / "runs").glob("member_hrrr_0_*.nc"))
        if not cached:
            self.skipTest("no cached HRRR member run to derive the grid from")
        import xarray as xr
        with xr.open_dataset(cached[-1]) as ds:
            rows, cols = members.api_bounds_slices(ds["latitude"].values, ds["longitude"].values)
            shape = [rows.stop - rows.start, cols.stop - cols.start]
        self.assertEqual(shape, list(county_cells()["grid_shape"]))


if __name__ == "__main__":
    unittest.main()
