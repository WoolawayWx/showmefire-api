import unittest
from datetime import datetime, timedelta, timezone

from services.station_precip import derive_all, derive_station_precip

NOW = datetime(2026, 10, 15, 0, 0, tzinfo=timezone.utc)


def _times(count, step_minutes=60):
    return [(NOW - timedelta(minutes=step_minutes * (count - 1 - i))).isoformat() for i in range(count)]


def _station(mnet, observations, state="MO", stid="TEST"):
    times = observations.pop("date_time", None) or _times(len(next(iter(observations.values()))))
    return {"STID": stid, "NAME": stid, "STATE": state, "LATITUDE": "38.0", "LONGITUDE": "-92.0",
            "MNET_ID": str(mnet), "OBSERVATIONS": {"date_time": times, **observations}}


def _asos_day(rain=None, minute=52, skip_hours=()):
    """25 h of ASOS/AWOS data, as Synoptic reports ``precip_accum_one_hour``.

    The value accumulates from the routine report and resets right after the next one, so a
    rainy hour is a 5-minute-style ramp up to the routine report at ``:minute`` and then a
    ~0 reset value just after it (e.g. 0.72 in at 11:49, 0.001 at 11:51). Dry hours have no
    precip group at all. ``rain`` maps hours-ago (0 = newest routine report) to the hour's
    final value; ``skip_hours`` are hours with no observation at all.
    """
    rain = rain or {}
    newest = NOW - timedelta(hours=1) + timedelta(minutes=minute)
    obs = [newest - timedelta(hours=k) for k in range(25) if k not in skip_hours]
    reports = []
    for k, peak in rain.items():
        end = newest - timedelta(hours=k)
        reports += [(end - timedelta(minutes=42), 0.3 * peak), (end - timedelta(minutes=22), 0.6 * peak),
                    (end, peak)]
        if k >= 1:
            reports.append((end + timedelta(minutes=3), 0.001))  # reset value after the routine report
    rows = sorted([(t, 60.0, None) for t in obs] + [(t, None, v) for t, v in reports], key=lambda row: row[0])
    return _station(1, {
        "date_time": [t.isoformat() for t, _, _ in rows],
        "air_temp_set_1": [a for _, a, _ in rows],
        "precip_accum_one_hour_set_1": [v for _, _, v in rows],
    }, stid="KTST")


class AsosAccumulationTests(unittest.TestCase):
    def _derive(self, station, now=NOW):
        return derive_station_precip(station, now, one_hour_sensors={"KTST"})

    def test_dry_day_is_zero_and_complete(self):
        record = self._derive(_asos_day())
        self.assertEqual(record["precip_24h_in"], 0.0)
        self.assertEqual(record["precip_1h_in"], 0.0)
        self.assertEqual(record["precip_24h_hours_missing"], 0)
        self.assertIn("asos_accum_one_hour", record["sources"])

    def test_ramp_within_an_hour_is_counted_once(self):
        self.assertAlmostEqual(self._derive(_asos_day(rain={5: 0.25}))["precip_24h_in"], 0.25, delta=0.005)

    def test_hours_are_summed_and_1h_is_the_latest_hour(self):
        record = self._derive(_asos_day(rain={0: 0.10, 5: 0.25, 23: 0.05}))
        self.assertAlmostEqual(record["precip_24h_in"], 0.40, delta=0.005)
        self.assertAlmostEqual(record["precip_1h_in"], 0.10, delta=0.005)

    def test_consecutive_rainy_hours_each_count(self):
        record = self._derive(_asos_day(rain={0: 0.05, 1: 0.30, 2: 0.20}))
        self.assertAlmostEqual(record["precip_24h_in"], 0.55, delta=0.005)

    def test_rain_older_than_24h_is_excluded(self):
        self.assertEqual(self._derive(_asos_day(rain={24: 0.50}))["precip_24h_in"], 0.0)

    def test_1h_is_rolling_not_since_last_reset(self):
        # 0.72 fell in the 30 minutes before a report at the end of the window, then the
        # accumulator reset: the last hour is 0.72 even though the latest value is ~0.
        newest = NOW - timedelta(minutes=11)
        rows = [(newest - timedelta(minutes=m), v) for m, v in ((29, 0.07), (20, 0.37), (10, 0.7), (2, 0.72))]
        rows += [(newest, 0.001)]
        obs = [newest - timedelta(minutes=5 * i) for i in range(0, 300)]
        rows += [(t, None) for t in obs]
        rows.sort(key=lambda row: row[0])
        station = _station(1, {"date_time": [t.isoformat() for t, _ in rows],
                               "air_temp_set_1": [60.0] * len(rows),
                               "precip_accum_one_hour_set_1": [v for _, v in rows]}, stid="KTST")
        self.assertAlmostEqual(self._derive(station)["precip_1h_in"], 0.72, delta=0.01)

    def test_hourly_only_station_with_rising_values_sums_each_hour(self):
        # KTBN on 2026-10-14: hourly reports 0.06, 0.01, 0.36, 0.57, 0.22 (no drop between the
        # rising ones) matched the NWS 24 h group of 1.22.
        newest = NOW - timedelta(minutes=5)
        hours = [(newest - timedelta(hours=4 - i), v) for i, v in enumerate((0.06, 0.01, 0.36, 0.57, 0.22))]
        obs = [newest - timedelta(hours=k) for k in range(25)]
        rows = sorted([(t, 60.0, None) for t in obs] + [(t, None, v) for t, v in hours], key=lambda row: row[0])
        station = _station(1, {"date_time": [t.isoformat() for t, _, _ in rows],
                               "air_temp_set_1": [a for _, a, _ in rows],
                               "precip_accum_one_hour_set_1": [v for _, _, v in rows]}, stid="KTST")
        self.assertAlmostEqual(self._derive(station)["precip_24h_in"], 1.22, delta=0.005)

    def test_awos_routine_minute_does_not_matter(self):
        record = self._derive(_asos_day(rain={0: 0.04, 3: 0.06}, minute=15))
        self.assertAlmostEqual(record["precip_24h_in"], 0.10, delta=0.005)
        self.assertEqual(record["precip_24h_hours_missing"], 0)

    def test_too_many_missing_hours_withholds_total(self):
        record = self._derive(_asos_day(rain={4: 0.1, 5: 0.1, 6: 0.1}, skip_hours={8, 9, 10}))
        self.assertIsNone(record["precip_24h_in"])
        self.assertEqual(record["precip_24h_hours_missing"], 3)

    def test_two_missing_hours_still_reports_with_count(self):
        record = self._derive(_asos_day(rain={0: 0.1, 1: 0.1, 2: 0.1}, skip_hours={8, 9}))
        self.assertAlmostEqual(record["precip_24h_in"], 0.30, delta=0.005)
        self.assertEqual(record["precip_24h_hours_missing"], 2)

    def test_missing_hour_next_to_rain_withholds_total(self):
        record = self._derive(_asos_day(rain={3: 0.5}, skip_hours={4}))
        self.assertIsNone(record["precip_24h_in"])
        self.assertEqual(record["precip_24h_hours_missing"], 1)

    def test_stale_station_has_no_24h_total(self):
        record = self._derive(_asos_day(), NOW + timedelta(hours=6))
        self.assertIsNone(record["precip_24h_in"])


class StationPrecipTests(unittest.TestCase):
    def test_asos_dry_station_is_explicit_zero(self):
        record = derive_station_precip(_station(1, {
            "precip_accum_one_hour_set_1": [0.0, 0.0, 0.001],
            "precip_accum_24_hour_set_1": [0.0, 0.0, 0.0]}), NOW)
        self.assertEqual(record["precip_1h_in"], 0.0)
        self.assertEqual(record["precip_24h_in"], 0.0)

    def test_station_without_precip_sensor_is_skipped(self):
        self.assertIsNone(derive_station_precip(_station(1, {"air_temp_set_1": [60, 61, 62]}), NOW))

    def test_raws_gauge_reset_never_goes_negative(self):
        # 2.00 -> 2.10 -> reset to 0.00 -> 0.05: rain is 0.10 + 0.05, not negative.
        record = derive_station_precip(_station(2, {"precip_accum_set_1": [2.00, 2.10, 0.0, 0.05]},
                                                ), NOW)
        self.assertAlmostEqual(record["precip_24h_in"], 0.15, places=3)
        self.assertAlmostEqual(record["precip_1h_in"], 0.05, places=3)

    def test_raws_with_one_reading_is_missing_not_dry(self):
        record = derive_station_precip(_station(2, {"precip_accum_set_1": [None, None, 1.0]}), NOW)
        self.assertIsNone(record["precip_1h_in"])
        self.assertIsNone(record["precip_24h_in"])

    def test_stale_reading_is_missing_not_dry(self):
        old = _times(3)
        old = [(NOW - timedelta(hours=10 + i)).isoformat() for i in range(3)][::-1]
        record = derive_station_precip(_station(1, {"precip_accum_24_hour_set_1": [0.0, 0.0, 0.0], "date_time": old}), NOW)
        self.assertIsNone(record["precip_24h_in"])

    def test_dry_asos_with_known_sensor_and_fresh_obs_is_zero(self):
        # METAR omits the precip group when dry: no precip values at all, but it is reporting.
        station = _station(1, {"air_temp_set_1": [60, 61, 62]}, stid="KDRY")
        record = derive_station_precip(station, NOW, one_hour_sensors={"KDRY"})
        self.assertEqual(record["precip_1h_in"], 0.0)

    def test_asos_old_rain_value_does_not_linger(self):
        times = [(NOW - timedelta(hours=h)).isoformat() for h in (6, 5, 4, 3, 0)]
        station = _station(1, {"precip_accum_one_hour_set_1": [0.3, 0.2, None, None, None],
                               "air_temp_set_1": [60, 60, 60, 60, 60], "date_time": times})
        self.assertEqual(derive_station_precip(station, NOW)["precip_1h_in"], 0.0)

    def test_asos_recent_rain_value_is_used(self):
        times = [(NOW - timedelta(minutes=m)).isoformat() for m in (30, 15, 0)]
        station = _station(1, {"precip_accum_one_hour_set_1": [0.1, 0.12, None],
                               "air_temp_set_1": [60, 60, 60], "date_time": times})
        self.assertEqual(derive_station_precip(station, NOW)["precip_1h_in"], 0.12)

    def test_unknown_sensor_or_silent_station_is_not_zero(self):
        quiet = _station(1, {"air_temp_set_1": [60, 61, 62]}, stid="KNOSENSOR")
        self.assertIsNone(derive_station_precip(quiet, NOW, one_hour_sensors={"OTHER"}))
        old = [(NOW - timedelta(hours=8 - i)).isoformat() for i in range(3)]
        silent = _station(1, {"air_temp_set_1": [60, 61, 62], "date_time": old}, stid="KSILENT")
        self.assertIsNone(derive_station_precip(silent, NOW, one_hour_sensors={"KSILENT"})["precip_1h_in"])

    def test_buffer_state_station_is_not_in_state(self):
        record = derive_station_precip(_station(1, {"precip_accum_24_hour_set_1": [0.2, 0.3, 0.4]}, state="IL"), NOW)
        self.assertFalse(record["in_state"])
        self.assertEqual(record["precip_24h_in"], 0.4)

    def test_implausible_value_is_dropped_and_ignored_stations_excluded(self):
        stations = [
            _station(1, {"precip_accum_24_hour_set_1": [90.0, 90.0, 90.0]}, stid="BAD"),
            _station(1, {"precip_accum_24_hour_set_1": [0.1, 0.1, 0.1]}, stid="IGN"),
            _station(1, {"precip_accum_24_hour_set_1": [0.1, 0.1, 0.1]}, stid="OK"),
        ]
        self.assertEqual([r["stid"] for r in derive_all(stations, {"IGN"}, NOW)], ["OK"])


if __name__ == "__main__":
    unittest.main()
