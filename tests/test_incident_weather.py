import unittest
from datetime import datetime, timedelta, timezone

from services import incident_weather

NOW = datetime(2026, 11, 13, 18, 20, tzinfo=timezone.utc)      # 12:20 PM Central


def _rows(count=14, start=None, **overrides):
    start = start or NOW.replace(minute=0) - timedelta(hours=2)
    rows = []
    for i in range(count):
        row = {"time_utc": start + timedelta(hours=i), "temp_f": 50.0 + i, "rh_pct": 60.0 - 2 * i, "wind_speed_mph": 8.0 + i,
               "wind_gust_mph": 14.0 + i, "wind_dir_deg": 200.0, "precip_mm_1h": 0.0, "fire_danger_est": 1.0 + (i > 5)}
        row.update({k: v(i) if callable(v) else v for k, v in overrides.items()})
        rows.append(row)
    return rows


def _forecast(rows):
    return lambda lat, lon, now: {"cycle": "c", "cycle_time_utc": "2026-11-13T17:00:00Z", "rows": rows}


class IncidentWeatherTests(unittest.TestCase):
    def test_returns_the_eight_hours_from_the_current_hour_on(self):
        result = incident_weather.outlook_at(36.6, -93.7, NOW, forecast=_forecast(_rows()))
        hours = result["hours"]
        self.assertEqual(len(hours), 8)
        self.assertEqual(hours[0]["time_utc"], "2026-11-13T18:00:00Z")          # this hour, not the earlier ones
        self.assertEqual(hours[-1]["time_utc"], "2026-11-14T01:00:00Z")
        self.assertEqual(hours[0]["hour_label"], "12 PM")
        self.assertEqual(hours[0]["wind_from"], "SSW")
        self.assertEqual(hours[0]["gust_mph"], 14.0 + 2)
        self.assertEqual(result["source"], "HRRR model forecast")

    def test_summary_covers_the_window_and_flags_rain_and_a_wind_shift(self):
        rows = _rows(wind_dir_deg=lambda i: 200.0 if i < 6 else 300.0, precip_mm_1h=lambda i: 0.4 if i == 7 else 0.0)
        summary = incident_weather.outlook_at(36.6, -93.7, NOW, forecast=_forecast(rows))["summary"]
        self.assertEqual(summary["temp_min_f"], 52.0)
        self.assertEqual(summary["temp_max_f"], 59.0)
        self.assertEqual(summary["rh_min_pct"], 42.0)
        self.assertEqual(summary["peak_fire_danger"], 2)
        self.assertTrue(summary["wind_shift"])
        self.assertTrue(summary["rain"])

    def test_missing_values_stay_none_and_short_forecasts_still_work(self):
        rows = _rows(count=5, temp_f=None, fire_danger_est=None)
        result = incident_weather.outlook_at(36.6, -93.7, NOW, forecast=_forecast(rows))
        self.assertEqual(len(result["hours"]), 3)
        self.assertIsNone(result["hours"][0]["temp_f"])
        self.assertIsNone(result["hours"][0]["fire_danger"])
        self.assertIsNone(result["summary"]["temp_max_f"])

    def test_no_forecast_or_nothing_in_the_window_is_none(self):
        self.assertIsNone(incident_weather.outlook_at(36.6, -93.7, NOW, forecast=lambda *a: None))
        old = _rows(count=4, start=NOW - timedelta(days=2))
        self.assertIsNone(incident_weather.outlook_at(36.6, -93.7, NOW, forecast=_forecast(old)))

    def test_compass_and_bad_input(self):
        self.assertEqual(incident_weather.cardinal(0), "N")
        self.assertEqual(incident_weather.cardinal(359), "N")
        self.assertEqual(incident_weather.cardinal(90), "E")
        self.assertIsNone(incident_weather.cardinal(None))
        self.assertIsNone(incident_weather.weather_outlook("x", None))


if __name__ == "__main__":
    unittest.main()
