import json
import tempfile
import unittest
from pathlib import Path

from ai.briefing import build_briefing, briefing_json, validate_briefing_text


class BriefingTests(unittest.TestCase):
    def test_builds_regional_daily_extrema_and_statewide_median(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "station_forecasts_20260828_12.json"
            path.write_text(json.dumps({
                "run_date": "2026-08-28T12:00:00Z",
                "stations": {
                    "central": {
                        "lat": 38.95, "lon": -92.33,
                        "forecasts": [
                            {"temp_c": 25, "rh": 40, "wind_speed_ms": 2, "precip_in": 0.01, "fuel_moisture": 12, "fire_danger": 1},
                            {"temp_c": 30, "rh": 30, "wind_speed_ms": 5, "precip_in": 0.15, "fuel_moisture": 8, "fire_danger": 2},
                        ],
                    },
                    "northwest": {
                        "lat": 40.0, "lon": -94.5,
                        "forecasts": [
                            {"temp_c": 20, "rh": 50, "wind_speed_ms": 3, "precip_in": 0.10, "fuel_moisture": 15, "fire_danger": 0},
                        ],
                    },
                    "outside": {
                        "lat": 35.0, "lon": -92.0,
                        "forecasts": [{"fire_danger": 4}],
                    },
                },
            }))

            briefing = build_briefing(directory)

        central = briefing["regions"]["Central"]
        self.assertEqual(central["station_count"], 1)
        self.assertEqual(central["precip_in"]["max"], 0.15)
        self.assertEqual(central["fuel_moisture"]["min"], 8)
        self.assertEqual(central["fire_danger_present"], ["Elevated"])
        self.assertEqual(briefing["statewide"]["station_count"], 2)
        self.assertEqual(briefing["statewide"]["precip_in"]["max"], 0.15)
        self.assertEqual(briefing["statewide"]["precipitation"], "measurable")
        json.loads(briefing_json(briefing))

    def test_trace_precipitation_is_labeled_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "station_forecasts_20260828_12.json"
            path.write_text(json.dumps({
                "stations": {
                    "station": {
                        "lat": 38.0, "lon": -92.0,
                        "forecasts": [{"precip_in": 0.02, "fire_danger": 0}],
                    },
                },
            }))
            briefing = build_briefing(directory)
        self.assertEqual(briefing["statewide"]["precipitation"], "trace")

    def test_county_classes_are_merged_into_allowed_statewide_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            forecast = root / "station_forecasts_20260828_12.json"
            forecast.write_text(json.dumps({
                "stations": {
                    "station": {
                        "lat": 38.0, "lon": -92.0,
                        "forecasts": [{"fire_danger": 1}],
                    },
                },
            }))
            county = root / "dangerbycounty.json"
            county.write_text(json.dumps([
                {"county": "Example", "max_fire_danger": 3},
            ]))
            briefing = build_briefing(root, forecast, county)
        self.assertEqual(briefing["statewide"]["highest_fire_danger"], "Critical")
        self.assertEqual(briefing["county_danger"]["county_count"], 1)

    def test_validation_rejects_unsupported_rain_but_allows_low_humidity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "station_forecasts_20260828_12.json"
            path.write_text(json.dumps({
                "stations": {
                    "station": {
                        "lat": 38.0, "lon": -92.0,
                        "forecasts": [
                            {"precip_in": 1, "rh": 25, "fire_danger": 2},
                        ],
                    },
                },
            }))
            briefing = build_briefing(directory)
        self.assertTrue(validate_briefing_text("Low relative humidity is possible.", briefing))
        self.assertFalse(validate_briefing_text("Rainfall could reach 5 inches.", briefing))
        self.assertFalse(validate_briefing_text("Critical fire danger is expected.", briefing))


if __name__ == "__main__":
    unittest.main()
