import json
import tempfile
import unittest
from pathlib import Path

from ai.briefing import build_briefing
from forecast.forecast_ai import generate_forecast_text, main


class FakeClient:
    configured = True

    def __init__(self):
        self.prompts = []

    def generate_text(self, prompt):
        self.prompts.append(prompt)
        if len(self.prompts) == 1:
            return "Elevated Fire Danger Across Missouri"
        return (
            "Elevated fire danger is forecast statewide with relative humidity "
            "near 25%, fuel moisture near 8%, and winds near 9 mph. "
            "Rainfall remains below 0.080 inches."
        )


class ForecastAITests(unittest.TestCase):
    def _forecast_file(self, directory):
        path = Path(directory) / "station_forecasts_20260828_12.json"
        path.write_text(json.dumps({
            "stations": {
                "station": {
                    "lat": 38.0,
                    "lon": -92.0,
                    "forecasts": [{
                        "temp_c": 30,
                        "rh": 25,
                        "wind_speed_ms": 4,
                        "precip_in": 0.08,
                        "fuel_moisture": 8,
                        "fire_danger": 2,
                    }],
                },
            },
        }))
        return path

    def test_test_mode_prints_without_persisting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._forecast_file(directory)
            client = FakeClient()
            headline, discussion = main(
                test_mode=True,
                forecast_path=str(path),
                client=client,
            )
        self.assertEqual(headline, "Elevated Fire Danger Across Missouri")
        self.assertIn("Elevated fire danger", discussion)
        self.assertEqual(len(client.prompts), 2)

    def test_forecast_generation_falls_back_when_client_is_unconfigured(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._forecast_file(directory)
            briefing = build_briefing(directory, forecast_path=path)
            headline, discussion = generate_forecast_text(
                briefing,
                client=type("UnavailableClient", (), {"configured": False})(),
            )
        self.assertEqual(headline, "Elevated Fire Danger Across Missouri")
        self.assertIn("0.08 inches", discussion)


class CurrentDispersionTests(unittest.TestCase):
    def _fire_weather_file(self, directory, valid_time):
        area = {
            "mixing_height_ft": {"min": 2000, "median": 4200, "max": 6000},
            "transport_speed_mph": {"min": 20, "median": 30, "max": 40},
            "transport_direction": "northwest", "ventilation_dominant": "Very Good",
            "ventilation_category_percent": {"Very Good": 70.0, "Good": 30.0},
            "surface_inversion_percent": 0.0, "haines": {"max": 5}, "hdw": {"max": 140}, "cell_count": 10,
        }
        path = Path(directory) / "fire_weather_latest.json"
        path.write_text(json.dumps({"valid_time": valid_time, "source": "HRRR",
                                    "statewide": area, "regions": {"Central": area}}))
        return path

    def _briefing(self, directory, valid_time):
        forecast = ForecastAITests()._forecast_file(directory)
        return build_briefing(directory, forecast_path=forecast,
                              fire_weather_path=self._fire_weather_file(directory, valid_time))

    def test_fresh_analysis_is_added_and_used_by_fallback(self):
        from datetime import datetime, timezone
        from forecast.forecast_ai import fallback_text

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:00:00Z")
        with tempfile.TemporaryDirectory() as directory:
            briefing = self._briefing(directory, now)
        self.assertEqual(briefing["current_dispersion"]["statewide"]["smoke_dispersal"], "Very Good")
        _headline, discussion = fallback_text(briefing)
        self.assertIn("mixing heights are near 4,200 ft", discussion)
        self.assertIn("very good smoke dispersal", discussion)

    def test_stale_or_missing_analysis_is_left_out(self):
        with tempfile.TemporaryDirectory() as directory:
            briefing = self._briefing(directory, "2020-01-01T00:00:00Z")
            self.assertNotIn("current_dispersion", briefing)
            forecast = ForecastAITests()._forecast_file(directory)
            briefing = build_briefing(directory, forecast_path=forecast,
                                      fire_weather_path=Path(directory) / "missing.json")
        self.assertNotIn("current_dispersion", briefing)


if __name__ == "__main__":
    unittest.main()
