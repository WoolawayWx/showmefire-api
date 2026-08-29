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
        return "Elevated fire danger is forecast statewide with low relative humidity. Rainfall remains below 0.080 inches."


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
        self.assertIn("0.080 inches", discussion)


if __name__ == "__main__":
    unittest.main()
