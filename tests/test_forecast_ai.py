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

    def generate_text(self, prompt, *, max_tokens):
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
            "run_date": "2026-08-28T12:00:00Z",
            "stations": {
                "station": {
                    "lat": 38.0,
                    "lon": -92.0,
                    "forecasts": [{
                        "time": f"2026-08-28T{hour}:00:00Z",
                        "temp_c": 30,
                        "rh": 25,
                        "wind_speed_ms": 4,
                        "precip_in": 0.08,
                        "fuel_moisture": 8,
                        "fire_danger": 2,
                    } for hour in (15, 20, 23)],
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


class ForecastDispersionTests(unittest.TestCase):
    def test_matching_forecast_hours_are_used_in_memo_and_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            forecast = ForecastAITests()._forecast_file(directory)
            for cycle, leads in (("12", [3]), ("11", [4, 9, 12, 13]), ("13", [2, 7, 10])):
                run = root / f"20260828_{cycle}z"
                run.mkdir()
                (run / "manifest.json").write_text(json.dumps({
                    "forecast_hours": leads,
                    "series": {
                        name: [{"lead_hour": lead, "median": value} for lead, value in zip(leads, values)]
                        for name, values in {
                            "mixing_height_ft": [1000, 4200, 2200, 2000],
                            "transport_speed_mph": [10, 30, 15, 10],
                            "ventilation_kt_ft": [10000, 110000, 35000, 20000],
                        }.items()
                    },
                }))
            briefing = build_briefing(root, forecast_path=forecast, hrrr_hourly_root=root)

        dispersion = briefing["forecast_dispersion"]
        self.assertEqual(dispersion["cycle"], "20260828_11z")
        self.assertEqual([hour["time"] for hour in dispersion["hours"]], ["10 AM CDT", "3 PM CDT", "6 PM CDT"])
        client = FakeClient()
        _headline, discussion = generate_forecast_text(briefing, client=client)
        self.assertIn('"forecast_dispersion"', client.prompts[-1])
        self.assertIn("poor dispersal at 10 AM CDT", discussion)
        self.assertIn("Ventilation peaks at 3 PM CDT, with very good dispersal", discussion)
        self.assertIn("By 6 PM CDT, HRRR forecasts poor dispersal", discussion)

    def test_missing_hourly_guidance_is_omitted(self):
        with tempfile.TemporaryDirectory() as directory:
            forecast = ForecastAITests()._forecast_file(directory)
            briefing = build_briefing(directory, forecast_path=forecast, hrrr_hourly_root=directory)
        self.assertNotIn("forecast_dispersion", briefing)


if __name__ == "__main__":
    unittest.main()
