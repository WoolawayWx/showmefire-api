import json
import tempfile
import unittest
from pathlib import Path

from ai.briefing import build_briefing, validate_briefing_text
from forecast.forecast_ai import generate_forecast_text, main


class FakeClient:
    configured = True

    def __init__(self):
        self.prompts = []
        self.systems = []
        self.session_ids = []

    def generate_text(self, prompt, system=None, *, max_tokens=320, session_id=None):
        self.prompts.append(prompt)
        self.systems.append(system)
        self.session_ids.append(session_id)
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


class HeadlineTests(unittest.TestCase):
    def test_classes_must_run_low_to_high(self):
        from forecast.forecast_ai import valid_headline

        self.assertTrue(valid_headline("Moderate to Elevated Fire Danger in Missouri"))
        self.assertFalse(valid_headline("Elevated to Moderate Fire Danger in Missouri"))
        self.assertTrue(valid_headline("Elevated Fire Danger Across Missouri"))


class CachingAndContentTests(unittest.TestCase):
    def _briefing(self, directory, **kwargs):
        forecast = ForecastAITests()._forecast_file(directory)
        return build_briefing(directory, forecast_path=forecast, alerts=[], **kwargs)

    def test_system_prompt_is_shared_stable_and_session_id_sent(self):
        with tempfile.TemporaryDirectory() as directory:
            briefing = self._briefing(directory)
            client = FakeClient()
            generate_forecast_text(briefing, client=client)
        self.assertEqual(len(set(client.systems)), 1)
        self.assertIn("Show Me Fire", client.systems[0])
        self.assertEqual(set(client.session_ids), {"smf-forecast-discussion-v1"})
        # Dynamic content (the briefing) is only in the user message, never the system prompt.
        self.assertNotIn(briefing["source_file"], client.systems[0])
        self.assertIn(briefing["source_file"], client.prompts[1])

    def test_retry_appends_note_and_keeps_prefix(self):
        class Rejecting(FakeClient):
            def generate_text(self, prompt, system=None, *, max_tokens=320, session_id=None):
                self.prompts.append(prompt)
                self.systems.append(system)
                return "Extreme fire danger with 0.50 inches of rain."

        with tempfile.TemporaryDirectory() as directory:
            briefing = self._briefing(directory)
            client = Rejecting()
            from forecast.forecast_ai import generate_text

            self.assertIsNone(generate_text(client, "TASK\n{}", briefing, require_core_facts=True))
        self.assertEqual(len(client.prompts), 2)
        self.assertTrue(client.prompts[1].startswith(client.prompts[0]))
        self.assertEqual(client.systems[0], client.systems[1])

    def test_dispersion_never_appears(self):
        from forecast.forecast_ai import DISCUSSION_SYSTEM_PROMPT, fallback_text

        with tempfile.TemporaryDirectory() as directory:
            briefing = self._briefing(directory)
        self.assertNotIn("current_dispersion", briefing)
        _headline, discussion = fallback_text(briefing)
        self.assertNotRegex(discussion, r"(?i)dispersal|mixing height|transport wind")
        self.assertFalse(validate_briefing_text("Smoke dispersal is poor statewide.", briefing))
        self.assertIn("never describe current conditions", DISCUSSION_SYSTEM_PROMPT.lower())

    def test_fallback_names_regions_counties_timing_rain_and_red_flag(self):
        from forecast.forecast_ai import fallback_text

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "station_forecasts_20260828_12.json"
            hours = [f"2026-08-28T{hour:02d}:00:00Z" for hour in (15, 18, 21, 23)]
            path.write_text(json.dumps({"stations": {"nw": {"lat": 39.5, "lon": -94.5, "forecasts": [
                {"time": hours[0], "temp_c": 25, "rh": 40, "wind_speed_ms": 3, "precip_in": 0.0, "fuel_moisture": 12, "fire_danger": 0},
                {"time": hours[1], "temp_c": 28, "rh": 22, "wind_speed_ms": 6, "precip_in": 0.0, "fuel_moisture": 9, "fire_danger": 2},
                {"time": hours[2], "temp_c": 27, "rh": 30, "wind_speed_ms": 4, "precip_in": 0.1, "fuel_moisture": 10, "fire_danger": 1},
                {"time": hours[3], "temp_c": 22, "rh": 45, "wind_speed_ms": 3, "precip_in": 0.28, "fuel_moisture": 12, "fire_danger": 1},
            ]}}}))
            counties = Path(directory) / "dangerbycounty.json"
            counties.write_text(json.dumps([
                {"county": "Buchanan", "max_fire_danger": 2}, {"county": "Platte", "max_fire_danger": 2},
                {"county": "Pulaski", "max_fire_danger": 0},
            ]))
            alerts = [{"event": "Red Flag Warning", "countyFips": ["29021"], "expires": "2026-08-29T00:00:00Z"}]
            briefing = build_briefing(directory, forecast_path=path, county_path=counties, alerts=alerts)
        northwest = briefing["regions"]["Northwest"]
        self.assertEqual(northwest["precipitation"], "measurable")
        self.assertIsNotNone(northwest["precip_window"])
        self.assertEqual(northwest["min_rh_time"]["from"], "1 PM CDT")
        self.assertIn("Northwest", briefing["county_danger"]["by_region"])
        self.assertEqual(briefing["red_flag"][0]["counties"], ["Buchanan"])
        _headline, discussion = fallback_text(briefing)
        self.assertIn("Buchanan", discussion)
        self.assertIn("Northwest", discussion)
        self.assertIn("0.28 inches", discussion)
        self.assertIn("Red Flag Warning is in effect for Buchanan County", discussion)
        self.assertNotIn("burn ban", discussion.lower())

    def test_no_red_flag_key_or_wording_when_none_active(self):
        from forecast.forecast_ai import fallback_text

        with tempfile.TemporaryDirectory() as directory:
            briefing = self._briefing(directory)
        self.assertNotIn("red_flag", briefing)
        self.assertNotRegex(fallback_text(briefing)[1].lower(), r"red flag|fire weather watch")

    def test_validator_rejects_invented_county_and_overcap_regional_rain(self):
        with tempfile.TemporaryDirectory() as directory:
            briefing = self._briefing(directory)
        self.assertFalse(validate_briefing_text("Elevated danger in Boone County.", briefing))
        self.assertFalse(validate_briefing_text("Rain up to 0.50 inches in the Central region.", briefing))
        self.assertTrue(validate_briefing_text("Rain up to 0.08 inches in the Central region.", briefing))
        # county named without the word "County", status talk about missing data, gusts, unsupported warnings
        self.assertFalse(validate_briefing_text("Boone and other counties will see these conditions.", briefing))
        self.assertFalse(validate_briefing_text("The Red Flag warning status is not available.", briefing))
        self.assertFalse(validate_briefing_text("Winds gust to 11 mph.", briefing))
        self.assertFalse(validate_briefing_text("Rain up to 0.08 inches is likely over Pulaski County.", briefing))
        self.assertFalse(validate_briefing_text("A Red Flag Warning is in effect for Boone County.", briefing))

    def test_stale_county_file_is_ignored(self):
        import os

        with tempfile.TemporaryDirectory() as directory:
            forecast = ForecastAITests()._forecast_file(directory)
            counties = Path(directory) / "gis" / "dangerbycounty.json"
            counties.parent.mkdir()
            counties.write_text(json.dumps([{"county": "Boone", "max_fire_danger": 3}]))
            old = forecast.stat().st_mtime - 86400
            os.utime(counties, (old, old))
            archive = Path(directory) / "a" / "b"
            archive.mkdir(parents=True)
            moved = archive / forecast.name
            moved.write_text(forecast.read_text())
            (Path(directory) / "gis").rename(Path(directory) / "gis")
            briefing = build_briefing(archive, forecast_path=moved, alerts=[],
                                      county_path=None, county_tif_path=Path(directory) / "none.tif")
        self.assertNotIn("county_danger", briefing)


if __name__ == "__main__":
    unittest.main()


class PlacementTests(unittest.TestCase):
    def _briefing(self):
        return {
            "statewide": {"fire_danger_present": ["Low", "Moderate"], "precip_in": {"max": None}},
            "regions": {"Northwest": {"precip_in": {}}, "Central": {"precip_in": {}}},
        }

    def test_too_many_counties_in_one_sentence(self):
        from unittest import mock
        from ai import briefing as b
        names = {n: {"region": "NW", "fips": "1"} for n in "ABCDEF"}
        with mock.patch.object(b, "county_regions", return_value=names):
            text = "In the Northwest region, A, B, C, D, E and F Counties reach Moderate."
            self.assertIn("more than", b.briefing_text_problem(text, self._briefing()) or "")

    def test_county_under_wrong_region(self):
        from unittest import mock
        from ai import briefing as b
        names = {"Boone": {"region": "Central", "fips": "29019"}}
        briefing = self._briefing()
        briefing["county_danger"] = {"Boone": 1}
        with mock.patch.object(b, "county_regions", return_value=names):
            bad = "In the Northwest region, Boone County reaches Moderate."
            self.assertEqual(b.briefing_text_problem(bad, briefing), "county named under the wrong region")
            self.assertIsNone(b.briefing_text_problem("In the Central region, Boone County reaches Moderate.", briefing))
