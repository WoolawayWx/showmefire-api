import sys
import unittest
from pathlib import Path

API_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(API_DIR))

from alerts import advisory_categories as ac


class CategorizeEventTests(unittest.TestCase):
    def test_known_event_strings_map_to_expected_categories(self):
        # Real NWS event strings (api.weather.gov vocabulary).
        cases = {
            "Tornado Warning": "severe_weather",
            "Severe Thunderstorm Warning": "severe_weather",
            "Special Weather Statement": "severe_weather",
            "Flood Warning": "flood",
            "Flash Flood Warning": "flood",
            "Flood Watch": "flood",
            "Red Flag Warning": "fire",
            "Fire Weather Watch": "fire",
            "High Wind Warning": "wind",
            "Wind Advisory": "wind",
            "Excessive Heat Warning": "heat",
            "Heat Advisory": "heat",
            "Winter Storm Warning": "winter_weather",
            "Winter Weather Advisory": "winter_weather",
            "Ice Storm Warning": "winter_weather",
            "Frost Advisory": "frost_freeze",
            "Freeze Warning": "frost_freeze",
            "Freeze Watch": "frost_freeze",
            "Dense Fog Advisory": "visibility",
        }
        for event, expected_category in cases.items():
            with self.subTest(event=event):
                self.assertEqual(ac.categorize_event(event), expected_category)

    def test_unmatched_event_returns_none(self):
        self.assertIsNone(ac.categorize_event("Small Craft Advisory"))

    def test_none_and_empty_return_none(self):
        self.assertIsNone(ac.categorize_event(None))
        self.assertIsNone(ac.categorize_event(""))

    def test_case_insensitive(self):
        self.assertEqual(ac.categorize_event("red flag warning"), "fire")
        self.assertEqual(ac.categorize_event("RED FLAG WARNING"), "fire")

    def test_every_category_has_a_label(self):
        for category in ac.CATEGORY_EVENT_KEYWORDS:
            self.assertIn(category, ac.CATEGORY_LABELS)


class FilterAlertsByCategoryTests(unittest.TestCase):
    def setUp(self):
        self.alerts = {
            "type": "FeatureCollection",
            "features": [
                {"properties": {"event": "Red Flag Warning"}, "geometry": None},
                {"properties": {"event": "Heat Advisory"}, "geometry": None},
                {"properties": {"event": "Flood Warning"}, "geometry": None},
                {"properties": {"event": "Small Craft Advisory"}, "geometry": None},
            ],
        }

    def test_filters_to_matching_category_only(self):
        result = ac.filter_alerts_by_category(self.alerts, "fire")
        self.assertEqual(len(result["features"]), 1)
        self.assertEqual(result["features"][0]["properties"]["event"], "Red Flag Warning")

    def test_does_not_mutate_input(self):
        original_count = len(self.alerts["features"])
        ac.filter_alerts_by_category(self.alerts, "fire")
        self.assertEqual(len(self.alerts["features"]), original_count)

    def test_no_matches_returns_empty_feature_list(self):
        result = ac.filter_alerts_by_category(self.alerts, "visibility")
        self.assertEqual(result["features"], [])

    def test_preserves_other_feature_collection_keys(self):
        result = ac.filter_alerts_by_category(self.alerts, "fire")
        self.assertEqual(result["type"], "FeatureCollection")


if __name__ == "__main__":
    unittest.main()
