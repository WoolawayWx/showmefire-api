import unittest
from unittest.mock import MagicMock, patch

from services import incident_wind


class ParsePeriodTests(unittest.TestCase):
    def test_parses_single_and_range_speeds(self):
        wind = incident_wind.parse_period({"windDirection": "SW", "windSpeed": "5 to 15 mph", "startTime": "t"})
        self.assertEqual(wind["from_cardinal"], "SW")
        self.assertEqual(wind["from_degrees"], 225.0)
        self.assertEqual(wind["toward_cardinal"], "NE")
        self.assertEqual(wind["speed_mph"], 15)
        self.assertEqual(wind["speed_range_mph"], [5, 15])
        self.assertEqual(incident_wind.parse_period({"windDirection": "N", "windSpeed": "10 mph"})["toward_cardinal"], "S")

    def test_rejects_unusable_periods(self):
        self.assertIsNone(incident_wind.parse_period({"windDirection": "", "windSpeed": "5 mph"}))
        self.assertIsNone(incident_wind.parse_period({"windDirection": "SW", "windSpeed": "calm"}))


class GetWindTests(unittest.TestCase):
    def setUp(self):
        incident_wind._cache.clear()

    def _client(self, periods):
        responses = [
            MagicMock(json=lambda: {"properties": {"forecastHourly": "https://x/hourly"}}, raise_for_status=lambda: None),
            MagicMock(json=lambda: {"properties": {"periods": periods}}, raise_for_status=lambda: None),
        ]
        client = MagicMock()
        client.get.side_effect = responses
        client.__enter__.return_value = client
        return client

    def test_returns_wind_and_caches_by_cell(self):
        client = self._client([{"windDirection": "S", "windSpeed": "12 mph", "startTime": "t"}])
        with patch.object(incident_wind.httpx, "Client", return_value=client) as factory:
            first = incident_wind.get_wind(36.90, -90.03)
            second = incident_wind.get_wind(36.91, -90.04)  # same ~5 km cell
        self.assertEqual(first["from_cardinal"], "S")
        self.assertEqual(first, second)
        self.assertEqual(factory.call_count, 1)

    def test_failure_returns_none_and_is_not_cached(self):
        client = MagicMock()
        client.get.side_effect = RuntimeError("nws down")
        client.__enter__.return_value = client
        with patch.object(incident_wind.httpx, "Client", return_value=client):
            self.assertIsNone(incident_wind.get_wind(36.9, -90.0))
        self.assertEqual(incident_wind._cache, {})


if __name__ == "__main__":
    unittest.main()
