import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from services import synoptic

URL = "https://api.synopticdata.com/v2/stations/metadata"
PARAMS = {"token": "secret", "state": "MO", "network": "1"}
PAYLOAD = {"STATION": [{"STID": "KSGF"}]}


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    async def json(self):
        return self.payload


class FakeSession:
    """Answers with the queued payloads (or raises queued exceptions) and counts requests."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = 0

    async def get(self, url, params=None, timeout=None):
        self.calls += 1
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return FakeResponse(result)


class MetadataCacheTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        synoptic._metadata_cache.clear()

    async def test_second_call_within_ttl_makes_no_request(self):
        session = FakeSession(PAYLOAD)
        first = await synoptic.fetch_metadata_cached(session, URL, PARAMS)
        second = await synoptic.fetch_metadata_cached(session, URL, PARAMS)
        self.assertEqual(first, second)
        self.assertEqual(session.calls, 1)

    async def test_expired_entry_is_requested_again(self):
        session = FakeSession(PAYLOAD, PAYLOAD)
        await synoptic.fetch_metadata_cached(session, URL, PARAMS)
        key, (fetched, payload) = next(iter(synoptic._metadata_cache.items()))
        synoptic._metadata_cache[key] = (fetched - synoptic.METADATA_CACHE_TTL - timedelta(minutes=1), payload)
        await synoptic.fetch_metadata_cached(session, URL, PARAMS)
        self.assertEqual(session.calls, 2)

    async def test_different_queries_are_cached_separately_and_token_is_not_part_of_the_key(self):
        session = FakeSession(PAYLOAD, PAYLOAD)
        await synoptic.fetch_metadata_cached(session, URL, PARAMS)
        await synoptic.fetch_metadata_cached(session, URL, {**PARAMS, "token": "other"})   # same query
        await synoptic.fetch_metadata_cached(session, URL, {**PARAMS, "state": "KS"})      # different query
        self.assertEqual(session.calls, 2)
        self.assertFalse(any("secret" in key for key in synoptic._metadata_cache))

    async def test_failed_refresh_serves_the_last_good_copy(self):
        session = FakeSession(PAYLOAD, RuntimeError("boom"))
        await synoptic.fetch_metadata_cached(session, URL, PARAMS)
        key, (fetched, payload) = next(iter(synoptic._metadata_cache.items()))
        synoptic._metadata_cache[key] = (fetched - synoptic.METADATA_CACHE_TTL - timedelta(minutes=1), payload)
        self.assertEqual(await synoptic.fetch_metadata_cached(session, URL, PARAMS), PAYLOAD)

    async def test_failure_with_nothing_cached_raises(self):
        with self.assertRaises(RuntimeError):
            await synoptic.fetch_metadata_cached(FakeSession(RuntimeError("boom")), URL, PARAMS)

    async def test_empty_response_is_not_cached(self):
        session = FakeSession({"STATION": []}, PAYLOAD)
        with self.assertRaises(ValueError):
            await synoptic.fetch_metadata_cached(session, URL, PARAMS)
        self.assertEqual(await synoptic.fetch_metadata_cached(session, URL, PARAMS), PAYLOAD)
        self.assertEqual(session.calls, 2)


if __name__ == "__main__":
    unittest.main()
