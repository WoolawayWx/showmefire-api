import json
import unittest

import httpx

from ai.cloudflare import CloudflareAIClient, CloudflareAIError


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeHttpClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.response


class CloudflareAIClientTests(unittest.TestCase):
    def _client(self, result):
        return CloudflareAIClient(api_key="k", account_id="a", model="m",
                                  http_client=FakeHttpClient(FakeResponse({"success": True, "result": result})))

    def test_an_already_parsed_json_response_comes_back_as_json_text(self):
        text = self._client({"response": {"event_type": "grass_fire", "fuels": ["grass"]}}).generate_text("p")
        self.assertEqual(json.loads(text), {"event_type": "grass_fire", "fuels": ["grass"]})

    def test_missing_response_error_says_what_came_back(self):
        with self.assertRaisesRegex(CloudflareAIError, r"response is NoneType.*usage"):
            self._client({"usage": {}}).generate_text("p")

    def test_posts_chat_messages_and_returns_workers_ai_response(self):
        http_client = FakeHttpClient(FakeResponse({
            "success": True,
            "result": {"response": "A factual headline"},
        }))
        client = CloudflareAIClient(
            api_key="test-token",
            account_id="account-123",
            model="@cf/meta/test-model",
            http_client=http_client,
        )

        result = client.generate_text("Write a headline", system="Be factual.")

        self.assertEqual(result, "A factual headline")
        args, kwargs = http_client.calls[0]
        self.assertEqual(
            args[0],
            "https://api.cloudflare.com/client/v4/accounts/account-123/ai/run/@cf/meta/test-model",
        )
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-token")
        self.assertEqual(kwargs["json"]["messages"], [
            {"role": "system", "content": "Be factual."},
            {"role": "user", "content": "Write a headline"},
        ])

    def test_reports_workers_ai_errors_without_leaking_credentials(self):
        http_client = FakeHttpClient(FakeResponse({
            "success": False,
            "errors": [{"message": "model unavailable"}],
        }))
        client = CloudflareAIClient(
            api_key="secret-token",
            account_id="account-123",
            http_client=http_client,
        )

        with self.assertRaisesRegex(CloudflareAIError, "model unavailable"):
            client.generate_text("hello")

    def test_rejects_missing_result_text(self):
        http_client = FakeHttpClient(FakeResponse({"success": True, "result": {}}))
        client = CloudflareAIClient(
            api_key="test-token",
            account_id="account-123",
            http_client=http_client,
        )

        with self.assertRaises(CloudflareAIError):
            client.generate_text("hello")

    def test_translates_http_errors(self):
        class BrokenHttpClient:
            def post(self, *args, **kwargs):
                raise httpx.ConnectError("offline")

        client = CloudflareAIClient(
            api_key="test-token",
            account_id="account-123",
            http_client=BrokenHttpClient(),
        )

        with self.assertRaises(CloudflareAIError):
            client.generate_text("hello")


if __name__ == "__main__":
    unittest.main()
