import os
import unittest
from unittest.mock import patch

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from core import rate_limit as rl


def make_client(limiter, extra=()):
    app = FastAPI()

    @app.get("/x", dependencies=[Depends(limiter), *extra])
    def x():
        return {"ok": True}

    return TestClient(app)


class RateLimitTests(unittest.TestCase):
    def test_blocks_after_limit_with_retry_after(self):
        now = [100.0]
        limiter = rl.RateLimiter("t", 3, env_var="RL_TEST", clock=lambda: now[0])
        client = make_client(limiter)
        for _ in range(3):
            self.assertEqual(client.get("/x").status_code, 200)
        blocked = client.get("/x")
        self.assertEqual(blocked.status_code, 429)
        self.assertGreaterEqual(int(blocked.headers["Retry-After"]), 1)
        now[0] += 61
        self.assertEqual(client.get("/x").status_code, 200)

    def test_api_key_gets_higher_limit(self):
        limiter = rl.RateLimiter("t", 2, env_var="RL_TEST", clock=lambda: 1.0)
        client = make_client(limiter)
        with patch.dict(os.environ, {"API_READ_KEYS": "good-key"}):
            headers = {"Authorization": "Bearer good-key"}
            self.assertTrue(all(client.get("/x", headers=headers).status_code == 200 for _ in range(10)))
            self.assertEqual(client.get("/x", headers={"Authorization": "Bearer bad"}).status_code, 200)
            self.assertEqual(client.get("/x", headers={"Authorization": "Bearer bad"}).status_code, 200)
            self.assertEqual(client.get("/x", headers={"Authorization": "Bearer bad"}).status_code, 429)

    def test_env_override(self):
        limiter = rl.RateLimiter("t", 100, env_var="RL_TEST", clock=lambda: 1.0)
        client = make_client(limiter)
        with patch.dict(os.environ, {"RL_TEST": "1"}):
            self.assertEqual(client.get("/x").status_code, 200)
            self.assertEqual(client.get("/x").status_code, 429)

    def test_station_gate_off_by_default_and_on_with_flag(self):
        limiter = rl.RateLimiter("t", 100, clock=lambda: 1.0)
        client = make_client(limiter, extra=[Depends(rl.require_station_data_access)])
        self.assertEqual(client.get("/x").status_code, 200)
        with patch.dict(os.environ, {"VERIFICATION_STATION_DATA_REQUIRES_KEY": "true", "API_READ_KEYS": "k"}):
            self.assertEqual(client.get("/x").status_code, 401)
            self.assertEqual(client.get("/x", headers={"Authorization": "Bearer k"}).status_code, 200)


if __name__ == "__main__":
    unittest.main()
