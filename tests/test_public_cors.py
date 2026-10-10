"""Tile responses carry a stable `Access-Control-Allow-Origin: *`, independent of the caller."""
import unittest

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from core.public_cors import PublicTileCorsMiddleware


def make_app() -> FastAPI:
    app = FastAPI()

    @app.get("/tiles/cog/tiles/1/0/0.png")
    def tile():
        return Response(content=b"png", media_type="image/png", headers={"Cache-Control": "public, max-age=60"})

    @app.get("/api/private")
    def private():
        return {"ok": True}

    # Same arrangement as main.py: credentialed CORS for the API, then the tile middleware outside it.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://showmefire.org"],
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(PublicTileCorsMiddleware)
    return app


class PublicTileCorsTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(make_app())

    def test_tiles_say_star_with_or_without_an_origin(self):
        for origin in (None, "https://showmefire.org", "http://127.0.0.1:8090", "https://anyone.example"):
            headers = {"Origin": origin} if origin else {}
            r = self.client.get("/tiles/cog/tiles/1/0/0.png", headers=headers)
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.headers["access-control-allow-origin"], "*", origin)
            self.assertNotIn("access-control-allow-credentials", r.headers, origin)
            self.assertNotIn("origin", r.headers.get("vary", "").lower(), origin)

    def test_every_caller_gets_the_same_cacheable_response(self):
        a = self.client.get("/tiles/cog/tiles/1/0/0.png")
        b = self.client.get("/tiles/cog/tiles/1/0/0.png", headers={"Origin": "http://127.0.0.1:8090"})
        for h in ("access-control-allow-origin", "cache-control", "content-type"):
            self.assertEqual(a.headers.get(h), b.headers.get(h), h)

    def test_preflight_for_tiles_is_also_open(self):
        r = self.client.options(
            "/tiles/cog/tiles/1/0/0.png",
            headers={"Origin": "https://anyone.example", "Access-Control-Request-Method": "GET"},
        )
        self.assertEqual(r.headers["access-control-allow-origin"], "*")

    def test_other_routes_keep_their_credentialed_cors(self):
        r = self.client.get("/api/private", headers={"Origin": "https://showmefire.org"})
        self.assertEqual(r.headers["access-control-allow-origin"], "https://showmefire.org")
        self.assertEqual(r.headers.get("access-control-allow-credentials"), "true")
        # An origin that is not allowed still gets no CORS header on the API.
        r = self.client.get("/api/private", headers={"Origin": "https://evil.example"})
        self.assertNotIn("access-control-allow-origin", r.headers)


if __name__ == "__main__":
    unittest.main()
