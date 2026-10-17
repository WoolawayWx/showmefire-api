"""Tile responses carry a stable `Access-Control-Allow-Origin: *`, independent of the caller."""
import unittest
from pathlib import Path

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


class CredentialedOriginTests(unittest.TestCase):
    def test_pages_previews_and_loopback_are_allowed(self):
        from core.cors_origins import origin_allowed
        for origin in ("https://showmefire-website.pages.dev", "https://3f2a1c9b.showmefire-website.pages.dev",
                       "https://my-branch.showmefire-website.pages.dev", "https://a.b.showmefire-website.pages.dev",
                       "http://localhost:3000", "http://127.0.0.1:5173", "http://[::1]:3001"):
            self.assertTrue(origin_allowed(origin), origin)

    def test_lookalikes_are_rejected(self):
        from core.cors_origins import origin_allowed
        for origin in ("https://evil-showmefire-website.pages.dev", "https://showmefire-website.pages.dev.evil.com",
                       "http://abc.showmefire-website.pages.dev", "https://showmefire-website.pages.dev:8443",
                       "https://pages.dev", "https://x.other.pages.dev", "https://localhost.evil.com",
                       "https://abc.showmefire-website.pages.dev/", "https://dev.showmefire.org"):
            self.assertFalse(origin_allowed(origin), origin)  # dev.showmefire.org is an exact entry in main.py

    def test_main_uses_the_shared_pattern_and_lists_dev(self):
        source = (Path(__file__).resolve().parent.parent / "main.py").read_text()
        self.assertIn("allow_origin_regex=ORIGIN_REGEX", source)
        self.assertIn('"https://dev.showmefire.org"', source)

    def test_middleware_answers_a_pages_preview_with_credentials(self):
        from core.cors_origins import ORIGIN_REGEX
        app = FastAPI()
        app.add_middleware(CORSMiddleware, allow_origins=["https://dev.showmefire.org"], allow_origin_regex=ORIGIN_REGEX,
                           allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

        @app.get("/ping")
        def ping():
            return {"ok": True}

        client = TestClient(app)
        for origin, allowed in (("https://abc123.showmefire-website.pages.dev", True), ("https://dev.showmefire.org", True),
                                ("https://evil-showmefire-website.pages.dev", False)):
            response = client.get("/ping", headers={"Origin": origin})
            self.assertEqual(response.headers.get("access-control-allow-origin"), origin if allowed else None, origin)
            if allowed:
                self.assertEqual(response.headers.get("access-control-allow-credentials"), "true")
        preflight = client.options("/ping", headers={"Origin": "https://abc123.showmefire-website.pages.dev",
                                                     "Access-Control-Request-Method": "POST",
                                                     "Access-Control-Request-Headers": "content-type"})
        self.assertEqual(preflight.status_code, 200)


if __name__ == "__main__":
    unittest.main()