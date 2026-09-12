import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from starlette.datastructures import Headers, UploadFile

from core.database import init_database
from routers import graphics
from services import graphic_renderer


SAMPLE_GEOJSON = json.dumps({
    "type": "FeatureCollection",
    "features": [{
        "type": "Feature",
        "properties": {"dn": 15, "fill": "#ffff00", "stroke": "#8b8000"},
        "geometry": {"type": "Polygon", "coordinates": [[[-100, 35], [-90, 35], [-90, 42], [-100, 42], [-100, 35]]]},
    }],
}).encode()


class GraphicsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.environment = patch.dict("os.environ", {"DATA_DIR": self.temporary.name})
        self.environment.start()
        graphics.ASSET_ROOT = Path(self.temporary.name) / "assets"
        init_database()

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    def _access(self):
        with patch.object(graphics, "verify_token", return_value="admin@example.com"):
            department = graphics.create_department(graphics.DepartmentCreate(name="Test Department", slug="test-department"), "admin")
            issued = graphics.issue_key(department["id"], "admin")
        return department, issued["key"]

    def test_official_spc_products_are_geojson(self):
        self.assertEqual(
            graphic_renderer.PRODUCT_URLS["spc_cat"],
            "https://www.spc.noaa.gov/products/outlook/day1otlk_cat.nolyr.geojson",
        )
        self.assertTrue(all(url.endswith(".nolyr.geojson") for url in graphic_renderer.PRODUCT_URLS.values()))

    def test_gis_renderer_produces_exact_png(self):
        with patch.object(graphic_renderer, "_fetch", return_value=SAMPLE_GEOJSON), patch.object(
            graphic_renderer, "_basemap", return_value=(Image.new("RGBA", (820, 350), (0, 0, 0, 0)), 8)
        ) as basemap:
            result = graphic_renderer.render_graphic({
                "id": "test", "product_id": "spc_four_panel", "header_text": "Day 1",
                "background_color": "#e8e8e8",
            })
        image = Image.open(io.BytesIO(result["bytes"]))
        self.assertEqual(image.size, (1920, 1080))
        self.assertEqual(result["renderer_version"], "graphics-gis-v8")
        self.assertEqual(len(result["source_urls"]), 4)
        self.assertEqual(basemap.call_count, 2)
        self.assertEqual(basemap.call_args_list[0].args[3], "rastertiles/voyager_nolabels")
        self.assertEqual(basemap.call_args_list[1].args[3], "rastertiles/voyager_only_labels")
        self.assertTrue(basemap.call_args_list[1].kwargs["transparent"])

    def test_png_logo_upload_is_bound_to_bundle(self):
        department, api_key = self._access()
        logo_bytes = io.BytesIO()
        Image.new("RGBA", (240, 90), (255, 0, 0, 180)).save(logo_bytes, format="PNG")
        upload = UploadFile(
            io.BytesIO(logo_bytes.getvalue()), filename="department.png",
            headers=Headers({"content-type": "image/png"}),
        )
        asset = asyncio.run(graphics.upload_logo(upload, f"Bearer {api_key}"))
        created = graphics.create_bundle(graphics.BundleCreate(
            id="logo-bundle", name="Logo Bundle", product_id="spc_cat",
            department_logo_asset_id=asset["asset_id"],
        ), f"Bearer {api_key}")
        self.assertEqual(created["config"]["department_logo_asset_id"], asset["asset_id"])
        self.assertEqual(len(list(graphics.ASSET_ROOT.glob("*.png"))), 1)
        session = graphics.graphics_session(f"Bearer {api_key}")
        self.assertEqual(session["department"]["name"], "Test Department")
        updated = graphics.update_bundle("logo-bundle", graphics.BundleCreate(
            id="logo-bundle", name="Updated Logo Bundle", product_id="spc_wind",
            department_logo_asset_id=asset["asset_id"],
        ), f"Bearer {api_key}")
        self.assertEqual(updated["version"], 2)
        self.assertEqual(updated["config"]["product_id"], "spc_wind")

    def test_geojson_asset_is_normalized_and_bound_to_department(self):
        department, api_key = self._access()
        upload = UploadFile(io.BytesIO(SAMPLE_GEOJSON), filename="bounds.geojson", headers=Headers({"content-type": "application/geo+json"}))
        asset = asyncio.run(graphics.upload_asset(upload, f"Bearer {api_key}"))
        created = graphics.create_bundle(graphics.BundleCreate(
            id="test-bundle", name="Test Bundle", product_id="spc_cat",
            jurisdiction_asset_id=asset["asset_id"], center=(-92.5, 38.4), zoom=7.0,
        ), f"Bearer {api_key}")
        self.assertEqual(created["bundle_id"], "test-bundle")
        self.assertEqual(created["image_url"], "https://cdn.showmefire.org/imggen/test-bundle/image.png")
        self.assertEqual(len(list(graphics.ASSET_ROOT.glob("*.geojson"))), 1)
        self.assertEqual(len(asset["suggested_center"]), 2)
        self.assertGreater(asset["suggested_zoom"], 1)

    def test_center_zoom_viewport_preserves_canvas_ratio(self):
        extent = graphic_renderer.viewport_extent((-92.5, 38.4), 6.3, 1920, 1080)
        west, south = graphic_renderer._mercator(extent[0], extent[2])
        east, north = graphic_renderer._mercator(extent[1], extent[3])
        self.assertAlmostEqual((east - west) / (north - south), 1920 / 1080, places=5)

    def test_invalid_api_key_is_rejected(self):
        with self.assertRaisesRegex(Exception, "Invalid or revoked API key"):
            graphics.list_bundles("Bearer not-a-real-key")

    def test_missing_auth_prompts_sign_in_instead_of_bearer_key(self):
        with self.assertRaisesRegex(Exception, "Sign in required"):
            graphics.list_bundles(None)


class GraphicsAccountTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.environment = patch.dict("os.environ", {"DATA_DIR": self.temporary.name})
        self.environment.start()
        graphics.ASSET_ROOT = Path(self.temporary.name) / "assets"
        init_database()
        with patch.object(graphics, "verify_token", return_value="admin@example.com"):
            self.department = graphics.create_department(
                graphics.DepartmentCreate(name="Invite Department", slug="invite-department"), "admin",
            )

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    def _invite(self, email="chief@department.gov"):
        with patch.object(graphics, "verify_token", return_value="admin@example.com"):
            return graphics.invite_department_user(
                self.department["id"], graphics.InviteCreate(email=email), "admin",
            )

    def test_invite_then_set_password_then_login(self):
        from fastapi import Response

        invite = self._invite()
        self.assertTrue(invite["token"].startswith("smf_"))

        verified = graphics.verify_invite(graphics.InviteVerify(email=invite["email"], token=invite["token"]))
        self.assertTrue(verified["valid"])

        response = Response()
        result = graphics.set_password(
            graphics.PasswordSet(email=invite["email"], token=invite["token"], password="correct-horse-battery9"),
            response,
        )
        self.assertTrue(result["success"])
        cookie_headers = [value.decode() for name, value in response.raw_headers if name == b"set-cookie"]
        self.assertTrue(any(header.startswith("graphics_access=") for header in cookie_headers))
        self.assertTrue(any(header.startswith("graphics_refresh=") for header in cookie_headers))

        # The token is now claimed: verifying it again must fail.
        with self.assertRaisesRegex(Exception, "invalid, expired, or already used"):
            graphics.verify_invite(graphics.InviteVerify(email=invite["email"], token=invite["token"]))

        login_response = Response()
        logged_in = graphics.graphics_login(
            graphics.PasswordLogin(email=invite["email"], password="correct-horse-battery9"), login_response,
        )
        self.assertTrue(logged_in["success"])

        with self.assertRaisesRegex(Exception, "Invalid email or password"):
            graphics.graphics_login(graphics.PasswordLogin(email=invite["email"], password="wrong-password-123"), Response())

    def test_weak_password_is_rejected(self):
        from fastapi import Response

        invite = self._invite("weak@department.gov")
        with self.assertRaisesRegex(Exception, "at least 12 characters"):
            graphics.set_password(
                graphics.PasswordSet(email=invite["email"], token=invite["token"], password="short1"), Response(),
            )

    def test_duplicate_invite_email_is_rejected(self):
        self._invite("duplicate@department.gov")
        with self.assertRaisesRegex(Exception, "already exists"):
            self._invite("duplicate@department.gov")

    def test_session_cookie_authorizes_department_endpoints(self):
        from fastapi import Response

        from core import security

        invite = self._invite("session@department.gov")
        response = Response()
        graphics.set_password(
            graphics.PasswordSet(email=invite["email"], token=invite["token"], password="correct-horse-battery9"),
            response,
        )
        # Simulate the cookie-renewal middleware exposing the freshly issued
        # access token to the request context, the way graphics_session.py does.
        access_cookie = next(
            header.decode() for name, header in response.raw_headers
            if name == b"set-cookie" and header.decode().startswith("graphics_access=")
        )
        token_value = access_cookie.split("graphics_access=", 1)[1].split(";", 1)[0]
        context = security.set_graphics_request_token(token_value)
        try:
            session = graphics.graphics_session(None)
            self.assertEqual(session["department"]["name"], "Invite Department")
        finally:
            security.reset_graphics_request_token(context)


if __name__ == "__main__":
    unittest.main()
