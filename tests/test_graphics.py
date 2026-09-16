import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from starlette.datastructures import Headers, UploadFile
from starlette.requests import Request

from core.database import init_database
from routers import graphics
from services import graphic_renderer
from services import graphics_email
from services import spc_graphics_watcher


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
            department = graphics.create_department(graphics.DepartmentCreate(
                name="Test Department", slug="test-department", email="contact@test.gov",
            ), "admin")
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
        self.assertEqual(result["renderer_version"], "graphics-gis-v9")
        self.assertEqual(len(result["source_urls"]), 4)
        self.assertEqual(basemap.call_count, 2)
        self.assertEqual(basemap.call_args_list[0].args[3], "rastertiles/voyager_nolabels")
        self.assertEqual(basemap.call_args_list[1].args[3], "rastertiles/voyager_only_labels")
        self.assertTrue(basemap.call_args_list[1].kwargs["transparent"])
        self.assertEqual(basemap.call_args_list[1].kwargs["zoom_bias"], 0)

    def test_bundle_supports_custom_graphic_theme(self):
        bundle = graphics.BundleCreate(
            id="theme-bundle", name="Theme Bundle", product_id="spc_cat",
            basemap_style="none", header_background_color="#102a43",
            header_text_color="#f0f4f8", accent_color="#2cb1bc",
            legend_background_color="#ffffff", legend_text_color="#102a43",
            border_color="#f0f4f8", border_width=1.8, outlook_opacity=0.65,
            show_town_labels=False, town_label_size="large",
        )
        self.assertEqual(bundle.basemap_style, "none")
        self.assertEqual(bundle.town_label_size, "large")
        self.assertFalse(bundle.show_town_labels)
        self.assertAlmostEqual(bundle.outlook_opacity, 0.65)

    def test_png_logo_upload_is_bound_to_bundle(self):
        department, api_key = self._access()
        logo_bytes = io.BytesIO()
        Image.new("RGBA", (240, 90), (255, 0, 0, 180)).save(logo_bytes, format="PNG")
        upload = UploadFile(
            io.BytesIO(logo_bytes.getvalue()), filename="department.png",
            headers=Headers({"content-type": "image/png"}),
        )
        with patch.object(graphics, "_r2_configured", return_value=True), patch.object(
            graphics, "_r2_client",
        ) as r2:
            asset = asyncio.run(graphics.upload_logo(upload, f"Bearer {api_key}"))
        upload_call = r2.return_value.put_object.call_args.kwargs
        self.assertEqual(
            upload_call["Key"],
            f"assets/departmentuploads/test-department/department-{asset['sha256'][:12]}.png",
        )
        self.assertEqual(upload_call["CacheControl"], "public,max-age=31536000,immutable")
        self.assertEqual(asset["url"], f"https://cdn.showmefire.org/{upload_call['Key']}")
        self.assertEqual(asyncio.run(graphics.list_logos(f"Bearer {api_key}"))["logos"][0]["asset_id"], asset["asset_id"])
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

    def test_admin_can_resume_existing_department_setup(self):
        department, _ = self._access()
        with patch.object(graphics, "verify_token", return_value="admin@example.com"):
            result = graphics.list_departments("admin")
            with self.assertRaisesRegex(Exception, "Department already exists"):
                graphics.create_department(
                    graphics.DepartmentCreate(
                        name="Test Department", slug="different-slug", email="other@test.gov",
                    ), "admin",
                )
        listed = next(item for item in result["departments"] if item["id"] == department["id"])
        self.assertEqual(listed["slug"], "test-department")
        self.assertEqual(listed["contact_email"], "contact@test.gov")
        self.assertEqual(listed["user_count"], 1)
        self.assertEqual(listed["bundle_count"], 0)

    def test_admin_can_edit_and_delete_department(self):
        department, _ = self._access()
        with patch.object(graphics, "verify_token", return_value="admin@example.com"):
            updated = graphics.update_department(
                department["id"],
                graphics.DepartmentCreate(
                    name="Renamed Department", slug="renamed-department", email="new-contact@test.gov",
                ),
                "admin",
            )
            self.assertTrue(updated["email_added"])
            with self.assertRaisesRegex(Exception, "Confirmation slug"):
                asyncio.run(graphics.delete_department(department["id"], "wrong-slug", "admin"))
            with patch.dict("os.environ", {
                "R2_ACCOUNT_ID": "", "R2_ACCESS_KEY_ID": "", "R2_SECRET_ACCESS_KEY": "",
            }):
                deleted = asyncio.run(graphics.delete_department(
                    department["id"], "renamed-department", "admin",
                ))
            self.assertTrue(deleted["deleted"])
            self.assertFalse(graphics.list_departments("admin")["departments"])

    def test_missing_auth_prompts_sign_in_instead_of_bearer_key(self):
        with self.assertRaisesRegex(Exception, "Sign in required"):
            graphics.list_bundles(None)

    def test_resend_login_email_uses_configured_sender(self):
        with patch.dict("os.environ", {
            "RESEND_API_KEY": "test-key",
            "GRAPHICS_EMAIL_FROM": "Show Me Fire <accounts@notify.showmefire.org>",
            "GRAPHICS_EMAIL_REPLY_TO": "support@showmefire.org",
        }), patch.object(graphics_email.requests, "post") as request:
            request.return_value.json.return_value = {"id": "email-id"}
            message_id = graphics_email.send_graphics_login_code("chief@example.gov", "123456")
        self.assertEqual(message_id, "email-id")
        payload = request.call_args.kwargs["json"]
        self.assertEqual(payload["to"], ["chief@example.gov"])
        self.assertEqual(payload["reply_to"], "support@showmefire.org")
        self.assertIn("123456", payload["text"])
        request.return_value.raise_for_status.assert_called_once()

    def test_spc_update_fans_out_only_affected_active_bundles(self):
        department, api_key = self._access()
        for bundle_id, product_id in (
            ("cat-bundle", "spc_cat"),
            ("wind-bundle", "spc_wind"),
            ("panel-bundle", "spc_four_panel"),
            ("alerts-bundle", "mo_alerts"),
        ):
            graphics.create_bundle(graphics.BundleCreate(
                id=bundle_id, name=bundle_id, product_id=product_id,
            ), f"Bearer {api_key}")

        # Establish the previous observation for every source, then change
        # only the categorical product.
        previous = {product_id: f"old-{product_id}" for product_id in graphic_renderer.PRODUCT_URLS}
        payload_by_product = {
            product_id: (b"new-cat" if product_id == "spc_cat" else f"old-{product_id}".encode())
            for product_id in graphic_renderer.PRODUCT_URLS
        }

        async def complete_job(job_id, bundle, department_id, api_key_id):
            with graphics._db() as db:
                db.execute(
                    "UPDATE graphic_jobs SET status='completed',finished_at=CURRENT_TIMESTAMP WHERE id=?",
                    (job_id,),
                )

        observed_hashes = {
            product_id: spc_graphics_watcher._fingerprint(payload)
            for product_id, payload in payload_by_product.items()
        }
        spc_graphics_watcher._record_processed(
            {product_id: spc_graphics_watcher._fingerprint(f"old-{product_id}".encode()) for product_id in previous},
            set(previous),
        )
        with patch.object(
            spc_graphics_watcher,
            "fetch_spc_product",
            side_effect=lambda product_id: payload_by_product[product_id],
        ), patch.object(graphics, "_run_job", side_effect=complete_job):
            result = asyncio.run(spc_graphics_watcher.refresh_spc_graphics())

        self.assertEqual(result["changed_products"], ["spc_cat"])
        self.assertEqual(result["queued"], 2)
        with graphics._db() as db:
            queued_bundles = {
                row[0] for row in db.execute(
                    "SELECT bundle_id FROM graphic_jobs ORDER BY bundle_id"
                ).fetchall()
            }
            state = db.execute(
                "SELECT source_fingerprint FROM graphic_source_state WHERE product_id='spc_cat'"
            ).fetchone()[0]
        self.assertEqual(queued_bundles, {"cat-bundle", "panel-bundle"})
        self.assertEqual(state, observed_hashes["spc_cat"])


class GraphicsAccountTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.environment = patch.dict("os.environ", {"DATA_DIR": self.temporary.name})
        self.environment.start()
        graphics.ASSET_ROOT = Path(self.temporary.name) / "assets"
        init_database()
        with patch.object(graphics, "verify_token", return_value="admin@example.com"):
            self.department = graphics.create_department(
                graphics.DepartmentCreate(
                    name="Invite Department", slug="invite-department", email="contact@invite.gov",
                ), "admin",
            )

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    def _request(self, ip="127.0.0.1"):
        return Request({
            "type": "http", "method": "POST", "path": "/", "query_string": b"",
            "headers": [], "client": (ip, 12345), "server": ("testserver", 80),
            "scheme": "http",
        })

    def _authorize(self, email="chief@department.gov"):
        with patch.object(graphics, "verify_token", return_value="admin@example.com"), patch.object(
            graphics, "send_graphics_login_code", return_value="email-id",
        ) as sender:
            result = asyncio.run(graphics.invite_department_user(
                self.department["id"], graphics.InviteCreate(email=email), self._request(), "admin",
            ))
        return result, sender.call_args.args[1]

    def test_authorize_then_verify_emailed_code(self):
        from fastapi import Response

        invite, code = self._authorize()
        self.assertTrue(invite["authorized"])
        self.assertTrue(invite["email_sent"])
        self.assertRegex(code, r"^\d{6}$")
        response = Response()
        result = graphics.verify_login_code(
            graphics.LoginCodeVerify(email=invite["email"], code=code), response,
        )
        self.assertTrue(result["success"])
        cookie_headers = [value.decode() for name, value in response.raw_headers if name == b"set-cookie"]
        self.assertTrue(any(header.startswith("graphics_access=") for header in cookie_headers))
        self.assertTrue(any(header.startswith("graphics_refresh=") for header in cookie_headers))

        # Codes are single-use.
        with self.assertRaisesRegex(Exception, "Invalid or expired"):
            graphics.verify_login_code(
                graphics.LoginCodeVerify(email=invite["email"], code=code), Response(),
            )

    def test_unknown_email_gets_generic_request_response(self):
        with patch.object(graphics, "send_graphics_login_code") as sender:
            result = asyncio.run(graphics.request_login_code(
                graphics.LoginCodeRequest(email="unknown@example.com"), self._request(),
            ))
        self.assertTrue(result["success"])
        sender.assert_not_called()

    def test_code_locks_after_five_bad_attempts(self):
        from fastapi import Response

        invite, code = self._authorize("attempts@department.gov")
        for _ in range(5):
            with self.assertRaisesRegex(Exception, "Invalid or expired"):
                graphics.verify_login_code(
                    graphics.LoginCodeVerify(email=invite["email"], code="999999"), Response(),
                )
        with self.assertRaisesRegex(Exception, "Invalid or expired"):
            graphics.verify_login_code(
                graphics.LoginCodeVerify(email=invite["email"], code=code), Response(),
            )

    def test_session_cookie_authorizes_department_endpoints(self):
        from fastapi import Response

        from core import security

        invite, code = self._authorize("session@department.gov")
        response = Response()
        graphics.verify_login_code(
            graphics.LoginCodeVerify(email=invite["email"], code=code), response,
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

    def test_existing_user_must_accept_versioned_terms_and_can_remove_access(self):
        from fastapi import Response
        from core import security

        invite, code = self._authorize("terms@department.gov")
        login_response = Response()
        graphics.verify_login_code(
            graphics.LoginCodeVerify(email=invite["email"], code=code), login_response,
        )
        access_cookie = next(
            header.decode() for name, header in login_response.raw_headers
            if name == b"set-cookie" and header.decode().startswith("graphics_access=")
        )
        token_value = access_cookie.split("graphics_access=", 1)[1].split(";", 1)[0]
        context = security.set_graphics_request_token(token_value)
        try:
            self.assertFalse(graphics.graphics_session(None)["terms"]["accepted"])
            with self.assertRaisesRegex(Exception, "terms acceptance required"):
                graphics.list_bundles(None)
            terms = graphics.graphics_terms(None)
            self.assertEqual(terms["version"], graphics.GRAPHICS_TERMS_VERSION)
            self.assertGreaterEqual(len(terms["sections"]), 4)
            self.assertTrue(graphics.accept_graphics_terms(None)["accepted"])
            self.assertEqual(graphics.list_bundles(None)["bundles"], [])

            with self.assertRaisesRegex(Exception, "REMOVE MY ACCESS"):
                graphics.decline_graphics_terms(
                    graphics.TermsDecline(confirmation="no"), Response(), None,
                )
            decline_response = Response()
            removed = graphics.decline_graphics_terms(
                graphics.TermsDecline(confirmation="REMOVE MY ACCESS"), decline_response, None,
            )
            self.assertTrue(removed["access_removed"])
            with graphics._db() as db:
                self.assertIsNone(db.execute(
                    "SELECT id FROM graphic_department_users WHERE lower(email)=lower(?)",
                    (invite["email"],),
                ).fetchone())
                event = db.execute(
                    "SELECT decision FROM graphic_terms_events WHERE lower(email)=lower(?) AND decision='declined'",
                    (invite["email"],),
                ).fetchone()
            self.assertEqual(event["decision"], "declined")
        finally:
            security.reset_graphics_request_token(context)


if __name__ == "__main__":
    unittest.main()
