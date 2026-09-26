import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from core import database, security
from routers import discord_admin
from services import discord_notifier, discord_rest

ENV = {"DISCORD_CLIENT_ID": "app1", "DISCORD_CLIENT_SECRET": "shh", "DISCORD_BOT_TOKEN": ""}


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class DiscordOAuthTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db = patch.object(database, "get_db_path", return_value=Path(self._tmpdir.name) / "t.db")
        self._db.start()
        database.init_database()
        self._env = patch.dict(os.environ, ENV)
        self._env.start()
        self.token = security.create_access_token({"sub": "staff@showmefire.org"})

    def tearDown(self):
        self._env.stop()
        self._db.stop()
        self._tmpdir.cleanup()

    def _start(self, **kwargs):
        response = run(discord_admin.discord_oauth_start(token=self.token, **kwargs))
        return response, parse_qs(urlparse(response.headers["location"]).query)

    def _callback(self, state, **kwargs):
        user = {"id": "u1", "username": "cade", "global_name": "Cade", "avatar": "abc"}
        guilds = [
            {"id": "g1", "name": "Show Me Fire", "owner": True, "permissions": "0"},
            {"id": "g2", "name": "Friends", "owner": False, "permissions": "0"},
            {"id": "g3", "name": "County EM", "owner": False, "permissions": str(0x20)},
        ]
        with patch.object(discord_admin, "_exchange_oauth_code", return_value={"access_token": "at"}), \
                patch.object(discord_rest, "get", side_effect=lambda path, **kw: user if path == "/users/@me" else guilds):
            return run(discord_admin.discord_oauth_callback(code="c", state=state, **kwargs))

    def test_connect_redirects_to_discord_with_state_and_callback(self):
        response, params = self._start()
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers["location"].startswith("https://discord.com/oauth2/authorize"))
        self.assertEqual(params["scope"], ["identify guilds"])
        self.assertEqual(params["redirect_uri"], [discord_admin._oauth_redirect_uri()])
        self.assertTrue(params["redirect_uri"][0].endswith("/api/admin/discord/oauth/callback"))
        with patch.dict(os.environ, {"DISCORD_OAUTH_REDIRECT_URI": "https://api.showmefire.org/api/admin/discord/oauth/callback"}):
            _, pinned = self._start()
        self.assertEqual(pinned["redirect_uri"], ["https://api.showmefire.org/api/admin/discord/oauth/callback"])
        self.assertTrue(params["state"][0])

    def test_invite_targets_guild_with_bot_scope(self):
        _, params = self._start(purpose="invite", guild_id="123")
        self.assertIn("bot", params["scope"][0].split())
        self.assertEqual(params["guild_id"], ["123"])
        self.assertEqual(params["permissions"], [str(discord_admin.BOT_INVITE_PERMISSIONS)])

    def test_not_configured_sends_admin_back_with_error(self):
        with patch.dict(os.environ, {"DISCORD_CLIENT_SECRET": ""}):
            response = run(discord_admin.discord_oauth_start(token=self.token))
        self.assertIn("discord_error=not_configured", response.headers["location"])

    def test_callback_links_account_and_lists_manageable_servers(self):
        _, params = self._start()
        response = self._callback(params["state"][0])
        self.assertIn("/admin/discord?discord=connected", response.headers["location"])
        with patch.object(discord_admin, "_fetch_discord_servers", return_value={"servers": [{"id": "g1"}], "stale": False}):
            account = run(discord_admin.get_discord_account(token=self.token))
        self.assertTrue(account["linked"])
        self.assertEqual(account["account"]["display_name"], "Cade")
        by_id = {g["id"]: g for g in account["guilds"]}
        self.assertEqual(set(by_id), {"g1", "g3"})  # g2: no manage permission
        self.assertTrue(by_id["g1"]["bot_present"])
        self.assertFalse(by_id["g3"]["bot_present"])
        self.assertIn("guild_id=g3", by_id["g3"]["add_url"])

    def test_state_is_single_use(self):
        _, params = self._start()
        self._callback(params["state"][0])
        response = self._callback(params["state"][0])
        self.assertIn("discord_error=expired", response.headers["location"])

    def test_callback_refuses_different_signed_in_admin(self):
        _, params = self._start()
        other = security.create_access_token({"sub": "someone-else@showmefire.org"})
        ctx = security.set_request_token(other)
        try:
            response = self._callback(params["state"][0])
        finally:
            security.reset_request_token(ctx)
        self.assertIn("discord_error=account_mismatch", response.headers["location"])
        self.assertIsNone(database.get_discord_admin_link("staff@showmefire.org"))

    def test_cancel_on_discord_returns_error(self):
        response = run(discord_admin.discord_oauth_callback(error="access_denied"))
        self.assertIn("discord_error=access_denied", response.headers["location"])


class DiscordRestDeliveryTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db = patch.object(database, "get_db_path", return_value=Path(self._tmpdir.name) / "t.db")
        self._db.start()
        database.init_database()

    def tearDown(self):
        self._db.stop()
        self._tmpdir.cleanup()

    def test_send_event_uses_direct_delivery_when_token_set_and_skips_secret(self):
        with patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "tok"}), \
                patch.object(discord_rest, "deliver_event", return_value=True) as deliver, \
                patch.object(discord_notifier.request, "urlopen") as webhook:
            self.assertTrue(discord_notifier._send_event({"event_type": "forecast_ready"}))
        deliver.assert_called_once()
        webhook.assert_not_called()

    def test_fire_alert_posts_embed_with_attached_map_and_role_mention(self):
        payload = {"event_type": "fire_alert", "event_id": "fire_alert:x", "event": "Red Flag Warning",
                   "area_description": "Boone", "headline": "RFW", "onset": "2026-09-26T18:00:00Z",
                   "image_url": "https://api.example/images/mo-firewx-alerts.png",
                   "target_channel_id": "555", "mention_role_ids": ["777"]}
        with patch.object(discord_rest, "resolve_channel", return_value="555"), \
                patch.object(discord_rest, "_fetch_image", return_value=b"\x89PNG"), \
                patch.object(discord_rest, "send_message") as send:
            self.assertTrue(discord_rest.deliver_event(payload, {}))
        args, kwargs = send.call_args
        self.assertEqual(args[0], "555")
        self.assertEqual(kwargs["content"], "<@&777>")
        self.assertEqual(kwargs["mention_role_ids"], ["777"])
        embed = kwargs["embeds"][0]
        self.assertEqual(embed["title"], "Red Flag Warning: Boone")
        self.assertTrue(embed["image"]["url"].startswith("attachment://"))
        self.assertEqual(kwargs["files"][0][2], "image/png")

    def test_staff_alert_never_falls_back_to_default_channel(self):
        def fake_get(path, **kwargs):
            if path == "/channels/111":
                return {"type": 0}
            raise discord_rest.DiscordRestError(404, "Unknown Channel")
        with patch.object(discord_rest, "get", side_effect=fake_get):
            with self.assertRaises(RuntimeError):
                discord_rest.resolve_channel("222", None, default_id="111", strict=True)
            self.assertEqual(discord_rest.resolve_channel("222", None, default_id="111"), "111")

    def test_multipart_body_contains_payload_and_file(self):
        body, content_type = discord_rest._multipart({"content": "hi"}, [("map.png", b"PNGDATA", "image/png")])
        self.assertIn("multipart/form-data; boundary=", content_type)
        self.assertIn(b'name="payload_json"', body)
        self.assertIn(b'name="files[0]"; filename="map.png"', body)
        self.assertIn(b"PNGDATA", body)

    def test_allowed_mentions_only_permit_configured_roles(self):
        with patch.object(discord_rest, "_call", return_value={}) as call:
            discord_rest.send_message("1", content="<@&9>", embeds=[], mention_role_ids=["9"])
        import json
        sent = json.loads(call.call_args.kwargs["body"])
        self.assertEqual(sent["allowed_mentions"], {"parse": [], "roles": ["9"]})


if __name__ == "__main__":
    unittest.main()
