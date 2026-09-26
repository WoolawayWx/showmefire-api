import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core import database
from routers import discord_admin

SERVERS = [{"id": "g1", "name": "Show Me Fire", "channels": [{"id": "c1", "name": "alerts"}], "roles": []}]


class DiscordServerDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_patcher = patch.object(database, "get_db_path", return_value=Path(self._tmpdir.name) / "t.db")
        self._db_patcher.start()
        database.init_database()
        self._env = patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "", "DISCORD_CLIENT_ID": ""})
        self._env.start()

    def tearDown(self):
        self._env.stop()
        self._db_patcher.stop()
        self._tmpdir.cleanup()

    def _bot_ok(self):
        return {"ok": True, "source": "bot", "application_id": "app1", "servers": SERVERS, "errors": []}

    def test_bot_result_is_cached_and_served_when_bot_goes_down(self):
        with patch.object(discord_admin, "_discover_servers_via_bot", return_value=self._bot_ok()):
            fresh = discord_admin._fetch_discord_servers()
        self.assertFalse(fresh["stale"])
        with patch.object(discord_admin, "_discover_servers_via_bot", side_effect=RuntimeError("bot unreachable")):
            stale = discord_admin._fetch_discord_servers()
        self.assertTrue(stale["stale"])
        self.assertEqual(stale["servers"], SERVERS)
        self.assertIn("bot unreachable", stale["error"])
        self.assertEqual(stale["application_id"], "app1")

    def test_no_cache_and_bot_down_reports_error(self):
        with patch.object(discord_admin, "_discover_servers_via_bot", side_effect=RuntimeError("bot unreachable")):
            result = discord_admin._fetch_discord_servers()
        self.assertFalse(result["ok"])
        self.assertEqual(result["servers"], [])

    def test_discord_api_preferred_when_token_configured(self):
        rest = {**self._bot_ok(), "source": "discord_api"}
        with patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "tok"}), \
                patch.object(discord_admin, "_discover_servers_via_rest", return_value=rest) as via_rest, \
                patch.object(discord_admin, "_discover_servers_via_bot") as via_bot:
            result = discord_admin._fetch_discord_servers()
        via_rest.assert_called_once_with("tok")
        via_bot.assert_not_called()
        self.assertEqual(result["source"], "discord_api")

    def test_falls_back_to_bot_when_discord_api_fails(self):
        with patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "tok"}), \
                patch.object(discord_admin, "_discover_servers_via_rest", side_effect=RuntimeError("401")), \
                patch.object(discord_admin, "_discover_servers_via_bot", return_value=self._bot_ok()):
            result = discord_admin._fetch_discord_servers()
        self.assertEqual(result["source"], "bot")

    def test_rest_discovery_filters_channels_and_roles_and_isolates_failures(self):
        responses = {
            "/users/@me": {"id": "app1"},
            "/users/@me/guilds": [{"id": "g1", "name": "B"}, {"id": "g2", "name": "A-broken"}],
            "/guilds/g1/channels": [{"id": "t", "name": "alerts", "type": 0}, {"id": "v", "name": "voice", "type": 2}],
            "/guilds/g1/roles": [{"id": "e", "name": "@everyone"}, {"id": "s", "name": "Staff"},
                                 {"id": "b", "name": "Bot", "managed": True}],
        }

        def fake_get(path, _token):
            if path not in responses:
                raise RuntimeError("HTTP 403: Missing Access")
            return responses[path]

        with patch.object(discord_admin, "_discord_rest_get", side_effect=fake_get):
            result = discord_admin._discover_servers_via_rest("tok")
        self.assertEqual(result["application_id"], "app1")
        self.assertEqual([c["id"] for c in result["servers"][0]["channels"]], ["t"])
        self.assertEqual([r["id"] for r in result["servers"][0]["roles"]], ["s"])
        self.assertEqual(result["errors"][0]["id"], "g2")

    def test_invite_url_targets_bot_with_send_permissions(self):
        url = discord_admin._bot_invite_url("app1")
        self.assertIn("client_id=app1", url)
        self.assertIn("scope=bot+applications.commands", url)
        self.assertIn("permissions=183296", url)


if __name__ == "__main__":
    unittest.main()
