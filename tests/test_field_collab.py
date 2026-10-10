"""Tests for the ShowMeFire Field team collaboration API (routers/field.py, core/field_store.py)."""
import os
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from core import field_store
from routers import field as field_router

CREATE_KEY = "operator-secret"


def _uid(n: int) -> str:
    return f"feature{n:04d}"


class FieldApiTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        env = patch.dict(
            os.environ, {"DATA_DIR": self._tmp.name, "FIELD_TEAM_CREATE_KEY": CREATE_KEY}
        )
        env.start()
        self.addCleanup(env.stop)
        field_router._attempts.clear()
        field_store.ensure_field_schema()
        app = FastAPI()
        app.include_router(field_router.router)
        self.client = TestClient(app)
        self.app = app

    # helpers -------------------------------------------------------------
    def make_team(self, team="Crew A", owner="Owner"):
        r = self.client.post(
            "/api/field/v1/teams",
            json={"team_name": team, "display_name": owner},
            headers={"X-Field-Create-Key": CREATE_KEY},
        )
        self.assertEqual(r.status_code, 201, r.text)
        return r.json()

    def join(self, code, name="Member"):
        r = self.client.post("/api/field/v1/teams/join", json={"join_code": code, "display_name": name})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    @staticmethod
    def auth(tok):
        return {"Authorization": f"Bearer {tok['token']}"}

    def make_map(self, tok, name="Search area", features=None):
        r = self.client.post(
            "/api/field/v1/maps",
            json={"name": name, "features": features or {}},
            headers=self.auth(tok),
        )
        self.assertEqual(r.status_code, 201, r.text)
        return r.json()

    def ops(self, tok, map_id, ops):
        return self.client.post(
            f"/api/field/v1/maps/{map_id}/ops", json={"ops": ops}, headers=self.auth(tok)
        )


class TeamAndAuthTests(FieldApiTestCase):
    def test_creation_requires_the_operator_key(self):
        body = {"team_name": "T", "display_name": "O"}
        # No credentials at all: told to use an account.
        r = self.client.post("/api/field/v1/teams", json=body)
        self.assertEqual(r.status_code, 401)
        r = self.client.post("/api/field/v1/teams", json=body, headers={"X-Field-Create-Key": "nope"})
        self.assertEqual(r.status_code, 403)
        # With no key configured at all, operator creation is off even for a matching empty header.
        with patch.dict(os.environ, {"FIELD_TEAM_CREATE_KEY": ""}):
            r = self.client.post("/api/field/v1/teams", json=body, headers={"X-Field-Create-Key": ""})
            self.assertEqual(r.status_code, 401)
        t = self.make_team()
        self.assertTrue(t["token"].startswith("fld_"))
        self.assertEqual(len(t["join_code"]), field_store.JOIN_CODE_LEN)

    def test_join_codes_are_forgiving_but_not_guessable(self):
        t = self.make_team()
        sloppy = t["join_code"][:5].lower() + " - " + t["join_code"][5:]
        j = self.join(sloppy, "  Pat   Lee ")
        me = self.client.get("/api/field/v1/me", headers=self.auth(j)).json()
        self.assertEqual(me["display_name"], "Pat Lee")
        self.assertFalse(me["is_owner"])
        self.assertEqual([m["display_name"] for m in me["members"]], ["Owner", "Pat Lee"])
        r = self.client.post("/api/field/v1/teams/join", json={"join_code": "AAAAAAAAAA", "display_name": "x"})
        self.assertEqual(r.status_code, 404)

    def test_guessing_is_throttled(self):
        codes = [f"WRONG{i:05d}" for i in range(field_router.AUTH_ATTEMPTS + 2)]
        statuses = [
            self.client.post("/api/field/v1/teams/join", json={"join_code": c, "display_name": "x"}).status_code
            for c in codes
        ]
        self.assertEqual(statuses[: field_router.AUTH_ATTEMPTS], [404] * field_router.AUTH_ATTEMPTS)
        self.assertEqual(statuses[field_router.AUTH_ATTEMPTS :], [429, 429])

    def test_tokens_are_required_and_checked(self):
        for h in ({}, {"Authorization": "Bearer junk"}, {"Authorization": "Basic abc"}, {"Authorization": "Bearer fld_" + "x" * 200}):
            r = self.client.get("/api/field/v1/me", headers=h)
            self.assertEqual(r.status_code, 401, h)
        t = self.make_team()
        self.assertEqual(self.client.get("/api/field/v1/me", headers=self.auth(t)).status_code, 200)

    def test_secrets_are_not_stored_in_plaintext(self):
        t = self.make_team()
        conn = sqlite3.connect(Path(self._tmp.name) / "showmefire.db")
        dump = "\n".join(conn.iterdump())
        conn.close()
        self.assertNotIn(t["token"], dump)
        self.assertNotIn(t["join_code"], dump)

    def test_owner_controls(self):
        t = self.make_team()
        m = self.join(t["join_code"], "Member")
        # Members can't rotate the code or remove people.
        self.assertEqual(self.client.post("/api/field/v1/teams/join-code", headers=self.auth(m)).status_code, 403)
        self.assertEqual(self.client.delete(f"/api/field/v1/members/{t['member_id']}", headers=self.auth(m)).status_code, 403)
        new = self.client.post("/api/field/v1/teams/join-code", headers=self.auth(t)).json()["join_code"]
        self.assertNotEqual(new, t["join_code"])
        self.assertEqual(
            self.client.post("/api/field/v1/teams/join", json={"join_code": t["join_code"], "display_name": "late"}).status_code,
            404,
        )
        self.join(new, "Newer")
        # Removing a member kills their token; owners cannot remove themselves.
        self.assertEqual(self.client.delete(f"/api/field/v1/members/{m['member_id']}", headers=self.auth(t)).status_code, 204)
        self.assertEqual(self.client.get("/api/field/v1/me", headers=self.auth(m)).status_code, 401)
        self.assertEqual(self.client.delete(f"/api/field/v1/members/{t['member_id']}", headers=self.auth(t)).status_code, 400)
        self.assertEqual(self.client.delete("/api/field/v1/members/nobody", headers=self.auth(t)).status_code, 404)

    def test_team_size_is_capped(self):
        t = self.make_team()
        with patch.object(field_store, "MAX_MEMBERS_PER_TEAM", 2):
            self.join(t["join_code"], "Two")
            r = self.client.post("/api/field/v1/teams/join", json={"join_code": t["join_code"], "display_name": "Three"})
            self.assertEqual(r.status_code, 409)


class MapTests(FieldApiTestCase):
    def test_maps_start_from_a_snapshot_and_follow_operations(self):
        t = self.make_team()
        feats = {_uid(1): {"name": "Camp", "geometry": {"type": "Marker", "lng": -92.1, "lat": 38.5}}}
        m = self.make_map(t, features=feats)
        self.assertEqual(m["rev"], 1)
        got = self.client.get(f"/api/field/v1/maps/{m['id']}", headers=self.auth(t)).json()
        self.assertEqual(got["features"], feats)
        r = self.ops(t, m["id"], [
            {"op": "upsert", "uid": _uid(2), "feature": {"name": "Hazard"}},
            {"op": "upsert", "uid": _uid(1), "feature": {"name": "Camp (moved)"}},
            {"op": "delete", "uid": _uid(2)},
        ])
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json(), {"rev": 4, "applied": 3})
        got = self.client.get(f"/api/field/v1/maps/{m['id']}", headers=self.auth(t)).json()
        self.assertEqual(got["rev"], 4)
        self.assertEqual(list(got["features"]), [_uid(1)])
        self.assertEqual(got["features"][_uid(1)]["name"], "Camp (moved)")
        # A deleted feature can be re-added (a teammate undoing a delete).
        self.ops(t, m["id"], [{"op": "upsert", "uid": _uid(2), "feature": {"name": "Hazard again"}}])
        got = self.client.get(f"/api/field/v1/maps/{m['id']}", headers=self.auth(t)).json()
        self.assertIn(_uid(2), got["features"])

    def test_bad_operations_change_nothing(self):
        t = self.make_team()
        m = self.make_map(t)
        for ops in (
            [{"op": "upsert", "uid": "short", "feature": {}}],
            [{"op": "upsert", "uid": "has spaces here", "feature": {}}],
            [{"op": "upsert", "uid": _uid(1)}],  # no feature
            [{"op": "explode", "uid": _uid(1)}],
            [{"op": "upsert", "uid": _uid(1), "feature": {"big": "x" * (field_store.MAX_FEATURE_BYTES + 10)}}],
            [],
        ):
            r = self.ops(t, m["id"], ops)
            self.assertIn(r.status_code, (413, 422), (ops[:1], r.status_code))
        # A batch with one bad operation is rejected whole: the good one must not apply.
        r = self.ops(t, m["id"], [
            {"op": "upsert", "uid": _uid(5), "feature": {"ok": True}},
            {"op": "upsert", "uid": "bad", "feature": {}},
        ])
        self.assertEqual(r.status_code, 422)
        got = self.client.get(f"/api/field/v1/maps/{m['id']}", headers=self.auth(t)).json()
        self.assertEqual((got["rev"], got["features"]), (0, {}))

    def test_teams_cannot_see_each_others_maps(self):
        a, b = self.make_team("A"), self.make_team("B")
        m = self.make_map(a, features={_uid(1): {"secret": True}})
        h = self.auth(b)
        self.assertEqual(self.client.get(f"/api/field/v1/maps/{m['id']}", headers=h).status_code, 404)
        self.assertEqual(self.ops(b, m["id"], [{"op": "delete", "uid": _uid(1)}]).status_code, 404)
        self.assertEqual(self.client.delete(f"/api/field/v1/maps/{m['id']}", headers=h).status_code, 404)
        self.assertEqual(self.client.get("/api/field/v1/events", params={"map_id": m["id"]}, headers=h).status_code, 404)
        self.assertEqual(self.client.get("/api/field/v1/maps", headers=h).json(), {"maps": []})

    def test_only_owners_delete_maps_and_map_counts_are_capped(self):
        t = self.make_team()
        member = self.join(t["join_code"])
        m = self.make_map(member)
        self.assertEqual(self.client.delete(f"/api/field/v1/maps/{m['id']}", headers=self.auth(member)).status_code, 403)
        self.assertEqual(self.client.delete(f"/api/field/v1/maps/{m['id']}", headers=self.auth(t)).status_code, 204)
        with patch.object(field_store, "MAX_MAPS_PER_TEAM", 1):
            self.make_map(t, "one")
            r = self.client.post("/api/field/v1/maps", json={"name": "two"}, headers=self.auth(t))
            self.assertEqual(r.status_code, 409)

    def test_oversized_requests_are_refused(self):
        t = self.make_team()
        big = "x" * (field_router.MAX_BODY_BYTES + 100)
        r = self.client.post("/api/field/v1/maps", content=big, headers={**self.auth(t), "Content-Type": "application/json"})
        self.assertEqual(r.status_code, 413)


class EventsTests(FieldApiTestCase):
    def test_events_return_ordered_operations_with_authors(self):
        t = self.make_team()
        u = self.join(t["join_code"], "Una")
        m = self.make_map(t)
        self.ops(t, m["id"], [{"op": "upsert", "uid": _uid(1), "feature": {"n": 1}}])
        self.ops(u, m["id"], [{"op": "upsert", "uid": _uid(2), "feature": {"n": 2}}, {"op": "delete", "uid": _uid(1)}])
        e = self.client.get("/api/field/v1/events", params={"map_id": m["id"], "since": 0}, headers=self.auth(t)).json()
        self.assertEqual([o["rev"] for o in e["ops"]], [1, 2, 3])
        self.assertEqual([o["by"] for o in e["ops"]], [t["member_id"], u["member_id"], u["member_id"]])
        self.assertEqual(e["ops"][2], {"rev": 3, "uid": _uid(1), "op": "delete", "feature": None, "by": u["member_id"]})
        self.assertEqual(e["rev"], 3)
        self.assertFalse(e["reset"])
        # Resuming from the returned revision yields nothing new.
        e2 = self.client.get("/api/field/v1/events", params={"map_id": m["id"], "since": 3}, headers=self.auth(t)).json()
        self.assertEqual((e2["ops"], e2["reset"]), ([], False))

    def test_stale_or_future_cursors_ask_for_a_reload(self):
        t = self.make_team()
        m = self.make_map(t)
        with patch.object(field_store, "OPS_KEEP", 3):
            for i in range(8):
                self.ops(t, m["id"], [{"op": "upsert", "uid": _uid(i), "feature": {"i": i}}])
            old = self.client.get("/api/field/v1/events", params={"map_id": m["id"], "since": 1}, headers=self.auth(t)).json()
            self.assertTrue(old["reset"])
            recent = self.client.get("/api/field/v1/events", params={"map_id": m["id"], "since": 6}, headers=self.auth(t)).json()
            self.assertFalse(recent["reset"])
            self.assertEqual([o["rev"] for o in recent["ops"]], [7, 8])
        future = self.client.get("/api/field/v1/events", params={"map_id": m["id"], "since": 999}, headers=self.auth(t)).json()
        self.assertTrue(future["reset"])
        bad = self.client.get("/api/field/v1/events", params={"since": -1}, headers=self.auth(t))
        self.assertEqual(bad.status_code, 422)

    def test_long_poll_waits_then_wakes_on_a_change(self):
        t = self.make_team()
        m = self.make_map(t)
        h = self.auth(t)
        # Quiet map: returns empty after roughly the wait.
        t0 = time.monotonic()
        r = self.client.get("/api/field/v1/events", params={"map_id": m["id"], "since": 0, "wait": 1}, headers=h).json()
        self.assertGreaterEqual(time.monotonic() - t0, 0.9)
        self.assertEqual(r["ops"], [])

        # A change made while a client is waiting releases it long before the wait ends.
        def later():
            time.sleep(0.8)
            TestClient(self.app).post(
                f"/api/field/v1/maps/{m['id']}/ops",
                json={"ops": [{"op": "upsert", "uid": _uid(9), "feature": {"hello": 1}}]},
                headers=h,
            )

        th = threading.Thread(target=later)
        th.start()
        t0 = time.monotonic()
        r = TestClient(self.app).get("/api/field/v1/events", params={"map_id": m["id"], "since": 0, "wait": 20}, headers=h).json()
        elapsed = time.monotonic() - t0
        th.join()
        self.assertLess(elapsed, 6)
        self.assertEqual([o["uid"] for o in r["ops"]], [_uid(9)])


class PositionTests(FieldApiTestCase):
    def post_pos(self, tok, lat=38.5, lng=-92.1, **kw):
        return self.client.post("/api/field/v1/positions", json={"lat": lat, "lng": lng, **kw}, headers=self.auth(tok))

    def events(self, tok, pos_since=0):
        return self.client.get("/api/field/v1/events", params={"pos_since": pos_since}, headers=self.auth(tok)).json()

    def test_teammates_see_positions_and_others_do_not(self):
        a, b = self.make_team("A"), self.make_team("B")
        u = self.join(a["join_code"], "Una")
        r = self.post_pos(u, speed=1.5, course=90, accuracy=5)
        self.assertEqual(r.status_code, 200, r.text)
        seen = self.events(a)["positions"]
        self.assertEqual(len(seen), 1)
        self.assertEqual((seen[0]["display_name"], seen[0]["speed"], seen[0]["cleared"]), ("Una", 1.5, False))
        self.assertEqual(self.events(b)["positions"], [])
        # Only changes after the cursor come back.
        cursor = self.events(a)["pos_seq"]
        self.assertEqual(self.events(a, cursor)["positions"], [])

    def test_validation_and_rate_limit(self):
        t = self.make_team()
        for body in ({"lat": 91, "lng": 0}, {"lat": 0, "lng": 181}, {"lat": 0, "lng": 0, "speed": -1}, {"lat": "x", "lng": 0}):
            r = self.client.post("/api/field/v1/positions", json=body, headers=self.auth(t))
            self.assertEqual(r.status_code, 422, body)
        self.assertEqual(self.post_pos(t).status_code, 200)
        self.assertEqual(self.post_pos(t).status_code, 429, "more than once a second is refused")

    def test_stopping_sharing_and_removal_tell_teammates(self):
        a = self.make_team()
        u = self.join(a["join_code"], "Una")
        self.post_pos(u)
        first = self.events(a)
        self.assertEqual(len(first["positions"]), 1)
        self.client.delete("/api/field/v1/positions", headers=self.auth(u))
        after = self.events(a, first["pos_seq"])["positions"]
        self.assertEqual([(p["member_id"], p["cleared"]) for p in after], [(u["member_id"], True)])
        # Sharing again clears the flag.
        time.sleep(1.1)
        self.post_pos(u)
        self.assertFalse(self.events(a)["positions"][0]["cleared"])
        # Removing the member makes their marker disappear for everyone else.
        seq = self.events(a)["pos_seq"]
        self.client.delete(f"/api/field/v1/members/{u['member_id']}", headers=self.auth(a))
        gone = self.events(a, seq)["positions"]
        self.assertTrue(gone and gone[0]["cleared"])


if __name__ == "__main__":
    unittest.main()
