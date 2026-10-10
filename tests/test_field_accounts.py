"""Tests for Field accounts, plans, quotas and usage metering."""
import sqlite3
import time
from pathlib import Path
from unittest.mock import patch

from tests.test_field_collab import FieldApiTestCase, _uid


class AccountTestCase(FieldApiTestCase):
    def setUp(self):
        super().setUp()
        # Stand in for the real admin JWT check.
        p = patch(
            "routers.field.verify_token",
            side_effect=lambda t=None: "admin@example.com" if t == "admin-jwt" else None,
        )
        p.start()
        self.addCleanup(p.stop)

    ADMIN = {"token": "admin-jwt"}

    def grant(self, name="Central Crossing FPD", plan="team", **extra):
        r = self.client.post(
            "/api/field/v1/admin/accounts",
            params=self.ADMIN,
            json={"name": name, "plan": plan, **extra},
        )
        self.assertEqual(r.status_code, 201, r.text)
        return r.json()

    def team_with(self, acct, name="Crew"):
        r = self.client.post(
            "/api/field/v1/teams",
            json={"team_name": name, "display_name": "Owner"},
            headers={"Authorization": f"Bearer {acct['token']}"},
        )
        return r

    def acct_headers(self, acct):
        return {"Authorization": f"Bearer {acct['token']}"}


class AdminAndCreationTests(AccountTestCase):
    def test_admin_endpoints_need_the_admin_token(self):
        for method, path in (("get", "/admin/accounts"), ("get", "/admin/usage")):
            self.assertEqual(getattr(self.client, method)("/api/field/v1" + path).status_code, 401)
        r = self.client.post("/api/field/v1/admin/accounts", json={"name": "x", "plan": "team"})
        self.assertEqual(r.status_code, 401)
        r = self.client.get("/api/field/v1/admin/accounts", params={"token": "wrong"})
        self.assertEqual(r.status_code, 401)

    def test_granting_an_account_and_creating_a_team_with_it(self):
        a = self.grant()
        self.assertTrue(a["token"].startswith("fldacct_"))
        self.assertEqual((a["plan"], a["status"]), ("team", "active"))
        self.assertEqual(a["limits"]["max_members"], 10)
        # The token is shown once: listings never include it.
        listed = self.client.get("/api/field/v1/admin/accounts", params=self.ADMIN).json()["accounts"]
        self.assertEqual(len(listed), 1)
        self.assertNotIn("token", listed[0])
        r = self.team_with(a)
        self.assertEqual(r.status_code, 201, r.text)
        owner = r.json()
        me = self.client.get("/api/field/v1/me", headers=self.auth(owner)).json()
        self.assertEqual((me["plan"]["plan"], me["plan"]["status"]), ("team", "active"))
        self.assertEqual(me["plan"]["usage"], {"members": 1, "maps": 0})
        # The plan allows one team; a second is refused with a clear message.
        r2 = self.team_with(a, "Second")
        self.assertEqual(r2.status_code, 403)
        self.assertIn("1 team", r2.json()["detail"])

    def test_bad_account_tokens_and_operator_teams(self):
        r = self.client.post(
            "/api/field/v1/teams",
            json={"team_name": "T", "display_name": "O"},
            headers={"Authorization": "Bearer fldacct_nonsense"},
        )
        self.assertEqual(r.status_code, 401)
        # A team token is not an account token.
        t = self.make_team()
        r = self.client.post(
            "/api/field/v1/teams",
            json={"team_name": "T", "display_name": "O"},
            headers={"Authorization": f"Bearer {t['token']}"},
        )
        self.assertEqual(r.status_code, 401)
        me = self.client.get("/api/field/v1/me", headers=self.auth(t)).json()
        self.assertEqual(me["plan"]["plan"], "operator", "operator-created teams keep working")

    def test_validation_of_plans_and_limits(self):
        for body in (
            {"name": "x", "plan": "free"},
            {"name": "x", "plan": "team", "limits": {"max_dragons": 3}},
            {"name": "x", "plan": "team", "limits": {"max_members": -1}},
            {"name": "x", "plan": "team", "limits": {"max_members": 100000}},
            {"name": "", "plan": "team"},
        ):
            r = self.client.post("/api/field/v1/admin/accounts", params=self.ADMIN, json=body)
            self.assertIn(r.status_code, (422,), body)
        a = self.grant(limits={"max_teams": 3, "max_members": 4})
        self.assertEqual((a["limits"]["max_teams"], a["limits"]["max_members"], a["limits"]["max_maps"]), (3, 4, 20))

    def test_account_tokens_are_hashed_and_can_be_rotated(self):
        a = self.grant()
        conn = sqlite3.connect(Path(self._tmp.name) / "showmefire.db")
        dump = "\n".join(conn.iterdump())
        conn.close()
        self.assertNotIn(a["token"], dump)
        new = self.client.post(f"/api/field/v1/admin/accounts/{a['id']}/token", params=self.ADMIN).json()["token"]
        self.assertNotEqual(new, a["token"])
        self.assertEqual(self.team_with(a).status_code, 401)
        self.assertEqual(self.team_with({"token": new}).status_code, 201)
        self.assertEqual(
            self.client.post("/api/field/v1/admin/accounts/nope/token", params=self.ADMIN).status_code, 404
        )

    def test_account_holders_see_their_own_plan_and_teams(self):
        a = self.grant()
        owner = self.team_with(a).json()
        self.join(owner["join_code"], "Una")
        self.make_map(owner)
        v = self.client.get("/api/field/v1/account", headers=self.acct_headers(a)).json()
        self.assertEqual((v["plan"], v["status"]), ("team", "active"))
        self.assertEqual(v["teams"][0]["members"], 2)
        self.assertEqual(v["teams"][0]["maps"], 1)
        self.assertNotIn("token", v)
        self.assertEqual(self.client.get("/api/field/v1/account").status_code, 401)


class PlanLifecycleTests(AccountTestCase):
    def test_suspended_and_expired_accounts_cannot_make_teams(self):
        a = self.grant()
        r = self.client.patch(f"/api/field/v1/admin/accounts/{a['id']}", params=self.ADMIN, json={"status": "suspended"})
        self.assertEqual(r.json()["status"], "suspended")
        self.assertEqual(self.team_with(a).status_code, 402)
        self.client.patch(f"/api/field/v1/admin/accounts/{a['id']}", params=self.ADMIN, json={"status": "active"})
        self.assertEqual(self.team_with(a, "Back").status_code, 201)
        e = self.grant("Lapsed", expires_at=int(time.time()) - 10)
        self.assertEqual(e["status"], "expired")
        r = self.team_with(e)
        self.assertEqual(r.status_code, 402)
        self.assertIn("expired", r.json()["detail"])

    def test_a_lapsed_plan_makes_the_team_read_only_not_gone(self):
        a = self.grant()
        owner = self.team_with(a).json()
        mate = self.join(owner["join_code"], "Una")
        m = self.make_map(owner, features={_uid(1): {"name": "Camp"}})
        self.client.patch(
            f"/api/field/v1/admin/accounts/{a['id']}", params=self.ADMIN, json={"expires_at": int(time.time()) - 5}
        )
        # Reads keep working so nothing is lost to a lapsed payment.
        me = self.client.get("/api/field/v1/me", headers=self.auth(mate)).json()
        self.assertEqual(me["plan"]["status"], "expired")
        self.assertIn("expired", me["plan"]["message"])
        got = self.client.get(f"/api/field/v1/maps/{m['id']}", headers=self.auth(mate)).json()
        self.assertEqual(list(got["features"]), [_uid(1)])
        self.assertEqual(self.client.get("/api/field/v1/events", params={"map_id": m["id"]}, headers=self.auth(mate)).status_code, 200)
        # Writes, new members and new maps are refused with 402.
        r = self.ops(mate, m["id"], [{"op": "upsert", "uid": _uid(2), "feature": {}}])
        self.assertEqual(r.status_code, 402)
        r = self.client.post("/api/field/v1/positions", json={"lat": 1, "lng": 1}, headers=self.auth(mate))
        self.assertEqual(r.status_code, 402)
        r = self.client.post("/api/field/v1/maps", json={"name": "n"}, headers=self.auth(owner))
        self.assertEqual(r.status_code, 402)
        r = self.client.post("/api/field/v1/teams/join", json={"join_code": owner["join_code"], "display_name": "Late"})
        self.assertEqual(r.status_code, 402)
        # Renewing restores everything.
        self.client.patch(
            f"/api/field/v1/admin/accounts/{a['id']}", params=self.ADMIN, json={"expires_at": int(time.time()) + 86400}
        )
        self.assertEqual(self.ops(mate, m["id"], [{"op": "upsert", "uid": _uid(2), "feature": {}}]).status_code, 200)

    def test_plan_changes_change_the_quotas(self):
        a = self.grant()
        owner = self.team_with(a).json()
        self.client.patch(f"/api/field/v1/admin/accounts/{a['id']}", params=self.ADMIN, json={"plan": "agency"})
        me = self.client.get("/api/field/v1/me", headers=self.auth(owner)).json()
        self.assertEqual((me["plan"]["plan"], me["plan"]["limits"]["max_members"]), ("agency", 50))
        # Overrides can be set and cleared.
        r = self.client.patch(f"/api/field/v1/admin/accounts/{a['id']}", params=self.ADMIN, json={"limits": {"max_members": 3}})
        self.assertEqual(r.json()["limits"]["max_members"], 3)
        r = self.client.patch(f"/api/field/v1/admin/accounts/{a['id']}", params=self.ADMIN, json={"limits": None})
        self.assertEqual(r.json()["limits"]["max_members"], 50)
        self.assertEqual(
            self.client.patch(f"/api/field/v1/admin/accounts/{a['id']}", params=self.ADMIN, json={"plan": "platinum"}).status_code, 422
        )
        self.assertEqual(
            self.client.patch("/api/field/v1/admin/accounts/nope", params=self.ADMIN, json={"plan": "team"}).status_code, 404
        )


class QuotaAndUsageTests(AccountTestCase):
    def test_members_maps_and_features_are_capped_by_plan(self):
        a = self.grant(limits={"max_members": 2, "max_maps": 1, "max_features": 3})
        owner = self.team_with(a).json()
        self.join(owner["join_code"], "Una")
        full = self.client.post("/api/field/v1/teams/join", json={"join_code": owner["join_code"], "display_name": "Third"})
        self.assertEqual(full.status_code, 409)
        m = self.make_map(owner)
        self.assertEqual(self.client.post("/api/field/v1/maps", json={"name": "two"}, headers=self.auth(owner)).status_code, 409)
        ok = self.ops(owner, m["id"], [{"op": "upsert", "uid": _uid(i), "feature": {}} for i in range(3)])
        self.assertEqual(ok.status_code, 200)
        over = self.ops(owner, m["id"], [{"op": "upsert", "uid": _uid(9), "feature": {}}])
        self.assertEqual(over.status_code, 413)
        # Editing an existing feature is still fine at the cap, and deleting frees room.
        self.assertEqual(self.ops(owner, m["id"], [{"op": "upsert", "uid": _uid(1), "feature": {"x": 1}}]).status_code, 200)
        self.ops(owner, m["id"], [{"op": "delete", "uid": _uid(0)}])
        self.assertEqual(self.ops(owner, m["id"], [{"op": "upsert", "uid": _uid(9), "feature": {}}]).status_code, 200)
        big = self.client.post(
            "/api/field/v1/maps",
            json={"name": "x", "features": {_uid(i): {} for i in range(4)}},
            headers=self.auth(self.team_with(self.grant("Other", limits={"max_features": 3, "max_maps": 5})).json()),
        )
        self.assertEqual(big.status_code, 413)

    def test_usage_is_metered_per_team_and_month(self):
        a = self.grant()
        owner = self.team_with(a, "Metered").json()
        m = self.make_map(owner, features={_uid(1): {}, _uid(2): {}})  # 2 operations
        self.ops(owner, m["id"], [{"op": "upsert", "uid": _uid(3), "feature": {}}, {"op": "delete", "uid": _uid(1)}])  # 2 more
        self.client.post("/api/field/v1/positions", json={"lat": 1, "lng": 1}, headers=self.auth(owner))
        rep = self.client.get("/api/field/v1/admin/usage", params=self.ADMIN).json()
        row = next(t for t in rep["teams"] if t["team_name"] == "Metered")
        self.assertEqual((row["ops"], row["positions"], row["members"], row["maps"]), (4, 1, 1, 1))
        self.assertEqual((row["account_name"], row["plan"]), ("Central Crossing FPD", "team"))
        self.assertRegex(rep["month"], r"^\d{4}-\d{2}$")
        empty = self.client.get("/api/field/v1/admin/usage", params={**self.ADMIN, "month": "1999-01"}).json()
        self.assertEqual(next(t for t in empty["teams"] if t["team_name"] == "Metered")["ops"], 0)
        self.assertEqual(self.client.get("/api/field/v1/admin/usage", params={**self.ADMIN, "month": "soon"}).status_code, 422)


if __name__ == "__main__":
    import unittest

    unittest.main()
