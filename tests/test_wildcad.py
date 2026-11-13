import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from core import database
from services import incident_history, mobile_push, wildcad

FIXTURES = Path(__file__).parent / "fixtures" / "wildcad"
NOW = datetime(2026, 11, 13, 3, 35, tzinfo=timezone.utc)   # 9:35 PM Central, just after the fixture capture


def _body(name):
    return (FIXTURES / name).read_bytes()


class FakeFeed:
    """Stands in for the network: records every request so tests can count them."""

    def __init__(self, bodies=None, status=200, headers=None):
        self.bodies = bodies or {"incidents": _body("incidents.json"), "resources": _body("resources.json")}
        self.status, self.headers, self.requests = status, headers or {}, []

    def __call__(self, url, extra_headers, client=None):
        self.requests.append((url, dict(extra_headers)))
        kind = url.rsplit("/", 1)[-1]
        return self.status, self.headers, self.bodies[kind]


class WildcadBase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary.name) / "showmefire.db"
        self.patches = [patch.object(module, "get_db_path", return_value=self.db_path)
                        for module in (database, mobile_push, incident_history, wildcad)]
        self.patches.append(patch.dict("os.environ", {"WILDCAD_ENABLED": "1"}))
        self.patches.append(patch("services.wildcad_health.STATUS_DIR", Path(self.temporary.name) / "status"))
        self.patches.append(patch("services.wildcad_health.STATE_PATH", Path(self.temporary.name) / "state.json"))
        for item in self.patches:
            item.start()
        database.init_database()

    def tearDown(self):
        for item in self.patches:
            item.stop()
        self.temporary.cleanup()

    def rows(self, sql, *args):
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute(sql, args)]


class ParsingTests(unittest.TestCase):
    def test_incident_fields_and_the_missing_minus_sign(self):
        records = wildcad.parse_incidents(json.loads(_body("incidents.json")))
        rockhouse = next(item for item in records if item["name"] == "Rockhouse")
        self.assertEqual((rockhouse["incident_key"], rockhouse["inc_num"], rockhouse["dispatch_unit"]), ("MOMOC:412:2026", "412", "MOMOS"))
        self.assertAlmostEqual(rockhouse["longitude"], -93.758117)
        self.assertEqual(rockhouse["acres"], 5.0)
        self.assertEqual(rockhouse["reported_at"], datetime(2026, 11, 13, 0, 23, 18, 749000, tzinfo=timezone.utc))  # 6:23 PM CST
        self.assertEqual(len(rockhouse["resources"]), 6)

    def test_masked_and_empty_resource_lists_are_dropped(self):
        records = wildcad.parse_incidents(json.loads(_body("incidents.json")))
        self.assertTrue(all("*******" not in item["resources"] and None not in item["resources"] for item in records))

    def test_a_payload_that_changed_shape_is_rejected_not_guessed(self):
        with self.assertRaises(wildcad.ParseError):
            wildcad.parse_incidents({"unexpected": True})
        with self.assertRaises(wildcad.ParseError):
            wildcad.parse_incidents([{"data": [{"foo": "bar"}]}])

    def test_resource_kinds_and_people(self):
        resources = {item["code"]: item for item in wildcad.parse_resources(json.loads(_body("resources.json")))}
        self.assertEqual(resources["E-862"]["kind"], "engine")
        self.assertEqual(resources["D-12"]["kind"], "dozer")
        self.assertEqual(resources["MTV C-3B"]["kind"], "utility")
        self.assertFalse(resources["MTV C-3B"]["is_person"])
        people = [item for item in resources.values() if item["is_person"] and item["kind"] == "person"]
        self.assertTrue(people)
        self.assertTrue(all(", " in item["code"] or " " in item["description"] for item in people))


class GuardTests(WildcadBase):
    def test_one_request_per_endpoint_per_five_minutes_even_when_called_ten_times(self):
        feed = FakeFeed()
        for step in range(10):
            wildcad.poll("incidents", NOW + timedelta(seconds=20 * step), fetch=feed)
        self.assertEqual(len(feed.requests), 1)
        wildcad.poll("incidents", NOW + timedelta(minutes=5, seconds=1), fetch=feed)
        self.assertEqual(len(feed.requests), 2)
        # The other endpoint has its own allowance, and every request carries the Show Me Fire user agent.
        wildcad.poll("resources", NOW + timedelta(seconds=150), fetch=feed)
        self.assertEqual(len(feed.requests), 3)
        self.assertEqual(wildcad.USER_AGENT, "ShowMeFire (contact@showmefire.org)")

    def test_the_guard_survives_a_restart_because_it_lives_in_the_database(self):
        feed = FakeFeed()
        wildcad.poll("incidents", NOW, fetch=feed)
        # "Restart": nothing in memory, same database.
        wildcad.poll("incidents", NOW + timedelta(minutes=2), fetch=feed)
        self.assertEqual(len(feed.requests), 1)

    def test_nothing_is_requested_unless_enabled(self):
        feed = FakeFeed()
        with patch.dict("os.environ", {"WILDCAD_ENABLED": ""}):
            result = wildcad.poll("incidents", NOW, fetch=feed)
        self.assertEqual((result["requested"], result["reason"], len(feed.requests)), (False, "disabled", 0))

    def test_failures_back_off_and_a_block_pauses_for_an_hour_and_alerts(self):
        failing = FakeFeed(status=500)
        wildcad.poll("incidents", NOW, fetch=failing)
        state = self.rows("SELECT * FROM wildcad_feed_state WHERE kind = 'incidents'")[0]
        self.assertEqual((state["consecutive_failures"], state["last_status"]), (1, 500))
        wildcad.poll("incidents", NOW + timedelta(minutes=5, seconds=1), fetch=failing)   # 5 min back-off over
        self.assertEqual(self.rows("SELECT consecutive_failures FROM wildcad_feed_state")[0]["consecutive_failures"], 2)
        wildcad.poll("incidents", NOW + timedelta(minutes=11), fetch=failing)             # inside the 10 min back-off
        self.assertEqual(len(failing.requests), 2)

        blocked = FakeFeed(status=429, headers={"retry-after": "7200"})
        with patch("services.wildcad_health.send") as send:
            wildcad.poll("resources", NOW, fetch=blocked)
            send.assert_called_once()
        paused = self.rows("SELECT next_allowed_at, paused_reason FROM wildcad_feed_state WHERE kind = 'resources'")[0]
        self.assertIn("blocked", paused["paused_reason"])
        self.assertGreaterEqual(datetime.fromisoformat(paused["next_allowed_at"].replace("Z", "+00:00")), NOW + timedelta(hours=2))

    def test_snapshots_are_kept_once_per_distinct_content(self):
        feed = FakeFeed()
        wildcad.poll("incidents", NOW, fetch=feed)
        wildcad.poll("incidents", NOW + timedelta(minutes=5, seconds=1), fetch=feed)
        self.assertEqual(len(self.rows("SELECT id FROM wildcad_snapshots WHERE kind = 'incidents'")), 1)


class IngestTests(WildcadBase):
    def _satellite_near_rockhouse(self):
        database.upsert_detection_event(
            source="viirs", external_id="v1", latitude=36.668, longitude=-93.757, occurred_at="2026-11-12T22:26:00Z",
            county_fips="29009", county_name="Barry", frp=10.0, confidence="n", satellite="N20")

    def test_rockhouse_joins_the_satellite_incident_and_gets_resources_and_smf_numbers(self):
        self._satellite_near_rockhouse()
        wildcad.poll("incidents", NOW - timedelta(minutes=30), fetch=FakeFeed())          # first run = baseline
        incidents = self.rows("SELECT id, smf_number FROM fire_incidents")
        self.assertEqual(len(incidents), 1)                                                # joined, not a second incident
        self.assertTrue(incidents[0]["smf_number"].startswith("SMF-2026-"))
        wildcad.poll("resources", NOW - timedelta(minutes=27), fetch=FakeFeed())
        public = wildcad.public_resources(incidents[0]["id"])
        names = sorted(item["name"] for item in public["equipment"])
        self.assertEqual(names, ["MTVC Truck A", "MTVC Truck B", "MTVC Truck E", "MTVC Truck F", "WY-HDD-E1605"])
        self.assertEqual(public["personnel"], 1)
        self.assertNotIn("Doe", json.dumps(public))                                        # no person's name, ever
        staff = wildcad.staff_resources(incidents[0]["id"])
        self.assertTrue(any(item["is_person"] for item in staff))
        dispatch = wildcad.dispatch_for(incidents[0]["id"])
        self.assertEqual((dispatch["incident_number"], dispatch["name"], dispatch["acres"]), ("MOMOS-412", "Rockhouse", 5.0))
        self.assertNotIn("ic", json.dumps(dispatch).lower().replace("incident", ""))        # the commander is staff-only
        timeline = [entry["summary"] for entry in incident_history.list_history(incidents[0]["id"])]
        self.assertTrue(any(line.startswith("Reported to dispatch as incident 412, Rockhouse") for line in timeline))
        self.assertTrue(any(line.startswith("On scene: ") for line in timeline))

    def test_a_dispatch_record_with_no_satellite_starts_an_incident_and_non_fires_do_not(self):
        wildcad.poll("incidents", NOW, fetch=FakeFeed())
        self.assertEqual(len(self.rows("SELECT id FROM fire_incidents")), 1)       # Rockhouse only
        stored = self.rows("SELECT name, incident_id FROM wildcad_incidents ORDER BY name")
        self.assertEqual(len(stored), 20)                                           # all kept for staff
        self.assertEqual(sum(1 for row in stored if row["incident_id"] is not None), 1)

    def test_updates_dedupe_on_number_and_year_and_do_not_re_alert(self):
        wildcad.poll("incidents", NOW, fetch=FakeFeed())
        body = json.loads(_body("incidents.json"))
        next(item for item in body[0]["data"] if item["name"] == "Rockhouse")["acres"] = "12"
        next(item for item in body[0]["data"] if item["name"] == "Rockhouse")["name"] = "Rockhouse Fire"
        wildcad.poll("incidents", NOW + timedelta(minutes=6), fetch=FakeFeed({"incidents": json.dumps(body).encode()}))
        self.assertEqual(len(self.rows("SELECT id FROM wildcad_incidents WHERE inc_num = '412'")), 1)
        self.assertEqual(len(self.rows("SELECT id FROM fire_incidents")), 1)
        timeline = [entry["summary"] for entry in incident_history.list_history(self.rows("SELECT id FROM fire_incidents")[0]["id"])]
        self.assertIn("Dispatch updated the size from 5 acres to 12 acres", timeline)
        self.assertIn("Dispatch changed the name from Rockhouse to Rockhouse Fire", timeline)
        # Same inc_num next year is a different fire.
        self.assertNotEqual(wildcad.parse_incident({"inc_num": "412", "date": "2027-01-02T10:00:00"})["incident_key"], "MOMOC:412:2026")

    def test_first_run_seeds_silently_and_a_later_new_fire_is_left_for_notification(self):
        wildcad.poll("incidents", NOW, fetch=FakeFeed())
        self.assertTrue(all(row["notified_at"] for row in self.rows("SELECT notified_at FROM wildcad_incidents")))
        body = json.loads(_body("incidents.json"))
        new = dict(next(item for item in body[0]["data"] if item["name"] == "Rockhouse"))
        new.update(inc_num="413", uuid="x", name="Another", latitude="37.2", longitude="91.4", date="2026-11-13T03:00:00")
        body[0]["data"].append(new)
        wildcad.poll("incidents", NOW + timedelta(minutes=6), fetch=FakeFeed({"incidents": json.dumps(body).encode()}))
        fresh = self.rows("SELECT name, notified_at FROM wildcad_incidents WHERE inc_num = '413'")[0]
        self.assertIsNone(fresh["notified_at"])


class SmfNumberTests(WildcadBase):
    def test_numbers_are_unique_sequential_and_backfilled_once(self):
        for n in range(3):
            database.upsert_detection_event(source="viirs", external_id=f"s{n}", latitude=36.6 + n, longitude=-93.7,
                                            occurred_at="2026-11-12T22:26:00Z", county_fips="29009", county_name="Barry")
        events = [row["smf_number"] for row in self.rows("SELECT smf_number FROM fire_events ORDER BY id")]
        self.assertEqual(events, ["SMFD-2026-0000001", "SMFD-2026-0000002", "SMFD-2026-0000003"])
        incidents = [row["smf_number"] for row in self.rows("SELECT smf_number FROM fire_incidents ORDER BY id")]
        self.assertEqual(incidents, ["SMF-2026-00001", "SMF-2026-00002", "SMF-2026-00003"])
        # Rows that predate the feature are numbered in id order on the next start, without touching the others.
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.execute("UPDATE fire_events SET smf_number = NULL WHERE id = 2")
            connection.execute("DELETE FROM smf_counters WHERE kind = 'event'")
            connection.execute("UPDATE fire_events SET smf_number = NULL")
        database.init_database()
        again = [row["smf_number"] for row in self.rows("SELECT smf_number FROM fire_events ORDER BY id")]
        self.assertEqual(again, ["SMFD-2026-0000001", "SMFD-2026-0000002", "SMFD-2026-0000003"])



class HealthAndPushTests(WildcadBase):
    def _heartbeat(self, kind, when):
        import os
        from services import wildcad_health

        wildcad_health.write_heartbeat(kind, when)
        os.utime(wildcad_health.heartbeat_path(kind), (when.timestamp(), when.timestamp()))

    def test_a_stale_feed_alerts_once_reminds_much_later_and_announces_recovery(self):
        from services import wildcad_health

        for kind in wildcad.KINDS:
            self._heartbeat(kind, NOW)
        with patch.object(wildcad_health, "send") as send:
            self.assertEqual(wildcad_health.run_health_check(NOW + timedelta(minutes=10)), [])            # fresh enough
            first = wildcad_health.run_health_check(NOW + timedelta(minutes=25))                         # 4+ cycles missed
            self.assertEqual(sorted(message for message, _ in first), ["WildCAD incidents is stale", "WildCAD resources is stale"])
            self.assertEqual(wildcad_health.run_health_check(NOW + timedelta(minutes=30)), [])           # no repeat
            self.assertEqual(send.call_count, 2)
            self._heartbeat("incidents", NOW + timedelta(minutes=31))
            recovered = wildcad_health.run_health_check(NOW + timedelta(minutes=32))
            self.assertEqual([message for message, _ in recovered], ["WildCAD incidents recovered"])

    def test_nothing_alerts_while_the_feed_is_switched_off(self):
        from services import wildcad_health

        with patch.dict("os.environ", {"WILDCAD_ENABLED": ""}), patch.object(wildcad_health, "send") as send:
            self.assertEqual(wildcad_health.run_health_check(NOW + timedelta(hours=5)), [])
            send.assert_not_called()

    def test_staff_get_a_push_for_a_feed_problem(self):
        from services import staff_push

        content = staff_push.content_for({"alert_type": "feed_stale", "title": "WildCAD: WildCAD incidents is stale"})
        self.assertEqual((content["title"], content["body"], content["url"]), ("Dispatch feed problem", "WildCAD incidents is stale", "/staff"))

    def test_a_dispatch_fire_notifies_once_and_a_later_nifc_listing_does_not_notify_again(self):
        mobile_push.upsert_subscription(
            installation_id="11111111-1111-4111-8111-111111111111", expo_push_token="ExpoPushToken[abcdefghijklmnopqrstuv]",
            platform="ios", app_version="1.2.0", forecast=False, sitrep=False, fire_weather=False, fire_detection=True,
            county_fips=["29203"])
        noon = datetime(2026, 11, 13, 17, 0, tzinfo=timezone.utc)                       # 11 AM Central: outside quiet hours
        wildcad.poll("incidents", noon - timedelta(hours=1), fetch=FakeFeed())           # baseline (seeds silently)
        body = json.loads(_body("incidents.json"))
        new = dict(next(item for item in body[0]["data"] if item["name"] == "Rockhouse"))
        new.update(inc_num="413", uuid="u413", name="Shannon Test", latitude="37.1", longitude="91.4", date="2026-11-12T20:30:00", acres="2")
        body[0]["data"].append(new)
        wildcad.poll("incidents", noon, fetch=FakeFeed({"incidents": json.dumps(body).encode()}))
        incident_id = self.rows("SELECT incident_id FROM wildcad_incidents WHERE inc_num = '413'")[0]["incident_id"]
        self.assertIsNotNone(incident_id)
        with patch.object(mobile_push, "_send_batch", return_value=[{"status": "ok", "id": "t1"}]) as send:
            self.assertEqual(mobile_push.notify_official_listing(noon), 1)
            message = send.call_args.args[0][0]
            self.assertEqual(message["title"], "Fire crews dispatched: Shannon Test")
            self.assertTrue(message["data"]["smfNumber"].startswith("SMF-2026-"))
            # NIFC lists the same incident later: no second push, but its row is marked handled.
            with closing(sqlite3.connect(self.db_path)) as connection, connection:
                connection.execute(
                    "INSERT INTO fire_incident_official (irwin_id, incident_id, name, first_seen_at) VALUES ('abc', ?, 'Shannon Test', ?)",
                    (incident_id, noon.strftime("%Y-%m-%d %H:%M:%S")))
            self.assertEqual(mobile_push.notify_official_listing(noon + timedelta(minutes=10)), 0)
            self.assertEqual(send.call_count, 1)
        self.assertIsNotNone(self.rows("SELECT notified_at FROM fire_incident_official")[0]["notified_at"])
        timeline = [e["summary"] for e in incident_history.list_history(incident_id)]
        self.assertIn("Told 1 person near Shannon County dispatch listed it", timeline)

    def test_merging_incidents_moves_dispatch_records_resources_and_history(self):
        from services import incident_merger

        for n, lat in enumerate((36.6, 36.6005), start=1):
            database.upsert_detection_event(source="viirs", external_id=f"m{n}", latitude=lat, longitude=-93.7 - n * 0.2,
                                            occurred_at="2026-11-12T22:26:00Z", county_fips="29009", county_name="Barry")
        ids = [row["id"] for row in self.rows("SELECT id FROM fire_incidents ORDER BY id")]
        self.assertEqual(len(ids), 2)
        survivor, loser = ids
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.row_factory = sqlite3.Row
            connection.execute("INSERT INTO wildcad_incidents (incident_key, center, inc_num, year, incident_id) VALUES ('K', 'MOMOC', '1', 2026, ?)", (loser,))
            connection.execute("INSERT INTO wildcad_resources (center, code, incident_id) VALUES ('MOMOC', 'E-1', ?)", (loser,))
            incident_history.record(connection, loser, "dispatch", "x", dedupe_key="d1")
            incident_merger._apply_merge(connection.cursor(), survivor, [loser])
        self.assertEqual(self.rows("SELECT incident_id FROM wildcad_incidents")[0]["incident_id"], survivor)
        self.assertEqual(self.rows("SELECT incident_id FROM wildcad_resources")[0]["incident_id"], survivor)
        self.assertEqual({row["incident_id"] for row in self.rows("SELECT incident_id FROM fire_incident_history WHERE kind = 'dispatch'")}, {survivor})

if __name__ == "__main__":
    unittest.main()
