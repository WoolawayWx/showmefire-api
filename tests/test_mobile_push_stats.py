"""Counting what became of each app alert: sent, accepted, delivered, failed, reached nobody."""
import json
import sqlite3
from contextlib import closing

import pytest

from core import database
from routers import admin_notifications
from services import mobile_push


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "showmefire.db"
    monkeypatch.setattr(database, "get_db_path", lambda: path)
    monkeypatch.setattr(mobile_push, "get_db_path", lambda: path)
    database.init_database()
    return path


def _device(n, forecast=True, fire_detection=False, counties=("29019",)):
    installation = f"{n:08d}-1111-4111-8111-111111111111"
    return mobile_push.upsert_subscription(
        installation_id=installation, expo_push_token=f"ExpoPushToken[{n:032d}]", platform="ios" if n % 2 else "android",
        app_version="1.0.0", forecast=forecast, sitrep=False, fire_weather=False, fire_detection=fire_detection,
        county_fips=list(counties))


def _send(key="forecast:1", event_type="forecast", **kwargs):
    return mobile_push.send_mobile_event(event_type=event_type, event_key=key, title="Forecast", body="Ready", url="/forecasts", **kwargs)


def _event(db, key):
    with closing(sqlite3.connect(db)) as connection:
        connection.row_factory = sqlite3.Row
        return dict(connection.execute("SELECT * FROM mobile_push_events WHERE event_key = ?", (key,)).fetchone())


def test_a_send_records_who_was_eligible_accepted_gone_and_failed(db, monkeypatch):
    for n in (1, 2, 3):
        _device(n)
    tickets = [{"status": "ok", "id": "t1"}, {"status": "error", "details": {"error": "DeviceNotRegistered"}},
               {"status": "error", "details": {"error": "MessageRateExceeded"}}]
    monkeypatch.setattr(mobile_push, "_send_batch", lambda messages: tickets)
    assert _send() == 1
    event = _event(db, "forecast:1")
    assert (event["recipients"], event["sent"], event["unregistered"], event["failed"]) == (3, 1, 1, 1)
    assert (event["delivered"], event["receipt_errors"]) == (0, 0)


def test_a_failed_batch_counts_every_device_as_failed(db, monkeypatch):
    for n in (1, 2):
        _device(n)
    monkeypatch.setattr(mobile_push, "_send_batch", lambda messages: [])     # Expo unreachable after retries
    assert _send() == 0
    event = _event(db, "forecast:1")
    assert (event["recipients"], event["sent"], event["failed"]) == (2, 0, 2)


def test_an_alert_nobody_could_receive_is_recorded_as_reaching_nobody(db, monkeypatch):
    _device(1, fire_detection=False)
    monkeypatch.setattr(mobile_push, "_send_batch", lambda messages: pytest.fail("nothing should be sent"))
    assert _send("fire_detection:9", "fire_detection", county_fips=["29019"]) == 0
    event = _event(db, "fire_detection:9")
    assert (event["recipients"], event["sent"], event["failed"]) == (0, 0, 0)
    assert mobile_push.push_stats()["totals"]["reached_nobody"] == 1


class _Receipts:
    def __init__(self, data):
        self.data = data

    def __call__(self, *args, **kwargs):
        outer = self

        class Client:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return None

            def post(self, *a, **k):
                class Response:
                    def raise_for_status(self):
                        return None

                    def json(self):
                        return {"data": outer.data}

                return Response()

        return Client()


def _age_tickets(db):
    with closing(sqlite3.connect(db)) as connection, connection:
        connection.execute("UPDATE mobile_push_tickets SET created_at = datetime('now', '-1 hour')")


def test_receipts_are_added_to_the_alert_and_counted_once(db, monkeypatch):
    for n in (1, 2, 3):
        _device(n)
    monkeypatch.setattr(mobile_push, "_send_batch", lambda messages: [{"status": "ok", "id": f"t{i}"} for i in range(3)])
    _send()
    _age_tickets(db)
    receipts = {"t0": {"status": "ok"}, "t1": {"status": "ok"},
                "t2": {"status": "error", "message": "too big", "details": {"error": "MessageTooBig"}}}
    monkeypatch.setattr(mobile_push.httpx, "Client", _Receipts(receipts))

    assert mobile_push.check_push_receipts() == 3
    event = _event(db, "forecast:1")
    assert (event["delivered"], event["receipt_errors"]) == (2, 1)
    assert json.loads(event["error_counts_json"]) == {"MessageTooBig": 1}

    assert mobile_push.check_push_receipts() == 0          # receipts are only fetched once per ticket
    assert _event(db, "forecast:1")["delivered"] == 2


def test_an_unregistered_receipt_is_counted_and_a_missing_event_row_is_tolerated(db, monkeypatch):
    _device(1)
    monkeypatch.setattr(mobile_push, "_send_batch", lambda messages: [{"status": "ok", "id": "t0"}])
    _send()
    with closing(sqlite3.connect(db)) as connection, connection:
        connection.execute("INSERT INTO mobile_push_tickets (ticket_id, installation_id, event_key) VALUES ('orphan', ?, 'gone:1')",
                           ("00000001-1111-4111-8111-111111111111",))
    _age_tickets(db)
    receipts = {"t0": {"status": "error", "details": {"error": "DeviceNotRegistered"}},
                "orphan": {"status": "ok"}}
    monkeypatch.setattr(mobile_push.httpx, "Client", _Receipts(receipts))
    mobile_push.check_push_receipts()
    event = _event(db, "forecast:1")
    assert event["receipt_errors"] == 1 and json.loads(event["error_counts_json"]) == {"DeviceNotRegistered": 1}


def test_the_counts_survive_the_ticket_purge(db, monkeypatch):
    _device(1)
    monkeypatch.setattr(mobile_push, "_send_batch", lambda messages: [{"status": "ok", "id": "t0"}])
    _send()
    _age_tickets(db)
    monkeypatch.setattr(mobile_push.httpx, "Client", _Receipts({"t0": {"status": "ok"}}))
    mobile_push.check_push_receipts()
    with closing(sqlite3.connect(db)) as connection, connection:
        connection.execute("UPDATE mobile_push_tickets SET created_at = datetime('now', '-30 days')")
    mobile_push.purge_delivery_records()
    event = _event(db, "forecast:1")
    assert (event["sent"], event["delivered"]) == (1, 1)


def test_stats_summarise_audience_alerts_and_leave_out_baseline_rows(db, monkeypatch):
    for n in (1, 2, 3):
        _device(n, fire_detection=(n == 1))
    issued = iter(range(1000))
    monkeypatch.setattr(mobile_push, "_send_batch", lambda messages: [{"status": "ok", "id": f"ticket-{next(issued)}"} for _ in messages])
    _send("forecast:1")
    _send("manual:abc", "forecast")
    mobile_push.record_event("fire_detection:baseline:v1", "fire_detection", {"seeded": 4})     # never sent
    mobile_push.record_event("fire_detection:old", "fire_detection", {"title": "seeded"})

    stats = mobile_push.push_stats(days=30)
    assert stats["audience"]["devices"] == 3 and stats["audience"]["forecast"] == 3 and stats["audience"]["fire_detection"] == 1
    assert stats["audience"]["platforms"] == {"ios": 2, "android": 1}
    assert stats["audience"]["new_in_period"] == 3
    assert stats["totals"]["alerts"] == 2 and stats["totals"]["recipients"] == 6 and stats["totals"]["sent"] == 6
    assert stats["totals"]["accepted_pct"] == 100.0
    kinds = {(row["event_type"], row["manual"]): row for row in stats["by_type"]}
    assert set(kinds) == {("forecast", False), ("forecast", True)} and kinds[("forecast", False)]["alerts"] == 1
    assert kinds[("forecast", False)]["awaiting_receipt"] == 3
    assert {item["event_key"] for item in stats["recent"]} == {"forecast:1", "manual:abc"}
    assert stats["recent"][0]["title"] == "Forecast" and stats["daily"][0]["alerts"] == 2
    assert stats["audience"]["awaiting_receipt"] == 6


def test_the_stats_endpoint_returns_the_summary(db, monkeypatch):
    _device(1)
    monkeypatch.setattr(mobile_push, "_send_batch", lambda messages: [{"status": "ok", "id": "t0"}])
    _send()
    body = admin_notifications.notification_stats(days=7, recent=5, _admin="staff@x.org")
    assert body["success"] and body["period_days"] == 7 and body["totals"]["sent"] == 1
