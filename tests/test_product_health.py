"""Product freshness checks, stale/recovered alert decisions, verification math and the new data endpoints."""
import json
import os
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import product_health as ph


class ProductHealthTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.object(ph, "PROJECT_DIR", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def touch(self, relative, minutes_ago):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
        stamp = time.time() - minutes_ago * 60
        os.utime(path, (stamp, stamp))

    def test_fresh_stale_and_missing(self):
        product = ph.Product("radar", "Radar", ["status/radar.json"], 45)
        self.assertEqual(ph.check_product(product)["status"], "missing")
        self.touch("status/radar.json", 10)
        self.assertEqual(ph.check_product(product)["status"], "ok")
        self.touch("status/radar.json", 90)
        result = ph.check_product(product)
        self.assertEqual(result["status"], "stale")
        self.assertIn("not updated for 90 min", result["problems"][0])

    def test_newest_of_several_files_counts(self):
        product = ph.Product("eddi", "EDDI", ["a.checked", "b.signature"], 180)
        self.touch("a.checked", 500)
        self.touch("b.signature", 20)
        self.assertEqual(ph.check_product(product)["status"], "ok")

    def test_extra_problems_make_a_product_stale(self):
        self.touch("x.json", 1)
        product = ph.Product("x", "X", ["x.json"], 45, extra=lambda: ["3 HRRR hours behind"])
        self.assertEqual(ph.check_product(product)["problems"], ["3 HRRR hours behind"])


class AlertPlanTests(unittest.TestCase):
    def setUp(self):
        from scripts import check_product_health as cph

        self.cph = cph
        self.now = datetime(2026, 10, 16, 20, tzinfo=timezone.utc)

    def report(self, status):
        return {"products": [{"key": "radar", "label": "Radar maps", "source": "RIDGE", "status": status,
                              "problems": [] if status == "ok" else ["not updated for 90 min"]}]}

    def test_alert_once_then_remind_then_recover(self):
        alerts, state = self.cph.plan_alerts(self.report("stale"), {}, self.now)
        self.assertEqual([a[0] for a in alerts], ["Radar maps is stale"])
        alerts, state = self.cph.plan_alerts(self.report("stale"), state, self.now + timedelta(hours=1))
        self.assertEqual(alerts, [])  # no repeat within the reminder window
        alerts, state = self.cph.plan_alerts(self.report("stale"), state, self.now + timedelta(hours=7))
        self.assertEqual([a[0] for a in alerts], ["Radar maps still stale"])
        alerts, state = self.cph.plan_alerts(self.report("ok"), state, self.now + timedelta(hours=8))
        self.assertEqual([a[0] for a in alerts], ["Radar maps recovered"])
        self.assertEqual(state, {})


class VerificationTests(unittest.TestCase):
    def test_bias_is_hrrr_minus_balloon(self):
        from services.fire_weather import verification as v

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(v, "PAIRS_PATH", Path(tmp) / "pairs.jsonl"):
            now = datetime(2026, 10, 16, 20, tzinfo=timezone.utc)
            rows = [{"station": "KSGF", "valid_time": "2026-10-16T12:00:00+00:00",
                     "transport_wind": {"balloon": 10.0, "hrrr": 14.0}},
                    {"station": "KOAX", "valid_time": "2026-10-16T18:00:00+00:00",
                     "transport_wind": {"balloon": 20.0, "hrrr": 18.0}}]
            v.PAIRS_PATH.write_text("\n".join(json.dumps(row) for row in rows))
            stats = v.summarize(now=now)["metrics"]["transport_wind"]
        self.assertEqual(stats["all"], {"count": 2, "bias": 1.0, "mae": 3.0})
        self.assertEqual(stats["by_station"]["KSGF"]["bias"], 4.0)


class EndpointTests(unittest.TestCase):
    def setUp(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from routers import fire_weather_data, map_archive, soundings

        app = FastAPI()
        for router in (fire_weather_data.router, soundings.router, map_archive.router):
            app.include_router(router)
        self.client = TestClient(app)
        self.soundings = soundings
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_sounding_site_aliases_and_validation(self):
        manifest = Path(self.tmp.name) / "soundings.json"
        manifest.write_text(json.dumps({"stations": {"USCOU": {"name": "Columbia, MO (Mizzou)", "soundings": []},
                                                     "KSGF": {"name": "Springfield, MO", "soundings": []}}}))
        with mock.patch.object(self.soundings, "MANIFEST_PATH", manifest):
            self.assertEqual(self.client.get("/api/soundings/cou").json()["station"], "USCOU")
            self.assertEqual(self.client.get("/api/soundings/KSGF").json()["name"], "Springfield, MO")
            self.assertEqual(self.client.get("/api/soundings/nope").status_code, 404)
            self.assertEqual(self.client.get("/api/soundings/sgf/notatime").status_code, 422)

    def test_fire_weather_point_lookup_rejects_bad_time(self):
        self.assertEqual(self.client.get("/api/fire-weather/at?lat=38&lon=-92&time=yesterday").status_code, 422)

    def test_map_archive_listing(self):
        from services import map_archive as ma

        with mock.patch.object(ma, "LEDGER_PATH", Path(self.tmp.name) / "ledger.json"):
            body = self.client.get("/api/map-archives").json()
        self.assertEqual(body["archives"], [])
        self.assertEqual(body["cdn_retention_days"], 14)


if __name__ == "__main__":
    unittest.main()
