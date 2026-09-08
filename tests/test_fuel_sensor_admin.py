import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from core import database
from core.security import create_access_token
from routers import fuel_sensor_admin


class FuelSensorAdminTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary.name) / "sensors.db"
        self.database_patch = patch.object(database, "get_db_path", return_value=self.database_path)
        self.router_patch = patch.object(fuel_sensor_admin, "get_db_path", return_value=self.database_path)
        self.database_patch.start()
        self.router_patch.start()
        database.init_database()
        self.token = create_access_token({"sub": "staff@showmefire.org"})

    def tearDown(self):
        self.router_patch.stop()
        self.database_patch.stop()
        self.temporary.cleanup()

    def _insert(self, *, device_id: str, received_at: datetime, recorded_at: str = "123"):
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO fuel_moisture_sensor_readings (
                    site_id, device_id, recorded_at, air_temp_c, relative_humidity_pct,
                    fuel_moisture_pct, rssi_dbm, firmware_version, enclosure_state, received_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "bench", device_id, recorded_at, 22.4, 38.2, 8.7, -62,
                    "0.1.0-bench", "open_air_bench",
                    received_at.strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )

    def test_status_uses_server_receipt_time_for_health(self):
        now = datetime.now(timezone.utc)
        self._insert(device_id="SMF-FMS-01", received_at=now, recorded_at="42")
        self._insert(device_id="SMF-FMS-02", received_at=now - timedelta(minutes=30), recorded_at="999999")
        self._insert(device_id="SMF-FMS-03", received_at=now - timedelta(hours=2))

        result = fuel_sensor_admin.fuel_sensor_status(self.token)

        self.assertEqual(result["summary"]["sensor_count"], 3)
        self.assertEqual(result["summary"]["online"], 1)
        self.assertEqual(result["summary"]["delayed"], 1)
        self.assertEqual(result["summary"]["offline"], 1)
        devices = {device["device_id"]: device for device in result["devices"]}
        self.assertEqual(devices["SMF-FMS-01"]["health"], "online")
        self.assertEqual(devices["SMF-FMS-02"]["health"], "delayed")
        self.assertEqual(devices["SMF-FMS-03"]["health"], "offline")
        self.assertEqual(str(devices["SMF-FMS-01"]["recorded_at"]), "42")
        self.assertEqual(devices["SMF-FMS-01"]["readings_last_24h"], 1)

    def test_status_requires_admin_session(self):
        with self.assertRaises(Exception) as context:
            fuel_sensor_admin.fuel_sensor_status()
        self.assertEqual(context.exception.status_code, 401)
