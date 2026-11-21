import math
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from core import database
from services.incident_models import features as F
from services.incident_models import incident_type, persistence

START = datetime(2026, 10, 1, 18, 0, tzinfo=timezone.utc)
FUTURE = datetime(2027, 6, 1, tzinfo=timezone.utc)


def _iso(when):
    return when.isoformat().replace("+00:00", "Z")


class ModelTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "smf.db"
        patcher = patch.object(database, "get_db_path", return_value=self.db)
        patcher.start()
        self.addCleanup(patcher.stop)
        env = patch.dict(os.environ, {"SMF_INCIDENT_MODEL_DIR": str(Path(self.tmp.name) / "models")})
        env.start()
        self.addCleanup(env.stop)
        database.init_database()
        self._slot = 0

    def tearDown(self):
        try:
            self.tmp.cleanup()
        except PermissionError:
            pass

    def add_incident(self, detections, *, hours_apart=(0.0,), source="ngfs", cause=None):
        """One separate incident (far from the others) with `detections` detections in its first minutes, plus one more
        detection each at the given hour offsets after the first."""
        self._slot += 1
        lat, lon = 36.0 + 0.2 * (self._slot % 30), -91.5 + 0.2 * (self._slot // 30)
        t0 = START + timedelta(hours=7 * self._slot)
        times = [t0 + timedelta(minutes=2 * i) for i in range(detections)] + [t0 + timedelta(hours=h) for h in hours_apart if h > 0]
        for index, when in enumerate(times):
            database.upsert_detection_event(source=source, external_id=f"s{self._slot}-{index}", latitude=lat, longitude=lon,
                                            occurred_at=_iso(when), county_fips="29023", county_name="Butler", frp=5.0 + index,
                                            confidence="n", satellite="G19")
        with closing(sqlite3.connect(self.db)) as conn:
            incident_id = conn.execute("SELECT incident_id FROM fire_events WHERE external_id = ?", (f"s{self._slot}-0",)).fetchone()[0]
            if cause:
                conn.execute("UPDATE fire_events SET cause_category = ? WHERE incident_id = ?", (cause, incident_id))
            conn.commit()
        return incident_id

    def conn(self):
        connection = sqlite3.connect(self.db)
        connection.row_factory = sqlite3.Row
        self.addCleanup(connection.close)
        return connection


class FeatureTests(ModelTestBase):
    def test_window_only_sees_the_first_thirty_minutes(self):
        incident_id = self.add_incident(3, hours_apart=(5.0,))
        members = database.list_fire_incident_members(incident_id)
        sat = F.satellite_members(members)
        t0, window = F.first_window(sat)
        self.assertEqual(len(sat), 4)
        self.assertEqual(len(window), 3)                       # the detection 5 h later is not in the window
        feats = F.window_features(t0, window)
        self.assertEqual(feats["n_w"], 3.0)
        self.assertEqual(feats["n_ngfs_w"], 3.0)
        self.assertAlmostEqual(feats["hour_sin"] ** 2 + feats["hour_cos"] ** 2, 1.0)

    def test_reports_and_agency_records_are_not_satellite_detections(self):
        rows = [{"source": "user_submission", "occurred_at": _iso(START)}, {"source": "wildcad", "occurred_at": _iso(START)},
                {"source": "viirs", "occurred_at": _iso(START)}, {"source": "viirs", "occurred_at": "not a date"}]
        self.assertEqual(len(F.satellite_members(rows)), 1)

    def test_missing_ground_cover_and_weather_are_nan_not_errors(self):
        self.assertTrue(all(math.isnan(v) for v in F.cover_shares(None).values()))
        weather = F.weather_at(36.6, -90.4, START)             # no RTMA archive in the test data directory
        self.assertEqual(set(weather), set(F.WEATHER_FEATURES))
        self.assertTrue(all(math.isnan(v) for v in weather.values()))

    def test_nlcd_cover_is_read_as_shares(self):
        cover = {"nlcd_groups": [{"group": "Cultivated crops", "pct": 70}, {"group": "Forest", "pct": 20}, {"group": "Pasture/hay", "pct": 10}],
                 "canopy_cover_pct": 12.5}
        shares = F.cover_shares(cover)
        self.assertEqual((shares["crop"], shares["forest"], shares["pasture_grass"], shares["canopy"]), (0.7, 0.2, 0.1, 12.5))

    def test_neighbours_count_only_incidents_that_began_before_and_nearby(self):
        first = self.add_incident(2)
        with closing(sqlite3.connect(self.db)) as conn:
            lat, lon = conn.execute("SELECT centroid_latitude, centroid_longitude FROM fire_incidents WHERE id = ?", (first,)).fetchone()
        # a second incident 1 km away, 2 hours LATER, and one 1 km away 3 days EARLIER
        for label, delta in (("later", timedelta(hours=2)), ("earlier", -timedelta(days=3))):
            database.upsert_detection_event(source="ngfs", external_id=f"n-{label}", latitude=lat + 0.045, longitude=lon, occurred_at=_iso(START + 7 * timedelta(hours=1) + delta),
                                            county_fips="29023", county_name="Butler", frp=3.0, confidence="n", satellite="G19")
        incident = {"id": first, "centroid_latitude": lat, "centroid_longitude": lon}
        t0 = START + timedelta(hours=7)
        counts = F.neighbor_counts(self.conn(), incident, t0)
        self.assertEqual(counts["n_near_7d"], 1.0)             # only the earlier one existed before this incident began
        self.assertEqual(counts["n_near_24h"], 0.0)


class PersistenceTests(ModelTestBase):
    def _history(self, count=420):
        """Incidents that start with many detections tend to persist; the rest rarely do (with some noise)."""
        for index in range(count):
            big = index % 2 == 0
            persists = (big and index % 10 != 0) or (not big and index % 9 == 0)
            self.add_incident(5 if big else 1, hours_apart=(4.5,) if persists else (0.0,))

    def test_it_learns_a_real_signal_on_a_time_ordered_holdout_and_passes_the_gate(self):
        self._history()
        report = persistence.train(self.conn(), since="2026-01-01", with_weather=False, save=False, now=FUTURE)
        self.assertEqual(report["status"], "ok", report)
        self.assertGreater(report["auc"], 0.8)
        self.assertTrue(report["gate_passed"])
        self.assertGreater(report["baseline_auc_first_window_count"], 0.7)      # the baselines it must beat are measured too
        self.assertIsNotNone(report["baseline_auc_rules_confidence"])
        self.assertEqual(report["train"] + report["holdout"], report["rows"])

    def test_holdout_is_the_newest_incidents(self):
        self._history(320)
        frame, extras = persistence.build_dataset(self.conn(), "2026-01-01", with_weather=False, now=FUTURE)
        self.assertTrue(frame["t0"].is_monotonic_increasing)
        cut = int(len(frame) * (1 - persistence.HOLDOUT_FRACTION))
        self.assertLess(frame["t0"].iloc[cut - 1], frame["t0"].iloc[cut])

    def test_too_little_history_trains_nothing(self):
        for _ in range(20):
            self.add_incident(2, hours_apart=(4.0,))
        report = persistence.train(self.conn(), since="2026-01-01", with_weather=False, save=True, now=FUTURE)
        self.assertEqual(report["status"], "not_enough_history")
        self.assertFalse((persistence.model_dir() / "model.json").exists())
        self.assertIsNone(persistence.outlook({"id": 1}, [], now=FUTURE))

    def test_recent_incidents_are_not_history_because_their_outcome_is_unknown(self):
        self.add_incident(3, hours_apart=(4.0,))
        now = START + timedelta(hours=7, minutes=30)           # the only incident began 30 minutes ago
        frame, _ = persistence.build_dataset(self.conn(), "2026-01-01", with_weather=False, now=now)
        self.assertTrue(frame.empty)

    def test_serving_predicts_inside_the_horizon_and_reports_the_outcome_after_it(self):
        self._history()
        report = persistence.train(self.conn(), since="2026-01-01", with_weather=False, save=True, now=FUTURE)
        self.assertTrue((persistence.model_dir() / "model.json").exists(), report)
        young = self.add_incident(5)
        incident = {"id": young, "centroid_latitude": 36.0, "centroid_longitude": -91.5}
        members = database.list_fire_incident_members(young)
        t0 = F.first_window(F.satellite_members(members))[0]
        predicted = persistence.outlook(incident, members, self.conn(), now=t0 + timedelta(minutes=45))
        self.assertEqual(predicted["status"], "predicted")
        self.assertGreater(predicted["probability"], 0.5)      # a big start persists in this data
        self.assertIsNone(predicted["persisted"])
        ended = persistence.outlook(incident, members, self.conn(), now=t0 + timedelta(hours=5))
        self.assertEqual((ended["status"], ended["persisted"], ended["probability"]), ("observed", False, None))
        lasting = self.add_incident(2, hours_apart=(4.0,))
        lasting_members = database.list_fire_incident_members(lasting)
        seen = persistence.outlook({"id": lasting}, lasting_members, self.conn(), now=FUTURE)
        self.assertEqual((seen["status"], seen["persisted"]), ("observed", True))

    def test_an_incident_with_no_satellite_detection_gets_nothing(self):
        self._history(320)
        persistence.train(self.conn(), since="2026-01-01", with_weather=False, save=True, now=FUTURE)
        self.assertIsNone(persistence.outlook({"id": 1}, [{"source": "user_submission", "occurred_at": _iso(START)}], self.conn(), now=FUTURE))


class IncidentTypeTests(ModelTestBase):
    def _labeled(self, per_class):
        """Wildfires: many detections over many hours. Planned burns: a few detections, over quickly."""
        for _ in range(per_class):
            self.add_incident(6, hours_apart=(3.0, 5.0, 8.0), cause="wildfire")
            self.add_incident(2, hours_apart=(0.0,), cause="prescribed")

    def test_with_few_labels_it_says_it_is_collecting_and_shows_no_number(self):
        self._labeled(10)                                      # 20 labelled incidents
        report = incident_type.train(self.conn(), save=True)
        self.assertEqual(report["status"], "collecting_labels")
        self.assertFalse(report["gate_passed"])
        result = incident_type.assess({"id": 1, "centroid_latitude": 36.0, "centroid_longitude": -91.5},
                                      database.list_fire_incident_members(1), self.conn())
        self.assertEqual(result["status"], "collecting_labels")
        self.assertNotIn("score", result)

    def test_a_model_that_is_no_better_than_one_input_is_not_served(self):
        # In this data one input (the number of detections) separates the labels perfectly, so a 19-input model adds nothing:
        # the gate must refuse it however accurate it looks. This is the real situation on production (cropland share alone).
        self._labeled(30)
        report = incident_type.train(self.conn(), save=True)
        self.assertGreater(report["cv_auc"], 0.9)
        self.assertGreaterEqual(report["best_single_input_auc"], report["cv_auc"] - 0.02)
        self.assertEqual(report["status"], "no_better_than_one_input", report)
        self.assertFalse(report["gate_passed"])
        self.assertFalse((incident_type.model_dir() / "model.joblib").exists())
        self.assertEqual(incident_type.assess({"id": 1}, [], self.conn())["status"], "no_better_than_one_input")

    def test_the_gate_wants_accuracy_a_safe_interval_and_a_win_over_the_best_single_input(self):
        good = {"cv_auc": 0.85, "cv_auc_low": 0.7, "best_single_input_auc": 0.78}
        self.assertTrue(incident_type.passes_gate(good))
        self.assertFalse(incident_type.passes_gate({**good, "best_single_input_auc": 0.84}))     # not better than one input
        self.assertFalse(incident_type.passes_gate({**good, "cv_auc_low": 0.55}))                # lucky small sample
        self.assertFalse(incident_type.passes_gate({**good, "cv_auc": 0.68}))                    # not accurate enough
        self.assertFalse(incident_type.passes_gate({"cv_auc": None}))

    def test_when_a_model_does_pass_it_serves_a_relative_score_with_its_own_accuracy_attached(self):
        self._labeled(30)
        with patch.object(incident_type, "passes_gate", return_value=True):           # exercise saving and serving
            report = incident_type.train(self.conn(), save=True)
        self.assertEqual(report["status"], "ok", report)
        wildfire = self.add_incident(7, hours_apart=(3.0, 6.0, 9.0))
        planned = self.add_incident(2)

        def assess(incident_id):
            with closing(sqlite3.connect(self.db)) as c:
                lat, lon = c.execute("SELECT centroid_latitude, centroid_longitude FROM fire_incidents WHERE id = ?", (incident_id,)).fetchone()
            return incident_type.assess({"id": incident_id, "centroid_latitude": lat, "centroid_longitude": lon},
                                        database.list_fire_incident_members(incident_id), self.conn())
        high, low = assess(wildfire), assess(planned)
        self.assertEqual(high["status"], "predicted")
        self.assertGreater(high["score"], low["score"])
        self.assertEqual((high["label"], low["label"]), ("wildfire", "planned burn"))
        self.assertIn("not a calibrated probability", high["note"])
        self.assertLessEqual(high["cv_auc_low"], high["cv_auc"])

    def test_labels_come_from_every_human_source_and_conflicts_are_dropped(self):
        review = self.add_incident(3, cause="wildfire")
        feedback = self.add_incident(3)
        cropland = self.add_incident(3)
        conflict = self.add_incident(3, cause="prescribed")
        unlabeled = self.add_incident(3)
        with closing(sqlite3.connect(self.db)) as conn:
            conn.execute("INSERT INTO fire_incident_feedback (incident_id, classification, status) VALUES (?, 'controlled_burn', 'approved')", (feedback,))
            conn.execute("INSERT INTO fire_incident_feedback (incident_id, classification, status) VALUES (?, 'confirmed_fire', 'approved')", (conflict,))
            conn.execute("INSERT INTO fire_incident_feedback (incident_id, classification, status) VALUES (?, 'controlled_burn', 'pending')", (unlabeled,))
            conn.execute("""CREATE TABLE IF NOT EXISTS cropland_burn_labels (incident_id INTEGER, answer INTEGER)""")
            conn.execute("INSERT INTO cropland_burn_labels VALUES (?, 1)", (cropland,))
            conn.commit()
        labels = incident_type.labeled_incidents(self.conn())
        self.assertEqual(labels, {review: incident_type.WILDFIRE, feedback: incident_type.PLANNED, cropland: incident_type.PLANNED})

    def test_when_labels_disappear_the_old_model_is_withdrawn_and_nothing_stale_is_shown(self):
        self._labeled(30)
        with patch.object(incident_type, "passes_gate", return_value=True):
            self.assertEqual(incident_type.train(self.conn(), save=True)["status"], "ok")
        self.assertTrue((incident_type.model_dir() / "model.joblib").exists())
        with closing(sqlite3.connect(self.db)) as conn:        # staff reset the labels
            conn.execute("UPDATE fire_events SET cause_category = 'unknown'")
            conn.commit()
        report = incident_type.train(self.conn(), save=True)
        self.assertEqual(report["status"], "collecting_labels")
        self.assertFalse((incident_type.model_dir() / "model.joblib").exists())
        self.assertEqual(incident_type.assess({"id": 1}, [], self.conn())["status"], "collecting_labels")


if __name__ == "__main__":
    unittest.main()
