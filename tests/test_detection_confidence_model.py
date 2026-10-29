import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from core import database
from services import detection_confidence_trainer as trainer

features = trainer.load_module("features")
evaluate = trainer.load_module("evaluate")


class LabelTests(unittest.TestCase):
    def test_only_not_a_fire_is_negative_every_kind_of_fire_is_positive(self):
        label = features.label_for
        for cause in ("wildfire", "prescribed", "agricultural", "debris_burn", "unknown"):
            self.assertEqual(label({"status": "approved", "cause_category": cause}), 1, cause)
        self.assertEqual(label({"status": "rejected", "cause_category": "unknown"}), 0)
        self.assertEqual(label({"status": "pending", "cause_category": "not_a_fire"}), 0)
        self.assertEqual(label({"status": "approved", "cause_category": "not_a_fire"}), 0)  # contradiction: negative wins
        self.assertEqual(label({"status": "pending", "verification_tier": "official_source_confirmed"}), 1)

    def test_an_undecided_detection_has_no_label(self):
        self.assertIsNone(features.label_for({"status": "pending", "cause_category": "unknown"}))
        self.assertIsNone(features.label_for({}))


class TrainingRowsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "smf.db"
        patcher = patch.object(database, "get_db_path", return_value=self.db)
        patcher.start()
        self.addCleanup(patcher.stop)
        database.init_database()

    def tearDown(self):
        try:
            self.tmp.cleanup()
        except PermissionError:
            pass

    def _event(self, source, status, cause="unknown", tier="unverified"):
        with sqlite3.connect(self.db) as conn:
            conn.execute("INSERT INTO fire_events (source, external_id, status, verification_tier, cause_category, "
                         "latitude, longitude, occurred_at) VALUES (?, ?, ?, ?, ?, 37, -92, 'x')",
                         (source, f"{source}-{status}-{cause}", status, tier, cause))

    def test_reviewed_detections_all_count_and_pending_ones_do_not(self):
        self._event("viirs", "rejected")                       # not a fire
        self._event("viirs", "approved")                       # approved with no label: a fire
        self._event("modis", "approved", cause="prescribed")   # a controlled burn is a fire
        self._event("ngfs", "pending", cause="not_a_fire")     # labelled not a fire by feedback
        self._event("viirs", "pending")                        # undecided: not training data
        self._event("user_submission", "rejected")             # a report, not a satellite detection
        rows = database.list_labeled_detection_events()
        self.assertEqual(sorted(features.label_for(r) for r in rows), [0, 0, 1, 1])
        self.assertTrue(all("incident_id" in r and "status" in r for r in rows))


class LoaderTests(unittest.TestCase):
    def test_loads_without_the_package_folder_on_the_path_and_restores_foreign_modules(self):
        foreign = types.ModuleType("features")
        foreign.marker = "someone else's features module"
        with patch.dict(sys.modules, {"features": foreign}):
            module = trainer.load_module("retrain")
            self.assertTrue(callable(module.retrain_and_gate))
            self.assertIs(sys.modules["features"], foreign)
        self.assertNotIn(str(trainer.PACKAGE_DIR), sys.path)  # the folder is only on the path while loading
        self.assertNotEqual(module.FEATURE_NAMES, getattr(foreign, "FEATURE_NAMES", None))

    def test_unknown_module_names_are_refused(self):
        with self.assertRaises(ValueError):
            trainer.load_module("os")


def _row(index, incident, fire):
    """A synthetic detection: fires are hot with high FRP and a high satellite confidence (but noisily so)."""
    hot = fire if index % 5 else not fire  # one in five breaks the pattern, so the labels are not trivially separable
    return {
        "id": index, "incident_id": incident, "source": "viirs", "status": "approved" if fire else "rejected",
        "cause_category": "wildfire" if fire else "unknown",
        "frp": 40.0 if hot else 2.0, "confidence": "high" if hot else "low", "bright_t7": 340.0 if hot else 300.0,
        "bright_t13": 300.0, "pixel_area": 0.14, "quality_flag": 0, "daynight": "D",
        "solar_zenith_angle": float(incident),  # unique per incident, so the split can be checked from the features
        "satellite_zenith_angle": 20.0, "land_cover": "Trees:100",
    }


def _rows(incidents=24, per_incident=3):
    rows = []
    for incident in range(1, incidents + 1):
        for _ in range(per_incident):
            rows.append(_row(len(rows) + 1, incident, fire=incident % 2 == 0))
    return rows


class RecordingFit:
    """A scikit-learn stand-in for XGBoost that records which incidents were in each training and test fold."""

    def __init__(self):
        self.folds = []

    def __call__(self, train_x, train_y, test_x):
        from sklearn.ensemble import HistGradientBoostingClassifier

        zenith = features.FEATURE_NAMES.index("solar_zenith_norm")
        self.folds.append(({round(r[zenith] * 180) for r in train_x}, {round(r[zenith] * 180) for r in test_x}))
        return HistGradientBoostingClassifier(max_iter=30, random_state=0).fit(train_x, train_y).predict_proba(test_x)[:, 1]


class EvaluationTests(unittest.TestCase):
    def test_one_fire_is_never_in_both_the_training_and_the_test_fold(self):
        fit = RecordingFit()
        result = evaluate.evaluate_rows(_rows(), fit_predict=fit)
        self.assertTrue(result["evaluated"])
        self.assertEqual((result["method"], result["groups"]), ("grouped_cv", 24))
        self.assertGreaterEqual(len(fit.folds), 2)
        for train_incidents, test_incidents in fit.folds:
            self.assertFalse(train_incidents & test_incidents)

    def test_reports_the_metrics_and_the_satellite_confidence_baseline(self):
        result = evaluate.evaluate_rows(_rows(), fit_predict=RecordingFit())
        self.assertGreater(result["auc"], 0.7)
        for key in ("pr_auc", "brier", "logloss", "baseline_auc", "auc_lift"):
            self.assertIsNotNone(result[key], key)
        self.assertAlmostEqual(result["auc_lift"], result["auc"] - result["baseline_auc"])
        self.assertEqual(result["positives"] + result["negatives"], result["samples"])

    def test_too_little_data_or_only_one_class_is_not_evaluated(self):
        self.assertFalse(evaluate.evaluate_rows(_rows(incidents=2, per_incident=2), fit_predict=RecordingFit())["evaluated"])
        only_fires = [dict(row, status="approved", cause_category="wildfire") for row in _rows()]
        self.assertFalse(evaluate.evaluate_rows(only_fires, fit_predict=RecordingFit())["evaluated"])
        self.assertFalse(evaluate.evaluate_rows([], fit_predict=RecordingFit())["evaluated"])


class GateTests(unittest.TestCase):
    """retrain_and_gate registers a model only when there is enough of each class and it beats the satellite's own
    confidence. XGBoost is replaced by a fake that just 'saves' a file."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.retrain = trainer.load_module("retrain")
        models = Path(self.tmp.name)
        for name, value in (("LATEST_MODEL_PATH", models / "m.json"), ("LATEST_MODEL_META_PATH", models / "meta.json"),
                            ("MODELS_DIR", models)):
            patcher = patch.object(self.retrain, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        class FakeBooster:
            def save_model(self, path):
                Path(path).write_text("model")

        patcher = patch.object(self.retrain, "fit_booster", lambda samples, labels: FakeBooster())
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run(self, rows, evaluation):
        with patch.object(self.retrain, "load_training_rows", lambda: rows), \
                patch.object(self.retrain, "evaluate_rows", lambda r: evaluation):
            return self.retrain.retrain_and_gate()

    def test_too_few_of_one_class_does_not_train(self):
        rows = _rows(incidents=30)
        rows = [r for r in rows if r["status"] == "approved"] + [r for r in rows if r["status"] == "rejected"][:3]
        result = self._run(rows, {"evaluated": False})
        self.assertEqual((result["trained"], result["registered"]), (False, False))
        self.assertIn("at least 10 of each", result["reason"])

    def test_a_model_that_does_not_beat_the_satellite_confidence_is_not_registered(self):
        result = self._run(_rows(), {"evaluated": True, "method": "grouped_cv", "auc": 0.70, "baseline_auc": 0.80})
        self.assertEqual((result["trained"], result["registered"]), (True, False))
        self.assertIn("does not beat the satellite's own confidence", result["reason"])
        self.assertFalse(self.retrain.LATEST_MODEL_PATH.exists())

    def test_a_better_model_is_registered_with_its_grouped_evaluation(self):
        result = self._run(_rows(), {"evaluated": True, "method": "grouped_cv", "auc": 0.9, "baseline_auc": 0.8})
        self.assertTrue(result["registered"])
        self.assertTrue(self.retrain.LATEST_MODEL_PATH.exists())
        self.assertIn('"grouped_cv"', self.retrain.LATEST_MODEL_META_PATH.read_text())

    def test_an_older_randomly_split_auc_is_not_used_to_reject_a_new_model(self):
        self.retrain.LATEST_MODEL_META_PATH.write_text('{"eval": {"auc": 0.99}}')  # no "method": the old, leaky way
        result = self._run(_rows(), {"evaluated": True, "method": "grouped_cv", "auc": 0.9, "baseline_auc": 0.8})
        self.assertTrue(result["registered"])
        # ...but a real grouped-CV regression still is
        self.retrain.LATEST_MODEL_META_PATH.write_text('{"eval": {"method": "grouped_cv", "auc": 0.99}}')
        self.assertFalse(self._run(_rows(), {"evaluated": True, "method": "grouped_cv", "auc": 0.9, "baseline_auc": 0.8})["registered"])


if __name__ == "__main__":
    unittest.main()


class NotAFireSurvivesDeletionTests(TrainingRowsTests):
    """The "Not a fire" button deletes the incident after 7 days; its detections must still count as negatives."""

    def _incident_with_events(self, count=2):
        with sqlite3.connect(self.db) as conn:
            incident = conn.execute("INSERT INTO fire_incidents (centroid_latitude, centroid_longitude, first_detected_at, "
                                    "last_detected_at, detection_count) VALUES (37, -92, 'a', 'b', ?)", (count,)).lastrowid
            ids = [conn.execute("INSERT INTO fire_events (source, external_id, status, latitude, longitude, occurred_at, "
                                "incident_id) VALUES ('viirs', ?, 'pending', 37, -92, 'x', ?)", (f"e{n}", incident)).lastrowid
                   for n in range(count)]
        return incident, ids

    def _negatives(self):
        return sorted(r["id"] for r in database.list_labeled_detection_events() if features.label_for(r) == 0)

    def test_the_full_not_a_fire_flow_keeps_every_detection_as_a_negative_after_the_delete(self):
        from services.fire_labeling import apply_feedback_label

        incident, first = self._incident_with_events()
        apply_feedback_label(incident, "not_a_fire", reviewed_by="admin")
        database.set_incident_review_decision(incident, "rejected", "admin", reason="not_a_fire")
        database.schedule_incident_deletion(incident, days=7)
        # a detection that joins during the 7-day hold inherits the rejection...
        with sqlite3.connect(self.db) as conn:
            conn.row_factory = sqlite3.Row
            late = conn.execute("INSERT INTO fire_events (source, external_id, status, latitude, longitude, occurred_at, "
                                "incident_id) VALUES ('viirs', 'late', 'pending', 37, -92, 'x', ?)", (incident,)).lastrowid
            database.apply_incident_review_to_new_event(conn.cursor(), late, incident)
            conn.commit()
        # ...and the hold ends
        with sqlite3.connect(self.db) as conn:
            conn.execute("UPDATE fire_incidents SET delete_after = datetime('now', '-1 hour') WHERE id = ?", (incident,))
        self.assertEqual(database.purge_due_fire_incidents(), 1)
        with sqlite3.connect(self.db) as conn:
            statuses = {r[0] for r in conn.execute("SELECT status FROM fire_events WHERE incident_id = ?", (incident,))}
        self.assertEqual(statuses, {"deleted"})
        self.assertEqual(self._negatives(), sorted([*first, late]))

    def test_a_straggler_that_never_got_the_label_is_labelled_when_the_hold_ends(self):
        incident, ids = self._incident_with_events(1)
        database.set_incident_review_decision(incident, "rejected", "admin", reason="not_a_fire")
        with sqlite3.connect(self.db) as conn:
            conn.execute("UPDATE fire_incidents SET delete_after = datetime('now', '-1 hour') WHERE id = ?", (incident,))
        database.purge_due_fire_incidents()
        self.assertEqual(self._negatives(), ids)

    def test_backfill_labels_earlier_deletions_once_and_never_touches_an_admin_approval(self):
        incident, ids = self._incident_with_events(3)
        database.set_incident_review_decision(incident, "rejected", "admin", reason="not_a_fire")
        with sqlite3.connect(self.db) as conn:
            conn.execute("UPDATE fire_events SET status = 'deleted' WHERE id IN (?, ?)", ids[:2])
            conn.execute("UPDATE fire_events SET status = 'approved' WHERE id = ?", (ids[2],))
        self.assertEqual(self._negatives(), [])  # deleted and unlabelled: lost, as before the fix
        self.assertEqual(database.backfill_not_a_fire_labels(dry_run=True), 2)
        self.assertEqual(database.backfill_not_a_fire_labels(), 2)
        self.assertEqual(database.backfill_not_a_fire_labels(), 0)
        self.assertEqual(self._negatives(), sorted(ids[:2]))

    def test_the_plain_delete_button_stays_unlabelled_it_could_be_a_duplicate_or_a_bogus_cluster(self):
        incident, ids = self._incident_with_events()
        database.delete_fire_incident_and_members(incident, actor="admin", reason="duplicate")
        self.assertEqual(self._negatives(), [])
