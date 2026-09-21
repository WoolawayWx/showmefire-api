import json
import tempfile
import unittest
from pathlib import Path

from services import shadow_observation_scoring as scoring


def _write_prediction(evidence_root: Path, run_id: str, *, family: str) -> None:
    if family == "v5":
        body = {
            "run_id": run_id, "row_keys": ["r1|STA1|2026-01-01T00:00:00Z", "r1|STA2|2026-01-01T00:00:00Z"],
            "v5_fm": [10.0, 12.0], "v5_category": [1, 2],
            "bundle_manifest_sha256": "abc",
        }
    else:
        body = {
            "run_id": run_id, "row_keys": ["r1|STA1|2026-01-01T00:00:00Z", "r1|STA2|2026-01-01T00:00:00Z"],
            "v4_quantiles": [[5, 6, 7, 10.0, 13, 14, 15], [7, 8, 9, 12.0, 15, 16, 17]],
            "v4_category": [1, 2],
            "bundle_manifest_sha256": "abc",
        }
    (evidence_root / f"{run_id}.prediction.json").write_text(json.dumps(body))


def _write_observation(evidence_root: Path, run_id: str, rows: dict) -> None:
    (evidence_root / f"{run_id}.observation.json").write_text(
        json.dumps({"run_id": run_id, "attached_at": "2026-01-01T01:00:00Z", "observations": rows}))


class FindUnscoredRunsTests(unittest.TestCase):
    def test_only_returns_runs_with_observation_but_no_score(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_observation(root, "run1", {})
            _write_observation(root, "run2", {})
            (root / "run2.scored.json").write_text("{}")
            self.assertEqual(scoring.find_unscored_runs(root), ["run1"])


class ScoreV5RunTests(unittest.TestCase):
    def test_computes_mae_and_category_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_prediction(root, "run1", family="v5")
            _write_observation(root, "run1", {
                "r1|STA1|2026-01-01T00:00:00Z": {"target_fm": 11.0, "actual_category": 1, "available": True},
                "r1|STA2|2026-01-01T00:00:00Z": {"target_fm": 10.0, "actual_category": 2, "available": True},
            })
            result = scoring.score_v5_run("run1", root)
            self.assertEqual(result["family"], "v5")
            self.assertEqual(result["n_matched"], 2)
            self.assertAlmostEqual(result["mae_vs_observation"], (1.0 + 2.0) / 2, places=4)
            self.assertEqual(result["category_match"]["n"], 2)
            self.assertAlmostEqual(result["category_match"]["match_rate"], 1.0, places=4)  # both categories match

    def test_missing_observation_returns_none(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_prediction(root, "run1", family="v5")
            self.assertIsNone(scoring.score_v5_run("run1", root))

    def test_unavailable_rows_are_excluded_from_category_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_prediction(root, "run1", family="v5")
            _write_observation(root, "run1", {
                "r1|STA1|2026-01-01T00:00:00Z": {"target_fm": 11.0, "actual_category": None, "available": False},
                "r1|STA2|2026-01-01T00:00:00Z": {"target_fm": 10.0, "actual_category": 2, "available": True},
            })
            result = scoring.score_v5_run("run1", root)
            self.assertEqual(result["category_match"]["n"], 1)
            # MAE is still computed from target_fm regardless of `available` -
            # only category_match filters on the available flag.
            self.assertEqual(result["n_matched"], 2)


class ScoreV4RunTests(unittest.TestCase):
    def test_uses_the_p50_quantile_as_the_point_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_prediction(root, "run1", family="v4")
            _write_observation(root, "run1", {
                "r1|STA1|2026-01-01T00:00:00Z": {"target_fm": 11.0, "actual_category": 1, "available": True},
                "r1|STA2|2026-01-01T00:00:00Z": {"target_fm": 13.0, "actual_category": 3, "available": True},
            })
            result = scoring.score_v4_run("run1", root)
            self.assertEqual(result["family"], "v4")
            self.assertAlmostEqual(result["mae_vs_observation"], (1.0 + 1.0) / 2, places=4)
            self.assertEqual(result["category_match"]["match_rate"], 0.5)  # row1 matches (1==1), row2 doesn't (2!=3)


class ScorePendingRunsTests(unittest.TestCase):
    def test_writes_a_scored_file_per_run_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_prediction(root, "run1", family="v5")
            _write_observation(root, "run1", {
                "r1|STA1|2026-01-01T00:00:00Z": {"target_fm": 11.0, "actual_category": 1, "available": True},
                "r1|STA2|2026-01-01T00:00:00Z": {"target_fm": 10.0, "actual_category": 2, "available": True},
            })
            summary = scoring.score_pending_runs("v5", root)
            self.assertEqual(summary["scored_count"], 1)
            self.assertTrue((root / "run1.scored.json").exists())

            # Re-running finds nothing new to score - the scored file already exists.
            summary2 = scoring.score_pending_runs("v5", root)
            self.assertEqual(summary2["scored_count"], 0)

    def test_a_malformed_prediction_is_recorded_as_an_error_not_a_crash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run1.prediction.json").write_text("{}")  # missing row_keys/v5_fm entirely
            _write_observation(root, "run1", {})
            summary = scoring.score_pending_runs("v5", root)
            self.assertEqual(summary["scored_count"], 0)
            self.assertEqual(len(summary["errors"]), 1)
            self.assertEqual(summary["errors"][0]["run_id"], "run1")


class RollingAccuracySummaryTests(unittest.TestCase):
    def test_averages_the_most_recent_scored_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i, mae in enumerate([1.0, 2.0, 3.0]):
                (root / f"run{i}.scored.json").write_text(json.dumps({
                    "mae_vs_observation": mae, "category_match": {"match_rate": 0.5},
                }))
            summary = scoring.rolling_accuracy_summary("v5", root, window=30)
            self.assertEqual(summary["scored_runs"], 3)
            self.assertAlmostEqual(summary["mean_mae_vs_observation"], 2.0, places=4)
            self.assertAlmostEqual(summary["mean_category_match_rate"], 0.5, places=4)

    def test_empty_evidence_root_returns_none_values(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = scoring.rolling_accuracy_summary("v5", Path(directory))
            self.assertEqual(summary["scored_runs"], 0)
            self.assertIsNone(summary["mean_mae_vs_observation"])
            self.assertIsNone(summary["mean_category_match_rate"])


if __name__ == "__main__":
    unittest.main()
