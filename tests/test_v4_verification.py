"""Covers Phase 4's new v4_verification.py - v4_shadow.py::attach_observations()
existed but was never called by anything (confirmed via repo-wide grep).
Mirrors test_v5_verification.py's coverage shape for the structurally
identical v5_verification.py this was mirrored from."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from services import v4_verification


class VerifyPendingTests(unittest.TestCase):
    def test_scores_a_backlog_run_even_when_no_fresh_raw_observations_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence_root = Path(directory) / "evidence"
            evidence_root.mkdir()
            raw_root = Path(directory) / "raw"
            (evidence_root / "run1.prediction.json").write_text(json.dumps({
                "run_id": "run1", "row_keys": ["r|STA1|2026-01-01T00:00:00Z"],
                "v4_quantiles": [[5, 6, 7, 10.0, 13, 14, 15]], "v4_category": [1],
                "bundle_manifest_sha256": "abc",
            }))
            (evidence_root / "run1.observation.json").write_text(json.dumps({
                "run_id": "run1", "attached_at": "2026-01-01T01:00:00Z",
                "observations": {"r|STA1|2026-01-01T00:00:00Z":
                                 {"target_fm": 11.0, "actual_category": 1, "available": True}},
            }))

            result = v4_verification.verify_pending(evidence_root=evidence_root, raw_root=raw_root)

            self.assertEqual(result["scoring"]["scored_count"], 1)
            self.assertTrue((evidence_root / "run1.scored.json").exists())

    def test_attaches_a_mature_prediction_against_a_real_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence_root = Path(directory) / "evidence"; evidence_root.mkdir()
            raw_root = Path(directory) / "raw"; raw_root.mkdir()

            valid_time = "2026-01-01T00:00:00Z"
            (evidence_root / "run1.prediction.json").write_text(json.dumps({
                "run_id": "run1", "recorded_at": "2025-12-31T20:00:00Z",
                "row_keys": [f"run1|STA1|{valid_time}"],
                "v4_quantiles": [[5, 6, 7, 10.0, 13, 14, 15]], "v4_category": [1],
                "bundle_manifest_sha256": "abc",
            }))
            (raw_root / "raw_data_20260101.json").write_text(json.dumps({
                "STATION": [{
                    "STID": "STA1",
                    "OBSERVATIONS": {
                        "date_time": ["2026-01-01T00:20:00Z"],
                        "fuel_moisture_set_1": [11.0],
                        "relative_humidity_set_1": [30.0],
                        "wind_speed_set_1": [10.0],
                    },
                }],
            }))

            now = pd.Timestamp("2026-01-01T02:00:00Z")
            result = v4_verification.verify_pending(evidence_root=evidence_root, raw_root=raw_root, now=now)

            self.assertEqual(result["attached"], 1)
            self.assertTrue((evidence_root / "run1.observation.json").exists())
            observation = json.loads((evidence_root / "run1.observation.json").read_text())
            row = observation["observations"][f"run1|STA1|{valid_time}"]
            self.assertEqual(row["target_fm"], 11.0)
            self.assertTrue(row["available"])

    def test_immature_prediction_stays_pending_not_attached(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence_root = Path(directory) / "evidence"; evidence_root.mkdir()
            raw_root = Path(directory) / "raw"; raw_root.mkdir()
            valid_time = "2026-01-01T05:00:00Z"  # far in the future relative to `now` below
            (evidence_root / "run1.prediction.json").write_text(json.dumps({
                "run_id": "run1", "recorded_at": "2026-01-01T00:00:00Z",
                "row_keys": [f"run1|STA1|{valid_time}"],
                "v4_quantiles": [[5, 6, 7, 10.0, 13, 14, 15]], "v4_category": [1],
                "bundle_manifest_sha256": "abc",
            }))
            (raw_root / "raw_data_20260101.json").write_text(json.dumps({
                "STATION": [{"STID": "STA1", "OBSERVATIONS": {
                    "date_time": ["2026-01-01T01:00:00Z"], "fuel_moisture_set_1": [11.0],
                    "relative_humidity_set_1": [30.0], "wind_speed_set_1": [10.0],
                }}],
            }))
            now = pd.Timestamp("2026-01-01T01:10:00Z")  # well before valid_time + tolerance
            result = v4_verification.verify_pending(evidence_root=evidence_root, raw_root=raw_root, now=now)
            self.assertEqual(result["pending"], 1)
            self.assertEqual(result["attached"], 0)
            self.assertFalse((evidence_root / "run1.observation.json").exists())


if __name__ == "__main__":
    unittest.main()
