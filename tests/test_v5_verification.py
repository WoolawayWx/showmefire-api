"""Covers Phase 4's wiring of shadow_observation_scoring into
v5_verification.py::verify_pending - the observation-attachment step
existed already; scoring what it attaches did not, until now."""
import json
import tempfile
import unittest
from pathlib import Path

from services import v5_verification


class VerifyPendingScoringTests(unittest.TestCase):
    def test_scores_a_backlog_run_even_when_no_fresh_raw_observations_exist(self):
        # A run that already has a prediction + an attached observation
        # (e.g. from before this feature existed) but no .scored.json yet -
        # must get scored even on a call where load_observations() finds
        # nothing new, since the early-return path used to skip scoring
        # entirely on a quiet raw-data day.
        with tempfile.TemporaryDirectory() as directory:
            evidence_root = Path(directory) / "evidence"
            evidence_root.mkdir()
            raw_root = Path(directory) / "raw"  # deliberately empty/nonexistent
            (evidence_root / "run1.prediction.json").write_text(json.dumps({
                "run_id": "run1", "row_keys": ["r|STA1|2026-01-01T00:00:00Z"],
                "v5_fm": [10.0], "v5_category": [1], "bundle_manifest_sha256": "abc",
            }))
            (evidence_root / "run1.observation.json").write_text(json.dumps({
                "run_id": "run1", "attached_at": "2026-01-01T01:00:00Z",
                "observations": {"r|STA1|2026-01-01T00:00:00Z":
                                 {"target_fm": 11.0, "actual_category": 1, "available": True}},
            }))

            result = v5_verification.verify_pending(evidence_root=evidence_root, raw_root=raw_root)

            self.assertEqual(result["scoring"]["scored_count"], 1)
            self.assertTrue((evidence_root / "run1.scored.json").exists())

    def test_scoring_failure_does_not_raise_or_break_the_response_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence_root = Path(directory) / "evidence"
            evidence_root.mkdir()
            raw_root = Path(directory) / "raw"
            # Malformed prediction - scoring will error internally per-run,
            # but score_pending_runs itself never raises.
            (evidence_root / "run1.prediction.json").write_text("{}")
            (evidence_root / "run1.observation.json").write_text(json.dumps({"observations": {}}))

            result = v5_verification.verify_pending(evidence_root=evidence_root, raw_root=raw_root)
            self.assertIn("scoring", result)
            self.assertEqual(result["scoring"]["scored_count"], 0)


if __name__ == "__main__":
    unittest.main()
