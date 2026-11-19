"""Shadow evidence is judged per candidate: an old beta's failure or days must not decide a new beta's gate."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services import model_shadow


def row(day, version, elevated=1, failed=False):
    base = {"timestamp": f"2026-10-{day:02d}T12:00:00+00:00", "beta_version": version}
    return {**base, "failed": True, "error": "x"} if failed else {**base, "failed": False, "elevated_samples": elevated}


class EvidencePerCandidateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "model_shadow.jsonl"

    def write(self, rows):
        self.path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    def test_an_old_betas_failure_does_not_block_a_new_candidate(self):
        self.write([row(d, "1.0.1-beta.1") for d in range(1, 8)] + [row(9, "1.0.1-beta.1", failed=True)] + [row(d, "1.0.1-beta.2") for d in range(10, 41)])
        pooled = model_shadow.evaluate_shadow_evidence(self.path)
        self.assertFalse(pooled["passed"])                                           # the old behaviour: one failure blocks everything
        fresh = model_shadow.evaluate_shadow_evidence(self.path, beta_version="1.0.1-beta.2")
        self.assertEqual((fresh["days"], fresh["failures"], fresh["passed"]), (31, 0, True))
        self.assertEqual(fresh["beta_version"], "1.0.1-beta.2")

    def test_days_from_an_earlier_beta_do_not_count_for_the_new_one(self):
        self.write([row(d, "1.0.1-beta.1") for d in range(1, 31)] + [row(31, "1.0.1-beta.2")])
        self.assertTrue(model_shadow.evaluate_shadow_evidence(self.path, beta_version="1.0.1-beta.1")["passed"])
        new = model_shadow.evaluate_shadow_evidence(self.path, beta_version="1.0.1-beta.2")
        self.assertEqual(new["days"], 1)
        self.assertFalse(new["passed"])

    def test_a_failure_of_this_candidate_still_blocks_it(self):
        self.write([row(d, "b2") for d in range(1, 32)] + [row(20, "b2", failed=True)])
        evidence = model_shadow.evaluate_shadow_evidence(self.path, beta_version="b2")
        self.assertEqual(evidence["failures"], 1)
        self.assertFalse(evidence["passed"])

    def test_old_failure_rows_without_a_version_are_reported_but_not_charged_to_anyone(self):
        legacy_failure = {"timestamp": "2026-09-01T00:00:00+00:00", "failed": True, "error": "old"}
        self.write([legacy_failure] + [row(d, "b2") for d in range(1, 32)])
        evidence = model_shadow.evaluate_shadow_evidence(self.path, beta_version="b2")
        self.assertEqual((evidence["failures"], evidence["unattributed_failures"], evidence["passed"]), (0, 1, True))

    def test_the_elevated_sample_requirement_still_applies_per_candidate(self):
        self.write([row(d, "b2", elevated=0) for d in range(1, 32)])
        self.assertFalse(model_shadow.evaluate_shadow_evidence(self.path, beta_version="b2")["passed"])

    def test_a_missing_log_is_empty_evidence_not_an_error(self):
        evidence = model_shadow.evaluate_shadow_evidence(self.path, beta_version="b2")
        self.assertEqual((evidence["days"], evidence["passed"]), (0, False))

    def test_record_shadow_gate_judges_the_current_beta_only(self):
        self.write([row(d, "old") for d in range(1, 5)] + [row(9, "old", failed=True)] + [row(d, "new") for d in range(10, 41)])
        entry = {"beta": {"version": "new"}}
        with patch.object(model_shadow, "get_model_entry", return_value=entry), \
                patch.object(model_shadow, "update_beta_metadata") as save:
            evidence = model_shadow.record_shadow_gate(self.path)
        self.assertTrue(evidence["passed"])
        self.assertEqual(save.call_args.args[1]["shadow"]["beta_version"], "new")


if __name__ == "__main__":
    unittest.main()
