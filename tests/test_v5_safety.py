import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.fire_danger import RULE_SPEC_SHA256
from core.precipitation import PRECIPITATION_CONTRACT_SHA256, PRECIPITATION_CONTRACT_VERSION
from services import v5_shadow


class V5ShadowTests(unittest.TestCase):
    @staticmethod
    def _bundle(root):
        bundle = root / "bundle"; bundle.mkdir(); assets = {}
        for name, value in (("base_xgboost.json", "base"), ("specialist_xgboost.json", "specialist"),
                            ("guard.json", "guard"), ("uncertainty.json", "uncertainty")):
            path = bundle / name; path.write_text(value); assets[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        contract = {"rule_spec_sha256": RULE_SPEC_SHA256, "manifest_sha256": "manifest",
                    "precipitation_contract_version": PRECIPITATION_CONTRACT_VERSION,
                    "precipitation_contract_sha256": PRECIPITATION_CONTRACT_SHA256,
                    "base_model_sha256": assets["base_xgboost.json"],
                    "specialist_model_sha256": assets["specialist_xgboost.json"],
                    "guard_sha256": assets["guard.json"], "uncertainty_sha256": assets["uncertainty.json"]}
        (bundle / "contract.json").write_text(json.dumps(contract))
        names = (*assets.keys(), "contract.json")
        shadow = {"status": "experimental_shadow_only", "registry_channel": None,
                  "rule_spec_sha256": RULE_SPEC_SHA256,
                  "precipitation_contract_version": PRECIPITATION_CONTRACT_VERSION,
                  "precipitation_contract_sha256": PRECIPITATION_CONTRACT_SHA256,
                  "assets": {name: hashlib.sha256((bundle / name).read_bytes()).hexdigest() for name in names}}
        (bundle / "shadow_bundle_manifest.json").write_text(json.dumps(shadow)); return bundle

    def setUp(self):
        v5_shadow._state.update(enabled=True, consecutive_failures=0, last_error=None, runs=0,
                                fallback_rows=0, unavailable=0)

    def test_prediction_is_immutable_and_observation_is_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); bundle = self._bundle(root); evidence = root / "evidence"
            arguments = ("run", ["row"], [9], [8.5], [8], [[6, 8, 10]], [-.5], [1],
                         ["oof_improvement"], ["summer_dry"], [20], [15])
            self.assertTrue(v5_shadow.record_predictions(*arguments, bundle, evidence_root=evidence))
            original = (evidence / "run.prediction.json").read_text()
            self.assertFalse(v5_shadow.record_predictions(*arguments, bundle, evidence_root=evidence))
            self.assertEqual((evidence / "run.prediction.json").read_text(), original)
            v5_shadow.attach_observations("run", {"target_fm": [7]}, evidence)
            self.assertEqual((evidence / "run.prediction.json").read_text(), original)

    def test_repeated_failure_disables_only_v5(self):
        with tempfile.TemporaryDirectory() as directory:
            for index in range(v5_shadow.MAX_FAILURES):
                self.assertFalse(v5_shadow.record_predictions(str(index), [], [], [], [], np.empty((0, 3)),
                                                               [], [], [], [], [], [], Path(directory) / "missing"))
            self.assertFalse(v5_shadow.diagnostics()["enabled"])


if __name__ == "__main__": unittest.main()
