import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.fire_danger import (RULE_SPEC_SHA256, calculate_fire_danger,
                              missing_input_diagnostics, reset_missing_input_diagnostics)
from services import v4_shadow
from core.precipitation import PRECIPITATION_CONTRACT_SHA256, PRECIPITATION_CONTRACT_VERSION


class MissingDangerTests(unittest.TestCase):
    def test_missing_is_unavailable_not_low(self):
        reset_missing_input_diagnostics()
        self.assertIsNone(calculate_fire_danger(np.nan, 20, 25))
        self.assertEqual(calculate_fire_danger(np.nan, 20, 25, missing_category=2), 2)
        diagnostics=missing_input_diagnostics();self.assertEqual(diagnostics["unavailable_outputs"],1)
        self.assertEqual(diagnostics["explicit_fallback_outputs"],1)


class V4ShadowTests(unittest.TestCase):
    @staticmethod
    def _bundle(root):
        bundle=root/"bundle";bundle.mkdir();assets={}
        for name,value in (("base_xgboost.json","base"),("guarded_gru.pt","gru"),("lead_guard.json","guard")):
            path=bundle/name;path.write_text(value);assets[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        contract={"rule_spec_sha256":RULE_SPEC_SHA256,"manifest_sha256":"manifest",
                  "precipitation_contract_version":PRECIPITATION_CONTRACT_VERSION,
                  "precipitation_contract_sha256":PRECIPITATION_CONTRACT_SHA256,
                  "base_model_sha256":assets["base_xgboost.json"],"residual_model_sha256":assets["guarded_gru.pt"],
                  "lead_guard_sha256":assets["lead_guard.json"]}
        (bundle/"contract.json").write_text(json.dumps(contract));(bundle/"calibration.json").write_text("{}")
        names=("base_xgboost.json","guarded_gru.pt","lead_guard.json","contract.json","calibration.json")
        shadow={"status":"experimental_shadow_only","registry_channel":None,"rule_spec_sha256":RULE_SPEC_SHA256,
                "precipitation_contract_version":PRECIPITATION_CONTRACT_VERSION,
                "precipitation_contract_sha256":PRECIPITATION_CONTRACT_SHA256,
                "assets":{name:hashlib.sha256((bundle/name).read_bytes()).hexdigest() for name in names}}
        (bundle/"shadow_bundle_manifest.json").write_text(json.dumps(shadow));return bundle

    def test_prediction_precedes_observation_and_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);bundle=self._bundle(root);evidence=root/"evidence"
            v4_shadow._state.update(enabled=True,consecutive_failures=0,last_error=None,runs=0,unavailable=0)
            quantiles=np.asarray([[5,6,7,8,9,10,11]],float)
            self.assertTrue(v4_shadow.record_predictions("run",["row"],[9],quantiles,[20],[15],[.8],[1],bundle,evidence_root=evidence))
            original=(evidence/"run.prediction.json").read_text()
            self.assertFalse(v4_shadow.record_predictions("run",["row"],[9],quantiles,[20],[15],[.8],[1],bundle,evidence_root=evidence))
            self.assertEqual((evidence/"run.prediction.json").read_text(),original)
            observation=v4_shadow.attach_observations("run",{"target_fm":[7]},evidence)
            self.assertTrue(observation.exists());self.assertEqual((evidence/"run.prediction.json").read_text(),original)

    def test_repeated_failures_disable_only_v4_shadow(self):
        with tempfile.TemporaryDirectory() as directory:
            v4_shadow._state.update(enabled=True,consecutive_failures=0,last_error=None,runs=0,unavailable=0)
            for index in range(v4_shadow.MAX_FAILURES):
                self.assertFalse(v4_shadow.record_predictions(str(index),[],[],np.empty((0,7)),[],[],[],[],Path(directory)/"missing"))
            self.assertFalse(v4_shadow.diagnostics()["enabled"])


if __name__=="__main__":unittest.main()
