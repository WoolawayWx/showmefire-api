import tempfile
import unittest
from pathlib import Path

from pipelines.import_model import REQUIRED_RISK_FUSION_ASSET_ROLES, _verify_generic_multiasset, _sha256


class GenericMultiassetVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.dir = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def _write(self, name: str, content: bytes = b"data") -> Path:
        path = self.dir / name
        path.write_bytes(content)
        return path

    def test_accepts_matching_declarations(self):
        glm_path = self._write("glm.json")
        files = {"glm.json": glm_path}
        declarations = {"glm": {"filename": "glm.json", "sha256": _sha256(glm_path)}}
        resolved = _verify_generic_multiasset(files, declarations, {"glm"})
        self.assertEqual(resolved["glm"], glm_path)

    def test_rejects_missing_required_role(self):
        with self.assertRaises(SystemExit):
            _verify_generic_multiasset({}, {}, {"glm", "guard"})

    def test_rejects_sha256_mismatch(self):
        glm_path = self._write("glm.json")
        files = {"glm.json": glm_path}
        declarations = {"glm": {"filename": "glm.json", "sha256": "0" * 64}}
        with self.assertRaises(SystemExit):
            _verify_generic_multiasset(files, declarations, {"glm"})

    def test_rejects_declared_file_not_present(self):
        declarations = {"glm": {"filename": "missing.json", "sha256": "0" * 64}}
        with self.assertRaises(SystemExit):
            _verify_generic_multiasset({}, declarations, {"glm"})

    def test_risk_fusion_required_roles_omit_optional_gbm(self):
        # The guard can legitimately weight the GBM residual to zero and ship
        # GLM-only (mirrors v5_guard.py's exact-incumbent-fallback pattern) -
        # gbm must stay optional, never required.
        self.assertNotIn("gbm", REQUIRED_RISK_FUSION_ASSET_ROLES)
        self.assertIn("glm", REQUIRED_RISK_FUSION_ASSET_ROLES)


if __name__ == "__main__":
    unittest.main()
