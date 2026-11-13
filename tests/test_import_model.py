import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from pipelines.import_model import (
    GROUND_COVER_VARIABLES,
    IMPORTABLE_MODEL_TYPES,
    ImportValidationError,
    _verify_ground_cover_assets,
    REQUIRED_RISK_FUSION_ASSET_ROLES,
    _sha256,
    _verify_fire_behavior_static_assets,
    _verify_generic_multiasset,
)


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
        with self.assertRaises(ImportValidationError):
            _verify_generic_multiasset({}, {}, {"glm", "guard"})

    def test_rejects_sha256_mismatch(self):
        glm_path = self._write("glm.json")
        files = {"glm.json": glm_path}
        declarations = {"glm": {"filename": "glm.json", "sha256": "0" * 64}}
        with self.assertRaises(ImportValidationError):
            _verify_generic_multiasset(files, declarations, {"glm"})

    def test_rejects_declared_file_not_present(self):
        declarations = {"glm": {"filename": "missing.json", "sha256": "0" * 64}}
        with self.assertRaises(ImportValidationError):
            _verify_generic_multiasset({}, declarations, {"glm"})

    def test_risk_fusion_required_roles_omit_optional_gbm(self):
        # The guard can legitimately weight the GBM residual to zero and ship
        # GLM-only (mirrors v5_guard.py's exact-incumbent-fallback pattern) -
        # gbm must stay optional, never required.
        self.assertNotIn("gbm", REQUIRED_RISK_FUSION_ASSET_ROLES)
        self.assertIn("glm", REQUIRED_RISK_FUSION_ASSET_ROLES)

    def _fire_behavior_release(self, *, synthetic=False):
        bundle = self.dir / "fire_behavior.nc"
        manifest_path = self.dir / "fire_behavior.json"
        shape = (256, 256)
        fields = {
            name: (("y", "x"), np.ones(shape, dtype="float32"))
            for name in (
                "elevation_m", "slope_degrees", "aspect_sin", "aspect_cos",
                "canopy_cover_pct", "canopy_height_m", "latitude", "longitude",
                "static_valid_mask", "fuel_model_fbfm40",
            )
        }
        xr.Dataset(
            fields,
            coords={"x": np.arange(256), "y": np.arange(256)},
            attrs={"grid_fingerprint": "grid-abc"},
        ).to_netcdf(bundle)
        manifest = {
            "sha256": _sha256(bundle),
            "grid_fingerprint": "grid-abc",
            "synthetic": synthetic,
            "validation": {
                "crs_validated": True,
                "units_validated": True,
                "nodata_validated": True,
            },
        }
        manifest_path.write_text(json.dumps(manifest))
        files = {bundle.name: bundle, manifest_path.name: manifest_path}
        declarations = {
            "static_bundle": {
                "filename": bundle.name,
                "sha256": _sha256(bundle),
                "grid_fingerprint": "grid-abc",
            },
            "static_manifest": {
                "filename": manifest_path.name,
                "sha256": _sha256(manifest_path),
            },
        }
        return files, declarations

    def test_accepts_valid_fire_behavior_static_release(self):
        files, declarations = self._fire_behavior_release()
        resolved = _verify_fire_behavior_static_assets(files, declarations)
        self.assertEqual(set(resolved), {"static_bundle", "static_manifest"})

    def test_rejects_synthetic_fire_behavior_static_release(self):
        files, declarations = self._fire_behavior_release(synthetic=True)
        with self.assertRaisesRegex(ImportValidationError, "Synthetic"):
            _verify_fire_behavior_static_assets(files, declarations)

    def _ground_cover_release(self, *, synthetic=False, cell=480, epsg=5070, drop=None, gates=True):
        import pyproj
        bundle = self.dir / "ground_cover.nc"
        manifest_path = self.dir / "ground_cover.json"
        shape = (4, 4)
        names = set(GROUND_COVER_VARIABLES) - {drop}
        fields = {n: (("y", "x"), np.ones(shape, dtype="float32")) for n in names}
        x = 480.0 * 100 + (np.arange(4) + 0.5) * cell
        y = 480.0 * 100 - (np.arange(4) + 0.5) * cell
        xr.Dataset(fields, coords={"x": x, "y": y}, attrs={
            "grid_fingerprint": "gc-abc", "crs_wkt": pyproj.CRS.from_epsg(epsg).to_wkt()}).to_netcdf(bundle)
        manifest_path.write_text(json.dumps({
            "sha256": _sha256(bundle), "grid_fingerprint": "gc-abc", "synthetic": synthetic,
            "validation": {"crs_validated": True, "units_validated": True, "nodata_validated": gates}}))
        files = {bundle.name: bundle, manifest_path.name: manifest_path}
        declarations = {
            "static_bundle": {"filename": bundle.name, "sha256": _sha256(bundle), "grid_fingerprint": "gc-abc"},
            "static_manifest": {"filename": manifest_path.name, "sha256": _sha256(manifest_path)},
        }
        return files, declarations

    def test_accepts_valid_ground_cover_release(self):
        files, declarations = self._ground_cover_release()
        self.assertEqual(set(_verify_ground_cover_assets(files, declarations)),
                         {"static_bundle", "static_manifest"})

    def test_rejects_bad_ground_cover_releases(self):
        cases = [({"synthetic": True}, "Synthetic"), ({"cell": 960}, "480 m"), ({"epsg": 4326}, "EPSG:5070"),
                 ({"drop": "nlcd_forest_pct"}, "variables missing"), ({"gates": False}, "validation gates")]
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs):
                files, declarations = self._ground_cover_release(**kwargs)
                with self.assertRaisesRegex(ImportValidationError, message):
                    _verify_ground_cover_assets(files, declarations)

    def test_rejects_ground_cover_checksum_mismatch(self):
        files, declarations = self._ground_cover_release()
        declarations["static_bundle"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ImportValidationError, "invalid release asset"):
            _verify_ground_cover_assets(files, declarations)

    def test_ground_cover_is_importable_and_needs_both_assets_to_promote(self):
        from models.versioning import validate_promotion_candidate
        self.assertIn("ground_cover_static", IMPORTABLE_MODEL_TYPES)
        blockers = validate_promotion_candidate("ground_cover_static", {"assets": {}})
        self.assertIn("static_bundle asset is required", blockers)
        self.assertIn("static_manifest asset is required", blockers)
        ok = {"assets": {"static_bundle": {"path": "b"}, "static_manifest": {"path": "m"}}}
        self.assertEqual(validate_promotion_candidate("ground_cover_static", ok), [])


if __name__ == "__main__":
    unittest.main()
