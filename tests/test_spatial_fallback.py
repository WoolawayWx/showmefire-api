import unittest
from unittest.mock import patch

from services import spatial_fm


class SpatialFallbackTests(unittest.TestCase):
    def test_missing_assets_returns_none_and_records_fallback(self):
        with patch.object(spatial_fm, "_load_runtime", side_effect=FileNotFoundError("no spatial stable")):
            result = spatial_fm.try_predict(object(), [(-93.0, 38.0, 8), (-92.0, 38.5, 9), (-91.0, 39.0, 10)])
        self.assertIsNone(result); self.assertTrue(spatial_fm.diagnostics()["fallback"])
        self.assertIn("no spatial stable", spatial_fm.diagnostics()["fallback_reason"])


if __name__ == "__main__": unittest.main()
