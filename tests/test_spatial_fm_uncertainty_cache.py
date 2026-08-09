import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from services import spatial_fm_uncertainty_cache as cache


class SpatialFmUncertaintyCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.cache_dir_patch = patch.object(cache, "CACHE_DIR", Path(self.temporary.name))
        self.cache_dir_patch.start()

    def tearDown(self):
        self.cache_dir_patch.stop()
        self.temporary.cleanup()

    def _prediction(self, steps=3, shape=(4, 4)):
        return {
            "p10": np.random.rand(steps, *shape).astype("float32"),
            "p50": np.random.rand(steps, *shape).astype("float32"),
            "p90": np.random.rand(steps, *shape).astype("float32"),
            "confidence": np.random.rand(steps, *shape).astype("float32"),
            "nearest_station_distance_deg": np.random.rand(*shape).astype("float32"),
            "effective_station_count": np.random.rand(*shape).astype("float32"),
        }

    def test_persist_returns_none_when_prediction_is_none(self):
        self.assertIsNone(cache.persist(None, run_date="2026-08-01T12:00:00Z"))
        self.assertEqual(list(Path(self.temporary.name).glob("*")), [])

    def test_persist_writes_npz_and_json_sidecar(self):
        path = cache.persist(self._prediction(), run_date="2026-08-01T12:00:00Z")
        self.assertIsNotNone(path)
        self.assertTrue(path.exists())
        sidecar = path.with_suffix(".json")
        self.assertTrue(sidecar.exists())

    def test_persisted_arrays_round_trip(self):
        prediction = self._prediction(steps=2, shape=(3, 3))
        path = cache.persist(prediction, run_date="2026-08-01T12:00:00Z")
        with np.load(path) as loaded:
            np.testing.assert_array_equal(loaded["p10"], prediction["p10"])
            np.testing.assert_array_equal(loaded["confidence"], prediction["confidence"])

    def test_run_id_is_used_as_filename(self):
        path = cache.persist(self._prediction(), run_date="2026-08-01T12:00:00Z")
        self.assertEqual(path.stem, "20260801_12")

    def test_never_raises_on_malformed_prediction(self):
        # Missing keys should be swallowed, not raised - forecast generation
        # must never fail because of this cache.
        result = cache.persist({"p10": None}, run_date="2026-08-01T12:00:00Z")
        self.assertIsNone(result)

    def test_purge_stale_removes_old_files_only(self):
        old_path = cache.persist(self._prediction(), run_date="2026-01-01T00:00:00Z")
        recent_path = cache.persist(self._prediction(), run_date="2026-08-01T00:00:00Z")
        old_time = time.time() - 30 * 86400
        for p in (old_path, old_path.with_suffix(".json")):
            import os
            os.utime(p, (old_time, old_time))

        removed = cache.purge_stale(retention_days=14)
        self.assertEqual(removed, 2)
        self.assertFalse(old_path.exists())
        self.assertTrue(recent_path.exists())


if __name__ == "__main__":
    unittest.main()
