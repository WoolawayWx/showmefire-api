import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from services import archive_bundler
from services.rtma_capture import _sanitize_dataset
import numpy as np
import xarray as xr


class ArchiveMergeTests(unittest.TestCase):
    def test_rtma_scalar_step_dtype_is_removed_before_serialization(self):
        ds = xr.Dataset({"t2m": (("y", "x"), np.ones((2, 2)))}, coords={"step": np.timedelta64(0, "h")})
        ds["step"].attrs["dtype"] = "timedelta64[ns]"
        sanitized = _sanitize_dataset(ds)
        self.assertNotIn("step", sanitized.coords)
        with tempfile.TemporaryDirectory() as root:
            sanitized.to_netcdf(Path(root) / "rtma.nc")

    def test_existing_members_survive_and_partial_hours_merge(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            output = root / "zips"
            rtma = root / "rtma"
            output.mkdir(); rtma.mkdir()
            final = output / "20260712.zip"
            with zipfile.ZipFile(final, "w") as zf:
                zf.writestr("hrrr/hrrr_20260712_12z.nc", b"historical")
            first = rtma / "rtma_20260712_00z.nc"
            second = rtma / "rtma_20260712_01z.nc"
            first.write_bytes(b"zero"); second.write_bytes(b"one")
            with patch.object(archive_bundler, "OUTPUT_DIR", output):
                archive_bundler.add_files_to_existing_zip("20260712", [first])
                archive_bundler.add_files_to_existing_zip("20260712", [second])
            with zipfile.ZipFile(final) as zf:
                self.assertIsNone(zf.testzip())
                self.assertEqual(zf.read("hrrr/hrrr_20260712_12z.nc"), b"historical")
                self.assertIn("rtma/rtma_20260712_00z.nc", zf.namelist())
                self.assertIn("rtma/rtma_20260712_01z.nc", zf.namelist())

    def test_conflicting_existing_member_preserves_source_and_archive(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); output = root / "zips"; rtma = root / "rtma"
            output.mkdir(); rtma.mkdir()
            final = output / "20260712.zip"
            with zipfile.ZipFile(final, "w") as zf:
                zf.writestr("rtma/rtma_20260712_00z.nc", b"old")
            source = rtma / "rtma_20260712_00z.nc"; source.write_bytes(b"different")
            with patch.object(archive_bundler, "OUTPUT_DIR", output):
                with self.assertRaises(RuntimeError):
                    archive_bundler.add_files_to_existing_zip("20260712", [source])
            self.assertTrue(source.exists())
            with zipfile.ZipFile(final) as zf:
                self.assertEqual(zf.read("rtma/rtma_20260712_00z.nc"), b"old")


if __name__ == "__main__":
    unittest.main()
