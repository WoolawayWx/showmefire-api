"""R2 writes are billed: an unchanged latest/ file is not uploaded again, but the dated archive always is."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.cdn_upload_state import MAX_AGE_SECONDS, UploadState

from tests.test_cdn_thumbnails import _load_upload_module               # stubs boto3/dotenv like the thumbnail tests do

upload_cdn = _load_upload_module()


class FakeS3:
    def __init__(self):
        self.keys = []

    def upload_file(self, path, bucket, key, ExtraArgs=None):
        self.keys.append(key)


class StateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.file = self.dir / "a.png"
        self.file.write_bytes(b"one")

    def test_same_bytes_are_unchanged_different_bytes_are_not(self):
        state = UploadState(self.dir / "s.json")
        self.assertFalse(state.unchanged("latest/a.png", self.file))
        state.remember("latest/a.png", self.file)
        self.assertTrue(state.unchanged("latest/a.png", self.file))
        self.file.write_bytes(b"two")
        self.assertFalse(state.unchanged("latest/a.png", self.file))

    def test_an_old_entry_is_not_trusted_so_a_deleted_object_comes_back(self):
        clock = [1000.0]
        state = UploadState(self.dir / "s.json", now=lambda: clock[0])
        state.remember("latest/a.png", self.file)
        clock[0] += MAX_AGE_SECONDS + 1
        self.assertFalse(state.unchanged("latest/a.png", self.file))

    def test_state_survives_a_save_and_a_corrupt_file_means_upload_everything(self):
        path = self.dir / "s.json"
        state = UploadState(path)
        state.remember("k", self.file)
        state.save()
        self.assertTrue(UploadState(path).unchanged("k", self.file))
        path.write_text("{ not json")
        self.assertFalse(UploadState(path).unchanged("k", self.file))


class UploadFlowTests(unittest.TestCase):
    def run_upload(self, s3, image, prefix="t"):
        with patch.object(upload_cdn, "R2_ACCESS_KEY_ID", "x"), patch.object(upload_cdn, "R2_SECRET_ACCESS_KEY", "x"), \
                patch.object(upload_cdn, "R2_ACCOUNT_ID", "x"), patch.object(upload_cdn, "get_r2_client", return_value=s3), \
                patch.object(upload_cdn, "save_image_timeline", side_effect=RuntimeError("skip")), \
                patch.object(upload_cdn, "thumbnails_enabled", return_value=False):
            upload_cdn.upload_quarter_hour_files([image], path_prefix=prefix)

    def test_a_repeat_with_identical_bytes_skips_only_the_latest_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "mo-rh.png"
            image.write_bytes(b"png-v1")
            with patch.dict("os.environ", {"CDN_UPLOAD_STATE": str(Path(tmp) / "state.json")}):
                first, second, third = FakeS3(), FakeS3(), FakeS3()
                self.run_upload(first, image)
                self.run_upload(second, image)
                image.write_bytes(b"png-v2")
                self.run_upload(third, image)
        self.assertEqual(sum(k.endswith("/latest/mo-rh.png") or k == "t/latest/mo-rh.png" for k in first.keys), 1)
        self.assertEqual(len(first.keys), 2)                                  # archive + latest the first time
        self.assertEqual(len(second.keys), 1)                                 # identical: archive only
        self.assertNotIn("t/latest/mo-rh.png", second.keys)
        self.assertIn("t/latest/mo-rh.png", third.keys)                       # changed again: latest is written


if __name__ == "__main__":
    unittest.main()
