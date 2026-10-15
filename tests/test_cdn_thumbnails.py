import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from services import cdn_thumbnails
from services.cdn_thumbnails import make_thumbnail, thumbnail_name, thumbnails_enabled

API_DIR = Path(__file__).resolve().parent.parent


def _map_like_png(path: Path, size=(2048, 1152), mode="RGB"):
    """A PNG with smooth colour fields and fine detail, like the rendered maps."""
    width, height = size
    x = np.linspace(0, 1, width)[None, :]
    y = np.linspace(0, 1, height)[:, None]
    rng = np.random.default_rng(1)
    channels = [(255 * (0.5 + 0.5 * np.sin(6 * x + k) * np.cos(4 * y))) + rng.integers(0, 8, (height, width)) for k in range(3)]
    data = np.clip(np.stack(channels, axis=-1), 0, 255).astype("uint8")
    image = Image.fromarray(data, "RGB")
    if mode == "RGBA":
        image = image.convert("RGBA")
        alpha = np.full((height, width), 255, "uint8")
        alpha[:, : width // 2] = 90
        image.putalpha(Image.fromarray(alpha))
    image.save(path, "PNG")
    return path


class ThumbnailNameTests(unittest.TestCase):
    def test_png_becomes_sm_webp(self):
        self.assertEqual(thumbnail_name("mo-rh.png"), "mo-rh-sm.webp")
        self.assertEqual(thumbnail_name("MO-RH.PNG"), "MO-RH-sm.webp")
        self.assertEqual(thumbnail_name("some/dir/mo-et-24hr.png"), "mo-et-24hr-sm.webp")

    def test_non_png_has_no_thumbnail(self):
        for name in ("a.jpg", "a.webp", "a-sm.webp", "a.geojson", "a", "a.png.bak"):
            self.assertIsNone(thumbnail_name(name), name)


class EnabledFlagTests(unittest.TestCase):
    def test_off_by_default_and_on_only_for_clear_yes_values(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CDN_THUMBNAILS", None)
            self.assertFalse(thumbnails_enabled())
        for value, expected in (("1", True), ("true", True), ("YES", True), ("on", True),
                                ("", False), ("0", False), ("false", False), ("maybe", False)):
            with mock.patch.dict(os.environ, {"CDN_THUMBNAILS": value}):
                self.assertEqual(thumbnails_enabled(), expected, value)


class MakeThumbnailTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        for name in ("CDN_THUMBNAIL_WIDTH", "CDN_THUMBNAIL_QUALITY"):
            os.environ.pop(name, None)

    def tearDown(self):
        self._tmp.cleanup()

    def test_resizes_to_800_wide_keeps_aspect_and_is_much_smaller(self):
        source = _map_like_png(self.tmp / "mo-rh.png")
        out = make_thumbnail(source, self.tmp)
        self.assertEqual(out.name, "mo-rh-sm.webp")
        with Image.open(out) as thumb:
            self.assertEqual(thumb.format, "WEBP")
            self.assertEqual(thumb.size, (800, 450))  # 2048x1152 is 16:9
        self.assertLess(out.stat().st_size * 5, source.stat().st_size)
        self.assertTrue(source.exists(), "the original PNG is never touched")

    def test_small_images_are_not_enlarged(self):
        source = _map_like_png(self.tmp / "small.png", size=(500, 300))
        with Image.open(make_thumbnail(source, self.tmp)) as thumb:
            self.assertEqual(thumb.size, (500, 300))

    def test_transparency_is_kept(self):
        source = _map_like_png(self.tmp / "alpha.png", size=(1000, 600), mode="RGBA")
        with Image.open(make_thumbnail(source, self.tmp)) as thumb:
            self.assertIn(thumb.mode, ("RGBA", "LA"))
            alpha = np.asarray(thumb.convert("RGBA"))[..., 3]
            self.assertLess(alpha[:, :100].mean(), 150)   # the translucent half stayed translucent
            self.assertGreater(alpha[:, -100:].mean(), 240)

    def test_width_setting_is_honoured_and_clamped(self):
        source = _map_like_png(self.tmp / "m.png", size=(1600, 900))
        with mock.patch.dict(os.environ, {"CDN_THUMBNAIL_WIDTH": "400"}):
            with Image.open(make_thumbnail(source, self.tmp)) as thumb:
                self.assertEqual(thumb.width, 400)
        with mock.patch.dict(os.environ, {"CDN_THUMBNAIL_WIDTH": "10"}):  # absurdly small: clamped up to 200
            with Image.open(make_thumbnail(source, self.tmp)) as thumb:
                self.assertEqual(thumb.width, 200)
        with mock.patch.dict(os.environ, {"CDN_THUMBNAIL_WIDTH": "not-a-number"}):
            with Image.open(make_thumbnail(source, self.tmp)) as thumb:
                self.assertEqual(thumb.width, 800)

    def test_bad_inputs_return_none_and_never_raise(self):
        self.assertIsNone(make_thumbnail(self.tmp / "missing.png", self.tmp))
        (self.tmp / "notes.txt").write_text("hi")
        self.assertIsNone(make_thumbnail(self.tmp / "notes.txt", self.tmp))
        broken = self.tmp / "broken.png"
        broken.write_bytes(b"\x89PNG\r\n\x1a\n" + b"not really a png")
        self.assertIsNone(make_thumbnail(broken, self.tmp))
        truncated = _map_like_png(self.tmp / "t.png", size=(600, 300))
        truncated.write_bytes(truncated.read_bytes()[:200])
        self.assertIsNone(make_thumbnail(truncated, self.tmp))


def _load_upload_module():
    """scripts/upload_cdn.py imports boto3 at the top; stand in for it if it is not installed."""
    for name in ("boto3", "botocore", "botocore.config"):
        if name not in sys.modules:
            try:
                __import__(name)
            except ImportError:
                module = types.ModuleType(name)
                module.Config = lambda **kwargs: None
                module.client = lambda *a, **k: None
                sys.modules[name] = module
    spec = importlib.util.spec_from_file_location("upload_cdn_under_test", API_DIR / "scripts" / "upload_cdn.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeS3:
    def __init__(self):
        self.uploads = []  # (key, content_type, cache_control)

    def upload_file(self, filename, bucket, key, ExtraArgs=None):
        self.uploads.append((key, (ExtraArgs or {}).get("ContentType"), (ExtraArgs or {}).get("CacheControl")))


class UploaderIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_upload_module()

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.png = _map_like_png(Path(self._tmp.name) / "mo-rh.png", size=(1600, 900))
        self.s3 = FakeS3()
        self.patches = [
            mock.patch.object(self.module, "R2_ACCESS_KEY_ID", "k"),
            mock.patch.object(self.module, "R2_SECRET_ACCESS_KEY", "s"),
            mock.patch.object(self.module, "R2_ACCOUNT_ID", "a"),
            mock.patch.object(self.module, "get_r2_client", lambda: self.s3),
            mock.patch.object(self.module, "save_image_timeline", side_effect=RuntimeError("skipped in test")),
            mock.patch.object(self.module, "get_forecast_path", return_value="2026/10/15/forecast"),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self._tmp.cleanup()

    def keys(self):
        return [key for key, _, _ in self.s3.uploads]

    def test_off_by_default_uploads_exactly_what_it_did_before(self):
        with mock.patch.dict(os.environ, {"CDN_THUMBNAILS": ""}):
            self.module.upload_quarter_hour_files([self.png], path_prefix="test")
        self.assertEqual([k for k in self.keys() if k.endswith(".png")], [k for k in self.keys()])
        self.assertEqual(len(self.keys()), 2)  # dated archive copy + latest, nothing else
        self.assertFalse(any(k.endswith(".webp") for k in self.keys()))

    def test_on_adds_a_latest_only_webp_with_the_right_headers(self):
        with mock.patch.dict(os.environ, {"CDN_THUMBNAILS": "true"}):
            self.module.upload_quarter_hour_files([self.png], path_prefix="test")
        thumbs = [u for u in self.s3.uploads if u[0].endswith(".webp")]
        self.assertEqual(thumbs, [("test/latest/mo-rh-sm.webp", "image/webp", "max-age=300")])
        self.assertEqual(len([k for k in self.keys() if k.endswith(".png")]), 2, "the PNG uploads are unchanged")
        self.assertFalse(any("/2026/" in k or k.count("/") > 2 for k in self.keys() if k.endswith(".webp")),
                         "thumbnails are not archived")

    def test_no_prefix_goes_to_plain_latest(self):
        with mock.patch.dict(os.environ, {"CDN_THUMBNAILS": "1"}):
            self.module.upload_quarter_hour_files([self.png])
        self.assertIn("latest/mo-rh-sm.webp", self.keys())

    def test_forecast_uploads_get_thumbnails_too(self):
        with mock.patch.dict(os.environ, {"CDN_THUMBNAILS": "1"}):
            self.module.upload_forecast_files([self.png])
        self.assertIn("latest/mo-rh-sm.webp", self.keys())

    def test_non_png_files_are_never_thumbnailed(self):
        other = Path(self._tmp.name) / "layers.geojson"
        other.write_text("{}")
        with mock.patch.dict(os.environ, {"CDN_THUMBNAILS": "1"}):
            self.module.upload_quarter_hour_files([other])
        self.assertFalse(any(k.endswith(".webp") for k in self.keys()))

    def test_a_thumbnail_failure_never_breaks_the_real_upload(self):
        for broken in (mock.patch.object(self.module, "make_thumbnail", side_effect=RuntimeError("boom")),):
            with broken, mock.patch.dict(os.environ, {"CDN_THUMBNAILS": "1"}):
                self.module.upload_quarter_hour_files([self.png], path_prefix="test")
        self.assertEqual(len([k for k in self.keys() if k.endswith(".png")]), 2)
        self.s3.uploads.clear()

        class FailingThumbUpload(FakeS3):
            def upload_file(self, filename, bucket, key, ExtraArgs=None):
                if key.endswith(".webp"):
                    raise OSError("R2 said no")
                super().upload_file(filename, bucket, key, ExtraArgs)

        self.s3 = FailingThumbUpload()
        with mock.patch.dict(os.environ, {"CDN_THUMBNAILS": "1"}):
            self.module.upload_quarter_hour_files([self.png], path_prefix="test")
        self.assertEqual(len(self.s3.uploads), 2, "both PNG copies still went up")


if __name__ == "__main__":
    unittest.main()
