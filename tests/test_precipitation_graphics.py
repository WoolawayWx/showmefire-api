from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from PIL import Image
from io import BytesIO

from maps import precipitation_graphics as graphics


class _Response:
    def __init__(self, payload=None, content=b"", status_error=None):
        self._payload = payload or {}
        self.content = content
        self.status_error = status_error

    def raise_for_status(self):
        if self.status_error:
            raise self.status_error

    def json(self):
        return self._payload


class _Session:
    def __init__(self, *, json_payload=None, content=None):
        self.json_payload = json_payload or {}
        if content is None:
            image = BytesIO()
            Image.new("RGBA", (16, 16), (40, 120, 180, 255)).save(image, format="PNG")
            content = image.getvalue()
        self.content = content
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if url.endswith("/export"):
            return _Response(content=self.content)
        return _Response(payload=self.json_payload)


def test_resolve_layer_ids_matches_all_six_named_products():
    metadata = {"layers": []}
    expected = {}
    for idx, (_, name, _, _) in enumerate(graphics.PRODUCTS, 10):
        metadata["layers"].extend([
            {"id": idx, "name": name, "parentLayerId": -1},
            {"id": idx + 100, "name": "Image", "parentLayerId": idx},
        ])
        expected[name] = idx + 100
    ids = graphics.resolve_layer_ids("https://example.test/MapServer", session=_Session(json_payload=metadata))
    assert ids == expected


def test_resolve_layer_ids_fails_when_a_product_layer_is_missing():
    metadata = {"layers": [{"id": 25, "name": graphics.PRODUCTS[0][1]}]}
    with pytest.raises(RuntimeError, match="NOAA QPE layers not found"):
        graphics.resolve_layer_ids("https://example.test/MapServer", session=_Session(json_payload=metadata))


def test_fetch_layer_png_rejects_error_payload():
    session = _Session(content=b"not an image")
    with pytest.raises(RuntimeError, match="did not return a PNG"):
        graphics.fetch_layer_png(25, service="https://example.test/MapServer", session=session)


def test_generate_graphics_writes_six_maps_at_established_dimensions(tmp_path, monkeypatch):
    session = _Session()
    layer_ids = {name: idx for idx, (_, name, _, _) in enumerate(graphics.PRODUCTS, 20)}
    monkeypatch.setattr(graphics, "resolve_layer_ids", lambda *args, **kwargs: layer_ids)
    monkeypatch.setattr(graphics, "fetch_legend_entries", lambda *args, **kwargs: [("0 to 1", (0.2, 0.5, 0.8, 1.0))])
    timestamp = datetime(2026, 9, 23, 6, 0, tzinfo=ZoneInfo("America/Chicago"))
    outputs = graphics.generate_graphics(output_dir=tmp_path, session=session, now=timestamp)
    assert len(outputs) == 6
    assert all(path.exists() for path in outputs)
    assert {path.name for path in outputs} == {f"mo-{stem}.png" for stem, *_ in graphics.PRODUCTS}
    from PIL import Image
    assert all(Image.open(path).size == (2048, 1152) for path in outputs)


def test_failed_render_keeps_cdn_untouched(monkeypatch):
    from scripts import publish_precipitation_graphics as publisher

    upload_called = False

    def fail_generation():
        raise RuntimeError("NOAA unavailable")

    class Client:
        def upload_file(self, *args, **kwargs):
            nonlocal upload_called
            upload_called = True

    monkeypatch.setattr(publisher, "generate_graphics", fail_generation)
    monkeypatch.setattr(publisher, "_get_r2_client_and_bucket", lambda: (Client(), "test-bucket"))
    with pytest.raises(RuntimeError, match="NOAA unavailable"):
        publisher.publish()
    assert upload_called is False


def test_publish_uploads_only_the_six_stable_graphic_keys(tmp_path, monkeypatch):
    from scripts import publish_precipitation_graphics as publisher

    outputs = [tmp_path / f"mo-{stem}.png" for stem, *_ in graphics.PRODUCTS]
    for output in outputs:
        output.write_bytes(b"PNG")
    uploads = []

    class Client:
        def upload_file(self, filename, bucket, key, ExtraArgs):
            uploads.append((Path(filename).name, bucket, key, ExtraArgs["ContentType"]))

    monkeypatch.setattr(publisher, "generate_graphics", lambda: outputs)
    monkeypatch.setattr(publisher, "_get_r2_client_and_bucket", lambda: (Client(), "test-bucket"))
    published = publisher.publish()
    assert len(uploads) == 6
    assert published == [f"latest/{output.name}" for output in outputs]
    assert all(content_type == "image/png" for _, _, _, content_type in uploads)
