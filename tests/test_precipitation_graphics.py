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


def test_generate_graphics_writes_every_product_at_established_dimensions(tmp_path, monkeypatch):
    session = _Session()
    layer_ids = {name: idx for idx, (_, name, _, _) in enumerate(graphics.PRODUCTS, 20)}
    monkeypatch.setattr(graphics, "resolve_layer_ids", lambda *args, **kwargs: layer_ids)
    monkeypatch.setattr(graphics, "fetch_legend_entries", lambda *args, **kwargs: [("0 to 1", (0.2, 0.5, 0.8, 1.0))])
    timestamp = datetime(2026, 9, 23, 6, 0, tzinfo=ZoneInfo("America/Chicago"))
    outputs = graphics.generate_graphics(output_dir=tmp_path, session=session, now=timestamp)
    assert len(outputs) == len(graphics.PRODUCTS)
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


def test_publish_uploads_only_the_stable_graphic_keys(tmp_path, monkeypatch):
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
    assert len(uploads) == len(graphics.PRODUCTS)
    assert published == [f"latest/{output.name}" for output in outputs]
    assert all(content_type == "image/png" for _, _, _, content_type in uploads)


def test_generate_graphics_can_render_only_the_hourly_products(tmp_path, monkeypatch):
    layer_ids = {name: idx for idx, (_, name, _, _) in enumerate(graphics.PRODUCTS, 20)}
    monkeypatch.setattr(graphics, "resolve_layer_ids", lambda *args, **kwargs: layer_ids)
    monkeypatch.setattr(graphics, "fetch_legend_entries", lambda *args, **kwargs: [("0 to 1", (0.2, 0.5, 0.8, 1.0))])
    timestamp = datetime(2026, 9, 23, 6, 0, tzinfo=ZoneInfo("America/Chicago"))
    outputs = graphics.generate_graphics(output_dir=tmp_path, session=_Session(), now=timestamp, stems=graphics.HOURLY_STEMS)
    assert {path.name for path in outputs} == {f"mo-{stem}.png" for stem in graphics.HOURLY_STEMS}
    assert len(outputs) == 5


def test_every_hourly_stem_is_a_defined_product():
    assert graphics.HOURLY_STEMS <= {stem for stem, *_ in graphics.PRODUCTS}


def test_departure_legend_labels_keep_negative_ranges_readable():
    assert graphics._shorten_legend_label("-1 to -0.5") == "-1 to -0.5"
    assert graphics._shorten_legend_label("-0.5 to 0.5") == "-0.5 to 0.5"
    assert graphics._shorten_legend_label("2 to 3") == "2–3"
    assert graphics._shorten_legend_label("Less than -8") == "< -8"
    assert graphics._shorten_legend_label("Greater than or equal to 8") == "≥ 8"


def test_scheduled_jobs_render_in_the_process_pool_then_upload(monkeypatch, tmp_path):
    import asyncio

    from core import executors, scheduler

    outputs = [tmp_path / "mo-precip-1hr.png"]
    pool_calls, uploads = [], []

    async def fake_pool(func, *args, **kwargs):
        # The render must hold the grid-job lock so it never shares the pool with an RTMA/spread-rate job.
        assert executors.get_rtma_job_lock().locked()
        pool_calls.append((func, args))
        return outputs

    monkeypatch.setattr(scheduler, "run_in_process_pool_async", fake_pool)
    monkeypatch.setattr(scheduler, "upload_precipitation_graphics", lambda paths: uploads.append(paths) or ["latest/x"])
    monkeypatch.setattr(executors, "_rtma_job_lock", None)

    asyncio.run(scheduler.publish_hourly_precipitation_graphics_job())
    asyncio.run(scheduler.publish_precipitation_graphics_job())
    assert pool_calls == [(scheduler.render_precipitation_graphics, (graphics.HOURLY_STEMS,)),
                          (scheduler.render_precipitation_graphics, ())]
    assert uploads == [outputs, outputs]


def test_scheduled_render_failure_uploads_nothing(monkeypatch):
    import asyncio

    from core import executors, scheduler

    async def failing_pool(func, *args, **kwargs):
        raise RuntimeError("NOAA unavailable")

    uploads = []
    monkeypatch.setattr(scheduler, "run_in_process_pool_async", failing_pool)
    monkeypatch.setattr(scheduler, "upload_precipitation_graphics", lambda paths: uploads.append(paths))
    monkeypatch.setattr(executors, "_rtma_job_lock", None)
    asyncio.run(scheduler.publish_hourly_precipitation_graphics_job())  # logs, never raises
    assert uploads == []
