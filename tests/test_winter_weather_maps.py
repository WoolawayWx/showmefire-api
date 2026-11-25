import json
from datetime import datetime, timezone

import numpy as np
from PIL import Image

from maps import winter_weather_maps as ww


def box(west, south, east, north):
    return {"type": "Polygon", "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]]}


def wssi_payload(items):
    props = {"valid_time": "12Z 11/25/26 - 12Z 11/26/26", "start_time": "12Z 11/25/26", "end_time": "12Z 11/26/26",
             "issue_time": "2026-11-25 1315Z"}
    return json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": geometry, "properties": {**props, "impact": impact}} for geometry, impact in items]}).encode()


def pixel_colors(path):
    image = Image.open(path).convert("RGB")
    return {"#%02x%02x%02x" % rgb for _count, rgb in image.getcolors(maxcolors=1_000_000)}


def test_products_have_unique_files_and_the_wssi_layers_published_by_wpc():
    assert len({p.key for p in ww.PRODUCTS}) == len(ww.PRODUCTS) == 10
    assert {p.kind for p in ww.PRODUCTS} == {"ndfd_snow", "ndfd_ice", "wssi"}
    assert ww.source_url(ww.BY_KEY["snow-72hr"]).endswith("/ds.snow.bin")
    assert ww.source_url(ww.BY_KEY["ice-72hr"]).endswith("/ds.iceaccum.bin")
    layers = {key: int(ww.BY_KEY[key].url.split("/MapServer/")[1].split("/")[0])
              for key in ("wssi-day1", "wssi-day2", "wssi-day3", "wssi-days1-3", "wssi-snow", "wssi-snow-load", "wssi-ice",
                          "wssi-blowing-snow")}
    assert layers == {"wssi-day1": 1, "wssi-day2": 2, "wssi-day3": 3, "wssi-days1-3": 4, "wssi-snow": 9, "wssi-snow-load": 14,
                      "wssi-ice": 19, "wssi-blowing-snow": 27}


def test_units_convert_to_inches():
    assert round(0.0254 * ww.METRES_TO_INCHES, 6) == 1.0          # NDFD snow is in metres
    assert round(25.4 * ww.MM_TO_INCHES, 6) == 1.0                # NDFD ice is in millimetres of liquid
    assert len(ww.SNOW_COLORS) == len(ww.SNOW_LEVELS) - 1 and len(ww.ICE_COLORS) == len(ww.ICE_LEVELS) - 1


def test_wssi_times_are_read_as_utc_and_shown_in_central():
    assert ww._wssi_time("12Z 11/25/26") == datetime(2026, 11, 25, 12, tzinfo=timezone.utc)
    assert ww._wssi_time("2026-11-25 1315Z") == datetime(2026, 11, 25, 13, 15, tzinfo=timezone.utc)
    assert ww._wssi_time("garbage") is None and ww._wssi_time(None) is None
    valid, issued = ww.wssi_valid({"start_time": "12Z 11/25/26", "end_time": "12Z 11/26/26", "issue_time": "2026-11-25 1315Z"})
    assert valid == "Valid: Nov 25, 6:00 AM – Nov 26, 6:00 AM CT"
    assert issued == "Issued: Nov 25, 7:15 AM CT"


def test_wssi_draws_the_classes_over_missouri_and_skips_unchanged_sources(tmp_path):
    payload = wssi_payload([(box(-95, 36, -90, 40), "WINTER WEATHER AREA"), (box(-93, 37, -91, 39), "MODERATE"),
                            (box(-110, 30, -105, 35), "MAJOR")])         # the last one is nowhere near Missouri
    product = ww.BY_KEY["wssi-day1"]
    changed, failed = ww.refresh([product], fetch_fn=lambda url: payload, images_dir=tmp_path, status_dir=tmp_path / "s")
    assert failed == [] and changed == [tmp_path / "mo-winter-wssi-day1.png"]
    status = json.loads((tmp_path / "s" / "winter_wssi-day1.json").read_text())
    assert status["levels_over_missouri"] == ["Winter Weather Area", "Moderate Impacts"]
    colors = pixel_colors(changed[0])
    assert "#f7962f" in colors and "#d2dfe7" in colors and "#e61f26" not in colors
    again, _ = ww.refresh([product], fetch_fn=lambda url: payload, images_dir=tmp_path, status_dir=tmp_path / "s")
    assert again == []
    forced, _ = ww.refresh([product], force=True, fetch_fn=lambda url: payload, images_dir=tmp_path, status_dir=tmp_path / "s")
    assert forced == changed


def test_wssi_with_no_polygons_says_so(tmp_path):
    payload = wssi_payload([])
    changed, failed = ww.refresh([ww.BY_KEY["wssi-ice"]], fetch_fn=lambda url: payload, images_dir=tmp_path, status_dir=tmp_path / "s")
    assert failed == [] and len(changed) == 1
    assert json.loads((tmp_path / "s" / "winter_wssi-ice.json").read_text())["levels_over_missouri"] == []


def test_one_source_down_does_not_stop_the_others(tmp_path):
    payload = wssi_payload([(box(-95, 36, -90, 40), "MINOR")])

    def fetch(url):
        if "/1/" in url:
            raise OSError("down")
        return payload

    changed, failed = ww.refresh([ww.BY_KEY["wssi-day1"], ww.BY_KEY["wssi-day2"]], fetch_fn=fetch, images_dir=tmp_path,
                                 status_dir=tmp_path / "s")
    assert failed == ["wssi-day1"] and [path.name for path in changed] == ["mo-winter-wssi-day2.png"]


def test_snow_is_colored_by_bin_not_by_value(tmp_path, monkeypatch):
    """contourf given only a colormap spreads colors over the value range, so 8 in came out in the 2-3 in color."""
    lon, lat = np.meshgrid(np.linspace(-96, -89, 140), np.linspace(35.5, 40.8, 110))
    blob = (abs(lon + 92.5) < 1) & (abs(lat - 38.5) < 0.7)
    grid = {"lon": lon, "lat": lat, "total": np.where(blob, 8.5, 1.5),
            "reference": datetime(2026, 11, 25, 14, tzinfo=timezone.utc), "end": datetime(2026, 11, 28, 12, tzinfo=timezone.utc)}
    monkeypatch.setattr(ww, "read_ndfd_total", lambda payload, scale: grid)
    changed, failed = ww.refresh([ww.BY_KEY["snow-72hr"]], fetch_fn=lambda url: b"x", images_dir=tmp_path, status_dir=tmp_path / "s")
    assert failed == []
    colors = pixel_colors(changed[0])
    assert ww.SNOW_COLORS[6].lower() in colors          # 8-12 in
    assert ww.SNOW_COLORS[1].lower() in colors          # 1-2 in
    assert ww.SNOW_COLORS[3].lower() in colors          # the colorbar swatch for 3-4 in: bins match the legend
    status = json.loads((tmp_path / "s" / "winter_snow-72hr.json").read_text())
    assert status["peak_inches_over_missouri"] == 8.5


def test_the_generator_the_catalog_and_the_page_list_exactly_the_same_maps():
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    catalog = json.loads((root / "core" / "map_catalog.json").read_text(encoding="utf-8"))
    cataloged = {m["file"] for g in catalog["groups"] for s in g["sections"] for m in s["maps"] if m["file"].startswith("mo-winter-")}
    generated = {f"mo-winter-{p.key}.png" for p in ww.PRODUCTS}
    assert cataloged == generated
    page = root.parent / "website" / "app" / "pages" / "mesonet" / "index.vue"
    if page.is_file():                                   # the api folder can be deployed without the website beside it
        assert set(re.findall(r"mo-winter-[a-z0-9-]+\.png", page.read_text(encoding="utf-8"))) == generated
