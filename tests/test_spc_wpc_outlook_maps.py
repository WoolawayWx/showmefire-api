import json
import re
from pathlib import Path
from datetime import timezone

import pytest

from maps import spc_wpc_outlook_maps as om


def box(west, south, east, north):
    return {"type": "Polygon", "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]]}


def collection(items, extra=None):
    return json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": geometry, "properties": {**(extra or {}), **props}} for geometry, props in items]}).encode()


TIMES = {"VALID_ISO": "2026-11-20T13:00:00+00:00", "EXPIRE_ISO": "2026-11-21T12:00:00+00:00", "ISSUE_ISO": "2026-11-20T12:56:00+00:00"}
MISSOURI = box(-95, 36, -90, 40)
ELSEWHERE = box(-110, 30, -105, 35)


def by_url(payloads):
    """A fetch function that answers by URL: the product's own URL, or its intensity overlay's ('<key>:overlay')."""
    lookup = {}
    for product in om.PRODUCTS:
        lookup[product.url] = product.key
        if product.overlay_url:
            lookup[product.overlay_url] = f"{product.key}:overlay"
    return lambda url: payloads[lookup[url]]


def test_every_day_of_every_outlook_has_a_unique_file_and_a_group():
    assert len({p.key for p in om.PRODUCTS}) == len(om.PRODUCTS) == 31
    by_group = {}
    for product in om.PRODUCTS:
        by_group.setdefault(product.day, []).append(product.key)
    assert {day: len(keys) for day, keys in by_group.items()} == {"Day 1": 7, "Day 2": 7, "Day 3": 5, "Days 4-8": 12}
    assert {p.hazard for p in om.PRODUCTS} == {"severe", "fire", "flood"}
    assert om.png_path(om.BY_KEY["day1-tornado"]).name == "mo-noaa-outlook-day1-tornado.png"
    # Days 4-8: severe and fire for each day, rainfall for Days 4-5 only (that is all WPC publishes)
    assert [k for k in by_group["Days 4-8"] if k.endswith("rainfall")] == ["day4-rainfall", "day5-rainfall"]
    assert sorted(k for k in by_group["Days 4-8"] if k.endswith("-severe")) == [f"day{d}-severe" for d in range(4, 9)]


def test_each_severe_hazard_carries_its_own_intensity_overlay_on_the_same_map():
    assert {key: om.BY_KEY[key].overlay_url for key in ("day1-tornado", "day1-wind", "day1-hail")} == {
        "day1-tornado": f"{om.SPC}/day1otlk_cigtorn.nolyr.geojson", "day1-wind": f"{om.SPC}/day1otlk_cigwind.nolyr.geojson",
        "day1-hail": f"{om.SPC}/day1otlk_cighail.nolyr.geojson"}
    assert om.BY_KEY["day2-tornado"].overlay_url == f"{om.SPC}/day2otlk_cigtorn.nolyr.geojson"
    assert om.BY_KEY["day2-tornado"].url == f"{om.SPC}/day2otlk_torn.nolyr.geojson"
    assert om.BY_KEY["day3-severe-prob"].overlay_url == f"{om.SPC}/day3otlk_cigprob.nolyr.geojson"
    no_overlay = [k for k in om.BY_KEY if k.endswith(("fire", "drythunder", "rainfall")) or re.fullmatch(r"day\d-severe", k)]
    assert no_overlay and all(om.BY_KEY[key].overlay_url is None for key in no_overlay)


def test_later_day_sources_are_the_ones_noaa_publishes():
    assert om.BY_KEY["day4-severe"].url == "https://www.spc.noaa.gov/products/exper/day4-8/day4prob.nolyr.geojson"
    assert om.BY_KEY["day3-fire"].url == "https://www.spc.noaa.gov/products/exper/fire_wx/day3fw_windrhcat.nolyr.geojson"
    assert om.BY_KEY["day8-fire"].url.endswith("/day8fw_windrhcat.nolyr.geojson")
    assert om.BY_KEY["day2-fire"].url.endswith("/MapServer/4/" + om.QUERY) and om.BY_KEY["day2-drythunder"].url.endswith("/MapServer/5/" + om.QUERY)
    assert om.BY_KEY["day5-rainfall"].url == "https://www.wpc.ncep.noaa.gov/exper/eromap/geojson/Day5_Latest.geojson"


def test_day_3_probabilities_are_labelled_from_the_percent_not_the_fraction_spc_writes():
    frame = om.read_polygons(collection([(MISSOURI, {"DN": 5, "LABEL": "0.05", "LABEL2": "5% Any Severe Risk", "fill": "#C5A392"})]))
    assert om.classify(om.BY_KEY["day3-severe-prob"], frame.iloc[0]) == (5, "5%", "#C5A392")


def test_later_day_fire_polygons_use_their_own_label_when_there_is_no_table():
    frame = om.read_polygons(collection([(MISSOURI, {"DN": 5, "LABEL": "Critical", "fill": "#FF0000"})]))
    assert om.classify(om.BY_KEY["day4-fire"], frame.iloc[0]) == (5, "Critical", "#FF0000")
    quiet = om.read_polygons(collection([(MISSOURI, {"DN": 0, "LABEL": "Predictability Too Low"})]))   # SPC's no-outlook placeholder
    assert om.classify(om.BY_KEY["day4-fire"], quiet.iloc[0])[0] == 0                                   # level 0 is never drawn


def test_the_generator_the_catalog_and_the_page_list_exactly_the_same_maps():
    root = Path(__file__).resolve().parents[1]
    catalog = json.loads((root / "core" / "map_catalog.json").read_text(encoding="utf-8"))
    cataloged = {m["file"] for g in catalog["groups"] for s in g["sections"] for m in s["maps"] if m["file"].startswith("mo-noaa-outlook-")}
    generated = {f"mo-noaa-outlook-{p.key}.png" for p in om.PRODUCTS}
    assert cataloged == generated
    page = root.parent / "website" / "app" / "pages" / "mesonet" / "index.vue"
    if page.is_file():                                   # the api folder can be deployed without the website beside it
        assert set(re.findall(r"mo-noaa-outlook-[a-z0-9-]+\.png", page.read_text(encoding="utf-8"))) == generated


def test_times_are_read_in_all_three_source_formats_and_shown_in_central_time():
    assert om.parse_time("202611191700") == om.parse_time("2026-11-19 17:00:00") == om.parse_time("2026-11-19T17:00:00+00:00")
    assert om.parse_time("2026-11-19T17:00:00Z").tzinfo is not None and om.parse_time("nonsense") is None and om.parse_time(None) is None
    valid, issued = om.valid_text(TIMES)                                           # 13z-12z next day = 7 AM-6 AM CST
    assert valid == "Valid: Nov 20, 7:00 AM – Nov 21, 6:00 AM CT" and issued == "Issued: Nov 20, 6:56 AM CT"
    same_day, _ = om.valid_text({"VALID_ISO": "2026-11-20T13:00:00+00:00", "EXPIRE_ISO": "2026-11-20T23:00:00+00:00"})
    assert same_day == "Valid: Nov 20, 7:00 AM – 5:00 PM CT"
    assert om.valid_text({}) == ("Valid: current outlook", None)
    fire, _ = om.valid_text({"valid": "202611201700", "expire": "202611211200"})    # NOAA's service uses lower-case names
    assert fire == "Valid: Nov 20, 11:00 AM – Nov 21, 6:00 AM CT"


def test_polygons_get_the_published_color_label_and_level():
    severe = om.BY_KEY["day1-severe"]
    frame = om.read_polygons(collection([(MISSOURI, {"DN": 4, "LABEL2": "Slight Risk", "fill": "#F6F67B"})]))
    level, label, color = om.classify(severe, frame.iloc[0])
    assert (level, label, color) == (4, "Slight Risk", "#F6F67B")
    bare = om.read_polygons(collection([(MISSOURI, {"DN": 5})]))                    # no label or fill: fall back to the table
    assert om.classify(severe, bare.iloc[0]) == (5, "Enhanced Risk (3 of 5)", "#E6B366")
    fire = om.read_polygons(collection([(MISSOURI, {"dn": 8})]))
    assert om.classify(om.BY_KEY["day1-fire"], fire.iloc[0]) == (8, "Critical", "#FF0000")
    assert om.classify(om.BY_KEY["day1-drythunder"], fire.iloc[0])[1:] == ("Scattered Dry Thunderstorms", "#FF0000")
    ero = om.read_polygons(collection([(MISSOURI, {"dn": 2, "OUTLOOK": "Slight (At Least 15%)"})]))
    assert om.classify(om.BY_KEY["day1-rainfall"], ero.iloc[0])[:2] == (2, "Slight Risk (at least 15%)")
    chance = om.read_polygons(collection([(MISSOURI, {"DN": 15, "LABEL": "15%", "fill": "#FFC800"})]))
    assert om.classify(om.BY_KEY["day1-wind"], chance.iloc[0]) == (15, "15%", "#FFC800")


def test_only_the_part_over_missouri_is_kept_and_a_quiet_day_is_empty():
    frame = om.read_polygons(collection([(MISSOURI, {"DN": 3}), (ELSEWHERE, {"DN": 3})]))
    assert len(frame) == 2 and len(om.clip_to_missouri(frame)) == 1
    quiet = collection([(None, {"DN": 0, **TIMES})])                                # SPC's placeholder feature has no geometry
    assert om.read_polygons(quiet).empty and om.clip_to_missouri(om.read_polygons(quiet)).empty
    assert om.first_properties(quiet)["ISSUE_ISO"] == TIMES["ISSUE_ISO"]            # but it still says when the outlook is valid


def test_a_map_is_drawn_once_per_outlook_and_again_only_when_it_changes(tmp_path):
    payloads = {"day1-severe": collection([(MISSOURI, {"DN": 3, "LABEL2": "Marginal Risk", "fill": "#66C57D", **TIMES})], TIMES)}
    products = (om.BY_KEY["day1-severe"],)
    kwargs = dict(fetch_fn=by_url(payloads), images_dir=tmp_path / "images", status_dir=tmp_path / "status")
    changed, failed = om.refresh(products, **kwargs)
    assert [p.name for p in changed] == ["mo-noaa-outlook-day1-severe.png"] and failed == [] and changed[0].stat().st_size > 10_000
    status = json.loads((tmp_path / "status" / "outlook_day1-severe.json").read_text())
    assert status["levels_over_missouri"] == ["Marginal Risk"] and status["valid"].startswith("Valid: Nov 20")
    assert om.refresh(products, **kwargs) == ([], [])                               # unchanged: not redrawn
    assert om.refresh(products, force=True, **kwargs)[0]                            # --force redraws
    payloads["day1-severe"] = collection([(MISSOURI, {"DN": 4, "LABEL2": "Slight Risk", "fill": "#F6F67B", **TIMES})], TIMES)
    assert om.refresh(products, **kwargs)[0]                                        # a new outlook is drawn


def test_probabilities_and_intensity_are_drawn_on_one_map_and_either_one_changing_redraws_it(tmp_path):
    payloads = {
        "day1-tornado": collection([(MISSOURI, {"DN": 5, "LABEL": "5%", "fill": "#8B4726", **TIMES})], TIMES),
        "day1-tornado:overlay": collection([(box(-93, 37, -91, 39), {"DN": 1, "LABEL": "CIG1", **TIMES})], TIMES),
    }
    products = (om.BY_KEY["day1-tornado"],)
    kwargs = dict(fetch_fn=by_url(payloads), images_dir=tmp_path / "images", status_dir=tmp_path / "status")
    changed, failed = om.refresh(products, **kwargs)
    assert [p.name for p in changed] == ["mo-noaa-outlook-day1-tornado.png"] and failed == []        # one file, not a second "-cig" map
    status = json.loads((tmp_path / "status" / "outlook_day1-tornado.json").read_text())
    assert status["levels_over_missouri"] == ["5%"] and status["intensity_levels_over_missouri"] == ["Level 1"]
    assert om.refresh(products, **kwargs) == ([], [])                                        # nothing changed
    payloads["day1-tornado:overlay"] = collection([(box(-93, 37, -91, 39), {"DN": 2, "LABEL": "CIG2", **TIMES})], TIMES)
    assert om.refresh(products, **kwargs)[0]                                                 # only the intensity areas changed: redraw
    # a source with no hatched areas (a quiet intensity layer) still draws the probabilities
    payloads["day1-tornado:overlay"] = collection([(None, {"DN": 0, **TIMES})], TIMES)
    assert om.refresh(products, **kwargs)[0]
    assert json.loads((tmp_path / "status" / "outlook_day1-tornado.json").read_text())["intensity_levels_over_missouri"] == []


def test_one_source_being_down_does_not_stop_the_others(tmp_path):
    quiet = collection([(None, {"DN": 0, **TIMES})], TIMES)

    def fetch(url):
        if url == om.BY_KEY["day1-tornado"].url:
            raise ConnectionError("source is down")
        return quiet

    products = (om.BY_KEY["day1-severe"], om.BY_KEY["day1-tornado"], om.BY_KEY["day1-hail"])
    changed, failed = om.refresh(products, fetch_fn=fetch, images_dir=tmp_path / "images", status_dir=tmp_path / "status")
    assert sorted(p.name for p in changed) == ["mo-noaa-outlook-day1-hail.png", "mo-noaa-outlook-day1-severe.png"] and failed == ["day1-tornado"]
    quiet = json.loads((tmp_path / "status" / "outlook_day1-severe.json").read_text())
    assert quiet["levels_over_missouri"] == []                                      # a quiet day still produces a (blank) map
