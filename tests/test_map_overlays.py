import importlib.util
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, MultiLineString

# scripts/ is not a package: load the build script by path.
_spec = importlib.util.spec_from_file_location("build_map_overlays", Path(__file__).resolve().parent.parent / "scripts" / "build_map_overlays.py")
overlays = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(overlays)


def places(rows):
    import pandas as pd
    return pd.DataFrame(rows, columns=["PLACEFP", "NAME", "INTPTLAT", "INTPTLON"])


def population(*pairs):
    return [{"PLACE": code, "POPESTIMATE2023": str(pop)} for code, pop in pairs]


def test_cities_are_joined_filtered_sorted_and_rounded():
    table = places([("38000", "Kansas City", "39.1234567", "-94.5678912"), ("00124", "Tiny", "38.0", "-92.0"),
                    ("65000", "St. Louis", "38.6", "-90.2"), ("99999", "No Population", "37.0", "-91.0")])
    cities = overlays.build_cities(table, population(("38000", 510704), ("65000", 281754), ("124", 300)))
    assert [(c["n"], c["p"]) for c in cities] == [("Kansas City", 510704), ("St. Louis", 281754)]    # the name is kept whole
    assert cities[0]["x"] == -94.5679 and cities[0]["y"] == 39.1235                                    # four decimals


def test_place_codes_match_even_when_one_side_lost_its_leading_zeros():
    table = places([("00124", "Abbeville", "38.0", "-92.0")])
    assert overlays.build_cities(table, population(("124", 2000)))[0]["n"] == "Abbeville"
    assert overlays.build_cities(table, population(("00124", 2000)))[0]["n"] == "Abbeville"


def test_bad_or_missing_population_rows_are_skipped_not_fatal():
    table = places([("00001", "A", "38", "-92")])
    rows = [{"PLACE": "00001", "POPESTIMATE2023": "n/a"}, {"POPESTIMATE2023": "5000"}, {"PLACE": "00001"}]
    assert overlays.build_cities(table, rows) == []


def test_the_minimum_population_is_inclusive_and_adjustable():
    table = places([("00001", "Edge", "38", "-92"), ("00002", "Below", "38", "-91")])
    rows = population(("00001", 1000), ("00002", 999))
    assert [c["n"] for c in overlays.build_cities(table, rows)] == ["Edge"]
    assert len(overlays.build_cities(table, rows, minimum=500)) == 2


def test_cities_with_equal_population_sort_by_name():
    table = places([("2", "Zed", "38", "-92"), ("1", "Abe", "38", "-91")])
    assert [c["n"] for c in overlays.build_cities(table, population(("1", 5000), ("2", 5000)))] == ["Abe", "Zed"]


def test_encode_stores_the_first_point_whole_then_steps_and_decodes_back():
    ints, points = overlays.encode([(-94.5, 39.25), (-94.4, 39.25), (-94.4, 39.3)])
    assert ints == [-945000, 392500, 1000, 0, 0, 500]            # 1/10,000 degree: the first point whole, then the steps
    assert points == [(-945000, 392500), (-944000, 392500), (-944000, 392500 + 500)]
    assert overlays.decode(ints) == [(-94.5, 39.25), (-94.4, 39.25), (-94.4, 39.3)]


def test_encode_drops_repeats_and_reports_a_degenerate_line_as_empty():
    ints, points = overlays.encode([(-94.00001, 39.0), (-94.00002, 39.00001), (-93.0, 39.0)])   # the first two round to one point
    assert len(points) == 2 and len(ints) == 4
    assert overlays.encode([(-94.0, 39.0), (-94.00001, 39.00001)]) == ([], [])                # one point once rounded: no line


def roads(rows):
    return gpd.GeoDataFrame([{"RTTYP": t, "FULLNAME": n, "geometry": g} for t, n, g in rows], geometry="geometry", crs="EPSG:4269")


def lines_of(result, cls):
    """[(decoded points, level string), ...] for a class."""
    return [(overlays.decode(ints), levels) for ints, levels in result[cls]]


def test_only_interstates_us_and_state_routes_are_kept_and_classified():
    frame = roads([("I", "I- 70", LineString([(-94, 39), (-92, 39)])), ("U", "US Hwy 63", LineString([(-92, 38), (-92, 40)])),
                   ("S", "State Rte 13", LineString([(-93, 37), (-93, 39)])), ("M", "Main St", LineString([(-91, 38), (-91, 39)])),
                   ("C", "Co Rd 5", LineString([(-90, 38), (-90, 39)]))])
    result = overlays.build_roads(frame)
    assert {k: len(v) for k, v in result.items()} == {"interstate": 1, "us": 1, "state": 1}
    points, levels = lines_of(result, "interstate")[0]
    assert points == [(-94.0, 39.0), (-92.0, 39.0)] and levels == "00"


def test_pieces_of_one_route_are_merged_into_one_line():
    frame = roads([("I", "I- 44", LineString([(-94, 37), (-93, 37)])), ("I", "I- 44", LineString([(-93, 37), (-92, 37)])),
                   ("I", "I- 44", LineString([(-92, 37), (-91, 37)]))])
    lines = lines_of(overlays.build_roads(frame), "interstate")
    assert len(lines) == 1 and lines[0][0][0] == (-94.0, 37.0) and lines[0][0][-1] == (-91.0, 37.0)


def test_a_route_that_is_not_connected_stays_separate_lines():
    frame = roads([("U", "US Hwy 60", LineString([(-94, 37), (-93, 37)])), ("U", "US Hwy 60", LineString([(-90, 37), (-89, 37)]))])
    assert len(overlays.build_roads(frame)["us"]) == 2


def wiggly(n=400, amplitude=0.002):
    """A long, gently curving road with small bumps, like a real highway."""
    import math
    return LineString([(-94.0 + i * 0.01, 38.0 + 0.3 * math.sin(i / 40.0) + (amplitude if i % 7 == 0 else 0.0)) for i in range(n)])


def test_every_point_has_a_level_the_ends_are_level_zero_and_the_levels_are_nested():
    result = overlays.build_roads(roads([("S", "State Rte 7", wiggly())]))
    points, levels = lines_of(result, "state")[0]
    assert len(levels) == len(points) and set(levels) <= {"0", "1", "2"}
    assert levels[0] == "0" and levels[-1] == "0"                           # a line's ends are never simplified away
    counts = [sum(1 for c in levels if int(c) <= level) for level in (0, 1, 2)]
    assert counts[0] < counts[1] < counts[2] == len(points)                 # each finer level adds points to the coarser one


def test_a_coarser_level_is_a_subset_of_the_points_of_a_finer_one():
    # Coarse points are picked from the fine ones, so drawing a coarse level only ever skips points: it never moves one.
    points, levels = lines_of(overlays.build_roads(roads([("S", "State Rte 7", wiggly())])), "state")[0]
    for level in (0, 1):
        coarse = [p for p, c in zip(points, levels) if int(c) <= level]
        assert set(coarse) <= set(points) and len(coarse) < len(points)


def hausdorff_point_to_line(original, kept):
    """The farthest any point of `original` is from the line drawn through `kept`."""
    from shapely.geometry import Point
    line = LineString(kept)
    return max(line.distance(Point(x, y)) for x, y in original.coords)


def test_each_level_stays_within_its_tolerance_of_the_original_road():
    original = wiggly()
    points, levels = lines_of(overlays.build_roads(roads([("S", "State Rte 7", original)])), "state")[0]
    rounding = 1.5 / overlays.COORD_SCALE                                    # a point can move by half a step each way
    for level, tolerance in enumerate(overlays.ROAD_TOLERANCES):
        kept = [p for p, c in zip(points, levels) if int(c) <= level]
        # Level n is a simplification of a simplification, so the error can add up: bounded by the sum of the tolerances to it.
        allowed = sum(overlays.ROAD_TOLERANCES[level:]) + rounding
        assert hausdorff_point_to_line(original, kept) <= allowed, (level, tolerance)
    assert hausdorff_point_to_line(original, points) <= overlays.ROAD_TOLERANCES[-1] + rounding   # the stored line itself


def test_a_road_is_never_dropped_just_because_it_was_simplified():
    # Regression: re-simplifying the rounded line used to leave fewer levels than points and the road vanished without a word.
    roads_in = [("I", f"I- {n}", wiggly(n=120 + n, amplitude=0.0004 * (n % 5 + 1))) for n in range(1, 30)]
    result = overlays.build_roads(roads(roads_in))
    assert len(result["interstate"]) == 29
    for ints, levels in result["interstate"]:
        assert len(levels) == len(ints) // 2


def test_multilinestrings_and_empty_geometry_are_handled():
    multi = MultiLineString([[(-94, 38), (-93, 38)], [(-90, 38), (-89, 38)]])
    frame = roads([("S", "State Rte 1", multi), ("S", "State Rte 2", LineString())])
    assert len(overlays.build_roads(frame)["state"]) == 2


def test_the_published_files_are_valid_and_inside_missouri():
    import json
    folder = Path(__file__).resolve().parents[2] / "website" / "public" / "json"
    cities_file, roads_file = folder / "cities_mo.json", folder / "roads_mo.json"
    if not (cities_file.is_file() and roads_file.is_file()):
        pytest.skip("the generated overlay files are not in the website folder")
    cities = json.loads(cities_file.read_text(encoding="utf-8"))["cities"]
    assert len(cities) > 200 and cities[0]["n"] == "Kansas City" and cities == sorted(cities, key=lambda c: (-c["p"], c["n"]))
    assert all(-96.0 < c["x"] < -89.0 and 35.8 < c["y"] < 40.7 and c["p"] >= 1000 for c in cities)
    document = json.loads(roads_file.read_text())
    assert document["v"] == 3 and document["scale"] == overlays.COORD_SCALE and document["levels"] == 3
    road_data = document["roads"]
    assert set(road_data) == {"interstate", "us", "state"} and all(road_data[k] for k in road_data)
    total_points = 0
    for lines in road_data.values():
        for ints, levels in lines:
            points = overlays.decode(ints)
            assert len(points) == len(levels) >= 2 and set(levels) <= {"0", "1", "2"} and levels[0] == levels[-1] == "0"
            assert all(-96.0 < x < -89.0 and 35.8 < y < 40.8 for x, y in points)
            total_points += len(points)
    assert total_points > 20_000                                           # the roads are really there (the lost-roads bug had ~15k)
    interstate = sum(len(overlays.decode(ints)) for ints, _ in road_data["interstate"])
    assert interstate > 1500
