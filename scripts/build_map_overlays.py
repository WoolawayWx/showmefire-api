#!/usr/bin/env python3
"""Build the small city and road files the data viewer can draw on its map (website/public/json).

    python -I scripts/build_map_overlays.py --roads <tl_2023_29_prisecroads.zip> --places <tl_2023_29_place.zip> \\
        --population <mo_places.csv> --out ../website/public/json

Inputs (all public, US Census Bureau):
  * TIGER/Line primary and secondary roads for Missouri (tl_YYYY_29_prisecroads.zip)
  * TIGER/Line places for Missouri (tl_YYYY_29_place.zip), for each place's internal point
  * The Vintage population estimates for incorporated places, Missouri rows only
    (https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/cities/totals/sub-est2023.csv, rows with
    SUMLEV 162 and STATE 29), which gives the PLACE code and POPESTIMATE2023.

Outputs: cities_mo.json and roads_mo.json, both compact (no GeoJSON wrapper) because they are loaded in the browser only when
a viewer turns the layer on. Roads are the interstates, US highways and state routes only, merged by route.

Each road is stored once, at its finest detail, with a level (0, 1 or 2) on every point: the zoom at which the point is first
needed. Drawing at a coarse level just skips the points with a higher level, so the map draws a few thousand points when the
whole state is in view and the full detail only when zoomed in (drawing is what costs, not downloading). The levels are nested
(each is a simplification of the next finer one), so every coarse point is also a fine point. Coordinates are whole numbers of
1/10,000 degree (about 11 m): the first point of a line is absolute and every other point is the step from the one before,
which is small and compresses well. The loader in website/app/utils/mapOverlays.ts undoes both.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Simplify tolerance in degrees for each level, finest last. Each is about half a screen pixel at the zoom where the level takes
# over (map zoom ~1, ~3 and ~6+), so the simplification cannot be seen. Level 2 is the file's actual geometry (about 65 m).
ROAD_TOLERANCES = (0.004, 0.0012, 0.0006)
COORD_SCALE = 10_000
MIN_POPULATION = 1000
# TIGER route type -> the class the viewer styles by. Common-name and county roads are left out: at state scale they are noise.
ROAD_CLASSES = {"I": "interstate", "U": "us", "S": "state"}


def build_cities(places, population_rows: list[dict[str, str]], minimum: int = MIN_POPULATION) -> list[dict[str, Any]]:
    """Places with a position (the Census internal point) and a 2023 population, largest first.

    `places` is a GeoDataFrame-like with PLACEFP, NAME, INTPTLAT and INTPTLON; `population_rows` are the CSV rows."""
    population = {}
    for row in population_rows:
        try:
            population[row["PLACE"].zfill(5)] = int(row["POPESTIMATE2023"])
        except (KeyError, ValueError):
            continue
    cities = []
    for _, place in places.iterrows():
        pop = population.get(str(place["PLACEFP"]).zfill(5))
        if pop is None or pop < minimum:
            continue
        # TIGER's NAME is the plain name ("Kansas City"); only the estimates file adds a legal suffix ("city"), which is not used.
        cities.append({"n": str(place["NAME"]).strip(), "p": pop,
                       "x": round(float(place["INTPTLON"]), 4), "y": round(float(place["INTPTLAT"]), 4)})
    cities.sort(key=lambda city: (-city["p"], city["n"]))
    return cities


def _lines(geometry) -> list[list[tuple[float, float]]]:
    """The plain coordinate lists of a LineString or MultiLineString."""
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [list(geometry.coords)]
    if geometry.geom_type == "MultiLineString":
        return [list(part.coords) for part in geometry.geoms]
    return []


def encode(coords: list[tuple[float, float]]) -> tuple[list[int], list[tuple[int, int]]]:
    """[(lon, lat), ...] -> ([x0, y0, dx1, dy1, ...], the points as whole numbers), in 1/10,000 degree.

    The first point is whole and each later one is the step from the last. Points that land on the same spot once rounded are
    dropped. Returns two empty lists for a line left with fewer than two points."""
    points: list[tuple[int, int]] = []
    for lon, lat in coords:
        point = (round(lon * COORD_SCALE), round(lat * COORD_SCALE))
        if not points or point != points[-1]:
            points.append(point)
    if len(points) < 2:
        return [], []
    ints = list(points[0])
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        ints.extend((x1 - x0, y1 - y0))
    return ints, points


def decode(ints: list[int]) -> list[tuple[float, float]]:
    """The inverse of encode (to the rounding): used by the tests, and the same sum the browser does."""
    points, x, y = [], 0, 0
    for i in range(0, len(ints), 2):
        x, y = (ints[i], ints[i + 1]) if i == 0 else (x + ints[i], y + ints[i + 1])
        points.append((x / COORD_SCALE, y / COORD_SCALE))
    return points


def vertex_levels(line, tolerances=ROAD_TOLERANCES) -> list[int]:
    """The level of each point of `line` (already the finest detail, so it is not simplified again): 0 if the point survives the
    coarsest simplification, 1 the middle one, 2 only the finest.

    Each coarser level is simplified from the finer one, and Douglas-Peucker only ever keeps points of its input, so the levels
    are nested and every coarse point is a fine point. The ends of a line are level 0 (they are never simplified away)."""
    from shapely.geometry import LineString

    finest = LineString(line.coords)
    rungs = [finest]
    for tolerance in reversed(tolerances[:-1]):
        rungs.append(rungs[-1].simplify(tolerance, preserve_topology=False))
    rungs.reverse()                                     # coarsest first
    kept = [{(round(x * COORD_SCALE), round(y * COORD_SCALE)) for x, y, *_ in rung.coords} for rung in rungs]
    levels = []
    for x, y, *_ in finest.coords:
        point = (round(x * COORD_SCALE), round(y * COORD_SCALE))
        levels.append(next((level for level, members in enumerate(kept) if point in members), len(kept) - 1))
    return levels


def build_roads(roads, tolerances=ROAD_TOLERANCES) -> dict[str, list[list]]:
    """{class: [[deltas, levels], ...]} for interstates, US highways and state routes, merged by route.

    `deltas` is the encoded point list (see encode) and `levels` is a string with one digit per point (see vertex_levels). A route
    that is not connected is several lines. `roads` is a GeoDataFrame-like with RTTYP, FULLNAME and geometry in degrees."""
    from shapely.geometry import LineString
    from shapely.ops import linemerge, unary_union

    out: dict[str, list[list]] = {name: [] for name in ROAD_CLASSES.values()}
    wanted = roads[roads["RTTYP"].isin(list(ROAD_CLASSES))]
    for (kind, _name), group in wanted.groupby(["RTTYP", "FULLNAME"], sort=True):
        merged = linemerge(unary_union(list(group.geometry))) if len(group) > 1 else group.geometry.iloc[0]
        for base in _lines(merged):
            finest = LineString(base).simplify(tolerances[-1], preserve_topology=False)
            ints, points = encode(list(finest.coords))
            if not ints:
                continue
            # Levels are worked out on the rounded points, so they line up one-to-one with what is stored.
            rounded = LineString([(x / COORD_SCALE, y / COORD_SCALE) for x, y in points])
            levels = vertex_levels(rounded, tolerances)
            if len(levels) != len(points):          # never drop a road quietly: the file would just be missing it
                raise ValueError(f"{_name}: {len(levels)} levels for {len(points)} points")
            out[ROAD_CLASSES[kind]].append([ints, "".join(str(level) for level in levels)])
    return out


def main() -> int:
    import csv

    import geopandas as gpd

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roads", required=True, help="tl_YYYY_29_prisecroads.zip")
    parser.add_argument("--places", required=True, help="tl_YYYY_29_place.zip")
    parser.add_argument("--population", required=True, help="Missouri rows of sub-est2023.csv")
    parser.add_argument("--out", required=True, help="folder to write cities_mo.json and roads_mo.json into")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(args.population, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    cities = build_cities(gpd.read_file(f"zip://{Path(args.places).resolve()}"), rows)
    roads = build_roads(gpd.read_file(f"zip://{Path(args.roads).resolve()}"))

    (out / "cities_mo.json").write_text(json.dumps(
        {"source": "US Census Bureau: TIGER/Line 2023 places, Vintage 2023 population estimates", "cities": cities},
        separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    (out / "roads_mo.json").write_text(json.dumps(
        {"v": 3, "scale": COORD_SCALE, "levels": len(ROAD_TOLERANCES),
         "source": "US Census Bureau: TIGER/Line 2023 primary and secondary roads", "roads": roads}, separators=(",", ":")),
        encoding="utf-8")
    print(f"cities: {len(cities)} (population {MIN_POPULATION}+), largest {cities[0]['n']} {cities[0]['p']}")
    print("roads (lines):", {name: len(lines) for name, lines in roads.items()})
    for level in range(len(ROAD_TOLERANCES)):
        points = sum(sum(1 for c in line[1] if int(c) <= level) for lines in roads.values() for line in lines)
        print(f"  drawn at detail level {level}: {points} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
