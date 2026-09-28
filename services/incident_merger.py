"""Merges fire incidents that are really one fire.

find_or_create_incident_for_detection() attaches each detection to the
nearest incident *centroid* within a small radius, and centroids drift, so a
fire larger than a few km (or one that spreads) ends up as several incidents
sitting side by side. This pass links incidents whose member detections
(real pixel footprints where stored, small buffered circles otherwise) come
within one pixel of each other and overlap in time, then folds each connected
group into its oldest incident.

The absorbed incidents are kept as status='merged' with merged_into_id set,
so old public slugs still resolve (get_public_fire_incident follows the
link) and feedback rows stay valid, but they never appear in listings or
absorb new detections.

Groups whose combined extent exceeds MAX_MERGED_EXTENT_KM are left alone and
reported - a long line of touching pixels along a ridge or road can chain
separate fires together, and that is a human call, not an automatic merge.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone

from shapely.ops import transform, unary_union
from shapely.strtree import STRtree

from core.database import _parse_occurred_at, get_db_path

logger = logging.getLogger(__name__)

MERGE_GAP_KM = float(os.getenv("FIRE_INCIDENT_MERGE_GAP_KM", "1.5"))
MERGE_WINDOW_HOURS = float(os.getenv("FIRE_INCIDENT_MERGE_WINDOW_HOURS", "48.0"))
MIN_FRAGMENT_DETECTIONS = 4
MAX_MERGED_EXTENT_KM = float(os.getenv("FIRE_INCIDENT_MAX_MERGED_EXTENT_KM", "12.0"))


def _km_projector(reference_lat: float):
    """Local equirectangular lon/lat -> km. Plenty accurate at incident scale."""
    kx = 111.320 * math.cos(math.radians(reference_lat))
    ky = 110.574
    return lambda x, y, z=None: (x * kx, y * ky)


def _to_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = _parse_occurred_at(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _spans_close_enough(a: dict, b: dict, window_hours: float) -> bool:
    a_first, a_last = _to_datetime(a["first_detected_at"]), _to_datetime(a["last_detected_at"])
    b_first, b_last = _to_datetime(b["first_detected_at"]), _to_datetime(b["last_detected_at"])
    if not all((a_first, a_last, b_first, b_last)):
        return True  # can't tell - fall back to spatial adjacency alone
    gap = max(a_first, b_first) - min(a_last, b_last)
    return gap <= timedelta(hours=window_hours)


class _DisjointSet:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a, b):
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def _load_incident_geometries(cursor: sqlite3.Cursor, incidents: list[dict]) -> dict[int, object]:
    """incident id -> union of its member geometries, in km."""
    from services.incident_shape_extractor import _member_polygons

    if not incidents:
        return {}
    reference_lat = sum(i["centroid_latitude"] for i in incidents) / len(incidents)
    project = _km_projector(reference_lat)
    ids = [i["id"] for i in incidents]
    members_by_incident: dict[int, list[dict]] = {i: [] for i in ids}
    # Chunked to stay well under SQLite's bound-variable limit.
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        placeholders = ",".join("?" * len(chunk))
        cursor.execute(
            f"SELECT incident_id, latitude, longitude, source, footprint_geojson FROM fire_events "
            f"WHERE incident_id IN ({placeholders}) AND latitude IS NOT NULL AND longitude IS NOT NULL",
            chunk,
        )
        for row in cursor.fetchall():
            members_by_incident[row["incident_id"]].append(dict(row))

    geometries = {}
    for incident in incidents:
        polygons = _member_polygons(members_by_incident[incident["id"]])
        if not polygons:
            continue
        geometries[incident["id"]] = transform(project, unary_union(polygons))
    return geometries


def _extent_km(geometries: list) -> float:
    minx, miny, maxx, maxy = unary_union(geometries).bounds
    return math.hypot(maxx - minx, maxy - miny)


def _recompute_incident_stats(cursor: sqlite3.Cursor, incident_id: int) -> None:
    """Rebuild an incident's centroid (FRP-weighted), counts, time span and
    county from its current member rows, and drop its cached shape/graphic so
    both regenerate."""
    cursor.execute(
        "SELECT latitude, longitude, frp, occurred_at, county_fips, county_name FROM fire_events WHERE incident_id = ?",
        (incident_id,),
    )
    rows = cursor.fetchall()
    weights = [max(1.0, float(r["frp"] or 0.0)) for r in rows]
    total = sum(weights)
    centroid_lat = sum(r["latitude"] * w for r, w in zip(rows, weights)) / total
    centroid_lon = sum(r["longitude"] * w for r, w in zip(rows, weights)) / total
    times = [r["occurred_at"] for r in rows if r["occurred_at"]]
    counties = Counter((r["county_fips"], r["county_name"]) for r in rows if r["county_fips"])
    county_fips, county_name = counties.most_common(1)[0][0] if counties else (None, None)

    cursor.execute(
        """UPDATE fire_incidents
           SET centroid_latitude = ?, centroid_longitude = ?, detection_count = ?,
               first_detected_at = ?, last_detected_at = ?,
               county_fips = COALESCE(?, county_fips), county_name = COALESCE(?, county_name),
               shape_geojson = NULL, shape_detection_count = NULL, graphic_filename = NULL,
               updated_at = CURRENT_TIMESTAMP
           WHERE id = ?""",
        (centroid_lat, centroid_lon, len(rows), min(times), max(times), county_fips, county_name, incident_id),
    )


def _apply_merge(cursor: sqlite3.Cursor, survivor_id: int, loser_ids: list[int]) -> None:
    placeholders = ",".join("?" * len(loser_ids))
    cursor.execute(
        f"UPDATE fire_events SET incident_id = ? WHERE incident_id IN ({placeholders})",
        [survivor_id, *loser_ids],
    )
    cursor.execute(
        f"UPDATE fire_incident_feedback SET incident_id = ? WHERE incident_id IN ({placeholders})",
        [survivor_id, *loser_ids],
    )

    _recompute_incident_stats(cursor, survivor_id)
    cursor.execute(
        f"UPDATE fire_incidents SET status = 'merged', merged_into_id = ?, updated_at = CURRENT_TIMESTAMP "
        f"WHERE id IN ({placeholders})",
        [survivor_id, *loser_ids],
    )


def merge_touching_incidents(
    dry_run: bool = False,
    since_hours: float | None = None,
    gap_km: float = MERGE_GAP_KM,
    window_hours: float = MERGE_WINDOW_HOURS,
    max_extent_km: float = MAX_MERGED_EXTENT_KM,
) -> dict:
    """Merge every group of touching incidents. since_hours limits the pass to
    incidents active recently (the scheduled call); None considers all active
    incidents (the one-off cleanup). Returns a summary; with dry_run nothing
    is written."""
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    summary = {"considered": 0, "groups": [], "oversized": [], "merged_away": 0, "dry_run": dry_run}
    try:
        query = "SELECT * FROM fire_incidents WHERE status = 'active'"
        params: list = []
        if since_hours is not None:
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=since_hours)).strftime("%Y-%m-%dT%H:%M:%S")
            query += " AND last_detected_at >= ?"
            params.append(cutoff)
        cursor.execute(query, params)
        incidents = [dict(r) for r in cursor.fetchall()]
        by_id = {i["id"]: i for i in incidents}
        summary["considered"] = len(incidents)

        geometries = _load_incident_geometries(cursor, incidents)
        ids = list(geometries)
        if len(ids) < 2:
            return summary

        tree = STRtree([geometries[i] for i in ids])
        sets = _DisjointSet(ids)
        for idx, incident_id in enumerate(ids):
            geom = geometries[incident_id]
            for other_idx in tree.query(geom.buffer(gap_km)):
                other_id = ids[int(other_idx)]
                if other_id <= incident_id:
                    continue
                if geom.distance(geometries[other_id]) <= gap_km and _spans_close_enough(by_id[incident_id], by_id[other_id], window_hours):
                    sets.union(incident_id, other_id)

        components: dict[int, list[int]] = {}
        for incident_id in ids:
            components.setdefault(sets.find(incident_id), []).append(incident_id)

        for members in components.values():
            if len(members) < 2:
                continue
            extent = _extent_km([geometries[i] for i in members])
            ordered = sorted(members, key=lambda i: (by_id[i]["first_detected_at"], i))
            survivor, losers = ordered[0], ordered[1:]
            entry = {
                "survivor_id": survivor,
                "survivor_slug": by_id[survivor].get("public_slug"),
                "merged_ids": losers,
                "detections": sum(int(by_id[i]["detection_count"] or 0) for i in members),
                "extent_km": round(extent, 1),
            }
            if extent > max_extent_km:
                summary["oversized"].append(entry)
                continue
            summary["groups"].append(entry)
            summary["merged_away"] += len(losers)
            if not dry_run:
                _apply_merge(cursor, survivor, losers)
        if not dry_run:
            conn.commit()
    finally:
        conn.close()

    logger.info(
        "incident_merger: considered=%d groups=%d merged_away=%d oversized=%d dry_run=%s",
        summary["considered"], len(summary["groups"]), summary["merged_away"], len(summary["oversized"]), dry_run,
    )
    return summary


def _split_member_groups(members: list[dict], gap_km: float, max_extent_km: float) -> list[list[int]]:
    """Group one incident's members (as lists of fire_events ids): first by
    touching footprints (single link, gap_km), then any group still wider than
    max_extent_km is cut with complete-linkage clustering, which by
    construction never produces a cluster wider than the threshold."""
    import numpy as np
    from scipy.cluster.hierarchy import fcluster, linkage

    from services.incident_shape_extractor import _member_polygons

    project = _km_projector(sum(float(m["latitude"]) for m in members) / len(members))
    geoms, kept = [], []
    for member in members:
        polygons = _member_polygons([member])
        if polygons:
            geoms.append(transform(project, unary_union(polygons)))
            kept.append(member)
    if len(kept) < 2:
        return [[m["id"] for m in members]]

    tree = STRtree(geoms)
    sets = _DisjointSet(range(len(kept)))
    for idx, geom in enumerate(geoms):
        for other in tree.query(geom.buffer(gap_km)):
            other = int(other)
            if other > idx and geom.distance(geoms[other]) <= gap_km:
                sets.union(idx, other)
    components: dict[int, list[int]] = {}
    for idx in range(len(kept)):
        components.setdefault(sets.find(idx), []).append(idx)

    groups: list[list[int]] = []
    for indexes in components.values():
        if len(indexes) > 1 and _extent_km([geoms[i] for i in indexes]) > max_extent_km:
            points = np.array([[geoms[i].centroid.x, geoms[i].centroid.y] for i in indexes])
            labels = fcluster(linkage(points, method="complete"), t=max_extent_km, criterion="distance")
            by_label: dict[int, list[int]] = {}
            for i, label in zip(indexes, labels):
                by_label.setdefault(int(label), []).append(i)
            groups.extend([[kept[i]["id"] for i in idxs] for idxs in by_label.values()])
        else:
            groups.append([kept[i]["id"] for i in indexes])
    # Fold tiny fragments back into the nearest bigger group when that keeps it
    # reasonably compact - a 1-3 detection straggler is noise, not a fire.
    geom_by_id = {kept[i]["id"]: geoms[i] for i in range(len(kept))}
    groups.sort(key=len, reverse=True)
    solid = [g for g in groups if len(g) >= MIN_FRAGMENT_DETECTIONS]
    if solid:
        result = [list(g) for g in solid]
        for fragment in (g for g in groups if len(g) < MIN_FRAGMENT_DETECTIONS):
            frag_geom = unary_union([geom_by_id[i] for i in fragment])
            nearest = min(result, key=lambda g: frag_geom.distance(unary_union([geom_by_id[i] for i in g])))
            merged = unary_union([geom_by_id[i] for i in nearest + fragment])
            minx, miny, maxx, maxy = merged.bounds
            if math.hypot(maxx - minx, maxy - miny) <= max_extent_km * 1.25:
                nearest.extend(fragment)
            else:
                result.append(fragment)
        groups = result
    dropped = [m["id"] for m in members if m not in kept]  # members with no usable geometry
    if dropped:
        groups[0].extend(dropped)
    return groups


def split_oversized_incidents(
    dry_run: bool = False,
    max_extent_km: float = MAX_MERGED_EXTENT_KM,
    gap_km: float = MERGE_GAP_KM,
) -> dict:
    """Undo over-merging: re-cluster every active incident wider than
    max_extent_km under the stricter rule. The largest resulting group keeps
    the original incident (id, slug, feedback); every other group becomes a
    new incident. Returns a summary; with dry_run nothing is written."""
    import secrets

    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    summary = {"examined": 0, "split": [], "dry_run": dry_run}
    try:
        cursor.execute("SELECT * FROM fire_incidents WHERE status = 'active' AND detection_count >= 2")
        incidents = [dict(r) for r in cursor.fetchall()]
        for incident in incidents:
            cursor.execute(
                "SELECT id, latitude, longitude, source, footprint_geojson FROM fire_events "
                "WHERE incident_id = ? AND latitude IS NOT NULL AND longitude IS NOT NULL",
                (incident["id"],),
            )
            members = [dict(r) for r in cursor.fetchall()]
            if len(members) < 2:
                continue
            summary["examined"] += 1
            from services.incident_shape_extractor import _member_polygons
            project = _km_projector(incident["centroid_latitude"])
            extent = _extent_km([transform(project, unary_union(_member_polygons([m]))) for m in members if _member_polygons([m])])
            if extent <= max_extent_km:
                continue

            groups = sorted(_split_member_groups(members, gap_km, max_extent_km), key=len, reverse=True)
            if len(groups) < 2:
                continue
            summary["split"].append({
                "incident_id": incident["id"], "slug": incident.get("public_slug"),
                "from_extent_km": round(extent, 1), "into": [len(g) for g in groups],
            })
            if dry_run:
                continue
            for group in groups[1:]:
                cursor.execute(
                    """INSERT INTO fire_incidents
                           (centroid_latitude, centroid_longitude, first_detected_at, last_detected_at,
                            detection_count, county_fips, county_name, public_slug)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (incident["centroid_latitude"], incident["centroid_longitude"], incident["first_detected_at"],
                     incident["last_detected_at"], len(group), incident.get("county_fips"), incident.get("county_name"),
                     secrets.token_urlsafe(9)),
                )
                new_id = cursor.lastrowid
                placeholders = ",".join("?" * len(group))
                cursor.execute(f"UPDATE fire_events SET incident_id = ? WHERE id IN ({placeholders})", [new_id, *group])
                _recompute_incident_stats(cursor, new_id)
            _recompute_incident_stats(cursor, incident["id"])
        if not dry_run:
            conn.commit()
    finally:
        conn.close()

    logger.info("incident_merger: split examined=%d split=%d dry_run=%s", summary["examined"], len(summary["split"]), dry_run)
    return summary
