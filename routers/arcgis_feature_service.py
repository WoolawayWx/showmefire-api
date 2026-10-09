"""
Read-only ArcGIS-compatible Feature Services (``.../FeatureServer``).

ArcGIS Online / Dashboards only consume feature layers, so a plain GeoJSON URL
can't be polled by a dashboard.  This router speaks enough of the ArcGIS REST
"FeatureServer" protocol (service + layer metadata, ``query`` with where /
outFields / statistics / distinct / extent / paging) for those clients to treat
Show Me Fire data as a live feature layer.  Nothing is stored: every request
reads the current source (the burn-ban database, or the latest published file).

Services (add via Content -> New item -> URL, "An ArcGIS Server web service"):

  /arcgis/rest/services/BurnBans/FeatureServer/0              live county burn bans
  /arcgis/rest/services/PeakFireDangerSmooth/FeatureServer/0  smooth danger polygons

Query-only (``capabilities: Query``); public data only.  ``where`` clauses are run
by an in-memory SQLite restricted to SELECT on one table (see ``_run_sql``).
"""
from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from fastapi import APIRouter, Request, Response
from pyproj import Transformer
from shapely.geometry import box, shape
from shapely.geometry.polygon import orient
from shapely.ops import transform as shp_transform

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/arcgis/rest", tags=["arcgis-feature-service"], include_in_schema=False)

VERSION = 11.1
MAX_RECORD_COUNT = 2000
MAX_WHERE_CHARS = 2000
MISSOURI_EXTENT = (-95.8, 35.9, -89.1, 40.7)  # xmin, ymin, xmax, ymax (fallback)
COPYRIGHT = "Show Me Fire (showmefire.org)"
_WEB_MERCATOR = {102100, 102113, 3857, 3785}

STRING, INTEGER, DOUBLE, DATE, OID = (
    "esriFieldTypeString", "esriFieldTypeInteger", "esriFieldTypeDouble",
    "esriFieldTypeDate", "esriFieldTypeOID",
)
_SQL_TYPE = {STRING: "TEXT", INTEGER: "INTEGER", DOUBLE: "REAL", DATE: "INTEGER", OID: "INTEGER PRIMARY KEY"}


@dataclass
class FieldDef:
    name: str
    type: str
    alias: str = ""
    length: int | None = None

    def json(self) -> dict:
        out = {"name": self.name, "type": self.type, "alias": self.alias or self.name,
               "sqlType": "sqlTypeOther", "nullable": self.type != OID, "editable": False,
               "domain": None, "defaultValue": None}
        if self.type == STRING:
            out["length"] = self.length or 255
        return out


@dataclass
class LayerDef:
    service: str
    name: str
    description: str
    display_field: str
    fields: list[FieldDef]
    load: Callable[[], list[dict]]           # -> [{"OBJECTID":..., <attrs>, "_geometry": shapely geom (EPSG:4326)}]
    renderer: Callable[[], dict] | None = None
    service_description: str = ""
    ttl: float = 0.0
    _cache: tuple = field(default=(0.0, None), repr=False)

    def records(self) -> list[dict]:
        if self.ttl > 0:
            stamp, cached = self._cache
            if cached is not None and time.monotonic() - stamp < self.ttl:
                return cached
        data = self.load()
        if self.ttl > 0:
            self._cache = (time.monotonic(), data)
        return data


# ── helpers ───────────────────────────────────────────────────────────────────
def _epoch_ms(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    try:
        text = str(value).strip().replace("Z", "+00:00").replace(" ", "T", 1)
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def _rgba(hex_color: str, alpha: int) -> list[int]:
    hex_color = hex_color.lstrip("#")
    return [int(hex_color[i:i + 2], 16) for i in (0, 2, 4)] + [alpha]


def _fill(color: list[int], outline: list[int] | None = None, width: float = 0.75) -> dict:
    return {"type": "esriSFS", "style": "esriSFSSolid", "color": color,
            "outline": {"type": "esriSLS", "style": "esriSLSSolid", "color": outline or [110, 110, 110, 255], "width": width}}


# ── layer sources ─────────────────────────────────────────────────────────────
def _load_burn_bans() -> list[dict]:
    """All Missouri counties, flagged active/inactive, straight from the database."""
    from services.burn_ban_map import burn_ban_feature_collection

    records = []
    for feature in burn_ban_feature_collection(include_all_counties=True)["features"]:
        p = feature["properties"]
        records.append({
            "OBJECTID": int(p["county_fips"]),
            "county_fips": p["county_fips"],
            "county_name": p.get("county_name"),
            "status": p["status"],
            "has_burn_ban": 1 if p.get("has_burn_ban") else 0,
            "effective_at": _epoch_ms(p.get("effective_at")),
            "expires_at": _epoch_ms(p.get("expires_at")),
            "published_at": _epoch_ms(p.get("published_at")),
            "updated_at": _epoch_ms(p.get("updated_at")),
            "proof_url": p.get("proof_url"),
            "_geometry": shape(feature["geometry"]),
        })
    return records


_SMOOTH_FILE = "peak_fire_danger_smooth_polygons.geojson"
_smooth_cache: tuple[float, list[dict]] = (-1.0, [])


def _load_smooth_polygons() -> list[dict]:
    """The latest smooth polygons file, re-read only when it changes."""
    global _smooth_cache
    from core.config import GIS_DIR

    path = GIS_DIR / _SMOOTH_FILE
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return []
    if mtime == _smooth_cache[0]:
        return _smooth_cache[1]
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for feature in payload.get("features", []):
        p = feature["properties"]
        level = int(p["danger_level"])
        records.append({
            "OBJECTID": level + 1,
            "danger_level": level,
            "label": p.get("label"),
            "color": p.get("color"),
            "model_run": _epoch_ms(p.get("model_run")),
            "resolution_m": p.get("resolution_m"),
            "area_km2": p.get("area_km2"),
            "_geometry": shape(feature["geometry"]),
        })
    _smooth_cache = (mtime, records)
    return records


def _burn_ban_renderer() -> dict:
    return {"type": "uniqueValue", "field1": "status", "fieldDelimiter": ", ", "defaultSymbol": None,
            "defaultLabel": None,
            "uniqueValueInfos": [
                {"value": "active", "label": "Active burn ban", "description": "",
                 "symbol": _fill([185, 28, 28, 200], [60, 60, 60, 255], 1.0)},
                {"value": "inactive", "label": "No burn ban", "description": "",
                 "symbol": _fill([255, 255, 255, 40], [150, 150, 150, 255], 0.6)},
            ]}


_DANGER = [(0, "Low", "#90EE90"), (1, "Moderate", "#FFED4E"), (2, "Elevated", "#FFA500"),
           (3, "Critical", "#FF0000"), (4, "Extreme", "#8B0000")]


def _danger_renderer() -> dict:
    return {"type": "uniqueValue", "field1": "danger_level", "fieldDelimiter": ", ", "defaultSymbol": None,
            "defaultLabel": None,
            "uniqueValueInfos": [{"value": str(level), "label": label, "description": "",
                                  "symbol": _fill(_rgba(color, 190), [90, 90, 90, 120], 0.4)}
                                 for level, label, color in _DANGER]}


LAYERS: dict[str, LayerDef] = {
    "BurnBans": LayerDef(
        service="BurnBans", name="County Burn Bans", display_field="county_name",
        description="Burn ban status for every Missouri county, read live from the Show Me Fire database.",
        service_description="Live Missouri county burn bans from Show Me Fire.",
        fields=[FieldDef("OBJECTID", OID, "OBJECTID"), FieldDef("county_fips", STRING, "County FIPS", 5),
                FieldDef("county_name", STRING, "County", 100), FieldDef("status", STRING, "Status", 20),
                FieldDef("has_burn_ban", INTEGER, "Has burn ban (1/0)"),
                FieldDef("effective_at", DATE, "Effective"), FieldDef("expires_at", DATE, "Expires"),
                FieldDef("published_at", DATE, "Published"), FieldDef("updated_at", DATE, "Updated"),
                FieldDef("proof_url", STRING, "Proof URL", 500)],
        load=_load_burn_bans, renderer=_burn_ban_renderer, ttl=10.0),
    "PeakFireDangerSmooth": LayerDef(
        service="PeakFireDangerSmooth", name="Peak Fire Danger Forecast (Smooth)", display_field="label",
        description="Forecast peak fire danger as smooth, high-resolution polygons clipped to Missouri.",
        service_description="Show Me Fire peak fire danger forecast, smooth polygons.",
        fields=[FieldDef("OBJECTID", OID, "OBJECTID"), FieldDef("danger_level", INTEGER, "Danger level (0-4)"),
                FieldDef("label", STRING, "Danger level", 20), FieldDef("color", STRING, "Color", 9),
                FieldDef("model_run", DATE, "Model run"), FieldDef("resolution_m", INTEGER, "Resolution (m)"),
                FieldDef("area_km2", DOUBLE, "Area (km²)")],
        load=_load_smooth_polygons, renderer=_danger_renderer),
}


# ── geometry ──────────────────────────────────────────────────────────────────
_transformers: dict[tuple[int, int], Transformer] = {}


def _wkid(value: Any, default: int = 4326) -> int:
    if isinstance(value, str) and value.strip().startswith("{"):
        # ArcGIS clients send inSR/outSR as a JSON spatial-reference object.
        try:
            value = json.loads(value)
        except ValueError:
            return default
    if isinstance(value, dict):
        value = value.get("latestWkid") or value.get("wkid")
    try:
        wkid = int(value)
    except (TypeError, ValueError):
        return default
    return 3857 if wkid in _WEB_MERCATOR else wkid


def _reproject(geom, source: int, target: int):
    if source == target:
        return geom
    key = (source, target)
    if key not in _transformers:
        _transformers[key] = Transformer.from_crs(f"EPSG:{source}", f"EPSG:{target}", always_xy=True)
    transformer = _transformers[key]
    return shp_transform(lambda x, y, z=None: transformer.transform(x, y), geom)


def _esri_rings(geom, precision: int) -> dict:
    polygons = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
    rings = []
    for polygon in polygons:
        if polygon.geom_type != "Polygon" or polygon.is_empty:
            continue
        polygon = orient(polygon, sign=-1.0)          # Esri: exterior clockwise, holes counter-clockwise
        for ring in [polygon.exterior, *polygon.interiors]:
            rings.append([[round(x, precision), round(y, precision)] for x, y, *_ in ring.coords])
    return {"rings": rings}


def _extent(records: list[dict], wkid: int) -> dict:
    geoms = [r["_geometry"] for r in records if r.get("_geometry") is not None]
    if geoms:
        xmin = min(g.bounds[0] for g in geoms); ymin = min(g.bounds[1] for g in geoms)
        xmax = max(g.bounds[2] for g in geoms); ymax = max(g.bounds[3] for g in geoms)
    else:
        xmin, ymin, xmax, ymax = MISSOURI_EXTENT
    if wkid != 4326:
        low = _reproject(box(xmin, ymin, xmax, ymax), 4326, wkid).bounds
        xmin, ymin, xmax, ymax = low
    sr = {"wkid": 102100, "latestWkid": 3857} if wkid == 3857 else {"wkid": wkid, "latestWkid": wkid}
    return {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "spatialReference": sr}


def _spatial_reference(wkid: int) -> dict:
    return {"wkid": 102100, "latestWkid": 3857} if wkid == 3857 else {"wkid": wkid, "latestWkid": wkid}


def _parse_filter_geometry(raw: str | None, in_sr: int):
    """Envelope (JSON or 'xmin,ymin,xmax,ymax') or polygon JSON -> shapely in EPSG:4326."""
    if not raw:
        return None
    try:
        if raw.strip().startswith("{"):
            data = json.loads(raw)
            in_sr = _wkid(data.get("spatialReference"), in_sr)
            if "rings" in data:
                from shapely.geometry import Polygon
                geom = Polygon(data["rings"][0], data["rings"][1:])
            else:
                geom = box(float(data["xmin"]), float(data["ymin"]), float(data["xmax"]), float(data["ymax"]))
        else:
            xmin, ymin, xmax, ymax = (float(part) for part in raw.split(","))
            geom = box(xmin, ymin, xmax, ymax)
        return _reproject(geom, in_sr, 4326)
    except (ValueError, KeyError, TypeError, IndexError):
        raise _QueryError("Invalid 'geometry' parameter.")


# ── SQL (where / statistics) ──────────────────────────────────────────────────
class _QueryError(Exception):
    pass


_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ALLOWED_FUNCTIONS = {"upper", "lower", "length", "abs", "round", "coalesce", "ifnull", "nullif", "substr",
                      "instr", "replace", "trim", "ltrim", "rtrim", "min", "max", "sum", "avg", "count", "total", "like"}
_ESRI_DATE = re.compile(r"(?i)\b(?:timestamp|date)\s+'([^']*)'")


def _authorizer(action, arg1, arg2, dbname, source):
    # SQLITE_SELECT=21, SQLITE_READ=20 (arg1 = table), SQLITE_FUNCTION=31
    if action == 21:
        return sqlite3.SQLITE_OK
    if action == 20 and arg1 == "f":
        return sqlite3.SQLITE_OK
    if action == 31 and str(arg2).lower() in _ALLOWED_FUNCTIONS:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def _where_sql(where: str | None) -> str:
    where = (where or "").strip()
    if where in ("", "1=1", "1 = 1"):
        return "1=1"
    if len(where) > MAX_WHERE_CHARS:
        raise _QueryError("'where' clause too long.")

    def literal(match: re.Match) -> str:
        ms = _epoch_ms(match.group(1))
        if ms is None:
            raise _QueryError(f"Invalid date literal '{match.group(1)}'.")
        return str(ms)

    return f"({_ESRI_DATE.sub(literal, where)})"


def _run_sql(layer: LayerDef, records: list[dict], sql: str) -> list[sqlite3.Row]:
    """Run one SELECT against an in-memory copy of the layer's attributes."""
    names = [f.name for f in layer.fields]
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE TABLE f (" + ", ".join(f'"{f.name}" {_SQL_TYPE[f.type]}' for f in layer.fields) + ")")
        conn.executemany(
            f"INSERT INTO f VALUES ({','.join('?' * len(names))})",
            [[r.get(n) for n in names] for r in records])
        conn.row_factory = sqlite3.Row
        conn.set_authorizer(_authorizer)
        steps = {"n": 0}

        def guard():
            steps["n"] += 1
            return 1 if steps["n"] > 200 else 0      # ~200k VM instructions, then abort
        conn.set_progress_handler(guard, 1000)
        return conn.execute(sql).fetchall()
    except sqlite3.Error as exc:
        raise _QueryError(f"Unable to complete operation. ({exc})")
    finally:
        conn.close()


def _columns(layer: LayerDef, text: str | None) -> list[str]:
    valid = {f.name.lower(): f.name for f in layer.fields}
    if not text or text.strip() == "*":
        return [f.name for f in layer.fields]
    cols = []
    for part in text.split(","):
        part = part.strip()
        if part.lower() not in valid:
            raise _QueryError(f"Invalid field '{part}'.")
        cols.append(valid[part.lower()])
    return cols


def _order_by(layer: LayerDef, text: str | None) -> str:
    if not text:
        return '"OBJECTID" ASC'
    valid = {f.name.lower(): f.name for f in layer.fields}
    parts = []
    for item in text.split(","):
        tokens = item.split()
        if not tokens or tokens[0].lower() not in valid or len(tokens) > 2 or \
                (len(tokens) == 2 and tokens[1].upper() not in ("ASC", "DESC")):
            raise _QueryError("Invalid 'orderByFields'.")
        parts.append(f'"{valid[tokens[0].lower()]}" {tokens[1].upper() if len(tokens) == 2 else "ASC"}')
    return ", ".join(parts)


_STAT_FUNCS = {"count": "COUNT", "sum": "SUM", "min": "MIN", "max": "MAX", "avg": "AVG"}


def _statistics(layer: LayerDef, records: list[dict], where: str, params: dict) -> dict:
    try:
        stats = json.loads(params["outStatistics"])
        assert isinstance(stats, list) and stats
    except (ValueError, AssertionError):
        raise _QueryError("Invalid 'outStatistics'.")
    valid = {f.name.lower(): f for f in layer.fields}
    group = [valid[g.strip().lower()].name for g in (params.get("groupByFieldsForStatistics") or "").split(",")
             if g.strip() and g.strip().lower() in valid]
    selects, out_fields = [f'"{g}"' for g in group], [valid[g.lower()].json() for g in group]
    for stat in stats:
        func = _STAT_FUNCS.get(str(stat.get("statisticType", "")).lower())
        source = valid.get(str(stat.get("onStatisticField", "")).lower())
        alias = str(stat.get("outStatisticFieldName") or "")
        if not func or source is None or not _IDENT.match(alias):
            raise _QueryError("Invalid 'outStatistics'.")
        selects.append(f'{func}("{source.name}") AS "{alias}"')
        out_fields.append({"name": alias, "type": INTEGER if func == "COUNT" else DOUBLE, "alias": alias,
                           "sqlType": "sqlTypeOther", "nullable": True, "editable": False,
                           "domain": None, "defaultValue": None})
    sql = f"SELECT {', '.join(selects)} FROM f WHERE {where}"
    if group:
        sql += " GROUP BY " + ", ".join(f'"{g}"' for g in group) + " ORDER BY " + ", ".join(f'"{g}"' for g in group)
    rows = _run_sql(layer, records, sql)
    return {"displayFieldName": layer.display_field, "fields": out_fields,
            "features": [{"attributes": {k: row[k] for k in row.keys()}} for row in rows]}


# ── query ─────────────────────────────────────────────────────────────────────
def _flag(params: dict, key: str, default: bool = False) -> bool:
    value = str(params.get(key, "")).strip().lower()
    return default if not value else value in ("true", "1", "yes")


def _query(layer: LayerDef, params: dict) -> dict:
    records = layer.records()
    by_id = {r["OBJECTID"]: r for r in records}
    where = _where_sql(params.get("where"))

    # object-id and spatial filters narrow the attribute filter (stats included)
    if params.get("objectIds"):
        try:
            wanted = [int(i) for i in str(params["objectIds"]).split(",") if i.strip()]
        except ValueError:
            raise _QueryError("Invalid 'objectIds'.")
        where = f'({where}) AND "OBJECTID" IN ({",".join(map(str, wanted)) or "NULL"})'
    flt = _parse_filter_geometry(params.get("geometry"), _wkid(params.get("inSR")))
    if flt is not None:
        hit = [r["OBJECTID"] for r in records if r["_geometry"].intersects(flt)]
        where = f'({where}) AND "OBJECTID" IN ({",".join(map(str, hit)) or "NULL"})'

    if params.get("outStatistics"):
        return _statistics(layer, records, where, params)

    distinct = _flag(params, "returnDistinctValues")
    columns = _columns(layer, params.get("outFields"))
    select = ", ".join(f'"{c}"' for c in columns)
    if distinct:
        sql = f"SELECT DISTINCT {select} FROM f WHERE {where} ORDER BY {_order_by(layer, params.get('orderByFields'))}"
    else:
        sql = (f'SELECT {select}, "OBJECTID" AS __oid FROM f WHERE {where} '
               f"ORDER BY {_order_by(layer, params.get('orderByFields'))}")
    rows = _run_sql(layer, records, sql)

    out_sr = _wkid(params.get("outSR"))
    if _flag(params, "returnCountOnly"):
        return {"count": len(rows)}
    if distinct:
        want_geometry = False
    else:
        want_geometry = _flag(params, "returnGeometry", True)
        if _flag(params, "returnIdsOnly"):
            return {"objectIdFieldName": "OBJECTID", "objectIds": [row["__oid"] for row in rows]}
        if _flag(params, "returnExtentOnly"):
            return {"count": len(rows), "extent": _extent([by_id[row["__oid"]] for row in rows], out_sr)}

    try:
        offset = max(0, int(params.get("resultOffset") or 0))
        limit = int(params.get("resultRecordCount") or MAX_RECORD_COUNT)
    except ValueError:
        raise _QueryError("Invalid paging parameters.")
    limit = max(1, min(limit, MAX_RECORD_COUNT))
    page = rows[offset:offset + limit]
    exceeded = offset + limit < len(rows)

    geojson = str(params.get("f", "")).lower() == "geojson"
    try:
        tolerance = float(params.get("maxAllowableOffset") or 0)
    except ValueError:
        tolerance = 0.0
    precision = 6 if out_sr == 4326 else 2

    features = []
    for row in page:
        attrs = {c: row[c] for c in columns}
        geom = by_id[row["__oid"]]["_geometry"] if want_geometry else None
        if geojson:
            features.append({"type": "Feature", "id": row["__oid"] if want_geometry else None, "properties": attrs,
                             "geometry": json.loads(json.dumps(geom.__geo_interface__)) if geom is not None else None})
            continue
        feature: dict[str, Any] = {"attributes": attrs}
        if geom is not None:
            geom = _reproject(geom, 4326, out_sr)
            if tolerance > 0:
                geom = geom.simplify(tolerance, preserve_topology=True)
            feature["geometry"] = _esri_rings(geom, precision)
        features.append(feature)

    if geojson:
        return {"type": "FeatureCollection", "features": features,
                **({"properties": {"exceededTransferLimit": True}} if exceeded else {})}
    by_name = {f.name: f for f in layer.fields}
    return {"objectIdFieldName": "OBJECTID", "globalIdFieldName": "", "displayFieldName": layer.display_field,
            "geometryType": "esriGeometryPolygon", "spatialReference": _spatial_reference(out_sr),
            "fields": [by_name[c].json() for c in columns],
            "features": features, "exceededTransferLimit": exceeded}


# ── metadata ──────────────────────────────────────────────────────────────────
def _service_json(layer: LayerDef) -> dict:
    extent = _extent(layer.records(), 4326)
    return {"currentVersion": VERSION, "serviceDescription": layer.service_description, "hasVersionedData": False,
            "supportsDisconnectedEditing": False, "hasStaticData": False, "maxRecordCount": MAX_RECORD_COUNT,
            "supportedQueryFormats": "JSON, geoJSON", "capabilities": "Query", "description": layer.service_description,
            "copyrightText": COPYRIGHT, "spatialReference": _spatial_reference(4326), "initialExtent": extent,
            "fullExtent": extent, "allowGeometryUpdates": False, "units": "esriDecimalDegrees",
            "layers": [{"id": 0, "name": layer.name, "parentLayerId": -1, "defaultVisibility": True,
                        "subLayerIds": None, "minScale": 0, "maxScale": 0, "geometryType": "esriGeometryPolygon"}],
            "tables": [], "syncEnabled": False, "supportsApplyEditsWithGlobalIds": False}


def _layer_json(layer: LayerDef) -> dict:
    out = {"currentVersion": VERSION, "id": 0, "name": layer.name, "type": "Feature Layer",
           "description": layer.description, "geometryType": "esriGeometryPolygon", "copyrightText": COPYRIGHT,
           "parentLayer": None, "subLayers": [], "minScale": 0, "maxScale": 0, "defaultVisibility": True,
           "extent": _extent(layer.records(), 4326), "hasAttachments": False,
           "htmlPopupType": "esriServerHTMLPopupTypeNone", "displayField": layer.display_field,
           "typeIdField": None, "subtypeField": None, "fields": [f.json() for f in layer.fields],
           "geometryProperties": None, "indexes": [], "types": [], "templates": [],
           "supportedQueryFormats": "JSON, geoJSON", "hasM": False, "hasZ": False, "objectIdField": "OBJECTID",
           "uniqueIdField": {"name": "OBJECTID", "isSystemMaintained": True}, "globalIdField": "",
           "capabilities": "Query", "maxRecordCount": MAX_RECORD_COUNT, "supportsPagination": True,
           "supportsStatistics": True, "supportsAdvancedQueries": True, "supportsOrderBy": True,
           "supportsDistinct": True, "supportsReturningQueryExtent": True, "supportsCoordinatesQuantization": False,
           "supportsCalculate": False, "hasStaticData": False, "canModifyLayer": False, "canScaleSymbols": False,
           "useStandardizedQueries": True, "supportsCurrentUserQueries": False}
    if layer.renderer:
        out["drawingInfo"] = {"renderer": layer.renderer(), "transparency": 0, "labelingInfo": None}
    return out


# ── HTTP ──────────────────────────────────────────────────────────────────────
_HEADERS = {"Access-Control-Allow-Origin": "*", "Cache-Control": "no-store"}


def _reply(payload: dict, status: int = 200) -> Response:
    return Response(json.dumps(payload, separators=(",", ":"), default=str), status_code=status,
                    media_type="application/json", headers=_HEADERS)


def _error(message: str, code: int = 400) -> Response:
    # ArcGIS clients read errors from the body, with HTTP 200.
    return _reply({"error": {"code": code, "message": message, "details": []}})


async def _params(request: Request) -> dict:
    params = dict(request.query_params)
    if request.method == "POST":
        try:
            params.update({k: v for k, v in (await request.form()).items() if isinstance(v, str)})
        except Exception:
            pass
    return params


def _layer_or_none(service: str) -> LayerDef | None:
    return LAYERS.get(next((k for k in LAYERS if k.lower() == service.lower()), ""))


@router.get("/info")
def server_info():
    return _reply({"currentVersion": VERSION, "fullVersion": "11.1.0", "authInfo": {"isTokenBasedSecurity": False}})


@router.get("/services")
def list_services():
    return _reply({"currentVersion": VERSION, "folders": [],
                   "services": [{"name": name, "type": "FeatureServer"} for name in LAYERS]})


@router.get("/services/{service}/FeatureServer")
def feature_server(service: str):
    layer = _layer_or_none(service)
    if layer is None:
        return _error("Service not found.", 404)
    return _reply(_service_json(layer))


@router.get("/services/{service}/FeatureServer/layers")
def feature_server_layers(service: str):
    layer = _layer_or_none(service)
    if layer is None:
        return _error("Service not found.", 404)
    return _reply({"layers": [_layer_json(layer)], "tables": []})


@router.get("/services/{service}/FeatureServer/{layer_id}")
def feature_layer(service: str, layer_id: int):
    layer = _layer_or_none(service)
    if layer is None or layer_id != 0:
        return _error("Layer not found.", 404)
    return _reply(_layer_json(layer))


@router.api_route("/services/{service}/FeatureServer/{layer_id}/query", methods=["GET", "POST"])
async def query_layer(service: str, layer_id: int, request: Request):
    layer = _layer_or_none(service)
    if layer is None or layer_id != 0:
        return _error("Layer not found.", 404)
    try:
        return _reply(_query(layer, await _params(request)))
    except _QueryError as exc:
        return _error(str(exc))
    except Exception:
        logger.error("Feature service query failed", exc_info=True)
        return _error("Unable to complete operation.", 500)
