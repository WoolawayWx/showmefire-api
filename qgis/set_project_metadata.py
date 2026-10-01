"""Fill in the metadata of the layers in showmefire.qgz without opening QGIS.

Everything you would type into QGIS (Layer Properties -> QGIS Server and
Metadata tabs) lives in the dictionaries below.  The script edits the
project's XML as text, so nothing else in the file changes and QGIS never
re-resolves the /data paths.  Empty values are skipped, so fill in what you
want published (contact, license, rights ...) and re-run.  Idempotent.

    python set_project_metadata.py                  # edits showmefire.qgz in place
    python set_project_metadata.py in.qgz out.qgz
"""
from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

PROJECT = Path(__file__).with_name("showmefire.qgz")

SITE = "https://showmefire.org"

# ── Shared by every layer; leave a value empty to omit it ────────────────────
COMMON = {
    "language": "eng",
    "attribution": "Show Me Fire — Cade Woolaway",
    "attribution_url": SITE,
    "links": [("Show Me Fire", "WWW:LINK", SITE, "Show Me Fire website")],
    "license": "",         # e.g. "CC BY 4.0"
    "rights": ["Copyright Show Me Fire — Cade Woolaway"],          # e.g. ["Copyright Show Me Fire"]
    "constraints": [],     # e.g. [("Access", "Public")]  (type, text)
    "fees": "",
    "contact": {           # name / organization / position / voice / email / role
        "name": "Cade Woolaway", "organization": "Show Me Fire", "position": "Founder/Metorologist", "voice": "", "email": "contact@showmefire.org", "role": "",
    },
}

# layer name (the stable WMS name, used in URLs) -> metadata
LAYERS = {
    "forecast_peak_fire_danger": {
        "title": "Peak Fire Danger Forecast",
        "abstract": (
            "Forecast peak fire danger across Missouri for the day (10:00-21:00 CT), "
            "from HRRR model forecasts and the Show Me Fire fire-weather model, as a "
            "3 km raster (UTM zone 15N) clipped to the state boundary.\n\n"
            "Pixel values: 0 Low, 1 Moderate, 2 Elevated, 3 High, 4 Extreme.\n"
            "Low: fuel moisture (FM) >= 15%.\n"
            "Moderate: FM < 15% and (RH < 45% or wind >= 10 kt).\n"
            "Elevated: FM < 9% with (RH < 35% and wind >= 12 kt) or (RH < 25% and wind >= 5 kt).\n"
            "High: FM < 9% with RH < 25% and wind >= 15 kt.\n"
            "Extreme: FM < 7% with RH < 20% and wind >= 25 kt.\n"
            "Updated after each forecast run."
        ),
        "keywords": ["fire danger", "wildfire", "fire weather", "forecast", "Missouri", "raster"],
        "history": "Generated from HRRR forecast fields by the Show Me Fire forecast pipeline and regridded to a 3 km EPSG:32615 grid.",
    },
    "forecast_peak_fire_danger_polygons": {
        "title": "Peak Fire Danger Forecast (Polygons)",
        "abstract": (
            "The Peak Fire Danger Forecast as polygon regions, one set per danger level "
            "(attributes danger_level 0-4 and label: Low, Moderate, Elevated, High, Extreme). "
            "Regions are buffered by 250 m and clipped to the Missouri state boundary. "
            "Updated after each forecast run."
        ),
        "keywords": ["fire danger", "wildfire", "fire weather", "forecast", "Missouri", "polygons"],
        "history": "Derived from the Peak Fire Danger Forecast raster by the Show Me Fire forecast pipeline.",
    },
    "burn_bans": {
        "title": "County Burn Bans",
        "abstract": (
            "Burn ban status for every Missouri county (attributes include county_fips, "
            "county_name and status: active or inactive)."
        ),
        "keywords": ["burn ban", "wildfire", "Missouri", "counties"],
        "history": "",
    },
}

# ── Project-level (service) text ─────────────────────────────────────────────
PROJECT_ABSTRACT = (
    "Missouri fire danger forecast layers from Show Me Fire: the peak fire danger "
    "forecast (raster and polygons) and county burn bans."
)
PROJECT_KEYWORDS = ["fire danger", "wildfire", "fire weather", "forecast", "Missouri", "burn ban"]

# ─────────────────────────────────────────────────────────────────────────────
BLOCK = re.compile(r"<maplayer\b.*?</maplayer>", re.S)
NAME = re.compile(r"<layername>([^<]*)</layername>")
INDENT = "      "


def _t(tag: str, value: str, **attrs: str) -> str:
    extra = "".join(f" {k}={quoteattr(v)}" for k, v in attrs.items())
    return f"<{tag}{extra}>{escape(value)}</{tag}>"


def _drop(text: str, *tags: str) -> str:
    for tag in tags:
        text = re.sub(rf"[ \t]*<{tag}\b[^>]*?/>[ \t]*\n?", "", text)
        text = re.sub(rf"[ \t]*<{tag}\b[^>]*>.*?</{tag}>[ \t]*\n?", "", text, flags=re.S)
    return text


def server_block(meta: dict, common: dict) -> str:
    lines = [_t("title", meta["title"]), _t("abstract", meta["abstract"])]
    lines.append("<keywordList>" + "".join(_t("value", k) for k in meta["keywords"]) + "</keywordList>")
    if common["attribution"]:
        lines.append(_t("attribution", common["attribution"]))
    if common["attribution_url"]:
        lines.append(_t("attributionUrl", common["attribution_url"]))
    if common["links"]:
        lines.append(_t("dataUrl", common["links"][0][2], format="text/html"))
    return "".join(f"\n{INDENT}{line}" for line in lines)


def resource_metadata(text: str, meta: dict, common: dict) -> str:
    """Rewrite the fields of one <resourceMetadata> block (the ISO metadata tab)."""
    text = _drop(text, "identifier", "language", "title", "abstract", "fees", "links", "rights",
                 "license", "history", "constraints", "contact", "keywords")
    pad = "        "
    parts = [
        _t("identifier", f"showmefire:{meta['name']}"),
        _t("language", common["language"]),
        _t("title", meta["title"]),
        _t("abstract", meta["abstract"]),
        _t("fees", common["fees"]),
    ]
    parts += [_t("constraints", text_, type=kind) for kind, text_ in common["constraints"]]
    parts += [_t("rights", r) for r in common["rights"]]
    if common["license"]:
        parts.append(_t("license", common["license"]))
    if meta["history"]:
        parts.append(_t("history", meta["history"]))
    contact = {k: v for k, v in common["contact"].items() if v}
    if contact:
        parts.append("<contact>" + "".join(_t(k, v) for k, v in contact.items()) + "</contact>")
    if common["links"]:
        parts.append("<links>" + "".join(
            f"<link name={quoteattr(n)} type={quoteattr(t)} url={quoteattr(u)} "
            f"description={quoteattr(d)} format=\"\" mimeType=\"\" size=\"\"/>"
            for n, t, u, d in common["links"]) + "</links>")
    parts.append("<keywords vocabulary=\"\">" + "".join(_t("keyword", k) for k in meta["keywords"]) + "</keywords>")
    # keep the QGIS-managed <type> first; everything else after it
    return re.sub(r"(<type>[^<]*</type>\s*)", lambda m: m.group(1) + "".join(f"{p}\n{pad}" for p in parts), text, count=1)


def retitle(block: str) -> str:
    match = NAME.search(block)
    if not match or match.group(1) not in LAYERS:
        return block
    meta = {**LAYERS[match.group(1)], "name": match.group(1)}

    # 1. WMS-visible server properties, between <datasource> and the layer's <keywordList>
    start = block.index("<datasource>")
    head, tail = block[:start], block[start:]
    end_ds = tail.index("</datasource>") + len("</datasource>")
    ds, rest = tail[:end_ds], tail[end_ds:]
    rest = _drop(rest.split("<resourceMetadata>")[0], "title", "abstract", "attribution", "attributionUrl", "dataUrl") \
        + ("<resourceMetadata>" + rest.split("<resourceMetadata>", 1)[1] if "<resourceMetadata>" in rest else "")
    rest = re.sub(r"[ \t]*<keywordList>.*?</keywordList>[ \t]*\n?", "", rest, count=1, flags=re.S)
    rest = server_block(meta, COMMON) + f"\n{INDENT}" + rest.lstrip()
    block = head + ds + rest

    # 2. ISO metadata tab
    return re.sub(r"<resourceMetadata>.*?</resourceMetadata>",
                  lambda m: resource_metadata(m.group(0), meta, COMMON), block, count=1, flags=re.S)


def project_level(text: str) -> str:
    text = re.sub(r"(<WMS>\s*)<ServiceAbstract type=\"QString\">[^<]*</ServiceAbstract>",
                  lambda m: m.group(1) + f'<ServiceAbstract type="QString">{escape(PROJECT_ABSTRACT)}</ServiceAbstract>',
                  text, count=1)

    def meta(m: re.Match) -> str:
        body = _drop(m.group(0), "abstract", "keywords")
        extra = f"<abstract>{escape(PROJECT_ABSTRACT)}</abstract>\n    " + \
            "<keywords vocabulary=\"\">" + "".join(_t("keyword", k) for k in PROJECT_KEYWORDS) + "</keywords>\n    "
        return re.sub(r"(<links/>)", lambda mm: extra + mm.group(1), body, count=1)

    return re.sub(r"<projectMetadata>.*?</projectMetadata>", meta, text, count=1, flags=re.S)


def main() -> None:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else source
    with zipfile.ZipFile(source) as archive:
        members = [(info, archive.read(info.filename)) for info in archive.infolist()]
    changed = 0
    rebuilt = []
    for info, data in members:
        if info.filename.endswith(".qgs"):
            text = data.decode("utf-8")
            updated = project_level(BLOCK.sub(lambda m: retitle(m.group(0)), text))
            changed = sum(1 for a, b in zip(BLOCK.findall(text), BLOCK.findall(updated)) if a != b)
            data = updated.encode("utf-8")
        rebuilt.append((info, data))
    temporary = target.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for info, data in rebuilt:
            archive.writestr(info, data)
    temporary.replace(target)
    print(f"{target}: updated metadata on {changed} layer(s)")


if __name__ == "__main__":
    main()
