"""Build the small production QGIS project from the tracked template."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT = Path(__file__).with_name("showmefire.qgz")
TARGETS = {
    "forecast_peak_fire_danger": "/data/latest/forecast_peak_fire_danger.tif",
    "forecast_peak_fire_danger_polygons": "/data/peak_fire_danger_polygons.geojson",
    "burn_bans": "/data/burn_bans.gpkg|layername=burn_bans",
}


def option(parent: ET.Element, name: str, value: str) -> None:
    ET.SubElement(parent, "Option", {"name": name, "type": "QString", "value": value})


def symbol(color: str, outline: str = "35,35,35,255", alpha: str = "0.72") -> ET.Element:
    item = ET.Element("symbol", {"alpha": alpha, "clip_to_extent": "1", "force_rhr": "0", "frame_rate": "10", "is_animated": "0", "name": "0", "type": "fill"})
    dd = ET.SubElement(item, "data_defined_properties")
    opt = ET.SubElement(dd, "Option", {"type": "Map"})
    option(opt, "name", "")
    ET.SubElement(opt, "Option", {"name": "properties"})
    option(opt, "type", "collection")
    layer = ET.SubElement(item, "layer", {"class": "SimpleFill", "enabled": "1", "id": "{" + str(uuid.uuid4()) + "}", "locked": "0", "pass": "0"})
    opts = ET.SubElement(layer, "Option", {"type": "Map"})
    option(opts, "border_width_map_unit_scale", "3x:0,0,0,0,0,0")
    option(opts, "color", color)
    option(opts, "joinstyle", "bevel")
    option(opts, "offset", "0,0")
    option(opts, "offset_map_unit_scale", "3x:0,0,0,0,0,0")
    option(opts, "offset_unit", "MM")
    option(opts, "outline_color", outline)
    option(opts, "outline_style", "solid")
    option(opts, "outline_width", "0.35")
    option(opts, "outline_width_unit", "MM")
    option(opts, "style", "solid")
    return item


def categorized(attr: str, categories: list[tuple[str, str, str]]) -> ET.Element:
    renderer = ET.Element("renderer-v2", {"enableorderby": "0", "forceraster": "0", "referencescale": "-1", "symbollevels": "0", "type": "categorizedSymbol", "attr": attr})
    cats = ET.SubElement(renderer, "categories")
    symbols = ET.SubElement(renderer, "symbols")
    for index, (value, label, color) in enumerate(categories):
        ET.SubElement(cats, "category", {"value": value, "label": label, "symbol": str(index)})
        sym = symbol(color)
        sym.set("name", str(index))
        symbols.append(sym)
    ET.SubElement(renderer, "source-symbol", {"name": "0"})
    ET.SubElement(renderer, "rotation")
    ET.SubElement(renderer, "sizescale")
    dd = ET.SubElement(renderer, "data-defined-properties")
    opt = ET.SubElement(dd, "Option", {"type": "Map"})
    option(opt, "name", "")
    ET.SubElement(opt, "Option", {"name": "properties"})
    option(opt, "type", "collection")
    return renderer


def main() -> None:
    with tempfile.TemporaryDirectory() as temp:
        temp_path = Path(temp)
        subprocess.run(["unzip", "-q", str(PROJECT), "-d", str(temp_path)], check=True)
        qgs = temp_path / "showmefire.qgs"
        tree = ET.parse(qgs)
        root = tree.getroot()

        layers = root.find("layer-tree-group")
        assert layers is not None
        for child in list(layers):
            if child.tag == "layer-tree-layer":
                name = child.get("name")
                if name == "peak_fire_danger_polygons":
                    child.set("name", "forecast_peak_fire_danger_polygons")
                    child.set("source", TARGETS["forecast_peak_fire_danger_polygons"])
                    child.set("id", "forecast_peak_fire_danger_polygons_8871e05c_c119_4e77_9ff4_1df870ab7a7f")
                elif name in TARGETS:
                    child.set("source", TARGETS[name])
                elif name != "forecast_peak_fire_danger_polygons":
                    layers.remove(child)

        legend = root.find("legend")
        if legend is not None:
            for child in list(legend):
                if child.tag == "legendlayer":
                    name = child.get("name")
                    if name == "peak_fire_danger_polygons":
                        child.set("name", "forecast_peak_fire_danger_polygons")
                        for f in child.findall("./filegroup/legendlayerfile"):
                            f.set("layerid", "forecast_peak_fire_danger_polygons_8871e05c_c119_4e77_9ff4_1df870ab7a7f")
                    if name not in {"forecast_peak_fire_danger", "burn_bans", "forecast_peak_fire_danger_polygons"}:
                        legend.remove(child)

        project_layers = root.find("projectlayers")
        assert project_layers is not None
        maplayers = project_layers.findall("maplayer")
        keep = {"forecast_peak_fire_danger", "burn_bans", "peak_fire_danger_polygons", "forecast_peak_fire_danger_polygons"}
        for layer in maplayers:
            layername = layer.findtext("layername")
            if layername not in keep:
                project_layers.remove(layer)
                continue
            source = layer.find("datasource")
            if source is not None:
                if layername == "peak_fire_danger_polygons":
                    layername = "forecast_peak_fire_danger_polygons"
                source.text = TARGETS[layername]
            name_node = layer.find("layername")
            if name_node is not None and layername == "forecast_peak_fire_danger_polygons":
                name_node.text = layername
            custom = layer.find("customproperties")
            if custom is not None:
                for item in custom.findall(".//Option"):
                    if item.get("name") == "ows_name":
                        item.set("value", layername)
            renderer = layer.find("renderer-v2")
            if layername == "forecast_peak_fire_danger_polygons":
                if renderer is not None:
                    layer.remove(renderer)
                layer.insert(list(layer).index(layer.find("selection")), categorized("danger_level", [
                    ("0", "Low", "144,238,144,190,rgb:0.564706,0.933333,0.564706,0.745098"),
                    ("1", "Moderate", "255,237,78,190,rgb:1,0.929412,0.305882,0.745098"),
                    ("2", "Elevated", "255,165,0,195,rgb:1,0.647059,0,0.764706"),
                    ("3", "Critical", "255,0,0,200,rgb:1,0,0,0.784314"),
                    ("4", "Extreme", "139,0,0,215,rgb:0.545098,0,0,0.843137"),
                ]))
            elif layername == "burn_bans":
                if renderer is not None:
                    layer.remove(renderer)
                layer.insert(list(layer).index(layer.find("selection")), categorized("status", [
                    ("active", "Active burn ban", "185,28,28,200,rgb:0.72549,0.109804,0.109804,0.784314"),
                    ("inactive", "No active burn ban", "235,235,235,95,rgb:0.921569,0.921569,0.921569,0.372549"),
                ]))

        ET.indent(tree, space="  ")
        tree.write(qgs, encoding="utf-8", xml_declaration=True)
        rebuilt = temp_path / "showmefire-new.qgz"
        subprocess.run(["zip", "-q", "-j", str(rebuilt), str(qgs)], check=True)
        shutil.copy2(rebuilt, PROJECT)


if __name__ == "__main__":
    main()
