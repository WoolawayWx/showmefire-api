import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
from PIL import Image

from services.fire_incident_graphics import _add_locator_map, _locator_extent, render_incident_graphic
from services.graphic_renderer import _mercator


class IncidentLocatorTests(unittest.TestCase):
    def test_regional_extent_contains_detail_with_fourfold_context(self):
        for detail in [(-92.07, -92.04, 37.20, 37.24), (-93, -91, 36, 38)]:
            extent = _locator_extent(detail)
            self.assertLess(extent[0], detail[0])
            self.assertGreater(extent[1], detail[1])
            self.assertLess(extent[2], detail[2])
            self.assertGreater(extent[3], detail[3])
            west, south = _mercator(extent[0], extent[2])
            east, north = _mercator(extent[1], extent[3])
            dw, ds = _mercator(detail[0], detail[2])
            de, dn = _mercator(detail[1], detail[3])
            self.assertGreaterEqual(north - south + 1e-6, max(64000, 4 * (dn - ds)))
            self.assertGreaterEqual(east - west + 1e-6, 4 * (de - dw))

    def test_locator_uses_labeled_basemap_and_marks_satellite_coverage(self):
        fig, ax = plt.subplots()
        detail = (-92.07, -92.04, 37.20, 37.24)
        try:
            with patch("services.graphic_renderer._basemap", return_value=(Image.new("RGB", (500, 400)), 4)) as basemap:
                locator = _add_locator_map(ax, detail, [-92.055], [37.22])
            self.assertEqual(basemap.call_args.kwargs["url_template"], "https://tile.openstreetmap.org/{z}/{x}/{y}.png")
            self.assertEqual(len(locator.images), 1)
            rectangle = locator.patches[0]
            west, south = _mercator(detail[0], detail[2])
            self.assertAlmostEqual(rectangle.get_x(), west)
            self.assertAlmostEqual(rectangle.get_y(), south)
            self.assertEqual(len(locator.collections[0].get_offsets()), 1)
            fig.canvas.draw()
        finally:
            plt.close(fig)

    def test_card_still_renders_when_both_basemaps_are_offline(self):
        rows = [{"latitude": 37.22, "longitude": -92.055, "source": "VIIRS", "county_name": "Texas"}]
        incident = {"centroid_latitude": 37.22, "centroid_longitude": -92.055, "public_slug": "test"}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "incident.png"
            with patch("cartopy.io.img_tiles.GoogleTiles", side_effect=RuntimeError("offline")), patch(
                "services.graphic_renderer._basemap", side_effect=RuntimeError("offline")
            ):
                render_incident_graphic(incident, rows, output)
            with Image.open(output) as image:
                self.assertGreater(image.width, 1500)
                self.assertGreater(image.height, 900)



class IncidentCardRefreshTests(unittest.TestCase):
    """Cards are redrawn only when what they show changes (the 5-minute job used to redraw every card every time)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.incident = {"id": 1, "public_slug": "mo-test", "detection_count": 6, "first_detected_at": "2026-10-20T15:00:00",
                         "last_detected_at": "2026-10-20T16:00:00", "centroid_latitude": 37.2, "centroid_longitude": -92.0,
                         "county_names": ["Texas"], "shape_geojson": None, "updated_at": "2026-10-20T16:05:00",
                         "graphic_filename": None}
        self.rows = [{"id": n, "latitude": 37.2 + n / 1000, "longitude": -92.0, "source": "viirs",
                      "detection_confidence_pct": 80.0} for n in range(6)]

    def refresh(self, **kwargs):
        from services import fire_incident_graphics as g

        renders, recorded = [], []

        def fake_render(incident, rows, output):
            renders.append(incident["id"])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"png")
            return output

        with patch.object(g, "IMAGES_DIR", self.tmp.name), \
                patch.object(g, "list_fire_incidents", return_value=[dict(self.incident)]), \
                patch.object(g, "list_fire_incident_members", side_effect=lambda _id: [dict(r) for r in self.rows]), \
                patch.object(g, "set_fire_incident_graphic", side_effect=lambda *a: recorded.append(a)), \
                patch.object(g, "render_incident_graphic", side_effect=fake_render):
            result = g.refresh_incident_graphics(**kwargs)
        return result, renders, recorded

    def test_unchanged_card_is_skipped_and_a_change_redraws_it(self):
        result, renders, recorded = self.refresh()
        self.assertEqual((result["rendered"], result["skipped"], renders), (1, 0, [1]))
        self.assertEqual(recorded, [(1, "mo-test.png")])

        self.incident["graphic_filename"] = "mo-test.png"
        self.incident["updated_at"] = "2026-10-20T16:10:00"  # bumped by the render itself; not drawn on the card
        result, renders, recorded = self.refresh()
        self.assertEqual((result["rendered"], result["skipped"], renders, recorded), (0, 1, [], []))

        self.rows[0]["detection_confidence_pct"] = 95.0  # shown on the card
        self.assertEqual(self.refresh()[0]["rendered"], 1)
        self.rows.append({"id": 9, "latitude": 37.3, "longitude": -92.1, "source": "goes", "detection_confidence_pct": None})
        self.incident["detection_count"] = 7
        self.assertEqual(self.refresh()[0]["rendered"], 1)

    def test_force_missing_png_and_unrecorded_filename(self):
        self.refresh()
        self.assertEqual(self.refresh(force=True)[0]["rendered"], 1)
        (Path(self.tmp.name) / "fire-incidents" / "mo-test.png").unlink()
        self.assertEqual(self.refresh()[0]["rendered"], 1)
        # Card on disk is current but the database never recorded it: record it without drawing again.
        result, renders, recorded = self.refresh()
        self.assertEqual((result["skipped"], renders, recorded), (1, [], [(1, "mo-test.png")]))

    def test_fingerprint_ignores_row_order_and_follows_the_design_version(self):
        from services import fire_incident_graphics as g

        base = g.card_fingerprint(self.incident, self.rows)
        self.assertEqual(base, g.card_fingerprint(self.incident, list(reversed(self.rows))))
        with patch.object(g, "CARD_VERSION", g.CARD_VERSION + 1):
            self.assertNotEqual(base, g.card_fingerprint(self.incident, self.rows))
        self.assertNotEqual(base, g.card_fingerprint({**self.incident, "shape_geojson": '{"type": "Point"}'}, self.rows))

    def test_roads_are_drawn_as_two_layers_not_two_per_road(self):
        from cartopy.mpl.geoaxes import GeoAxes
        from services import fire_incident_graphics as g

        roads = Path(g.__file__).resolve().parents[1] / "maps" / "shapefiles"
        if not any((roads / name).is_dir() for name in ("MO_MoDOT_Roads_Arcs", "MO_TIGER_Primary_Roads")):
            self.skipTest("no road shapefile in this checkout")
        rows = [{"latitude": 38.95, "longitude": -92.33, "source": "VIIRS"},
                {"latitude": 38.97, "longitude": -92.30, "source": "VIIRS"}]  # Columbia, on I-70 / US-63
        incident = {"centroid_latitude": 38.96, "centroid_longitude": -92.31, "public_slug": "roads"}
        with patch.object(GeoAxes, "add_image"), patch("services.graphic_renderer._basemap", side_effect=RuntimeError("offline")), \
                patch.object(GeoAxes, "add_geometries", autospec=True, side_effect=GeoAxes.add_geometries) as add:
            g.render_incident_graphic(incident, rows, Path(self.tmp.name) / "roads.png")
        self.assertEqual(add.call_count, 2)
        self.assertGreater(len(list(add.call_args_list[0].args[1])), 1)


if __name__ == "__main__":
    unittest.main()
