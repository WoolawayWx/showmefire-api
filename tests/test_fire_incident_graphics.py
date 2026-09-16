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


if __name__ == "__main__":
    unittest.main()
