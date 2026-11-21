import json
import unittest

from shapely.geometry import Point, shape

from services import incident_shape_extractor as extractor


def _square(lon, lat, half=0.01):
    return json.dumps({"type": "Polygon", "coordinates": [[[lon - half, lat - half], [lon + half, lat - half], [lon + half, lat + half], [lon - half, lat + half], [lon - half, lat - half]]]})


class IncidentShapeTests(unittest.TestCase):
    def test_no_members_means_no_shape(self):
        self.assertIsNone(extractor.compute_incident_shape({"id": 1}, []))

    def test_real_footprints_and_point_detections_become_one_polygon_covering_them(self):
        members = [
            {"source": "ngfs", "latitude": 36.6, "longitude": -90.4, "footprint_geojson": _square(-90.4, 36.6)},
            {"source": "ngfs", "latitude": 36.6, "longitude": -90.38, "footprint_geojson": _square(-90.38, 36.6)},
            {"source": "viirs", "latitude": 36.61, "longitude": -90.39},       # no footprint: a small circle
        ]
        geometry = extractor.compute_incident_shape({"id": 2}, members)
        self.assertEqual(geometry["type"], "Polygon")
        outline = shape(geometry)
        for member in members:
            self.assertTrue(outline.buffer(0.001).contains(Point(member["longitude"], member["latitude"])))

    def test_detections_that_do_not_touch_still_give_a_single_polygon(self):
        members = [
            {"source": "viirs", "latitude": 36.6, "longitude": -90.4},
            {"source": "viirs", "latitude": 36.9, "longitude": -90.0},           # ~45 km away
        ]
        self.assertEqual(extractor.compute_incident_shape({"id": 3}, members)["type"], "Polygon")

    def test_the_old_satellite_image_step_is_gone(self):
        # compute_incident_shape must not fetch imagery (no network in this test would otherwise slow or fail it).
        for name in ("extract_hot_mask", "MICROPHYSICS_TILE_URL", "IMAGE_SIZE", "BLUE_MAX"):
            self.assertFalse(hasattr(extractor, name), name)


if __name__ == "__main__":
    unittest.main()
