import unittest

from services import county_lookup
from services.mobile_content import county_catalog


class CountyLookupTests(unittest.TestCase):
    def test_columbia_missouri_resolves_to_boone_county(self):
        # Columbia, MO - well inside Boone County, FIPS 29019. This is the
        # regression test for the EPSG:3857 projection trap: a naive
        # implementation that skips the WGS84 -> Web Mercator transform
        # returns (None, None) here every time.
        fips, name = county_lookup.county_for_point(38.9517, -92.3341)
        self.assertEqual(fips, "29019")
        self.assertEqual(name, "Boone")

    def test_every_resolvable_fips_exists_in_county_catalog(self):
        known = {c["fips"] for c in county_catalog()}
        for _, _, fips, _name in county_lookup._county_polygons():
            self.assertIn(fips, known)

    def test_point_outside_missouri_resolves_to_none(self):
        # Wichita, KS
        fips, name = county_lookup.county_for_point(37.6872, -97.3301)
        self.assertIsNone(fips)
        self.assertIsNone(name)


if __name__ == "__main__":
    unittest.main()
