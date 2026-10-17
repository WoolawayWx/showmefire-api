import sys
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

API_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(API_DIR))

from maps import drought_indices as di
from maps import satellite_maps as sat


class EddiCategoryTests(unittest.TestCase):
    def classify(self, value):
        return di.EDDI_CLASSES[int(np.digitize(value, di.EDDI_BREAKS[1:-1]))][0]

    def test_percentile_edges_map_to_official_categories(self):
        self.assertIsNone(self.classify(0.0))
        self.assertEqual(self.classify(0.6), "ED0")    # 70-80th percentile
        self.assertEqual(self.classify(1.0), "ED1")
        self.assertEqual(self.classify(1.5), "ED2")
        self.assertEqual(self.classify(1.9), "ED3")
        self.assertEqual(self.classify(2.5), "ED4")    # 98th+
        self.assertEqual(self.classify(-2.5), "EW4")

    def test_class_table_lines_up_with_breaks(self):
        self.assertEqual(len(di.EDDI_CLASSES), len(di.EDDI_BREAKS) - 1)


class VegdriCodeTests(unittest.TestCase):
    def name_for(self, code):
        return next(name for low, high, name, _color in di.VEGDRI_CLASSES if low <= code <= high)

    def test_index_scale_is_128_plus_16_per_unit(self):
        self.assertEqual(self.name_for(128), "Near Normal")       # index 0
        self.assertEqual(self.name_for(104), "Pre-Drought Stress")  # index -1.5
        self.assertEqual(self.name_for(90), "Moderate Drought")     # index -2.4
        self.assertEqual(self.name_for(40), "Extreme Drought")

    def test_flag_codes(self):
        self.assertEqual(self.name_for(253), "Water")
        self.assertEqual(self.name_for(254), "Out of Season")

    def test_ranges_cover_every_code_once(self):
        covered = [code for low, high, _name, _color in di.VEGDRI_CLASSES for code in range(low, high + 1)]
        self.assertEqual(sorted(covered), list(range(1, 256)))


class WeekDateTests(unittest.TestCase):
    def test_processing_date_rolls_back_to_week_ending_sunday(self):
        self.assertEqual(di._sunday_on_or_before(date(2026, 10, 12)), date(2026, 10, 11))
        self.assertEqual(di._sunday_on_or_before(date(2026, 10, 11)), date(2026, 10, 11))
        self.assertEqual(di._tiff_valid_date({"TIFFTAG_DATETIME": "2026:10:05 02:02:25"}), "Week ending Oct 04, 2026")


class SatelliteTests(unittest.TestCase):
    def test_scan_start_parsed_from_key(self):
        key = "ABI-L2-MCMIPC/2026/289/18/OR_ABI-L2-MCMIPC-M6_G19_s20262891831174_e20262891833547_c20262891834044.nc"
        self.assertEqual(sat._scan_start(key), datetime(2026, 10, 16, 18, 31, 17, tzinfo=timezone.utc))

    def test_stretch_clips_and_can_invert(self):
        np.testing.assert_allclose(sat._stretch(np.array([0.0, 5.0, 20.0]), 0.0, 10.0), [0.0, 0.5, 1.0])
        np.testing.assert_allclose(sat._stretch(np.array([280.65, 219.65]), 280.65, 219.65), [0.0, 1.0])

    def test_rgb_alpha_hides_missing_pixels(self):
        image = sat._rgb(np.array([[0.5, np.nan]]), np.array([[0.5, 0.5]]), np.array([[0.5, 0.5]]))
        self.assertEqual(image.shape, (1, 2, 4))
        self.assertEqual(image[0, 0, 3], 1.0)
        self.assertEqual(image[0, 1, 3], 0.0)

    def test_day_night_gating(self):
        self.assertTrue(sat._wanted("day", 30))
        self.assertFalse(sat._wanted("day", 0))
        self.assertTrue(sat._wanted("night", -10))
        self.assertFalse(sat._wanted("night", 0))
        self.assertTrue(sat._wanted("any", 0))

    def test_band7_reflectance_removes_emitted_heat(self):
        t13 = np.array([290.0, 290.0, 290.0])
        r39 = sat.band7_reflectance(np.array([290.0, 300.0, 320.0]), t13, np.array([40.0, 40.0, 40.0]))
        self.assertAlmostEqual(r39[0], 0.0)  # all emitted, nothing reflected
        self.assertTrue(0 < r39[1] < r39[2] <= 1)
        self.assertTrue(np.isnan(sat.band7_reflectance(np.array([300.0]), np.array([290.0]), np.array([-5.0]))[0]))

    def test_product_keys_and_pngs_are_unique(self):
        specs = sat.CHANNELS + sat.RGBS
        self.assertEqual(len({spec["key"] for spec in specs}), len(specs))
        self.assertEqual(len({spec["png"] for spec in specs}), len(specs))

    def test_every_fire_mask_code_has_one_confidence(self):
        codes = [code for _name, _color, group in sat.FIRE_CLASSES for code in group]
        self.assertEqual(len(codes), len(set(codes)))
        self.assertEqual(sat._confidence(10), 0)
        self.assertEqual(sat._confidence(35), 2)


if __name__ == "__main__":
    unittest.main()
