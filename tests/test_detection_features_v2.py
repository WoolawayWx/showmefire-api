import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from services import detection_features_v2 as f2


def _row(i, source="viirs", occurred="2026-09-01T18:00:00Z", arrived=None, lat=38.0, lon=-92.0, frp=5.0, **extra):
    row = {"id": i, "source": source, "occurred_at": occurred, "first_seen_at": arrived or occurred,
           "latitude": lat, "longitude": lon, "frp": frp, "confidence": "n", "bright_t7": 330.0, "bright_t13": 295.0,
           "pixel_area": 0.14, "solar_zenith_angle": 40.0, "satellite_zenith_angle": 30.0, "daynight": "D",
           "land_cover": "Cropland:60,Trees:40", "exclusion_zone_id": None, "recurring_source_id": None}
    row.update(extra)
    return row


class DetectionFeaturesV2Tests(unittest.TestCase):
    def test_only_arrived_prior_detections_count(self):
        target = pd.DataFrame([_row(1)])
        history = pd.DataFrame([
            _row(2, occurred="2026-09-01T10:00:00Z"),                                   # earlier, arrived in time
            _row(3, occurred="2026-09-01T10:00:00Z", arrived="2026-09-02T00:00:00Z"),   # earlier but arrived after scoring
            _row(4, occurred="2026-09-01T20:00:00Z"),                                   # later than the detection
            _row(1),                                                                    # the detection itself
        ])
        X, _ = f2.build_features(target, history)
        self.assertEqual(X.loc[0, "n_1km_24h"], 1)
        self.assertEqual(X.loc[0, "n_5km_365d"], 1)

    def test_updated_score_sees_late_arrivals_but_initial_does_not(self):
        history = pd.DataFrame([_row(3, occurred="2026-09-01T10:00:00Z", arrived="2026-09-01T22:00:00Z")])
        initial = pd.DataFrame([_row(1)])
        updated = initial.assign(as_of="2026-09-02T00:00:00Z")
        self.assertEqual(f2.build_features(initial, history)[0].loc[0, "n_1km_24h"], 0)
        self.assertEqual(f2.build_features(updated, history)[0].loc[0, "n_1km_24h"], 1)

    def test_corroboration_needs_another_sensor_and_look_ahead_only_when_updated(self):
        target = pd.DataFrame([_row(1, source="viirs")])
        history = pd.DataFrame([
            _row(2, source="modis", occurred="2026-09-01T17:00:00Z", frp=20.0),
            _row(3, source="viirs", occurred="2026-09-01T17:30:00Z"),                    # same sensor: not corroboration
            _row(4, source="ngfs", occurred="2026-09-01T19:00:00Z", frp=9.0),            # after the detection
        ])
        X0, _ = f2.build_features(target, history)
        self.assertEqual(X0.loc[0, "n_other_sensor_3h"], 1)
        self.assertEqual(X0.loc[0, "max_other_frp_3h"], 20.0)
        updated = target.assign(as_of="2026-09-01T23:00:00Z")
        X1, _ = f2.build_features(updated, history, lookahead_h=3.0)
        self.assertEqual(X1.loc[0, "n_other_sensor_3h"], 2)

    def test_missing_values_stay_missing_not_zero(self):
        target = pd.DataFrame([_row(1, frp=None, bright_t7=None, land_cover=None, confidence=None, daynight=None)])
        X, avail = f2.build_features(target, pd.DataFrame(columns=target.columns))
        for col in ("frp_log", "bright_diff", "native_confidence", "is_day", "lc_cropland"):
            self.assertTrue(np.isnan(X.loc[0, col]), col)
        self.assertEqual(X.loc[0, "n_1km_24h"], 0)
        self.assertEqual(avail.loc[0, "arrival_known"], 1)

    def test_unknown_arrival_falls_back_and_is_flagged(self):
        target = pd.DataFrame([_row(1)]).assign(first_seen_at=None)
        _, avail = f2.build_features(target, pd.DataFrame(columns=target.columns))
        self.assertEqual(avail.loc[0, "arrival_known"], 0)

    def test_recurring_source_only_counts_once_confirmed(self):
        target = pd.DataFrame([_row(1)])
        rec = pd.DataFrame({"latitude": [38.001], "longitude": [-92.0], "detection_count": [40],
                            "reviewed_at": ["2026-09-05T00:00:00Z"]})
        self.assertTrue(np.isnan(f2.build_features(target, pd.DataFrame(columns=target.columns), rec)[0].loc[0, "recurring_source_km"]))
        later = target.assign(as_of="2026-09-10T00:00:00Z")
        X, _ = f2.build_features(later, pd.DataFrame(columns=target.columns), rec)
        self.assertLess(X.loc[0, "recurring_source_km"], 1.0)
        self.assertEqual(X.loc[0, "recurring_source_detections"], 40)

    def test_excluded_inputs_are_not_features(self):
        self.assertFalse(set(f2.EXCLUDED) & set(f2.FEATURE_NAMES))
        self.assertEqual(len(set(f2.FEATURE_NAMES)), len(f2.FEATURE_NAMES))


if __name__ == "__main__":
    unittest.main()
