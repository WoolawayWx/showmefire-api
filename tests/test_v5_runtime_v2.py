import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from services.v5_scorer import build_features
from services.v5_verification import verify_pending


class V5RuntimeV2Tests(unittest.TestCase):
    def test_feature_builder_uses_interval_rain_causally(self):
        frame = pd.DataFrame([
            {"run_id":"run", "station_id":"A", "valid_time":"2026-08-01T16:00:00Z", "initial_fm":10,
             "initial_age_hours":.5, "lead_hour":4, "rtma_temp_c":20, "rtma_rh":50, "rtma_wind_ms":2,
             "hrrr_temp_c":30, "hrrr_rh":30, "hrrr_wind_ms":3, "hrrr_precip_mm":1,
             "hrrr_precip_accum_mm":1, "hrrr_precip_increment_mm":1, "precip_interval_hours":1,
             "precip_available":1, "lat":38, "lon":-92},
            {"run_id":"run", "station_id":"A", "valid_time":"2026-08-01T17:00:00Z", "initial_fm":10,
             "initial_age_hours":.5, "lead_hour":5, "rtma_temp_c":20, "rtma_rh":50, "rtma_wind_ms":2,
             "hrrr_temp_c":31, "hrrr_rh":25, "hrrr_wind_ms":4, "hrrr_precip_mm":1,
             "hrrr_precip_accum_mm":1, "hrrr_precip_increment_mm":0, "precip_interval_hours":1,
             "precip_available":1, "lat":38, "lon":-92},
        ])
        result = build_features(frame)
        self.assertEqual(result.precip_3h_mm.tolist(), [1, 1])
        self.assertEqual(result.active_rain_indicator.tolist(), [1, 0])
        self.assertEqual(result.post_rain_3h_indicator.tolist(), [0, 1])
        self.assertEqual(result.temp_lead_change_c.tolist(), [0, 1])

    def test_verification_is_separate_and_same_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); evidence = root / "evidence"; raw = root / "raw"; evidence.mkdir(); raw.mkdir()
            valid = "2026-08-01T16:00:00+00:00"; row_key = f"run|A|{valid}"
            (evidence / "run.prediction.json").write_text(json.dumps({"run_id":"run", "row_keys":[row_key],
                                                                       "recorded_at":"2026-08-01T12:00:00Z"}))
            (raw / "raw_data_20260801.json").write_text(json.dumps({"STATION":[{"STID":"A", "OBSERVATIONS":{
                "date_time":[valid], "fuel_moisture_set_1":[7], "relative_humidity_set_1":[25], "wind_speed_set_1":[10]}}]}))
            report = verify_pending(evidence, raw, now="2026-08-01T18:00:00Z")
            self.assertEqual(report["attached"], 1); self.assertEqual(report["matched_rows"], 1)
            observation = json.loads((evidence / "run.observation.json").read_text())["observations"][row_key]
            self.assertEqual(observation["source_station_id"], "A")
            self.assertEqual(verify_pending(evidence, raw, now="2026-08-01T18:00:00Z")["attached"], 0)


if __name__ == "__main__": unittest.main()
