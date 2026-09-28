import unittest

from services.incident_confidence import HIGH_THRESHOLD, MODERATE_THRESHOLD, score_incident


def _member(minutes, source="ngfs", frp=20.0, confidence="nominal", land_cover=None):
    hour, minute = divmod(minutes, 60)
    return {"occurred_at": f"2026-09-28T{12 + hour:02d}:{minute:02d}:00Z", "source": source,
            "frp": frp, "confidence": confidence, "land_cover": land_cover}


def _fire(scans=10, **kwargs):
    return [_member(i * 15, **kwargs) for i in range(scans)]


class IncidentConfidenceTests(unittest.TestCase):
    def test_label_always_matches_score(self):
        for n in range(0, 12):
            result = score_incident({}, _fire(n) if n else [])
            expected = "high" if result["score"] >= HIGH_THRESHOLD else "moderate" if result["score"] >= MODERATE_THRESHOLD else "low"
            self.assertEqual(result["label"], expected)
            self.assertTrue(0 <= result["score"] <= 100)

    def test_persistent_multi_sensor_fire_scores_high_with_reasons(self):
        members = _fire(12, frp=50.0) + [_member(30, source="viirs", frp=40.0, confidence="high")]
        result = score_incident({}, members)
        self.assertEqual(result["label"], "high")
        self.assertTrue(any("more than one satellite" in r for r in result["reasons"]))
        self.assertTrue(any("scans" in r for r in result["reasons"]))

    def test_single_weak_detection_is_low(self):
        result = score_incident({}, [_member(0, frp=2.0, confidence="low")])
        self.assertEqual(result["label"], "low")

    def test_more_scans_never_lowers_score(self):
        scores = [score_incident({}, _fire(n))["score"] for n in range(1, 12)]
        self.assertEqual(scores, sorted(scores))

    def test_second_sensor_never_lowers_score(self):
        base = _fire(6)
        self.assertGreaterEqual(
            score_incident({}, base + [_member(20, source="viirs")])["score"],
            score_incident({}, base)["score"],
        )

    def test_cropland_penalizes(self):
        plain = score_incident({}, _fire(6))["score"]
        farm = score_incident({}, _fire(6, land_cover="Cropland:100"))["score"]
        self.assertLess(farm, plain)

    def test_heavy_cropland_cannot_reach_high_and_long_lived_is_flagged(self):
        farm = score_incident({}, _fire(16, frp=60.0, land_cover="Cropland:95"))
        self.assertNotEqual(farm["label"], "high")
        weeks = _fire(6) + [{"occurred_at": "2026-10-20T12:00:00Z", "source": "ngfs", "frp": 20.0, "confidence": "nominal"}]
        result = score_incident({}, weeks)
        self.assertTrue(any("recurring heat source" in f["label"] for f in result["factors"]))

    def test_public_feedback_moves_score(self):
        members = _fire(6)
        base = score_incident({}, members)["score"]
        confirmed = score_incident({"approved_feedback_counts": {"confirmed_fire": 1}}, members)["score"]
        denied = score_incident({"approved_feedback_counts": {"not_a_fire": 1}}, members)["score"]
        self.assertGreater(confirmed, base)
        self.assertLess(denied, base)

    def test_handles_empty_and_malformed_members(self):
        self.assertEqual(score_incident({}, [])["label"], "low")
        result = score_incident({}, [{"occurred_at": "garbage", "frp": None}])
        self.assertIn(result["label"], {"low", "moderate", "high"})


if __name__ == "__main__":
    unittest.main()
