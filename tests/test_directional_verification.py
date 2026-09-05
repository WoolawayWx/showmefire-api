import unittest
from services.verification_metrics import directional_metrics, summarize


class DirectionalTests(unittest.TestCase):
    def test_matrix_orientation_and_critical_denominators(self):
        m = [[0]*5 for _ in range(5)]
        m[1][3] = 2  # Two overforecasts and critical false alarms.
        m[3][1] = 1  # One underforecast and missed critical.
        m[3][3] = 1
        result = directional_metrics({'confusion_matrix': {'matrix': m}})['Fire Danger Index']
        self.assertEqual(result['count'], 4)
        self.assertEqual(result['bias'], .5)
        self.assertEqual(result['over_rate'], .5)
        self.assertEqual(result['under_rate'], .25)
        self.assertEqual(result['critical_miss_rate'], .5)
        self.assertAlmostEqual(result['critical_false_alarm_rate'], 2/3)
        self.assertEqual(result['large_under_count'], 1)

    def test_balanced_bias_does_not_hide_offsetting_misses(self):
        result = summarize([(0, 2, 1), (4, 2, 1)], True)
        self.assertEqual(result['direction'], 'balanced')
        self.assertEqual(result['exact_rate'], 0)
        self.assertEqual(result['over_rate'], .5)
        self.assertIsNone(result['critical_miss_rate'])

    def test_invalid_values_and_aggregate_fallback(self):
        self.assertIsNone(summarize([(None, 1, 1), (float('nan'), 2, 1)]))
        r = directional_metrics({'metrics': {'Fire Danger Index': {'count': 12, 'bias': -.2}}})['Fire Danger Index']
        self.assertEqual(r['direction'], 'under')
        self.assertNotIn('over_rate', r)

    def test_full_rows_are_used_beyond_table_limit(self):
        row = {'forecast': {'temperature_c': 3}, 'observed': {'temperature_c': 1}}
        result = directional_metrics({'comparison_rows': [row]*600})['Temperature (C)']
        self.assertEqual(result['count'], 600)
        self.assertEqual(result['bias'], 2)


if __name__ == '__main__':
    unittest.main()
