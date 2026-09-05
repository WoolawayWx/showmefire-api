import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
from services import verification_feedback as f


class FeedbackTests(unittest.TestCase):
    def test_generate_and_reload_without_changing_report(self):
        report = {'date': '2026-01-01', 'metrics': {}, 'record_count': 1}
        with tempfile.TemporaryDirectory() as root, patch.object(f, 'REPORTS_DIR', Path(root)), patch.object(f, '_read_summary', return_value=report), patch.object(f, '_load_history', return_value=[]), patch.object(f, 'CloudflareAIClient', return_value=Mock(configured=True)), patch.object(f, 'generate_verification_summary', return_value='Today: verification feedback'):
            result = f.generate_feedback(report['date'])
            self.assertEqual(result['status'], 'ready')
            self.assertEqual(f.feedback_status(report['date'])['ai_summary'], result['ai_summary'])
            self.assertNotIn('ai_summary', report)
            report['record_count'] = 2
            self.assertEqual(f.feedback_status(report['date'])['status'], 'not_generated')

    def test_failed_generation_does_not_publish_or_erase_feedback(self):
        report = {'date': '2026-01-01', 'metrics': {}}
        with tempfile.TemporaryDirectory() as root, patch.object(f, 'REPORTS_DIR', Path(root)), patch.object(f, '_read_summary', return_value=report), patch.object(f, '_load_history', return_value=[]), patch.object(f, 'CloudflareAIClient', return_value=Mock(configured=True)), patch.object(f, 'generate_verification_summary', side_effect=RuntimeError('upstream unavailable')):
            with self.assertRaises(RuntimeError): f.generate_feedback(report['date'])
            self.assertFalse(list(Path(root).rglob('ai_feedback.json')))

    def test_missing_credentials_are_visible(self):
        with patch.object(f, '_read_summary', return_value={'date': '2026-01-01'}), patch.object(f, 'CloudflareAIClient', return_value=Mock(configured=False)):
            self.assertEqual(f.feedback_status('2026-01-01')['status'], 'not_configured')
            with self.assertRaises(RuntimeError): f.generate_feedback('2026-01-01')


if __name__ == '__main__':
    unittest.main()
