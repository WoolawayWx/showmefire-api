import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from fastapi import HTTPException
from routers import verification as v


class VerificationHistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name, value in [('REPORTS_DIR', self.root), ('HISTORY_FILE', self.root / 'validation_history.json')]:
            patcher = patch.object(v, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def report(self, day='2026-01-01', generated='2026-01-01T22:30:00Z'):
        return {'date': day, 'generated_at': generated, 'record_count': 12,
                'metrics': {'Temperature (C)': {'mae': 2.0}}}

    def write_daily(self, report):
        path = self.root / report['date'] / 'validation_summary.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report))

    def test_early_historical_report_is_visible_and_flagged(self):
        v.HISTORY_FILE.write_text(json.dumps([self.report()]))
        rows = v.get_verification_history(90)['dates']
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['window_status'], 'early')
        self.assertEqual(v._read_summary('2026-01-01')['record_count'], 12)

    def test_daily_reports_recover_missing_or_corrupt_index(self):
        self.write_daily(self.report())
        self.assertEqual(len(v._load_history()), 1)
        v.HISTORY_FILE.write_text('{broken')
        with self.assertLogs(v.logger, level='WARNING'):
            self.assertEqual(len(v._load_history()), 1)

    def test_daily_summary_overrides_stale_index_without_duplicate(self):
        v.HISTORY_FILE.write_text(json.dumps([self.report()]))
        daily = self.report(generated='2026-01-02T05:00:00Z')
        daily['record_count'] = 24
        self.write_daily(daily)
        rows = v.get_verification_history(90)['dates']
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['record_count'], 24)
        self.assertEqual(rows[0]['window_status'], 'closed')

    def test_future_reports_do_not_consume_history_limit(self):
        v.HISTORY_FILE.write_text(json.dumps([self.report('2099-01-01'), self.report()]))
        rows = v.get_verification_history(1)['dates']
        self.assertEqual([row['date'] for row in rows], ['2026-01-01'])

    def test_compact_date_directory_and_invalid_path(self):
        report = self.report('20260101')
        self.write_daily(report)
        self.assertEqual(v._load_history()[0]['date'], '2026-01-01')
        self.assertEqual(v._read_summary('2026-01-01')['date'], '2026-01-01')
        with self.assertRaises(HTTPException):
            v._read_summary('../secret')

    def test_canonical_daily_directory_wins_over_legacy_duplicate(self):
        self.write_daily(self.report('20260101'))
        canonical = self.report(generated='2026-01-02T05:00:00Z')
        canonical['record_count'] = 24
        self.write_daily(canonical)
        self.assertEqual(v._load_history()[0]['record_count'], 24)
        self.assertEqual(v._read_summary('2026-01-01')['record_count'], 24)

    def test_history_is_parsed_again_only_when_a_report_changes(self):
        self.write_daily(self.report())
        with patch.object(v, '_parse_history', wraps=v._parse_history) as parse:
            self.assertEqual(len(v._load_history()), 1)
            self.assertEqual(len(v._load_history()), 1)
            self.assertEqual(parse.call_count, 1)
            self.write_daily(self.report('2026-01-02'))
            self.assertEqual(len(v._load_history()), 2)
            self.assertEqual(parse.call_count, 2)

    def test_peak_history_reads_each_archived_file_once_and_only_the_days_it_returns(self):
        dirs = {name: self.root / name for name in ('forecast', 'station', 'rtma')}
        for directory in dirs.values():
            directory.mkdir()
            for day in range(1, 6):
                (directory / f'2026-01-0{day}.tif').write_bytes(b'tif')
        reads = []

        def fake_single(path):
            reads.append(path.name)
            return 2

        def fake_rgba(path, colors):
            reads.append(path.name)
            return 1

        with patch.object(v, '_forecast_peak_archive_dir', return_value=dirs['forecast']), \
                patch.object(v, '_station_peak_archive_dir', return_value=dirs['station']), \
                patch.object(v, '_rtma_peak_archive_dir', return_value=dirs['rtma']), \
                patch.object(v, '_max_class_single_band', side_effect=fake_single), \
                patch.object(v, '_max_class_rgba', side_effect=fake_rgba), \
                patch.object(v, '_peak_class_cache', {}):
            days = v.get_peak_verification_history(limit=2)['days']
            self.assertEqual([d['date'] for d in days], ['2026-01-04', '2026-01-05'])
            self.assertEqual(days[0]['station_bias'], 1)
            self.assertEqual(sorted(set(reads)), ['2026-01-04.tif', '2026-01-05.tif'])
            self.assertEqual(len(reads), 6)

            v.get_peak_verification_history(limit=2)
            self.assertEqual(len(reads), 6)  # served from the cache

            (dirs['rtma'] / '2026-01-05.tif').write_bytes(b'rewritten')
            v.get_peak_verification_history(limit=2)
            self.assertEqual(len(reads), 7)  # only the changed file is read again

    def test_original_closed_window_checks_still_reject_early_generation(self):
        now = datetime(2026, 1, 2, 12, tzinfo=timezone.utc)
        self.assertFalse(v._report_covers_closed_window(self.report(), now=now))
        self.assertTrue(v._report_covers_closed_window(self.report(generated='2026-01-02T05:00:00Z'), now=now))
        self.assertFalse(v._report_is_available(self.report('2099-01-01')))


if __name__ == '__main__':
    unittest.main()
