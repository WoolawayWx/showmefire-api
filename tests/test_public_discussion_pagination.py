import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from core.database import list_forecast_discussions


class PublicDiscussionPaginationTests(unittest.TestCase):
    def test_all_public_history_is_pageable_without_exposing_drafts(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'test.db'
            with sqlite3.connect(path) as db:
                db.execute('CREATE TABLE forecast_discussions(id INTEGER, status TEXT, issued_at TEXT, created_at TEXT)')
                db.executemany('INSERT INTO forecast_discussions VALUES (?,?,?,?)',
                    [(i, 'archived' if i % 2 else 'published', '2026-09-01', '2026-09-01') for i in range(1, 251)] +
                    [(999, 'draft', '2026-09-05', '2026-09-05')])
            with patch('core.database.get_db_path', return_value=str(path)):
                pages = [list_forecast_discussions(public_only=True, limit=100, offset=i) for i in (0, 100, 200, 300)]
            self.assertEqual([len(page) for page in pages], [100, 100, 50, 0])
            ids = [row['id'] for page in pages for row in page]
            self.assertEqual(ids, list(range(250, 0, -1)))
            self.assertNotIn(999, ids)


if __name__ == '__main__':
    unittest.main()
