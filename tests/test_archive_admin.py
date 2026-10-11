import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from core import security
from routers import archive_admin
from services.archive_store import ArchiveStore


class ArchiveAdminTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = ArchiveStore(Path(self.tmp.name))
        app = FastAPI()
        app.include_router(archive_admin.router)
        self.client = TestClient(app)

    def test_requires_authentication(self):
        # archive_admin's routes now authenticate via Depends(require_admin),
        # resolved against core.security.require_admin at route-registration
        # time - require_admin calls verify_token(token) by bare name against
        # core.security's own globals at request time, so that's what needs
        # patching (patching archive_admin.verify_token no longer exists/applies).
        with patch.object(security, 'verify_token', return_value=None):
            self.assertEqual(self.client.get('/api/admin/archive/status').status_code, 401)
            self.assertEqual(self.client.get('/api/admin/archive/processing').status_code, 401)
            self.assertEqual(self.client.post('/api/admin/archive/sync').status_code, 401)

    def test_bounded_status_and_missing_dates(self):
        with patch.object(security, 'verify_token', return_value='admin'), patch.object(archive_admin, 'ArchiveStore', return_value=self.store):
            response = self.client.get('/api/admin/archive/status?days=7')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(response.json()['days']), 7)
            self.assertEqual(self.client.get('/api/admin/archive/status?days=10000').status_code, 422)


if __name__ == '__main__':
    unittest.main()
