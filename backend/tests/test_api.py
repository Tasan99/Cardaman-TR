import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from regchain.api import app

class ApiTests(unittest.TestCase):
    def test_liveness(self):
        response = TestClient(app).get('/health/live')
        self.assertEqual(response.json(), {"status":"ok"})
        self.assertIn('X-Request-ID', response.headers)

    def test_missing_database_is_not_ready(self):
        with patch.dict('os.environ', {}, clear=True):
            self.assertEqual(TestClient(app).get('/health/ready').status_code, 503)

    def test_regulatory_database_errors_are_sanitized(self):
        with patch.dict('os.environ', {}, clear=True):
            response = TestClient(app).get('/regulations')
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()['detail'], 'Regulatory database unavailable')

    def test_invalid_uuid_and_unbounded_page_rejected(self):
        client = TestClient(app)
        self.assertEqual(client.get('/versions/not-a-uuid').status_code,422)
        self.assertEqual(client.get('/regulations?limit=100000').status_code,422)

    def test_ingestion_is_not_a_public_write_endpoint(self):
        self.assertEqual(TestClient(app).post('/regulations',json={}).status_code,405)
