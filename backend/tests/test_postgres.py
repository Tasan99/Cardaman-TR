"""Use disposable PostgreSQL only; each test and migration are rolled back."""
import os
from pathlib import Path
import unittest
from uuid import uuid4

import psycopg


@unittest.skipUnless(os.getenv('TEST_DATABASE_URL'), 'Requires disposable PostgreSQL + pgvector')
class PostgresTests(unittest.TestCase):
    def setUp(self):
        self.conn = psycopg.connect(os.environ['TEST_DATABASE_URL'])
        self.addCleanup(self.conn.close)
        self.addCleanup(self.conn.rollback)
        ddl = Path(__file__).resolve().parents[1] / 'migrations' / '001_initial.sql'
        self.conn.execute(ddl.read_text(encoding='utf-8'))
        self.conn.execute('CREATE ROLE regchain_test_reader NOSUPERUSER NOBYPASSRLS')
        self.conn.execute('GRANT USAGE ON SCHEMA public TO regchain_test_reader')
        self.conn.execute('GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA public TO regchain_test_reader')
        self.a, self.b, self.company_a, self.company_b = [uuid4() for _ in range(4)]
        for tenant, company in [(self.a, self.company_a), (self.b, self.company_b)]:
            self.conn.execute('INSERT INTO tenants(id,name) VALUES(%s,%s)', (tenant, str(tenant)))
            self.conn.execute('INSERT INTO companies(id,tenant_id,name) VALUES(%s,%s,%s)', (company, tenant, str(company)))

    def scope(self, tenant):
        self.conn.execute('SET LOCAL ROLE regchain_test_reader')
        self.conn.execute("SELECT set_config('app.tenant_id',%s,true)", (str(tenant),))

    def test_no_tenant_is_fail_closed(self):
        self.conn.execute('SET LOCAL ROLE regchain_test_reader')
        self.assertEqual(self.conn.execute('SELECT count(*) FROM companies').fetchone()[0], 0)

    def test_tenant_isolation(self):
        self.scope(self.a)
        self.assertEqual(self.conn.execute('SELECT id FROM companies').fetchall(), [(self.company_a,)])

    def test_cross_tenant_insert_rejected(self):
        self.scope(self.a)
        with self.assertRaises(psycopg.errors.InsufficientPrivilege):
            self.conn.execute('INSERT INTO companies(tenant_id,name) VALUES(%s,%s)', (self.b, 'forged'))

    def test_cross_tenant_company_reference_rejected(self):
        self.scope(self.a)
        with self.assertRaises(psycopg.errors.ForeignKeyViolation):
            self.conn.execute('INSERT INTO company_profiles(tenant_id,company_id,version_no,profile_hash) VALUES(%s,%s,1,%s)', (self.a, self.company_b, 'a'*64))

    def test_history_is_immutable(self):
        profile = self.conn.execute('INSERT INTO company_profiles(tenant_id,company_id,version_no,profile_hash) VALUES(%s,%s,1,%s) RETURNING id', (self.a, self.company_a, 'a'*64)).fetchone()[0]
        with self.assertRaises(psycopg.errors.RaiseException):
            self.conn.execute('UPDATE company_profiles SET jurisdiction=%s WHERE id=%s', ('UK',profile))
