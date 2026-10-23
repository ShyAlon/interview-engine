"""Startup session-persistence selection: database is the default.

With INTERVIEW_ENGINE_SESSIONS_DSN set, session persistence uses the
delivery-api-owned Postgres registry. Without it the boot fails clear and
names the explicit file opt-in (INTERVIEW_ENGINE_SESSIONS_FILE=1): file
persistence is never a silent fallback. An unreachable database, a missing
placement migration, or a single-tenant deployment without an owner scope
also stop the boot instead of failing mid-interview.

Schema-present cases run against the REAL delivery-api migrated schema (its
own pg_harness, no DDL mirror) when the checkout is provided via
DELIVERY_TEST_DELIVERY_SOURCE, and skip otherwise.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

pgserver = pytest.importorskip('pgserver', reason='pgserver not installed: startup database-mode tests skip')
from conftest import managed_pg_server

import psycopg
import psycopg.rows

DELIVERY = os.getenv('DELIVERY_TEST_DELIVERY_SOURCE')
requires_delivery = pytest.mark.skipif(not DELIVERY, reason='requires DELIVERY_TEST_DELIVERY_SOURCE (delivery-api checkout)')

OWNER = '00000000-0000-0000-0000-000000000001'

PROBE = """
import os
from interview_engine import http_service as hs
sessions = hs._build_service_app().state.sessions
store = getattr(sessions, '_store', None)
print('type=%s store=%s owner=%s state_dir=%s' % (
    type(sessions).__name__,
    type(store).__name__ if store is not None else None,
    getattr(sessions, '_owner_id', None),
    getattr(sessions, '_state_dir', None)))
"""

PROBE_TENANT = """
import os
from interview_engine import http_service as hs
router = hs._build_service_app().state.sessions
print('type=%s dsn_set=%s' % (type(router).__name__, bool(getattr(router, '_session_dsn', None))))
"""


def run_startup(env, probe=PROBE):
    full = {k: v for k, v in os.environ.items() if not k.startswith('INTERVIEW_ENGINE_')}
    full.update(env)
    return subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True, env=full, timeout=120)


@pytest.fixture(scope='module')
def migrated_dsn():
    """The real delivery-api schema, applied by its own pg_harness."""
    if not DELIVERY:
        pytest.skip('requires DELIVERY_TEST_DELIVERY_SOURCE (delivery-api checkout)')
    sys.path.insert(0, str(Path(DELIVERY) / 'tests'))
    import pg_harness
    pg_harness.start()
    dsn = pg_harness.STATE.get('dsn')
    if not dsn:
        pytest.skip('pg_harness unavailable: ' + pg_harness.STATE.get('reason', ''))
    yield dsn


def test_startup_without_dsn_or_explicit_file_fails_clear():
    result = run_startup({})
    assert result.returncode != 0
    assert 'database session persistence is the default' in result.stderr
    assert 'INTERVIEW_ENGINE_SESSIONS_FILE=1' in result.stderr


def test_file_mode_runs_when_explicitly_chosen():
    result = run_startup({'INTERVIEW_ENGINE_SESSIONS_FILE': '1'})
    assert result.returncode == 0, result.stderr
    assert 'type=SessionRegistry store=None' in result.stdout
    assert 'state_dir=None' not in result.stdout  # state_dir is the artifacts root


def test_database_mode_wires_store_and_owner(migrated_dsn):
    result = run_startup({'INTERVIEW_ENGINE_SESSIONS_DSN': migrated_dsn, 'INTERVIEW_ENGINE_OWNER_ID': OWNER})
    assert result.returncode == 0, result.stderr
    assert 'type=SessionRegistry store=PgSessionStore' in result.stdout
    assert ('owner=%s' % OWNER) in result.stdout


def test_database_mode_single_tenant_requires_owner_scope():
    # The owner check fails before any database contact, so no server is needed.
    result = run_startup({'INTERVIEW_ENGINE_SESSIONS_DSN': 'postgresql://127.0.0.1:1/none'})
    assert result.returncode != 0
    assert 'INTERVIEW_ENGINE_OWNER_ID is required' in result.stderr


def test_database_mode_fails_clear_without_placement_migration(tmp_path):
    server = managed_pg_server(str(tmp_path / 'pgdata'))
    result = run_startup({'INTERVIEW_ENGINE_SESSIONS_DSN': server.get_uri(), 'INTERVIEW_ENGINE_OWNER_ID': OWNER})
    assert result.returncode != 0
    assert '20261022090000_session_placement_leases.sql has not run' in result.stderr


def test_database_mode_fails_clear_when_database_unreachable():
    result = run_startup({'INTERVIEW_ENGINE_SESSIONS_DSN': 'postgresql://127.0.0.1:1/none', 'INTERVIEW_ENGINE_OWNER_ID': OWNER})
    assert result.returncode != 0
    assert 'database is unreachable' in result.stderr


def test_tenant_mode_passes_dsn_to_router(migrated_dsn):
    result = run_startup({
        'INTERVIEW_ENGINE_TENANTS': '1',
        'INTERVIEW_ENGINE_TOKEN': 't#startup-test-token',
        'INTERVIEW_ENGINE_SESSIONS_DSN': migrated_dsn,
    }, probe=PROBE_TENANT)
    assert result.returncode == 0, result.stderr
    assert 'type=TenantRegistryRouter dsn_set=True' in result.stdout


def test_tenant_mode_without_dsn_or_explicit_file_fails_clear():
    result = run_startup({'INTERVIEW_ENGINE_TENANTS': '1', 'INTERVIEW_ENGINE_TOKEN': 't#startup-test-token'}, probe=PROBE_TENANT)
    assert result.returncode != 0
    assert 'database session persistence is the default' in result.stderr


def test_tenant_mode_without_dsn_keeps_file_registries_when_explicit():
    result = run_startup({'INTERVIEW_ENGINE_TENANTS': '1', 'INTERVIEW_ENGINE_TOKEN': 't#startup-test-token',
                          'INTERVIEW_ENGINE_SESSIONS_FILE': '1'}, probe=PROBE_TENANT)
    assert result.returncode == 0, result.stderr
    assert 'type=TenantRegistryRouter dsn_set=False' in result.stdout


def test_tenant_router_builds_database_registries_per_owner(migrated_dsn):
    from interview_engine import tenant_worker
    from interview_engine.http_service import SessionRegistry
    router = tenant_worker.TenantRegistryRouter(SessionRegistry, lambda *a: None, Path('/tmp/unused'), session_dsn=migrated_dsn)
    token = tenant_worker._owner.set(OWNER)
    try:
        registry = router._registry()
    finally:
        tenant_worker._owner.reset(token)
    assert registry._store is not None
    assert registry._owner_id == OWNER
    assert registry._state_dir is None
    assert registry._allow_db_dispatch is True
    # Tenant claims travel in the conninfo, so row security admits this owner only.
    row = registry._store.conn.execute("select current_setting('request.jwt.claim.sub', true) as claim").fetchone()
    assert row['claim'] == OWNER


def test_owner_scoped_dsn_normalizes_uri_and_keyword_forms():
    from interview_engine.tenant_worker import owner_scoped_dsn
    for dsn in ('postgresql://u@h:5432/db', 'host=x dbname=y user=z'):
        scoped = owner_scoped_dsn(dsn, OWNER)
        assert 'request.jwt.claim.sub=%s' % OWNER in scoped
        assert 'request.jwt.claim.session_id=%s' % OWNER in scoped
        assert 'postgres://' not in scoped  # normalized to keyword conninfo


def test_tenant_router_without_dsn_requires_explicit_file_opt_in(tmp_path):
    from interview_engine import tenant_worker
    from interview_engine.http_service import SessionRegistry
    router = tenant_worker.TenantRegistryRouter(SessionRegistry, lambda *a: None, tmp_path)
    token = tenant_worker._owner.set(OWNER)
    try:
        with pytest.raises(RuntimeError, match='database session persistence is the default'):
            router._registry()
    finally:
        tenant_worker._owner.reset(token)


def test_tenant_router_file_mode_unchanged(tmp_path):
    from interview_engine import tenant_worker
    from interview_engine.http_service import SessionRegistry
    router = tenant_worker.TenantRegistryRouter(SessionRegistry, lambda *a: None, tmp_path, allow_file_sessions=True)
    token = tenant_worker._owner.set(OWNER)
    try:
        registry = router._registry()
    finally:
        tenant_worker._owner.reset(token)
    assert registry._store is None
    assert registry._state_dir is not None
