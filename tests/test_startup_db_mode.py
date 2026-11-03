"""Startup session-persistence selection: database is the default.

With INTERVIEW_ENGINE_SESSIONS_DSN set, session persistence uses the
delivery-api-owned Postgres registry. Without it the boot fails clear and
refuses even the retired file opt-in: file persistence is never a fallback. An unreachable database, a missing
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
# Matches the seeded live auth session for OWNER in delivery's pg_harness (distinct
# from the owner id: session authority is never the owner UUID).
SESSION = '00000000-0000-0000-0000-0000000000a1'

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
    assert 'PostgreSQL session persistence is required' in result.stderr
    assert 'INTERVIEW_ENGINE_SESSIONS_DSN' in result.stderr


def test_retired_file_mode_refuses_even_when_explicitly_chosen():
    result = run_startup({'INTERVIEW_ENGINE_SESSIONS_FILE': '1'})
    assert result.returncode != 0
    assert 'PostgreSQL session persistence is required' in result.stderr


def test_database_mode_wires_store_and_owner(migrated_dsn):
    result = run_startup({'INTERVIEW_ENGINE_SESSIONS_DSN': migrated_dsn, 'INTERVIEW_ENGINE_OWNER_ID': OWNER, 'INTERVIEW_ENGINE_SESSION_ID': SESSION})
    assert result.returncode == 0, result.stderr
    assert 'type=ScopedInstalledRegistry store=PgSessionStore' in result.stdout
    assert ('owner=%s' % OWNER) in result.stdout


def test_database_mode_single_tenant_requires_owner_scope():
    # The owner check fails before any database contact, so no server is needed.
    result = run_startup({'INTERVIEW_ENGINE_SESSIONS_DSN': 'postgresql://127.0.0.1:1/none'})
    assert result.returncode != 0
    assert 'INTERVIEW_ENGINE_OWNER_ID is required' in result.stderr


def test_database_mode_single_tenant_requires_session_scope():
    # The session check also fails before any database contact: synthetic
    # session authority is never substituted for a real auth session.
    result = run_startup({'INTERVIEW_ENGINE_SESSIONS_DSN': 'postgresql://127.0.0.1:1/none', 'INTERVIEW_ENGINE_OWNER_ID': OWNER})
    assert result.returncode != 0
    assert 'INTERVIEW_ENGINE_SESSION_ID is required' in result.stderr


def test_database_mode_fails_clear_without_placement_migration(tmp_path):
    server = managed_pg_server(str(tmp_path / 'pgdata'))
    with psycopg.connect(server.get_uri(),autocommit=True) as conn:
        conn.execute('CREATE ROLE requi_backend NOLOGIN')
    result = run_startup({'INTERVIEW_ENGINE_SESSIONS_DSN': server.get_uri(), 'INTERVIEW_ENGINE_OWNER_ID': OWNER, 'INTERVIEW_ENGINE_SESSION_ID': SESSION})
    assert result.returncode != 0
    assert '20261022090000_session_placement_leases.sql has not run' in result.stderr


def test_database_mode_fails_clear_when_database_unreachable():
    result = run_startup({'INTERVIEW_ENGINE_SESSIONS_DSN': 'postgresql://127.0.0.1:1/none', 'INTERVIEW_ENGINE_OWNER_ID': OWNER, 'INTERVIEW_ENGINE_SESSION_ID': SESSION})
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
    assert 'PostgreSQL session persistence is required' in result.stderr


def test_tenant_mode_without_dsn_refuses_old_explicit_file_flag():
    result = run_startup({'INTERVIEW_ENGINE_TENANTS': '1', 'INTERVIEW_ENGINE_TOKEN': 't#startup-test-token',
                          'INTERVIEW_ENGINE_SESSIONS_FILE': '1'}, probe=PROBE_TENANT)
    assert result.returncode != 0
    assert 'PostgreSQL session persistence is required' in result.stderr


def test_tenant_router_builds_database_registries_per_owner(migrated_dsn):
    from interview_engine import tenant_worker
    from interview_engine.http_service import SessionRegistry
    router = tenant_worker.TenantRegistryRouter(SessionRegistry, lambda *a: None, Path('/tmp/unused'), session_dsn=migrated_dsn)
    token = tenant_worker._owner.set(OWNER)
    stoken = tenant_worker._session.set(SESSION)
    try:
        registry = router._registry()
    finally:
        tenant_worker._session.reset(stoken)
        tenant_worker._owner.reset(token)
    assert registry._store is not None
    assert registry._owner_id == OWNER
    assert registry._state_dir is None
    assert registry._allow_db_dispatch is True
    # Tenant claims ride each transaction (pg_claims), never the conninfo, so
    # row security admits this owner only, and only while this auth session
    # stays live.
    with registry._store._borrow() as c:
        row = c.execute("select current_setting('request.jwt.claim.sub', true) as sub, current_setting('request.jwt.claim.session_id', true) as sid").fetchone()
    assert row['sub'] == OWNER
    assert row['sid'] == SESSION


def test_pg_session_store_claims_per_transaction_never_conninfo(migrated_dsn):
    # The claim transport is per-transaction set_config (pg_claims): the
    # conninfo carries no options payload at all, because poolers drop
    # startup-option GUCs (2026-10-25 live probe: claims invisible in both
    # sampled transactions through the Supabase pooler).
    from interview_engine.tenant_worker import pg_session_store
    from conftest import managed_pool
    store = pg_session_store(managed_pool(migrated_dsn), OWNER, SESSION)
    assert 'request.jwt' not in store.pool.conninfo
    assert 'options' not in store.pool.conninfo
    with store._borrow() as c:
        row = c.execute("select current_setting('request.jwt.claim.sub', true) as sub,"
                        " current_setting('request.jwt.claim.session_id', true) as sid").fetchone()
        assert row['sub'] == OWNER
        assert row['sid'] == SESSION
        c.commit()
        # Claims are transaction-local: after COMMIT they are gone, and the
        # next transaction re-applies them before any statement.
        row = c.execute("select current_setting('request.jwt.claim.sub', true) as sub").fetchone()
        assert row['sub'] == OWNER
        c.rollback()


def test_database_registry_requires_session_scope(migrated_dsn):
    # Router-level defense in depth: a database-backed registry is never built
    # without a session scope, even if the middleware were bypassed.
    from interview_engine import tenant_worker
    from interview_engine.http_service import SessionRegistry
    router = tenant_worker.TenantRegistryRouter(SessionRegistry, lambda *a: None, Path('/tmp/unused'), session_dsn=migrated_dsn)
    token = tenant_worker._owner.set(OWNER)
    try:
        with pytest.raises(PermissionError, match='worker session scope required'):
            router._registry()
    finally:
        tenant_worker._owner.reset(token)


def test_parallel_sessions_of_one_owner_coexist(migrated_dsn):
    # Two simultaneous valid logins of the same owner get separate registries;
    # neither closes the other's connections.
    from interview_engine import tenant_worker
    from interview_engine.http_service import SessionRegistry
    other = '00000000-0000-0000-0000-0000000000a2'
    import psycopg as _pg
    with _pg.connect(migrated_dsn, autocommit=True) as conn:
        conn.execute("INSERT INTO auth.sessions(id,user_id) VALUES (%s,%s) ON CONFLICT (id) DO NOTHING", (other, OWNER))
    router = tenant_worker.TenantRegistryRouter(SessionRegistry, lambda *a: None, Path('/tmp/unused'), session_dsn=migrated_dsn)
    token = tenant_worker._owner.set(OWNER)
    try:
        stoken = tenant_worker._session.set(SESSION)
        try:first = router._registry()
        finally:tenant_worker._session.reset(stoken)
        stoken = tenant_worker._session.set(other)
        try:second = router._registry()
        finally:tenant_worker._session.reset(stoken)
        assert first is not second
        assert len(router._registries) == 2
        stoken = tenant_worker._session.set(SESSION)
        try:assert router._registry() is first
        finally:tenant_worker._session.reset(stoken)
    finally:
        tenant_worker._owner.reset(token)
        router.close()


def test_tenant_router_without_dsn_refuses(tmp_path):
    from interview_engine import tenant_worker
    from interview_engine.http_service import SessionRegistry
    router = tenant_worker.TenantRegistryRouter(SessionRegistry, lambda *a: None, tmp_path)
    token = tenant_worker._owner.set(OWNER)
    try:
        with pytest.raises(RuntimeError, match='PostgreSQL session persistence is required'):
            router._registry()
    finally:
        tenant_worker._owner.reset(token)



def test_relogin_builds_fresh_registry_and_writes_succeed(migrated_dsn):
    # Re-login recovery: the next request carries the new session id, the
    # router builds a fresh registry under the new claims, and its writes pass
    # the placement policy as delivery_app - the old registry is untouched.
    from interview_engine import tenant_worker
    from interview_engine.http_service import SessionRegistry
    import psycopg as _pg
    relogin = '00000000-0000-0000-0000-0000000000b1'
    with _pg.connect(migrated_dsn, autocommit=True) as conn:
        conn.execute("INSERT INTO auth.sessions(id,user_id) VALUES (%s,%s)", (relogin, OWNER))
        conn.execute("INSERT INTO tasks(task_id,title,state,created_at,updated_at,owner_id) VALUES ('relogin-task','relogin-title','pending',0,0,%s) ON CONFLICT (task_id) DO NOTHING", (OWNER,))
    app_dsn = migrated_dsn.replace('user=sandbox', 'user=delivery_app')
    router = tenant_worker.TenantRegistryRouter(SessionRegistry, lambda *a: None, Path('/tmp/unused'), session_dsn=app_dsn)
    token = tenant_worker._owner.set(OWNER)
    try:
        stoken = tenant_worker._session.set(SESSION)
        try:
            first = router._registry()
        finally:
            tenant_worker._session.reset(stoken)
        stoken = tenant_worker._session.set(relogin)
        try:
            second = router._registry()
            assert second is not first
            second._store.create('relogin-sess', 'relogin-task', OWNER)
        finally:
            tenant_worker._session.reset(stoken)
    finally:
        tenant_worker._owner.reset(token)
        router.close()
