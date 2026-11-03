import pytest

import interview_engine.model_config as mc
@pytest.fixture(autouse=True)
def _reset_llm_call_budget(monkeypatch):
    """Each test is one run: the run-wide LLM call budget starts clean."""
    monkeypatch.setattr(mc, "_CALLS_USED", 0)


@pytest.fixture(autouse=True)
def _no_dotenv_files(monkeypatch):
    """No test reads a developer's .env files: model routes come from the code defaults and the test's own settings.
    A test that needs a .env builds its own and resets mc._DOTENV itself."""
    monkeypatch.setattr(mc, "_DOTENV", {})


_PG_SERVERS = []
_PG_POOLS = []


def managed_pool(dsn, max_size=4, name='test'):
    """One psycopg_pool per throwaway server, tracked for per-test shutdown
    (closed BEFORE the server underneath goes away, mirroring managed_pg_server)."""
    from interview_engine.pg_pool import build_pool
    pool = build_pool(dsn, max_size=max_size, name=name)
    _PG_POOLS.append(pool)
    return pool


def managed_pg_server(path):
    """pgserver.get_server tracked for per-test shutdown. Each test's tmp_path
    yields a NEW postgres instance; with only atexit cleanup a suite run kept
    every instance alive concurrently and OOMed the host (~100 orphaned
    postmasters). Per-test cleanup keeps concurrent instances bounded to one
    test's own servers."""
    import pgserver
    server = pgserver.get_server(str(path))
    _PG_SERVERS.append(server)
    # Mirrored-schema unit fixtures still model the runtime SET role. Real
    # delivery schema tests use delivery's separate migration-owned harness.
    import psycopg
    with psycopg.connect(server.get_uri(),autocommit=True) as conn:
        if not conn.execute("SELECT 1 FROM pg_roles WHERE rolname='requi_backend'").fetchone():
            conn.execute('CREATE ROLE requi_backend NOLOGIN NOSUPERUSER NOBYPASSRLS')
        conn.execute('GRANT USAGE ON SCHEMA public TO requi_backend')
        conn.execute('ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO requi_backend')
        conn.execute('ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE ON SEQUENCES TO requi_backend')
    return server


@pytest.fixture(autouse=True)
def _cleanup_pg_servers():
    yield
    while _PG_POOLS:
        pool = _PG_POOLS.pop()
        try:
            pool.close()
        except Exception:
            pass
    while _PG_SERVERS:
        server = _PG_SERVERS.pop()
        try:
            server.cleanup()
        except Exception:
            pass


@pytest.fixture
def pg_registry(tmp_path):
    """Real disposable PG registry port with explicit owner and lifecycle."""
    import psycopg,psycopg.rows
    from test_pg_session_store import MIRRORED_REGISTRY_DDL,MIRRORED_JOURNAL_DDL
    from interview_engine.pg_session_store import PgSessionStore
    from interview_engine.http_service import SessionRegistry
    server=managed_pg_server(tmp_path/'registry-pg')
    conn=psycopg.connect(server.get_uri(),row_factory=psycopg.rows.dict_row)
    conn.execute(MIRRORED_REGISTRY_DDL);conn.execute(MIRRORED_JOURNAL_DDL);conn.commit()
    pool=managed_pool(conn.info.dsn);regs=[]
    owner='00000000-0000-0000-0000-000000000001'
    store=PgSessionStore(pool)
    def make(*args,**kwargs):
        kwargs.setdefault('session_store',store);kwargs.setdefault('owner_id',owner)
        kwargs.setdefault('instance_id','fixture#'+str(len(regs)))
        registry=SessionRegistry(*args,**kwargs);regs.append(registry);return registry
    make.store=store;make.owner=owner;make.conn=conn
    yield make
    for registry in regs:registry.close()
    conn.close()
