"""PR A parity proof: the delivery-api checkpoint migration must produce
EXACTLY the schema the pinned langgraph-checkpoint-postgres package's own
migrations produce - same tables, columns, indexes, and the recorded final
migration version - so the runtime never depends on the saver's DDL.
"""
import os
import sys
from pathlib import Path

import pytest

pgserver = pytest.importorskip('pgserver', reason='pgserver not installed: parity test skips')
from conftest import managed_pg_server
psycopg = pytest.importorskip('psycopg', reason='psycopg not installed')
pgcp = pytest.importorskip('langgraph.checkpoint.postgres', reason='langgraph-checkpoint-postgres not installed')
from langgraph.checkpoint.postgres import PostgresSaver

DELIVERY = os.getenv('DELIVERY_TEST_DELIVERY_SOURCE')
pytestmark = pytest.mark.skipif(not DELIVERY, reason='requires DELIVERY_TEST_DELIVERY_SOURCE (delivery-api checkout)')

_COLUMNS = ("SELECT table_name, column_name, data_type, is_nullable, coalesce(column_default,'')"
            " FROM information_schema.columns WHERE table_name LIKE 'checkpoint%' ORDER BY 1,2")
_INDEXES = ("SELECT tablename, indexname, regexp_replace(pg_get_indexdef("
            "(quote_ident(schemaname)||'.'||quote_ident(indexname))::regclass), '\\s+', ' ', 'g')"
            " FROM pg_indexes WHERE tablename LIKE 'checkpoint%' ORDER BY 1,2")


def _dump(conn):
    return (conn.execute(_COLUMNS).fetchall(), conn.execute(_INDEXES).fetchall())


def test_checkpoint_schema_matches_pinned_library(tmp_path):
    # Library-final schema on a scratch server, built by the pinned package.
    server = managed_pg_server(str(tmp_path / 'libpg'))
    lib = psycopg.connect(server.get_uri(), autocommit=True)
    PostgresSaver(lib).setup()
    lib_cols, lib_idx = _dump(lib)
    lib.close()

    # Migration-built schema from the real delivery-api migrations.
    sys.path.insert(0, str(Path(DELIVERY) / 'tests'))
    import pg_harness
    pg_harness.start()
    dsn = pg_harness.STATE.get('dsn')
    if not dsn:
        pytest.skip('pg_harness unavailable: ' + pg_harness.STATE.get('reason', ''))
    mig = psycopg.connect(dsn, autocommit=True)
    mig_cols, mig_idx = _dump(mig)

    assert mig_cols == lib_cols, 'column drift: only-in-migration=%s only-in-library=%s' % (
        sorted(set(mig_cols) - set(lib_cols)), sorted(set(lib_cols) - set(mig_cols)))
    assert mig_idx == lib_idx, 'index drift: only-in-migration=%s only-in-library=%s' % (
        sorted(set(mig_idx) - set(lib_idx)), sorted(set(lib_idx) - set(mig_idx)))
    version = mig.execute("SELECT max(v) FROM checkpoint_migrations").fetchone()[0]
    assert version == len(PostgresSaver.MIGRATIONS) - 1 == 9
