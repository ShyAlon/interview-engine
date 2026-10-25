"""Parallel nodes share the store's dedicated staging connection. Each staged
call runs one short transaction, and two overlapping Transaction contexts on
a single connection nest and fail out of order (psycopg
OutOfOrderTransactionNesting) - the CI regression concealed by full-suite
ordering. This test forces the overlap directly so no suite ordering can
mask it again."""
import threading

import pytest

pgserver = pytest.importorskip('pgserver', reason='pgserver not installed')
from conftest import managed_pg_server
import psycopg
import psycopg.rows

from interview_engine.pg_session_store import PgSessionStore
from test_pg_session_store import MIRRORED_REGISTRY_DDL, MIRRORED_STAGING_DDL

OWNER = '00000000-0000-0000-0000-000000000001'
CYCLES = 25


def _store(tmp_path):
    server = managed_pg_server(str(tmp_path / 'pgdata'))
    conn = psycopg.connect(server.get_uri(), autocommit=False, row_factory=psycopg.rows.dict_row)
    conn.execute(MIRRORED_REGISTRY_DDL)
    conn.execute(MIRRORED_STAGING_DDL)
    conn.commit()
    store = PgSessionStore(conn)
    store.create('s1', 'task1', OWNER)
    claim = store.claim('s1', OWNER, 'host#1', ttl=600)
    assert claim is not None
    return store, claim


def _staged_cycle(store, claim, node, n):
    store.record_dispatch(claim, execution_point='anchor:1', node=node, stage='llm', item='', ordinal=n,
                          request_sha256=f'sha-{node}-{n}', provider='fake', model='m1')
    store.bind_attempt(claim, execution_point='anchor:1', node=node, stage='llm', item='', ordinal=n,
                       attempt_id=f'att-{node}-{n}')
    store.complete_staged(claim, execution_point='anchor:1', node=node, stage='llm', item='',
                          ordinal=n, response={'ok': node}, usage={'total_tokens': 1})


def test_parallel_staging_transactions_on_the_shared_connection(tmp_path):
    store, claim = _store(tmp_path)
    barrier = threading.Barrier(2)
    failures = []

    def worker(node):
        try:
            barrier.wait(timeout=10)
            for n in range(CYCLES):
                _staged_cycle(store, claim, node, n)
        except Exception as exc:
            failures.append(exc)

    threads = [threading.Thread(target=worker, args=('x',)), threading.Thread(target=worker, args=('y',))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not failures, f"parallel staging failed: {failures[0]!r}"
    for node in ('x', 'y'):
        rows = store.staged_log('s1', claim.generation, execution_point='anchor:1', node=node, stage='llm')
        completed = [r for r in rows if r.get('response') is not None]
        assert len(completed) == CYCLES, f"node {node}: {len(completed)} completed of {CYCLES} - lost writes"
