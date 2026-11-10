"""pg_session_store proves the placement table is reachable once per pool, not once per registry.

delivery-api creates a registry (and so a store) on every answer step: 12 per warm answer, each probe a pool checkout,
claims, SELECT and commit - about five round trips on a ~164 ms link. The probe answers a pool-wide question (does the
table exist, may the role read it), so one success per pool is enough. Failure is never remembered."""
from contextlib import contextmanager
from types import SimpleNamespace

import psycopg.errors
import pytest

from interview_engine import tenant_worker


class Pool:
    pass


@pytest.fixture
def probes(monkeypatch):
    calls = []
    fail = {'with': None}

    @contextmanager
    def borrow(pool, claims):
        calls.append((pool, claims))
        if fail['with'] is not None:
            raise fail['with']
        yield SimpleNamespace(execute=lambda *a, **k: None, commit=lambda: None)

    monkeypatch.setattr('interview_engine.pg_pool.borrow_claimed', borrow)
    monkeypatch.setattr('interview_engine.pg_session_store.PgSessionStore', lambda pool, claims, close_pool=False: SimpleNamespace(pool=pool))
    monkeypatch.setattr(tenant_worker, '_probed_pools', type(tenant_worker._probed_pools)(), raising=False)
    return calls, fail


def test_probe_runs_once_per_pool(probes):
    calls, _ = probes
    pool = Pool()
    for sid in ('s1', 's2', 's3'):
        tenant_worker.pg_session_store(pool, 'owner', sid)
    assert len(calls) == 1


def test_each_pool_is_probed(probes):
    calls, _ = probes
    tenant_worker.pg_session_store(Pool(), 'owner', 's1')
    tenant_worker.pg_session_store(Pool(), 'owner', 's1')
    assert len(calls) == 2


def test_failure_is_raised_and_not_remembered(probes):
    calls, fail = probes
    pool = Pool()
    fail['with'] = psycopg.errors.UndefinedTable('engine_sessions')
    for _ in range(2):
        with pytest.raises(RuntimeError, match='placement migration'):
            tenant_worker.pg_session_store(pool, 'owner', 's1')
    assert len(calls) == 2
    fail['with'] = None
    tenant_worker.pg_session_store(pool, 'owner', 's1')
    tenant_worker.pg_session_store(pool, 'owner', 's2')
    assert len(calls) == 3
