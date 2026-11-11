import pytest

def test_entry_verified_skips_only_construction_probe(monkeypatch):
    from interview_engine import tenant_worker
    from interview_engine import pg_session_store as module
    class Pool:pass
    pool=Pool();calls=[]
    monkeypatch.setattr(module,'PgSessionStore',lambda pool,**kw:calls.append(kw) or kw)
    from interview_engine import pg_pool
    def forbidden(*a,**kw):raise AssertionError('duplicate preflight')
    monkeypatch.setattr(pg_pool,'borrow_claimed',forbidden)
    got=tenant_worker.pg_session_store(pool,'owner','auth',entry_verified=True)
    assert got['claims']==('owner','auth')
    assert pool not in tenant_worker._probed_pools
    with pytest.raises(RuntimeError,match='AssertionError'):
        tenant_worker.pg_session_store(pool,'owner','auth')
