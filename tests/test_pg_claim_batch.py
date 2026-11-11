"""Claims must stay local and preserve error rollback while sharing one barrier."""
import os
import pytest
from conftest import managed_pg_server


def test_claims_one_wire_barrier_and_transaction_reset(tmp_path):
    import psycopg
    from interview_engine.pg_claims import apply_claims
    server=managed_pg_server(tmp_path/'claims-pg')
    with psycopg.connect(server.get_uri(),autocommit=True) as conn:
        login=conn.execute('SELECT current_user').fetchone()[0]
        conn.execute('BEGIN')
        fd=os.memfd_create('claim-protocol',os.MFD_CLOEXEC)
        try:
            conn.pgconn.trace(fd)
            apply_claims(conn,('owner-a','session-a'))
            conn.pgconn.untrace();os.lseek(fd,0,0)
            wire=os.read(fd,65536)
            assert sum(b'\tF\t' in row and (b'\tSync' in row or b'\tQuery\t' in row) for row in wire.splitlines())==1
        finally:conn.pgconn.untrace();os.close(fd)
        assert conn.execute("SELECT current_user,current_setting('request.jwt.claim.sub'),current_setting('request.jwt.claim.session_id')").fetchone()==('requi_backend','owner-a','session-a')
        conn.execute('ROLLBACK')
        assert conn.execute("SELECT current_user,coalesce(current_setting('request.jwt.claim.sub',true),'')").fetchone()==(login,'')
        conn.execute('BEGIN');apply_claims(conn,('owner-b','session-b'));conn.execute('COMMIT')
        assert conn.execute("SELECT current_user,coalesce(current_setting('request.jwt.claim.session_id',true),'')").fetchone()==(login,'')


def test_first_claimed_query_shares_identity_barrier(tmp_path):
    import psycopg
    from interview_engine.pg_claims import ClaimedConnection
    server=managed_pg_server(tmp_path/'first-query-pg')
    with psycopg.connect(server.get_uri(),autocommit=False) as raw:
        conn=ClaimedConnection(raw,'owner-a','session-a')
        fd=os.memfd_create('first-query-protocol',os.MFD_CLOEXEC)
        try:
            raw.pgconn.trace(fd)
            result=conn.execute("SELECT current_user,current_setting('request.jwt.claim.sub')").fetchone()
            raw.pgconn.untrace();os.lseek(fd,0,0);wire=os.read(fd,65536)
            assert result==('requi_backend','owner-a')
            assert sum(b'\tF\t' in row and (b'\tSync' in row or b'\tQuery\t' in row) for row in wire.splitlines())==2  # implicit BEGIN, then the combined claims/query batch
        finally:raw.pgconn.untrace();os.close(fd)


def test_explicit_claimed_transaction_setup_is_one_barrier(tmp_path):
    import psycopg
    from interview_engine.pg_claims import ClaimedConnection
    server=managed_pg_server(tmp_path/'explicit-transaction-pg')
    with psycopg.connect(server.get_uri(),autocommit=False) as raw:
        conn=ClaimedConnection(raw,'owner-a','session-a')
        fd=os.memfd_create('transaction-protocol',os.MFD_CLOEXEC)
        try:
            raw.pgconn.trace(fd)
            with conn.transaction():
                raw.pgconn.untrace();os.lseek(fd,0,0);wire=os.read(fd,65536)
                assert sum(b'\tF\t' in row and (b'\tSync' in row or b'\tQuery\t' in row) for row in wire.splitlines())==1
                assert conn.execute("SELECT current_user,current_setting('request.jwt.claim.sub')").fetchone()==('requi_backend','owner-a')
        finally:raw.pgconn.untrace();os.close(fd)


def test_nested_claimed_transactions_keep_savepoint_rollback(tmp_path):
    import psycopg
    from interview_engine.pg_claims import ClaimedConnection
    server=managed_pg_server(tmp_path/'nested-pg')
    with psycopg.connect(server.get_uri(),autocommit=False) as raw:
        raw.execute('CREATE TABLE nested_probe(v int)');raw.commit()
        conn=ClaimedConnection(raw,'owner','auth')
        with conn.transaction():
            conn.execute('INSERT INTO nested_probe VALUES(1)')
            with pytest.raises(ValueError):
                with conn.transaction():
                    conn.execute('INSERT INTO nested_probe VALUES(2)')
                    raise ValueError('undo nested')
            assert conn.execute('SELECT v FROM nested_probe').fetchall()==[(1,)]
        assert raw.autocommit is False
        assert raw.execute('SELECT v FROM nested_probe').fetchall()==[(1,)]
