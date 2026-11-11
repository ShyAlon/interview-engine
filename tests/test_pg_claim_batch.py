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
