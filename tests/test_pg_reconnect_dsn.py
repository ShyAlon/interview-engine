"""Reconnect DSN carries the password (fe_sendauth regression).

psycopg's Connection.info.dsn never returns the password, so every reconnect
that reuses it (heartbeat spawn, staging connection, checkpoint write
connection, application pg connect) failed against a password-authenticated
server with 'fe_sendauth: no password supplied' - the 2026-10-25/26 start
failures. These tests pin the exact conninfo handed to the reconnect calls,
hermetically: no server, no live database.
"""
import psycopg.conninfo
import pytest

from interview_engine.pg_claims import reconnect_dsn
from interview_engine.pg_session_store import PgSessionStore

REAL_DSN = 'postgresql://engine_user@db.example.supabase.co:5432/postgres'
PASSWORD = 'p@ss word\'"\\ with:chars,=&'


class _Info:
    dsn = REAL_DSN
    password = PASSWORD


class _FakeConn:
    """Only what the reconnect paths touch: .info.dsn, .info.password, .closed."""

    def __init__(self, password=PASSWORD):
        self.info = type('I', (), {'dsn': REAL_DSN, 'password': password})()
        self.closed = False


def test_reconnect_dsn_adds_the_password():
    out = reconnect_dsn(_FakeConn())
    assert psycopg.conninfo.conninfo_to_dict(out)['password'] == PASSWORD


def test_reconnect_dsn_round_trips_special_characters():
    for password in ("plain", "with space", "with'quote", 'with"dquote',
                     'with\\backslash', 'with,comma=equals', PASSWORD):
        out = reconnect_dsn(_FakeConn(password))
        assert psycopg.conninfo.conninfo_to_dict(out)['password'] == password


def test_reconnect_dsn_without_password_returns_the_original():
    conn = _FakeConn('')
    assert reconnect_dsn(conn) == REAL_DSN
    conn = _FakeConn(None)
    assert reconnect_dsn(conn) == REAL_DSN


def _capture_connect(monkeypatch):
    seen = {}
    def fake_connect(dsn, **kwargs):
        seen['dsn'] = psycopg.conninfo.conninfo_to_dict(dsn)
        seen['kwargs'] = kwargs
        return _FakeConn()
    monkeypatch.setattr('interview_engine.pg_session_store.psycopg.connect', fake_connect)
    import interview_engine.pg_claims as claims
    monkeypatch.setattr(claims.psycopg, 'connect', fake_connect)
    return seen


def test_spawn_reconnect_carries_the_password(monkeypatch):
    seen = _capture_connect(monkeypatch)
    store = PgSessionStore(_FakeConn())
    store.spawn()
    assert seen['dsn'].get('password') == PASSWORD


def test_spawn_with_claims_reconnect_carries_the_password(monkeypatch):
    seen = _capture_connect(monkeypatch)
    store = PgSessionStore(_FakeConn(), claims=('owner', 'session'))
    store.spawn()
    assert seen['dsn'].get('password') == PASSWORD


def test_staging_connection_reconnect_carries_the_password(monkeypatch):
    seen = _capture_connect(monkeypatch)
    store = PgSessionStore(_FakeConn())
    store._stage_conn  # property: the reconnect happens on access
    assert seen['dsn'].get('password') == PASSWORD


def test_staging_connection_with_claims_reconnect_carries_the_password(monkeypatch):
    seen = _capture_connect(monkeypatch)
    store = PgSessionStore(_FakeConn(), claims=('owner', 'session'))
    store._stage_conn  # property: the reconnect happens on access
    assert seen['dsn'].get('password') == PASSWORD
