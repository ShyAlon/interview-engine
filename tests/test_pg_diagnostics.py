"""The operator 503 block is sanitized by construction: these tests pin the
redaction shape, the honest UNKNOWNs, and the no-claim-values rule before the
feature ships."""
import psycopg
import pytest

from interview_engine import pg_diagnostics
from interview_engine.pg_diagnostics import UNKNOWN, build_operator_block, sanitize_message, tag_stage


def test_sanitize_message_redacts_quoted_identifiers_keeps_class_and_type():
    text = 'new row violates row-level security policy for table "engine_sessions"'
    out = sanitize_message(text)
    assert 'row-level security policy' in out
    assert 'table' in out
    assert 'engine_sessions' not in out
    assert '"[identifier]"' in out


def test_sanitize_message_redacts_bare_uuids_and_collapses_whitespace():
    text = 'claim 3fa85f64-5717-4562-b3fc-2c963f66afa6\n\tfailed'
    out = sanitize_message(text)
    assert '3fa85f64' not in out
    assert '[uuid]' in out
    assert '\n' not in out


def test_sanitize_message_bounded_and_empty_safe():
    assert sanitize_message('') == ''
    assert sanitize_message(None) == ''
    assert len(sanitize_message('x' * 1000)) <= 300


def test_block_without_connection_records_unknowns_and_never_values():
    exc = psycopg.errors.InsufficientPrivilege('permission denied for table "engine_sessions"')
    block = build_operator_block(exc)
    assert block['error_class'] == 'InsufficientPrivilege'
    assert block['sqlstate'] == '42501'
    assert block['stage'] is None
    assert block['role'] == UNKNOWN
    assert block['sub_present'] == UNKNOWN
    assert block['session_id_present'] == UNKNOWN
    assert block['session_active_for_sub'] == UNKNOWN
    assert 'engine_sessions' not in block['message']


def test_stage_and_connection_travel_on_the_error_through_wrappers():
    class FakeConn:
        claims = ('owner-sub-value', 'session-id-value')
        def execute(self, sql):
            raise RuntimeError('connection is aborted')
    inner = psycopg.errors.InsufficientPrivilege('denied')
    tag_stage(inner, 'session insert', FakeConn())
    wrapped = RuntimeError('engine start wrapper')
    wrapped.__cause__ = inner
    block = build_operator_block(wrapped)
    assert block['stage'] == 'session insert'
    assert block['sqlstate'] == '42501'
    # A claimed connection reports presence from what it carries, never the values.
    assert block['sub_present'] is True
    assert block['session_id_present'] is True
    assert block['session_active_for_sub'] == UNKNOWN
    assert block['role'] == UNKNOWN  # the aborted connection cannot answer
    blob = str(block)
    assert 'owner-sub-value' not in blob and 'session-id-value' not in blob


def test_plain_connection_query_failure_yields_unknowns():
    class DeadConn:
        def execute(self, sql):
            raise psycopg.OperationalError('connection closed')
    exc = psycopg.errors.InsufficientPrivilege('denied')
    tag_stage(exc, 'claim setup', DeadConn())
    block = build_operator_block(exc)
    assert block['stage'] == 'claim setup'
    assert block['role'] == UNKNOWN
    assert block['sub_present'] == UNKNOWN
    assert block['session_id_present'] == UNKNOWN


def test_block_is_json_ascii_safe_and_bounded():
    import base64, json
    exc = psycopg.errors.InsufficientPrivilege('denied for table "t" near uuid 3fa85f64-5717-4562-b3fc-2c963f66afa6')
    raw = json.dumps(build_operator_block(exc), separators=(',', ':')).encode()
    assert len(raw) <= 1800
    base64.urlsafe_b64encode(raw).decode('ascii')


# --- HTTP-level integration: the block must actually travel on the start 503
# response header, through the real app, store and handler - and customer
# error contracts (409, minimal 503 detail) must stay untouched.

import base64
import json
from uuid import uuid4

import psycopg.rows
from fastapi.testclient import TestClient
from vmodel_contracts.interactive import ProtocolErrorCode

from interview_engine.http_service import SessionRegistry, create_app
from interview_engine.pg_session_store import PgSessionStore

_TOKEN = 'test-service-token'
_OWNER = '00000000-0000-0000-0000-000000000001'


def _pg_registry(tmp_path):
    from conftest import managed_pg_server
    from test_pg_session_store import MIRRORED_REGISTRY_DDL, MIRRORED_JOURNAL_DDL
    server = managed_pg_server(str(tmp_path / 'pgdata'))
    conn = psycopg.connect(server.get_uri(), autocommit=False, row_factory=psycopg.rows.dict_row)
    conn.execute(MIRRORED_REGISTRY_DDL)
    conn.execute(MIRRORED_JOURNAL_DDL)
    conn.commit()
    registry = SessionRegistry(session_store=PgSessionStore(conn), owner_id=_OWNER,
                               instance_id='t#diag', lease_ttl=60.0)
    return registry, conn


def test_start_503_carries_sanitized_operator_header(tmp_path):
    """A real database failure during start surfaces the operator block on the
    response header: tagged stage, real sqlstate, sanitized message - while the
    customer-facing detail keeps its minimal-disclosure shape."""
    registry, conn = _pg_registry(tmp_path)
    try:
        # Force a genuine NotNullViolation in the session insert.
        conn.execute("ALTER TABLE engine_sessions ADD COLUMN forced_failure TEXT NOT NULL")
        conn.commit()
        client = TestClient(create_app(registry, service_token=_TOKEN),
                            headers={'X-Interview-Engine-Token': _TOKEN})
        r = client.post('/v1/sessions', json={'protocol_version': 1, 'task_id': 't',
                                              'idea': 'i', 'model_route': 'm'})
        assert r.status_code == 503
        detail = r.json()['detail']
        assert detail['code'] == ProtocolErrorCode.INTERNAL_ERROR
        assert detail['retryable'] is True
        assert detail['detail'] == 'engine program failed during start: NotNullViolation'
        header = r.headers.get('x-engine-operator-diagnostics')
        assert header, 'operator diagnostics header missing on start 503'
        assert len(header) <= 2400  # bounded by construction before base64
        block = json.loads(base64.urlsafe_b64decode(header))
        assert block['error_class'] == 'NotNullViolation'
        assert block['sqlstate'] == '23502'
        assert block['stage'] == 'session insert'
        # Sanitized by construction: no relation/column identifiers, no values.
        assert 'engine_sessions' not in block['message']
        assert 'forced_failure' not in block['message']
        assert '[identifier]' in block['message']
        # Presence fields are booleans or an honest UNKNOWN - never values.
        assert block['sub_present'] in (True, False, 'UNKNOWN')
        assert block['session_id_present'] in (True, False, 'UNKNOWN')
        assert block['session_active_for_sub'] in (True, False, 'UNKNOWN')
        assert isinstance(block['role'], str) and block['role']
    finally:
        registry.close()
        conn.close()


def test_start_conflict_409_carries_no_operator_header(tmp_path):
    """A typed protocol failure (session ID conflict) is a 409 with the plain
    contract - the operator header is added only on unexpected 503s."""
    registry, conn = _pg_registry(tmp_path)
    try:
        sid = str(uuid4())
        conn.execute(
            "INSERT INTO engine_sessions(session_id,owner_id,task_id,generation,created_at,lease_epoch,lease_expires_at)"
            " VALUES (%s,%s,%s,1,1,0,0)",
            (sid, '00000000-0000-0000-0000-000000000002', 'other-task'))
        conn.commit()
        client = TestClient(create_app(registry, service_token=_TOKEN),
                            headers={'X-Interview-Engine-Token': _TOKEN})
        r = client.post('/v1/sessions', json={'protocol_version': 1, 'task_id': 't',
                                              'idea': 'i', 'model_route': 'm',
                                              'session_id': sid})
        assert r.status_code == 409
        assert 'x-engine-operator-diagnostics' not in r.headers
    finally:
        registry.close()
        conn.close()
