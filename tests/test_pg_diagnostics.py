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
