"""HTTP/registry session-contract sweep in database persistence mode.

These are the same session contracts the in-memory suite proves in
test_http_service.py - typed start, input idempotency, stale-input conflict,
typed completion, retryable failure, restart restore and stop - run against a
SessionRegistry backed by PgSessionStore on a throwaway Postgres, so the
protocol surface behaves identically on the new persistence path.
"""
from uuid import uuid4

import pytest

pgserver = pytest.importorskip('pgserver', reason='pgserver not installed: PG sweep tests skip')
from conftest import managed_pg_server

import psycopg
import psycopg.rows
from fastapi.testclient import TestClient
from vmodel_contracts.interactive import (
    EventType, InputKind, InputOption, InputRequest, InputResponse, NextStep,
    ProcessStatus, ProtocolErrorCode, StartSessionRequest,
)

from interview_engine.http_service import SessionRegistry, create_app
from interview_engine.pg_session_store import PgSessionStore
from test_http_service import TOKEN, FakeProgram, FailOnceProgram, authed
from test_pg_session_store import MIRRORED_REGISTRY_DDL, MIRRORED_JOURNAL_DDL

OWNER = '00000000-0000-0000-0000-000000000001'
_OPEN = {'conns': [], 'registries': [], 'servers': []}


@pytest.fixture(autouse=True)
def _cleanup_pg():
    yield
    for r in _OPEN['registries']:
        r.close()
    for c in _OPEN['conns']:
        try:
            c.close()
        except Exception:
            pass
    for k in _OPEN:
        _OPEN[k].clear()


def make_conn(tmp_path, name='pgdata', init=True):
    server = managed_pg_server(str(tmp_path / name))
    _OPEN['servers'].append(server)
    conn = psycopg.connect(server.get_uri(), autocommit=False, row_factory=psycopg.rows.dict_row)
    if init:
        conn.execute(MIRRORED_REGISTRY_DDL)
        conn.execute(MIRRORED_JOURNAL_DDL)
        conn.commit()
    _OPEN['conns'].append(conn)
    return conn


def make_registry(conn, factory=None, ttl=60.0):
    registry = SessionRegistry(factory, session_store=PgSessionStore(conn),
                               owner_id=OWNER, instance_id='sweep#1', lease_ttl=ttl)
    _OPEN['registries'].append(registry)
    return registry


def start_json(client, **over):
    payload = {'protocol_version': 1, 'task_id': 't', 'idea': 'i', 'model_route': 'm'}
    payload.update(over)
    return client.post('/v1/sessions', json=payload)


def test_start_returns_typed_contract(tmp_path):
    client = authed(create_app(make_registry(make_conn(tmp_path)), service_token=TOKEN))
    r = start_json(client)
    assert r.status_code == 201
    body = r.json()
    assert body['accepted_event']['event_type'] == EventType.RUN_STARTED
    assert body['snapshot']['process_status'] == ProcessStatus.RUNNING


def test_program_driven_to_typed_input_and_completion(tmp_path):
    program = FakeProgram()
    registry = make_registry(make_conn(tmp_path), lambda session_id, request: program)
    client = authed(create_app(registry, service_token=TOKEN))
    started = start_json(client).json()
    assert started['snapshot']['process_status'] == ProcessStatus.WAITING_INPUT
    pending = started['snapshot']['open_input']
    reply = {'protocol_version': 1, 'input_id': pending['input_id'], 'value': 'players', 'actor_id': 'founder'}
    done = client.post(f"/v1/sessions/{started['snapshot']['session_id']}/inputs/{pending['input_id']}", json=reply).json()
    assert done['snapshot']['process_status'] == ProcessStatus.COMPLETED
    row = PgSessionStore(_OPEN['conns'][-1]).load_published(started['snapshot']['session_id'], OWNER)
    assert row['control']['status'] == int(ProcessStatus.COMPLETED)


def test_input_replay_is_idempotent(tmp_path):
    registry = make_registry(make_conn(tmp_path))
    client = authed(create_app(registry, service_token=TOKEN))
    sid = start_json(client).json()['snapshot']['session_id']
    pending = InputRequest(input_id=uuid4(), kind=InputKind.APPROVAL, prompt='Approve intent?',
                           options=[InputOption(value=1, label='Approve'), InputOption(value=0, label='Reject')])
    registry.publish(sid, EventType.INPUT_REQUESTED, phase_id=10, status=ProcessStatus.WAITING_INPUT,
                     next_step=NextStep.WAIT_INPUT, input_request=pending)
    body = {'protocol_version': 1, 'input_id': str(pending.input_id), 'value': 1, 'actor_id': 'founder'}
    first = client.post(f'/v1/sessions/{sid}/inputs/{pending.input_id}', json=body)
    replay = client.post(f'/v1/sessions/{sid}/inputs/{pending.input_id}', json=body)
    assert first.status_code == 200 and replay.status_code == 200


def test_stale_input_conflict(tmp_path):
    registry = make_registry(make_conn(tmp_path))
    client = authed(create_app(registry, service_token=TOKEN))
    sid = start_json(client).json()['snapshot']['session_id']
    current = InputRequest(input_id=uuid4(), kind=InputKind.TEXT, prompt='Who uses it?')
    registry.publish(sid, EventType.INPUT_REQUESTED, phase_id=10, status=ProcessStatus.WAITING_INPUT,
                     next_step=NextStep.WAIT_INPUT, input_request=current)
    stale = uuid4()
    r = client.post(f'/v1/sessions/{sid}/inputs/{stale}', json={
        'protocol_version': 1, 'input_id': str(stale), 'value': 'players', 'actor_id': 'founder'})
    assert r.status_code == 409 and r.json()['detail']['code'] == ProtocolErrorCode.STALE_INPUT


def test_failed_advance_is_typed_retryable(tmp_path):
    program = FailOnceProgram()
    registry = make_registry(make_conn(tmp_path), lambda session_id, request: program)
    client = authed(create_app(registry, service_token=TOKEN))
    started = start_json(client).json()
    pending = started['snapshot']['open_input']
    response = {'protocol_version': 1, 'input_id': pending['input_id'], 'value': 'players', 'actor_id': 'founder'}
    failed = client.post(f"/v1/sessions/{started['snapshot']['session_id']}/inputs/{pending['input_id']}", json=response)
    assert failed.status_code == 503 and failed.json()['detail']['retryable'] is True
    retried = client.post(f"/v1/sessions/{started['snapshot']['session_id']}/inputs/{pending['input_id']}", json=response)
    assert retried.status_code == 200 and retried.json()['snapshot']['process_status'] == ProcessStatus.COMPLETED


def test_restart_over_same_database_restores_open_input(tmp_path):
    conn_a = make_conn(tmp_path)
    first_program = FakeProgram()
    first = make_registry(conn_a, lambda session_id, request: first_program)
    session, _ = first.start(StartSessionRequest(task_id='t', idea='i', model_route='m'))
    pending = session.open_input
    first.close()

    second_program = FakeProgram()
    restarted = make_registry(make_conn(tmp_path, init=False), lambda session_id, request: second_program)
    restored = restarted.get(session.session_id)
    assert restored.open_input == pending and restored.status == ProcessStatus.WAITING_INPUT
    _, snapshot = restarted.respond(session.session_id,
                                    InputResponse(input_id=pending.input_id, value='players', actor_id='founder'))
    assert snapshot.process_status == ProcessStatus.COMPLETED


def test_stop_blocks_inputs_and_persists(tmp_path):
    registry = make_registry(make_conn(tmp_path), lambda session_id, request: FakeProgram())
    client = authed(create_app(registry, service_token=TOKEN))
    started = start_json(client).json()
    sid = started['snapshot']['session_id']
    pending = started['snapshot']['open_input']
    registry.stop(sid)
    row = PgSessionStore(_OPEN['conns'][-1]).load_published(sid, OWNER)
    assert row['control']['stopped'] is True
    r = client.post(f"/v1/sessions/{sid}/inputs/{pending['input_id']}", json={
        'protocol_version': 1, 'input_id': pending['input_id'], 'value': 'players', 'actor_id': 'founder'})
    assert r.status_code == 422 and r.json()['detail']['code'] == ProtocolErrorCode.INVALID_RESPONSE


def test_http_cross_owner_session_invisible(tmp_path):
    # Full-entrypoint tenant boundary: a session row owned by another owner is
    # invisible through the real HTTP app + auth route, not only registry calls.
    OWNER_B = '00000000-0000-0000-0000-000000000002'
    conn = make_conn(tmp_path)
    store = PgSessionStore(conn)
    store.create('s-foreign', 'task-b', OWNER_B)
    # Published for its owner: the 404 below is owner scoping, not an
    # unpublished-row artifact.
    claim = store.claim('s-foreign', OWNER_B, 'foreign#1', ttl=60)
    assert claim is not None
    store.publish_step(claim, {'session_id': 's-foreign', 'sequence': 1, 'status': 1})
    client = authed(create_app(make_registry(conn), service_token=TOKEN))  # serves OWNER
    r = client.get('/v1/sessions/s-foreign')
    assert r.status_code == 404, r.text
    r = client.get('/v1/sessions/s-foreign/progress')
    assert r.status_code == 404, r.text


def test_http_unpublished_row_never_leaks(tmp_path):
    # A row that was created but has no published step must not leak through
    # the ordinary session or progress APIs.
    conn = make_conn(tmp_path)
    PgSessionStore(conn).create('s-draft', 'task-d', OWNER)
    client = authed(create_app(make_registry(conn), service_token=TOKEN))
    assert client.get('/v1/sessions/s-draft').status_code == 404
    assert client.get('/v1/sessions/s-draft/progress').status_code == 404
