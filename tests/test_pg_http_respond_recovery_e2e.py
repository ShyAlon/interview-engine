"""HTTP respond recovery acceptance (PR B): the real registry respond entrypoint
behind the actual FastAPI surface over database persistence - mid-advance
provider crash, typed retryable error, heal + retry commits once, idempotent
replay, and a full process restart over the same database. No leg bypasses the
HTTP surface; the graph and models are the real interview graph with a scripted
fake provider."""
import time

import pytest

pgserver = pytest.importorskip('pgserver', reason='pgserver not installed')

from conftest import managed_pool
from vmodel_contracts.interactive import ProtocolErrorCode, ProcessStatus

from interview_engine.http_service import SessionRegistry, default_program_factory, create_app
from interview_engine.pg_session_store import PgSessionStore
from test_http_service import TOKEN, authed
from test_pg_application_wiring import make_conn, fake_ollama, fake_models, OWNER, SID, _OPEN


def _make(registry):
    _OPEN['registries'].append(registry)
    return authed(create_app(registry, service_token=TOKEN))


def _start(client):
    started = client.post('/v1/sessions', json={'protocol_version': 1, 'session_id': SID, 'task_id': 't',
                                                'idea': 'booking idea', 'model_route': 'ollama',
                                                'initial_model': 'phi4'})
    assert started.status_code == 201, started.text
    return started.json()['snapshot']


def _respond(client, input_id, value='customers book online'):
    return client.post(f'/v1/sessions/{SID}/inputs/{input_id}',
                       json={'protocol_version': 1, 'input_id': input_id, 'value': value, 'actor_id': 'founder'})


def test_http_respond_crash_retry_restart_and_idempotent_replay(tmp_path, fake_ollama, fake_models):
    conn = make_conn(tmp_path)
    store = PgSessionStore(managed_pool(conn.info.dsn))
    registry = SessionRegistry(default_program_factory, allow_db_provider_dispatch=True,
                               session_store=store, owner_id=OWNER, instance_id='w#1', lease_ttl=1.0)
    client = _make(registry)
    snapshot = _start(client)
    open_input = snapshot['open_input']
    prior = store.load_published(SID, OWNER)['control']['sequence']

    # 1. The provider crashes mid-advance: the client gets the typed retryable
    # error (never a masking PublicationConflict), the durable step stays put,
    # and the question stays open.
    class Boom(Exception):
        pass
    original_invoke = fake_models['probing'].invoke
    fake_models['probing'].invoke = lambda m: (_ for _ in ()).throw(Boom('provider crashed mid-advance'))
    failed = _respond(client, open_input['input_id'])
    assert failed.status_code == 503, failed.text
    detail = failed.json()['detail']
    assert detail['code'] == ProtocolErrorCode.INTERNAL_ERROR and detail['retryable'] is True
    state = store.load_published(SID, OWNER)
    assert state['control']['sequence'] == prior
    assert state['control']['open_input']['input_id'] == open_input['input_id']

    # 2. Heal: the same answer retries over HTTP and the step commits exactly once.
    fake_models['probing'].invoke = original_invoke
    retried = _respond(client, open_input['input_id'])
    assert retried.status_code == 200, retried.text
    after = store.load_published(SID, OWNER)['control']['sequence']
    assert after > prior

    # 3. Idempotent replay of the committed answer: accepted, no advance.
    replay = _respond(client, open_input['input_id'])
    assert replay.status_code == 200, replay.text
    assert store.load_published(SID, OWNER)['control']['sequence'] == after

    # 4. Full process boundary: new registry + app over the same database. The
    # committed answer replayed after restart is still idempotent (durable
    # response journal), the session keeps waiting on its CURRENT input, and a
    # fresh answer advances it.
    registry.close()
    time.sleep(1.2)  # let w#1's lease expire so w#2 can claim the restore
    registry2 = SessionRegistry(default_program_factory, allow_db_provider_dispatch=True,
                                session_store=PgSessionStore(managed_pool(conn.info.dsn)), owner_id=OWNER,
                                instance_id='w#2', lease_ttl=60.0)
    client2 = _make(registry2)
    restored = client2.get(f'/v1/sessions/{SID}')
    assert restored.status_code == 200, restored.text
    restored_snapshot = restored.json()
    assert restored_snapshot['process_status'] == ProcessStatus.WAITING_INPUT
    replayed_after_restart = _respond(client2, open_input['input_id'])
    assert replayed_after_restart.status_code == 200, replayed_after_restart.text
    assert store.load_published(SID, OWNER)['control']['sequence'] == after, \
        'a committed answer replayed after restart must never double-advance'

    current_input = restored_snapshot['open_input']
    assert current_input['input_id'] != open_input['input_id'], 'a fresh question must be open after the committed advance'
    advanced = _respond(client2, current_input['input_id'], value='they pick a time slot')
    assert advanced.status_code == 200, advanced.text
    assert store.load_published(SID, OWNER)['control']['sequence'] > after
