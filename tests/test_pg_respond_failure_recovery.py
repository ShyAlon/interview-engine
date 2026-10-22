"""Diagnosis + regression: registry.respond failure semantics when an advance
crashes AFTER the response acceptance committed. Reproduced on merged-A code
(http_service.py is byte-identical 2719232..HEAD)."""
import time
import pytest

pgserver = pytest.importorskip('pgserver', reason='pgserver not installed')

from vmodel_contracts.interactive import StartSessionRequest, InputResponse
from interview_engine.http_service import SessionRegistry, default_program_factory
from interview_engine.pg_session_store import PgSessionStore, PublicationConflict
from test_pg_application_wiring import make_conn, fake_ollama, fake_models, FakeModel, OWNER, SID, _OPEN


def test_respond_crash_mid_advance_original_error_and_clean_retry(tmp_path, fake_ollama, fake_models):
    """Failure BEFORE the step commit (provably: publish_step is only reachable
    from registry methods, never inside _advance): the ORIGINAL provider error
    must surface (never a masking PublicationConflict), durable control stays at
    the prior published step with the question open, and after the model heals
    the SAME answer retries successfully - the response idempotency cache
    reconciles, no double advance."""
    conn = make_conn(tmp_path)
    store = PgSessionStore(conn)
    registry = SessionRegistry(default_program_factory, allow_db_provider_dispatch=True, session_store=store, owner_id=OWNER,
                               instance_id='w#1', lease_ttl=60.0)
    _OPEN['registries'].append(registry)
    session, _ = registry.start(StartSessionRequest(session_id=SID, task_id='t', idea='idea',
                                                    model_route='ollama', initial_model='phi4'))
    prior = store.load_published(SID, OWNER)['control']['sequence']

    class Boom(Exception):
        pass
    original_invoke = fake_models['probing'].invoke
    def crashing_invoke(messages):
        raise Boom('crash during provider call')
    fake_models['probing'].invoke = crashing_invoke

    open_input = session.open_input
    answer = InputResponse(input_id=open_input.input_id, actor_id='founder', value='customers book online')
    with pytest.raises(Boom):
        registry.respond(SID, answer)
    state = store.load_published(SID, OWNER)
    assert state['control']['sequence'] == prior, 'a pre-commit failure must not move the durable step'
    assert state['control']['open_input']['input_id'] == str(open_input.input_id), 'question must stay open'

    # Heal the model: the same answer retries and the step commits exactly once.
    fake_models['probing'].invoke = original_invoke
    event, snapshot = registry.respond(SID, answer)
    after = store.load_published(SID, OWNER)
    assert after['control']['sequence'] > prior
    # Idempotent replay of the same response is accepted without re-advancing.
    event2, _ = registry.respond(SID, answer)
    assert store.load_published(SID, OWNER)['control']['sequence'] == after['control']['sequence']


def test_respond_crash_after_restart_same_behavior(tmp_path, fake_ollama, fake_models):
    """The failure path survives a process boundary: crash mid-advance, restore on
    a new registry, re-answer, heal, retry - same durable guarantees."""
    conn = make_conn(tmp_path)
    store = PgSessionStore(conn)
    registry = SessionRegistry(default_program_factory, allow_db_provider_dispatch=True, session_store=store, owner_id=OWNER,
                               instance_id='w#1', lease_ttl=1.0)
    _OPEN['registries'].append(registry)
    session, _ = registry.start(StartSessionRequest(session_id=SID, task_id='t', idea='idea',
                                                    model_route='ollama', initial_model='phi4'))
    prior = store.load_published(SID, OWNER)['control']['sequence']

    class Boom(Exception):
        pass
    original_invoke = fake_models['probing'].invoke
    fake_models['probing'].invoke = lambda m: (_ for _ in ()).throw(Boom())
    open_input = session.open_input
    answer = InputResponse(input_id=open_input.input_id, actor_id='founder', value='customers book online')
    with pytest.raises(Boom):
        registry.respond(SID, answer)
    assert store.load_published(SID, OWNER)['control']['sequence'] == prior

    registry.close()
    time.sleep(1.2)
    registry2 = SessionRegistry(default_program_factory, allow_db_provider_dispatch=True, session_store=PgSessionStore(conn),
                                owner_id=OWNER, instance_id='w#2', lease_ttl=60.0)
    _OPEN['registries'].append(registry2)
    restored = registry2.get(SID)
    assert restored.open_input is not None and str(restored.open_input.input_id) == str(open_input.input_id)
    with pytest.raises(Boom):
        registry2.respond(SID, answer)
    assert store.load_published(SID, OWNER)['control']['sequence'] == prior
    fake_models['probing'].invoke = original_invoke
    registry2.respond(SID, answer)
    assert store.load_published(SID, OWNER)['control']['sequence'] > prior
