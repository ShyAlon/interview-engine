"""Diagnosis + regression: registry.respond failure semantics when an advance
crashes AFTER the response acceptance committed. Reproduced on merged-A code
(http_service.py is byte-identical 2719232..HEAD)."""
import time
import pytest

pgserver = pytest.importorskip('pgserver', reason='pgserver not installed')

from vmodel_contracts.interactive import StartSessionRequest, InputResponse
from interview_engine.http_service import SessionRegistry, default_program_factory
from interview_engine.pg_session_store import PgSessionStore, PublicationConflict
from conftest import managed_pool
from test_pg_application_wiring import make_conn, fake_ollama, fake_models, FakeModel, OWNER, SID, _OPEN


def test_respond_crash_mid_advance_original_error_and_clean_retry(tmp_path, fake_ollama, fake_models):
    """Failure BEFORE the step commit (provably: publish_step is only reachable
    from registry methods, never inside _advance): the ORIGINAL provider error
    must surface (never a masking PublicationConflict), durable control stays at
    the prior published step with the question open, and after the model heals
    the SAME answer retries successfully - the response idempotency cache
    reconciles, no double advance."""
    conn = make_conn(tmp_path)
    store = PgSessionStore(managed_pool(conn.info.dsn))
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
    store = PgSessionStore(managed_pool(conn.info.dsn))
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
    registry2 = SessionRegistry(default_program_factory, allow_db_provider_dispatch=True, session_store=PgSessionStore(managed_pool(conn.info.dsn)),
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


def test_respond_stale_input_fails_before_advance(tmp_path, fake_ollama, fake_models):
    """Failure location 1 - before acceptance/advance: a stale input id fails as a
    protocol error; nothing moves, no model call."""
    from interview_engine.http_service import ProtocolFailure
    conn = make_conn(tmp_path)
    store = PgSessionStore(managed_pool(conn.info.dsn))
    registry = SessionRegistry(default_program_factory, allow_db_provider_dispatch=True, session_store=store, owner_id=OWNER,
                               instance_id='w#1', lease_ttl=60.0)
    _OPEN['registries'].append(registry)
    session, _ = registry.start(StartSessionRequest(session_id=SID, task_id='t', idea='idea',
                                                    model_route='ollama', initial_model='phi4'))
    prior = store.load_published(SID, OWNER)['control']['sequence']
    probing_before = len(fake_models['probing'].replies)
    with pytest.raises(ProtocolFailure):
        registry.respond(SID, InputResponse(input_id='00000000-0000-0000-0000-000000000000',
                                            actor_id='founder', value='x'))
    assert store.load_published(SID, OWNER)['control']['sequence'] == prior
    assert len(fake_models['probing'].replies) == probing_before


def test_respond_tail_precommit_failure_evicts_and_retry_advances_once(tmp_path, fake_ollama, fake_models, monkeypatch):
    """Failure location 2 - the step transaction fails BEFORE commit: the injected
    error surfaces, durable stays at the prior step, and the session is evicted so
    the retry cannot be short-circuited by stale in-memory response idempotency;
    the reloaded session re-advances exactly once."""
    conn = make_conn(tmp_path)
    store = PgSessionStore(managed_pool(conn.info.dsn))
    registry = SessionRegistry(default_program_factory, allow_db_provider_dispatch=True, session_store=store, owner_id=OWNER,
                               instance_id='w#1', lease_ttl=60.0)
    _OPEN['registries'].append(registry)
    session, _ = registry.start(StartSessionRequest(session_id=SID, task_id='t', idea='idea',
                                                    model_route='ollama', initial_model='phi4'))
    prior = store.load_published(SID, OWNER)['control']['sequence']
    answer = InputResponse(input_id=session.open_input.input_id, actor_id='founder', value='customers book online')

    class CommitRefused(Exception):
        pass
    real_publish = store.publish_step
    monkeypatch.setattr(store, 'publish_step', lambda *a, **k: (_ for _ in ()).throw(CommitRefused('db down before commit')))
    with pytest.raises(CommitRefused):
        registry.respond(SID, answer)
    assert store.load_published(SID, OWNER)['control']['sequence'] == prior
    assert SID not in registry._sessions, 'session serving from disproved memory must be evicted'

    monkeypatch.setattr(store, 'publish_step', real_publish)  # heal only the store; keep fake models patched
    event, snapshot = registry.respond(SID, answer)  # reloads from durable, re-advances
    after = store.load_published(SID, OWNER)
    assert after['control']['sequence'] > prior
    assert after['control']['open_input']['input_id'] != str(answer.input_id)


def test_respond_tail_lost_ack_serves_committed_outcome(tmp_path, fake_ollama, fake_models, monkeypatch):
    """Failure location 3 - the step COMMITS but the ack/connection fails: the
    registry reconciles from the durable store and serves the committed outcome;
    the retry is an idempotent accept, never a double advance."""
    conn = make_conn(tmp_path)
    store = PgSessionStore(managed_pool(conn.info.dsn))
    registry = SessionRegistry(default_program_factory, allow_db_provider_dispatch=True, session_store=store, owner_id=OWNER,
                               instance_id='w#1', lease_ttl=60.0)
    _OPEN['registries'].append(registry)
    session, _ = registry.start(StartSessionRequest(session_id=SID, task_id='t', idea='idea',
                                                    model_route='ollama', initial_model='phi4'))
    prior = store.load_published(SID, OWNER)['control']['sequence']
    answer = InputResponse(input_id=session.open_input.input_id, actor_id='founder', value='customers book online')

    real_publish = store.publish_step
    class LostAck(Exception):
        pass
    def commit_then_lose(*a, **k):
        result = real_publish(*a, **k)
        raise LostAck('connection dropped after commit')
    monkeypatch.setattr(store, 'publish_step', commit_then_lose)

    event, snapshot = registry.respond(SID, answer)  # no exception: the committed outcome is served
    committed = store.load_published(SID, OWNER)['control']
    assert committed['sequence'] > prior
    assert str(answer.input_id) in [k for k in committed['responses']] or any(
        r['input_id'] == str(answer.input_id) for r in committed['responses']), committed
    probing_after_first = len(fake_models['probing'].replies)
    event2, _ = registry.respond(SID, answer)
    assert store.load_published(SID, OWNER)['control']['sequence'] == committed['sequence'], 'retry double-advanced'
    assert len(fake_models['probing'].replies) == probing_after_first, 'retry re-ran the model'


def test_journal_acknowledge_failure_serves_committed_without_duplicate_pages(tmp_path, fake_ollama, fake_models, monkeypatch):
    """Failure location 4 - journal acknowledge raises after the publish commit:
    the committed outcome is served and the committed pages are not duplicated on
    the next step."""
    conn = make_conn(tmp_path)
    store = PgSessionStore(managed_pool(conn.info.dsn))
    registry = SessionRegistry(default_program_factory, allow_db_provider_dispatch=True, session_store=store, owner_id=OWNER,
                               instance_id='w#1', lease_ttl=60.0)
    _OPEN['registries'].append(registry)
    session, _ = registry.start(StartSessionRequest(session_id=SID, task_id='t', idea='idea',
                                                    model_route='ollama', initial_model='phi4'))
    answer = InputResponse(input_id=session.open_input.input_id, actor_id='founder', value='customers book online')
    journal = session.program.progress_journal
    class AckBoom(Exception):
        pass
    monkeypatch.setattr(journal, 'acknowledge', lambda n: (_ for _ in ()).throw(AckBoom('ack failed after commit')))

    event, snapshot = registry.respond(SID, answer)  # committed outcome served despite the ack failure
    pages = store.read_journal(SID, OWNER, after=0, limit=200)
    cursors = [p['cursor'] for p in pages]
    assert len(cursors) == len(set(cursors)), f'duplicate journal cursors: {cursors}'
    assert store.load_published(SID, OWNER)['control']['responses']
