"""PR B staged provider responses: durable staging contract + invoke_checked replay.

A usable provider response is staged BEFORE interpretation; replaying the exact same
request returns the staged bytes without a new provider call (no rebill). Mirrors the
table shape of delivery-api 20261022160000_staged_provider_responses.sql without the
auth FK/RLS (tenant isolation is covered delivery-api-side against the real migration).
"""
import psycopg, psycopg.rows, pytest
from types import SimpleNamespace

pgserver = pytest.importorskip('pgserver', reason='pgserver not installed')
from interview_engine.pg_session_store import PgSessionStore, SessionClaim, PublicationConflict

MIRRORED_STAGING_DDL = """CREATE TABLE engine_sessions(
 session_id TEXT PRIMARY KEY, owner_id UUID NOT NULL, task_id TEXT NOT NULL,
 generation INTEGER NOT NULL CHECK(generation > 0), created_at DOUBLE PRECISION NOT NULL,
 placement_instance TEXT, lease_epoch BIGINT NOT NULL DEFAULT 0,
 lease_expires_at DOUBLE PRECISION NOT NULL DEFAULT 0, updated_at DOUBLE PRECISION,
 control JSONB, published_thread_id TEXT,
 published_checkpoint_ns TEXT NOT NULL DEFAULT '', published_checkpoint_id TEXT);
"""
MIRRORED_STAGING_DDL += """CREATE TABLE staged_provider_responses(
 owner_id UUID NOT NULL, session_id TEXT NOT NULL, stage TEXT NOT NULL, item TEXT,
 request_sha256 TEXT NOT NULL, response JSONB NOT NULL, usage JSONB,
 provider TEXT NOT NULL, model TEXT NOT NULL, attempt INTEGER NOT NULL DEFAULT 1,
 created_at DOUBLE PRECISION NOT NULL,
 PRIMARY KEY (session_id, stage, request_sha256))"""

OWNER_A = '00000000-0000-0000-0000-000000000001'


@pytest.fixture
def store(tmp_path):
    server = pgserver.get_server(str(tmp_path / 'pgdata'))
    conn = psycopg.connect(server.get_uri(), autocommit=False, row_factory=psycopg.rows.dict_row)
    for statement in MIRRORED_STAGING_DDL.split(';'):
        if statement.strip(): conn.execute(statement)
    conn.commit()
    s = PgSessionStore(conn)
    s.create('s1', 'task1', OWNER_A)
    return s


def claim(store):
    won = store.claim('s1', OWNER_A, 'host#1', ttl=60)
    assert won is not None
    return won


def test_stage_and_load_roundtrip(store):
    c = claim(store)
    store.stage_response(c, stage='scope', item=None, request_sha256='sha-1',
                         payload={'lc': 1, 'type': 'constructor', 'kwargs': {'content': 'answer'}},
                         usage={'input_tokens': 3, 'output_tokens': 4}, provider='fake', model='fake-1')
    store.commit() if hasattr(store, 'commit') else store.conn.commit()
    row = store.load_staged_response('s1', OWNER_A, stage='scope', request_sha256='sha-1')
    assert row['usage'] == {'input_tokens': 3, 'output_tokens': 4}
    assert row['response']['kwargs']['content'] == 'answer'
    assert store.load_staged_response('s1', OWNER_A, stage='scope', request_sha256='other') is None
    assert store.load_staged_response('other-session', OWNER_A, stage='scope', request_sha256='sha-1') is None


def test_same_key_restage_same_payload_is_idempotent(store):
    c = claim(store)
    payload = {'lc': 1, 'kwargs': {'content': 'x'}}
    store.stage_response(c, stage='scope', item=None, request_sha256='sha-1', payload=payload,
                         usage=None, provider='fake', model='m')
    store.stage_response(c, stage='scope', item=None, request_sha256='sha-1', payload=payload,
                         usage=None, provider='fake', model='m')
    store.conn.commit()


def test_same_key_different_payload_conflicts_loudly(store):
    c = claim(store)
    store.stage_response(c, stage='scope', item=None, request_sha256='sha-1',
                         payload={'v': 1}, usage=None, provider='fake', model='m')
    with pytest.raises(PublicationConflict):
        store.stage_response(c, stage='scope', item=None, request_sha256='sha-1',
                             payload={'v': 2}, usage=None, provider='fake', model='m')
    store.conn.rollback()


def test_invoke_checked_replays_staged_response_without_rebill(store, monkeypatch):
    """Crash-and-retry at the call level: the first usable response is staged; the
    identical retried request is served from the staged row - provider call counter
    stays 1. A different request is a different paid call."""
    import interview_engine.model_config as mc
    from langchain_core.messages import HumanMessage, AIMessage
    c = claim(store)
    calls = []

    def fake_invoke(chat_model, messages, **kwargs):
        calls.append(messages)
        return AIMessage(content='{"answer": 42}', usage_metadata={'input_tokens': 5, 'output_tokens': 6, 'total_tokens': 11})

    import interview_engine.attempt_admission as admission
    monkeypatch.setattr(admission, 'checked_model_invoke', fake_invoke)
    monkeypatch.setattr(mc, '_start_heartbeat', lambda *a, **k: (lambda: None))
    token = mc._response_staging.set(('s1', store, c))
    try:
        messages = [HumanMessage(content='name the scope')]
        first = mc.invoke_checked(object(), messages, 'scope_tiebreak')
        store.conn.commit()
        second = mc.invoke_checked(object(), messages, 'scope_tiebreak')  # retry after crash
        assert first.content == second.content == '{"answer": 42}'
        assert len(calls) == 1, f'provider re-called on replay: {len(calls)}'
        different = mc.invoke_checked(object(), [HumanMessage(content='other question')], 'scope_tiebreak')
        assert len(calls) == 2  # a different request is a new legitimate call
    finally:
        mc._response_staging.reset(token)


def test_failed_attempt_is_never_staged(store, monkeypatch):
    """Provider-empty responses are failures: they raise and are NOT staged, so a later
    retry makes a fresh call instead of replaying the failure."""
    import interview_engine.model_config as mc
    from langchain_core.messages import HumanMessage, AIMessage
    c = claim(store)
    calls = []

    def empty_invoke(chat_model, messages, **kwargs):
        calls.append(messages)
        return AIMessage(content='   ', usage_metadata={'input_tokens': 1, 'output_tokens': 0, 'total_tokens': 1})

    import interview_engine.attempt_admission as admission
    monkeypatch.setattr(admission, 'checked_model_invoke', empty_invoke)
    monkeypatch.setattr(mc, '_start_heartbeat', lambda *a, **k: (lambda: None))
    token = mc._response_staging.set(('s1', store, c))
    try:
        from interview_engine.errors import UserFacingError
        with pytest.raises(UserFacingError):
            mc.invoke_checked(object(), [HumanMessage(content='q')], 'scope_tiebreak')
        assert store.load_staged_response('s1', OWNER_A, stage='scope_tiebreak',
                                          request_sha256=mc._staging_request_sha(mc.for_stage('scope_tiebreak'), [HumanMessage(content='q')], 'scope_tiebreak', None)) is None
        assert len(calls) == 2  # both attempts happened, neither staged
    finally:
        mc._response_staging.reset(token)
