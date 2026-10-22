"""PR B staged provider responses: logical-call identity, gateway-boundary lifecycle,
and crash-window behavior.

One row per LOGICAL call (session, generation, stage, item, ordinal). Dispatch opens
the row before the ledger attempt; the raw result + usage complete it BEFORE settle.
Replay consumes the log by ordinal: identical INTENTIONAL repeats are new paid calls;
only a crash-replay of the same logical call reuses staged bytes. A row stuck
dispatched-without-response pauses (unknown outcome), never blind-reruns.
"""
import psycopg, psycopg.rows, pytest
from types import SimpleNamespace
from contextlib import contextmanager

pgserver = pytest.importorskip('pgserver', reason='pgserver not installed')
from interview_engine.pg_session_store import PgSessionStore, PublicationConflict
from test_pg_session_store import MIRRORED_REGISTRY_DDL, MIRRORED_STAGING_DDL

OWNER_A = '00000000-0000-0000-0000-000000000001'


@pytest.fixture
def store(tmp_path):
    server = pgserver.get_server(str(tmp_path / 'pgdata'))
    conn = psycopg.connect(server.get_uri(), autocommit=False, row_factory=psycopg.rows.dict_row)
    conn.execute(MIRRORED_REGISTRY_DDL)
    conn.execute(MIRRORED_STAGING_DDL)
    conn.commit()
    s = PgSessionStore(conn)
    s.create('s1', 'task1', OWNER_A)
    return s


@pytest.fixture
def claim(store):
    won = store.claim('s1', OWNER_A, 'host#1', ttl=60)
    assert won is not None
    return won


class FakeGateway:
    def __init__(self):
        self.reserved = []; self.settled = []; self.uncertain_calls = []

    def reserve(self, attempt_id, **kw):
        self.reserved.append(attempt_id)
        return SimpleNamespace(attempt_id=attempt_id)

    def start(self, attempt): pass
    def settle(self, attempt, response): self.settled.append(attempt.attempt_id)
    def uncertain(self, attempt): self.uncertain_calls.append(attempt.attempt_id)


@contextmanager
def staging_env(store, claim, gateway, monkeypatch):
    """Cloud admission path + staging context + a counting fake transport."""
    import interview_engine.model_config as mc
    import interview_engine.attempt_admission as aa
    from langchain_core.messages import AIMessage
    calls = []

    def transport(messages, content):
        calls.append(messages)
        return AIMessage(content=content, usage_metadata={'input_tokens': 5, 'output_tokens': 6, 'total_tokens': 11})

    ctx = mc.StagingContext('s1', store, claim)
    token = mc._response_staging.set(ctx)
    with aa.admission_gateway(gateway, required=True):
        monkeypatch.setenv('VMODEL_TENANT_OWNER', OWNER_A)
        yield ctx, calls
    mc._response_staging.reset(token)


def invoke(messages, *, stage='scope_tiebreak', item=None, content='{"answer": 42}', monkeypatch=None, transport_calls=None):
    import interview_engine.attempt_admission as aa
    from langchain_core.messages import AIMessage
    return aa.admitted_invoke(lambda: transport_calls(messages, content), stage=stage, item=item,
                              provider='fake', model='fake-1', messages=messages, max_output_tokens=100)


def test_identical_intentional_calls_both_invoke(store, claim, monkeypatch):
    from langchain_core.messages import HumanMessage
    gw = FakeGateway()
    with staging_env(store, claim, gw, monkeypatch) as (ctx, calls):
        msgs = [HumanMessage(content='name the scope')]
        invoke(msgs, transport_calls=lambda m, c: _transport(calls, m, c))
        invoke(msgs, transport_calls=lambda m, c: _transport(calls, m, c))
    assert len(calls) == 2 and len(gw.reserved) == 2 and len(gw.settled) == 2
    log = store.staged_log('s1', claim.generation, stage='scope_tiebreak')
    assert [r['ordinal'] for r in log] == [0, 1] and all(r['response'] is not None for r in log)


def _transport(calls, messages, content):
    from langchain_core.messages import AIMessage
    calls.append(messages)
    return AIMessage(content=content, usage_metadata={'input_tokens': 5, 'output_tokens': 6, 'total_tokens': 11})


def test_crash_replay_same_logical_call_reuses_staged_bytes(store, claim, monkeypatch):
    from langchain_core.messages import HumanMessage
    import interview_engine.model_config as mc
    import interview_engine.attempt_admission as aa
    gw = FakeGateway()
    msgs = [HumanMessage(content='name the scope')]
    with staging_env(store, claim, gw, monkeypatch) as (ctx, calls):
        first = invoke(msgs, transport_calls=lambda m, c: _transport(calls, m, c))
        store.conn.commit()
    # Process/crash boundary: brand-new staging context (cursors reset), same store.
    with staging_env(store, claim, gw, monkeypatch) as (ctx2, calls2):
        second = invoke(msgs, transport_calls=lambda m, c: _transport(calls2, m, c))
    assert second.content == first.content == '{"answer": 42}'
    assert len(calls) == 1 and len(calls2) == 0, 'provider re-called on crash replay'
    assert len(gw.reserved) == 1 and len(gw.settled) == 1, 'ledger moved on replay'


def test_dispatched_without_response_pauses_no_rerun(store, claim, monkeypatch):
    """Crash after dispatch (provider outcome unknown): the replay PAUSES with the
    attempt evidence; no new provider call, no new ledger attempt."""
    from langchain_core.messages import HumanMessage
    import interview_engine.model_config as mc
    import interview_engine.attempt_admission as aa
    gw = FakeGateway()
    msgs = [HumanMessage(content='name the scope')]
    with staging_env(store, claim, gw, monkeypatch) as (ctx, calls):
        fp = mc._staging_request_sha(type('C', (), {'provider': 'fake', 'model': 'fake-1', 'max_tokens': 100})(), msgs, 'scope_tiebreak', None)
        store.record_dispatch(claim, stage='scope_tiebreak', item='', ordinal=0,
                              request_sha256=fp, provider='fake', model='fake-1')
        store.conn.commit()
        from interview_engine.errors import UserFacingError
        with pytest.raises(UserFacingError, match='result is missing'):
            invoke(msgs, transport_calls=lambda m, c: _transport(calls, m, c))
    assert len(calls) == 0 and len(gw.reserved) == 0


def test_diverged_replay_pauses(store, claim, monkeypatch):
    """Same ordinal, different request (execution path changed across the crash):
    never mix staged results into a diverged path."""
    from langchain_core.messages import HumanMessage
    gw = FakeGateway()
    with staging_env(store, claim, gw, monkeypatch) as (ctx, calls):
        invoke([HumanMessage(content='original')], transport_calls=lambda m, c: _transport(calls, m, c))
        store.conn.commit()
    with staging_env(store, claim, gw, monkeypatch) as (ctx2, calls2):
        from interview_engine.errors import UserFacingError
        with pytest.raises(UserFacingError, match='does not match'):
            invoke([HumanMessage(content='DIFFERENT prompt same position')],
                   transport_calls=lambda m, c: _transport(calls2, m, c))
    assert len(calls2) == 0 and len(gw.reserved) == 1


def test_billable_empty_response_is_staged_and_retry_is_new_attempt(store, claim, monkeypatch):
    """A completed-but-empty (billable, unusable) response is staged raw; the deliberate
    retry is a NEW logical attempt, not a lost history overwrite."""
    from langchain_core.messages import HumanMessage
    gw = FakeGateway()
    msgs = [HumanMessage(content='q')]
    with staging_env(store, claim, gw, monkeypatch) as (ctx, calls):
        invoke(msgs, transport_calls=lambda m, c: _transport(calls, m, '   '))
        invoke(msgs, transport_calls=lambda m, c: _transport(calls, m, '{"ok": true}'))
        store.conn.commit()
    log = store.staged_log('s1', claim.generation, stage='scope_tiebreak')
    assert len(log) == 2 and log[0]['response'] is not None and log[1]['response'] is not None
    assert len(gw.reserved) == 2


def test_serialization_exactness_roundtrip(store, claim, monkeypatch):
    """tool_calls, response_metadata and usage survive the staged round-trip exactly."""
    from langchain_core.messages import AIMessage, HumanMessage
    gw = FakeGateway()
    rich = AIMessage(content='', tool_calls=[{'name': 'lookup', 'args': {'q': 'x'}, 'id': 'call_1'}],
                     response_metadata={'finish_reason': 'tool_calls', 'model_name': 'fake-1'},
                     usage_metadata={'input_tokens': 9, 'output_tokens': 8, 'total_tokens': 17})
    with staging_env(store, claim, gw, monkeypatch) as (ctx, calls):
        invoke([HumanMessage(content='use a tool')], transport_calls=lambda m, c: rich)
        store.conn.commit()
    with staging_env(store, claim, gw, monkeypatch) as (ctx2, calls2):
        replayed = invoke([HumanMessage(content='use a tool')], transport_calls=lambda m, c: _transport(calls2, m, c))
    assert len(calls2) == 0
    assert replayed.tool_calls == rich.tool_calls
    assert replayed.response_metadata == rich.response_metadata
    assert replayed.usage_metadata == rich.usage_metadata


def test_completion_is_append_once(store, claim):
    c = claim
    store.record_dispatch(c, stage='s', item='', ordinal=0, request_sha256='x', provider='p', model='m')
    store.complete_staged(c, stage='s', item='', ordinal=0, response={'v': 1}, usage=None)
    with pytest.raises(PublicationConflict):
        store.complete_staged(c, stage='s', item='', ordinal=0, response={'v': 2}, usage=None)
    store.conn.rollback()
