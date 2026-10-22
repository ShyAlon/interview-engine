"""PR B staged provider responses: logical-call identity, gateway-boundary lifecycle,
and crash-window behavior.

One row per LOGICAL call (session, generation, execution_point, node, stage, item,
ordinal). Dispatch opens
the row before the ledger attempt; the raw result + usage complete it BEFORE settle.
Replay consumes the log by ordinal: identical INTENTIONAL repeats are new paid calls;
only a crash-replay of the same logical call reuses staged bytes. A row stuck
dispatched-without-response pauses (unknown outcome), never blind-reruns.
"""
import psycopg, psycopg.rows, pytest
from types import SimpleNamespace
from contextlib import contextmanager

pgserver = pytest.importorskip('pgserver', reason='pgserver not installed')
from conftest import managed_pg_server
from interview_engine.pg_session_store import PgSessionStore, PublicationConflict
from test_pg_session_store import MIRRORED_REGISTRY_DDL, MIRRORED_STAGING_DDL

OWNER_A = '00000000-0000-0000-0000-000000000001'


@pytest.fixture
def store(tmp_path):
    server = managed_pg_server(str(tmp_path / 'pgdata'))
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
    """Attempt-keyed idempotent ledger fake, mirroring the DA ledger contract:
    settling an already-settled attempt with the same usage is a no-op;
    conflicting usage raises."""

    def __init__(self):
        self.reserved = []; self.settled = []; self.uncertain_calls = []
        self.settle_staged_calls = []; self._settled_usage = {}
        self.fail_settle_staged = None

    def reserve(self, attempt_id, **kw):
        self.reserved.append(attempt_id)
        return SimpleNamespace(attempt_id=attempt_id)

    def start(self, attempt): pass
    def settle(self, attempt, response):
        self.settled.append(attempt.attempt_id)
        self._settled_usage[attempt.attempt_id] = getattr(response, 'usage_metadata', None)
    def uncertain(self, attempt): self.uncertain_calls.append(attempt.attempt_id)

    def settle_staged(self, attempt_id, usage):
        """Settle from staged trusted usage: no-op when already settled with the
        same usage, conflict raises, unsettled settles."""
        if self.fail_settle_staged is not None:
            raise self.fail_settle_staged
        self.settle_staged_calls.append((attempt_id, usage))
        prior = self._settled_usage.get(attempt_id, 'unset')
        if prior != 'unset' and prior != usage:
            raise RuntimeError('conflicting usage for settled attempt')
        self._settled_usage[attempt_id] = usage


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
    log = store.staged_log('s1', claim.generation, execution_point='bootstrap:0', node='', stage='scope_tiebreak')
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
        store.record_dispatch(claim, execution_point='bootstrap:0', node='', stage='scope_tiebreak', item='', ordinal=0,
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
    log = store.staged_log('s1', claim.generation, execution_point='bootstrap:0', node='', stage='scope_tiebreak')
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
    store.record_dispatch(c, execution_point='bootstrap:0', node='', stage='s', item='', ordinal=0, request_sha256='x', provider='p', model='m')
    store.complete_staged(c, execution_point='bootstrap:0', node='', stage='s', item='', ordinal=0, response={'v': 1}, usage=None)
    with pytest.raises(PublicationConflict):
        store.complete_staged(c, execution_point='bootstrap:0', node='', stage='s', item='', ordinal=0, response={'v': 2}, usage=None)
    store.conn.rollback()


def test_replay_settles_unsettled_attempt_from_staged_usage_before_returning(store, claim, monkeypatch):
    """Crash after complete_staged but before settle: the replay completes the
    existing attempt's settlement from the STAGED trusted usage BEFORE the graph
    gets the result. No graph progress while billing is unreconciled."""
    from langchain_core.messages import HumanMessage
    gw = FakeGateway()
    msgs = [HumanMessage(content='name the scope')]
    with staging_env(store, claim, gw, monkeypatch) as (ctx, calls):
        invoke(msgs, transport_calls=lambda m, c: _transport(calls, m, c))
        # Simulate the complete/before-settle crash window: the staged row holds the
        # trusted result+usage, but the ledger attempt never settled.
        gw._settled_usage.clear()
        store.conn.commit()
    with staging_env(store, claim, gw, monkeypatch) as (ctx2, calls2):
        second = invoke(msgs, transport_calls=lambda m, c: _transport(calls2, m, c))
    assert second.content == '{"answer": 42}'
    assert len(calls2) == 0 and len(gw.reserved) == 1
    assert gw.settle_staged_calls == [(gw.reserved[0], {'input_tokens': 5, 'output_tokens': 6, 'total_tokens': 11})], gw.settle_staged_calls
    assert gw._settled_usage[gw.reserved[0]] == {'input_tokens': 5, 'output_tokens': 6, 'total_tokens': 11}


def test_replay_pauses_when_staged_settlement_fails(store, claim, monkeypatch):
    """Settlement conflict on replay (ledger says something else): pause with the
    attempt evidence; the result is NOT handed to the graph."""
    from langchain_core.messages import HumanMessage
    gw = FakeGateway()
    msgs = [HumanMessage(content='name the scope')]
    with staging_env(store, claim, gw, monkeypatch) as (ctx, calls):
        invoke(msgs, transport_calls=lambda m, c: _transport(calls, m, c))
        gw._settled_usage[gw.reserved[0]] = {'input_tokens': 999}  # ledger disagrees
        store.conn.commit()
    from interview_engine.errors import UserFacingError
    with staging_env(store, claim, gw, monkeypatch) as (ctx2, calls2):
        with pytest.raises(UserFacingError, match='billing is unreconciled'):
            invoke(msgs, transport_calls=lambda m, c: _transport(calls2, m, c))
    assert len(calls2) == 0


def test_identical_prompt_under_new_execution_point_is_a_new_paid_call(store, claim, monkeypatch):
    """A new turn resumes from a NEW committed checkpoint: the identical prompt is a
    new logical call (new paid call), never a replay of the old point's row."""
    from langchain_core.messages import HumanMessage
    gw = FakeGateway()
    msgs = [HumanMessage(content='same question, later turn')]
    with staging_env(store, claim, gw, monkeypatch) as (ctx, calls):
        invoke(msgs, transport_calls=lambda m, c: _transport(calls, m, c))
        store.conn.commit()
        ctx.set_execution_point('checkpoint-2', 0)  # the step committed; a new turn begins
        invoke(msgs, transport_calls=lambda m, c: _transport(calls, m, c))
    assert len(calls) == 2 and len(gw.reserved) == 2
    old = store.staged_log('s1', claim.generation, execution_point='bootstrap:0', node='', stage='scope_tiebreak')
    new = store.staged_log('s1', claim.generation, execution_point='checkpoint-2:0', node='', stage='scope_tiebreak')
    assert len(old) == 1 and len(new) == 1


def test_parallel_nodes_same_execution_point_do_not_collide(store, claim, monkeypatch):
    """Two nodes issuing the same stage+item under one execution point stage under
    their own node identity; neither replays the other's row."""
    from langchain_core.messages import HumanMessage
    gw = FakeGateway()
    with staging_env(store, claim, gw, monkeypatch) as (ctx, calls):
        monkeypatch.setattr('interview_engine.model_config.StagingContext.current_node', staticmethod(lambda: 'node_a'))
        invoke([HumanMessage(content='from A')], transport_calls=lambda m, c: _transport(calls, m, c))
        monkeypatch.setattr('interview_engine.model_config.StagingContext.current_node', staticmethod(lambda: 'node_b'))
        invoke([HumanMessage(content='from B')], transport_calls=lambda m, c: _transport(calls, m, c))
    a = store.staged_log('s1', claim.generation, execution_point='bootstrap:0', node='node_a', stage='scope_tiebreak')
    b = store.staged_log('s1', claim.generation, execution_point='bootstrap:0', node='node_b', stage='scope_tiebreak')
    assert len(a) == 1 and len(b) == 1 and len(calls) == 2


def test_fingerprint_binds_generation_params_and_tool_call_ids(store, claim):
    """The staged fingerprint must bind provider-significant content: generation
    parameters and tool-call ids change it; locally random message ids do not."""
    import interview_engine.model_config as mc
    from langchain_core.messages import AIMessage, HumanMessage
    cfg = type('C', (), {'provider': 'fake', 'model': 'fake-1', 'max_tokens': 100})()
    msgs = [HumanMessage(content='q')]
    base = mc._staging_request_sha(cfg, msgs, 'scope_tiebreak', None, params={'temperature': 0.5})
    assert mc._staging_request_sha(cfg, msgs, 'scope_tiebreak', None, params={'temperature': 0.7}) != base
    assert mc._staging_request_sha(cfg, msgs, 'scope_tiebreak', None, params={'temperature': 0.5}) == base
    assert mc._staging_request_sha(cfg, msgs, 'scope_tiebreak', None, params={'temperature': 0.5, 'response_format': {'type': 'json_schema'}}) != base
    # Tool-call ids are provider-significant; the random message id is not.
    a = AIMessage(content='', tool_calls=[{'name': 'lookup', 'args': {'q': 'x'}, 'id': 'call_1'}], id='msg-1')
    b = AIMessage(content='', tool_calls=[{'name': 'lookup', 'args': {'q': 'x'}, 'id': 'call_2'}], id='msg-1')
    assert mc._staging_request_sha(cfg, [a], 'scope_tiebreak', None) != mc._staging_request_sha(cfg, [b], 'scope_tiebreak', None)
    # A resume rebuilds the message with a fresh local id: same paid call.
    a2 = AIMessage(content='', tool_calls=[{'name': 'lookup', 'args': {'q': 'x'}, 'id': 'call_1'}], id='msg-2')
    assert mc._staging_request_sha(cfg, [a2], 'scope_tiebreak', None) == mc._staging_request_sha(cfg, [a], 'scope_tiebreak', None)
