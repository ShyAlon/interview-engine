"""InterviewApplication in database mode: the default factory wires the real
application onto the session's Postgres lease.

Graph checkpoints write into Postgres through the fenced saver, progress
signals buffer per step and publish atomically with the control commit and
the checkpoint pointer, and restore pins the published pointer instead of
trusting saver-latest. No provider dispatch: scripted fake models answer
every stage and the Ollama preflight endpoint is stubbed.
"""
import time

import pytest

pgserver = pytest.importorskip('pgserver', reason='pgserver not installed: PG application wiring tests skip')
from conftest import managed_pg_server

import psycopg
import psycopg.rows
from langchain_core.messages import AIMessage

from vmodel_contracts.interactive import StartSessionRequest

from interview_engine import model_config
from interview_engine.application import InterviewApplication
from interview_engine.http_service import SessionRegistry, default_program_factory
from interview_engine.pg_checkpoint import FencedPostgresSaver
from interview_engine.pg_journal import BufferedJournal
from interview_engine.pg_session_store import PgSessionStore
from interview_engine.stop_fence import StopRequested
from test_pg_session_store import MIRRORED_REGISTRY_DDL, MIRRORED_JOURNAL_DDL, MIRRORED_STAGING_DDL

OWNER = '00000000-0000-0000-0000-000000000001'
SID = '11111111-1111-4111-8111-111111111111'
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
        conn.execute(MIRRORED_STAGING_DDL)
        conn.commit()
    # Tests bootstrap the checkpoint schema explicitly as superuser; the
    # runtime saver setup never runs DDL. Library DDL uses CREATE INDEX
    # CONCURRENTLY, so it runs on its own autocommit connection.
    from interview_engine.pg_checkpoint import bootstrap_checkpoint_schema
    boot = psycopg.connect(server.get_uri(), autocommit=True)
    bootstrap_checkpoint_schema(boot)
    boot.close()
    _OPEN['conns'].append(conn)
    return conn


class _OllamaResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"data": [{"id": "phi4"}]}


@pytest.fixture
def fake_ollama(monkeypatch):
    monkeypatch.setattr('httpx.get', lambda *a, **k: _OllamaResponse())


class FakeModel:
    def __init__(self, replies):
        self.replies = list(replies)

    def invoke(self, messages):
        content = self.replies.pop(0) if self.replies else "[INTERVIEW COMPLETE]"
        return AIMessage(content=content)


@pytest.fixture
def fake_models(monkeypatch):
    scripts = {
        'probing': ['{"question": "What does a booking look like for your customers?", "covered": [], "thin": [], "skipped": [], "complete": false}'] * 6,
        'scope_tiebreak': ['{"is_web_app": true, "archetype": "saas_crud", "confidence": 0.9, "reason": "booking app is interactive web software"}'] * 4,
    }
    box = {}

    def build(stage, **kwargs):
        # Cache per stage like the graph does: one model, one reply queue.
        if stage not in box:
            box[stage] = FakeModel(scripts.get(stage, ['[INTERVIEW COMPLETE]']))
        return box[stage]

    monkeypatch.setattr(model_config, 'build_chat_model', build)
    return box


def test_application_pg_builds_fenced_saver_and_buffered_journal(tmp_path, fake_ollama, fake_models):
    store = PgSessionStore(make_conn(tmp_path))
    store.create('s1', 'task-1', OWNER)
    claim = store.claim('s1', OWNER, 'inst#1', ttl=60)
    app = InterviewApplication('s1', 'task-1', 'idea', model_route='ollama', initial_model='phi4',
                               artifact_root=str(tmp_path / 'artifacts'),
                               pg_store=store, pg_claim=claim)
    try:
        assert isinstance(app.saver, FencedPostgresSaver)
        assert app.saver.conn is not store.conn  # own write connection
        assert isinstance(app.progress_journal, BufferedJournal)
        app.progress_journal.append(phase_key='intent', stage='interview_turn', label='t', status='started')
        assert len(app.progress_journal.pending()) == 1
        # Buffered, not published: nothing is durable before the step commits.
        assert store.journal_head('s1', OWNER) == 0
    finally:
        app.saver.conn.close()


def test_registry_pg_real_application_publishes_step_and_restores_pointer(tmp_path, fake_ollama, fake_models):
    conn = make_conn(tmp_path)
    store = PgSessionStore(conn)
    registry = SessionRegistry(default_program_factory, allow_db_provider_dispatch=True, session_store=store, owner_id=OWNER,
                               instance_id='w#1', lease_ttl=1.0)
    _OPEN['registries'].append(registry)
    session, _ = registry.start(StartSessionRequest(session_id=SID, task_id='t', idea='idea',
                                                    model_route='ollama', initial_model='phi4'))
    app = session.program
    assert isinstance(app, InterviewApplication)
    assert isinstance(app.saver, FencedPostgresSaver) and isinstance(app.progress_journal, BufferedJournal)

    # start() runs the first advance and publishes it: the buffered journal
    # pages and the graph checkpoint pointer committed with the step.
    row = store.load_published(SID, OWNER)
    assert row['control']['sequence'] >= 1, row['control']
    assert row['published_thread_id'] == SID and row['published_checkpoint_id']
    pages = store.read_journal(SID, OWNER, after=0, limit=100)
    assert pages and pages[0]['cursor'] == 1
    assert {page['page']['stage'] for page in pages} >= {'scope_gate', 'interview_turn'}
    assert app.progress_journal.pending() == []  # acknowledged only after commit

    published_id = row['published_checkpoint_id']
    # Expire the lease and take over from a second instance: the restored
    # program resumes at the published pointer, never at saver-latest.
    registry.close()
    time.sleep(1.2)
    registry2 = SessionRegistry(default_program_factory, allow_db_provider_dispatch=True, session_store=PgSessionStore(conn),
                                owner_id=OWNER, instance_id='w#2', lease_ttl=60.0)
    _OPEN['registries'].append(registry2)
    restored = registry2.get(SID)
    app2 = restored.program
    assert isinstance(app2, InterviewApplication)
    cfg = app2.cfg['configurable']
    assert cfg.get('checkpoint_id') == published_id, cfg
    state = app2.graph.get_state(app2.cfg)
    assert state.values.get('submission'), 'restored graph resumes with the published gate state'
    # No provider dispatch on restore: construction with pinned models never
    # preflights, and get_state is a local read.


def test_pg_application_stop_blocks_writes_and_survives_takeover(tmp_path, fake_ollama, fake_models):
    conn = make_conn(tmp_path)
    store = PgSessionStore(conn)
    registry = SessionRegistry(default_program_factory, allow_db_provider_dispatch=True, session_store=store, owner_id=OWNER,
                               instance_id='w#1', lease_ttl=1.0)
    _OPEN['registries'].append(registry)
    session, _ = registry.start(StartSessionRequest(session_id=SID, task_id='t', idea='idea',
                                                    model_route='ollama', initial_model='phi4'))
    committed = store.load_published(SID, OWNER)['control']['sequence']

    # Stop after a completed advance: the quiesced republish is a newer step,
    # the generation bump fences every later write from this process.
    registry.stop(SID)
    assert session.pg_stopped
    row = store.load_published(SID, OWNER)
    assert row['control']['stopped'] is True and row['control']['open_input'] is None
    assert row['control']['sequence'] > committed
    with pytest.raises(StopRequested):
        store.publish_step(session.pg_claim, dict(row['control'], sequence=row['control']['sequence'] + 1))

    # Takeover after expiry: the stopped marker is honored, no graph is built.
    registry.close()
    time.sleep(1.2)
    registry2 = SessionRegistry(default_program_factory, allow_db_provider_dispatch=True, session_store=PgSessionStore(conn),
                                owner_id=OWNER, instance_id='w#2', lease_ttl=60.0)
    _OPEN['registries'].append(registry2)
    restored = registry2.get(SID)
    assert restored.pg_stopped and restored.open_input is None and restored.program is None


def test_pg_application_publish_rolls_back_all_or_nothing(tmp_path, fake_ollama, fake_models, monkeypatch):
    conn = make_conn(tmp_path)
    store = PgSessionStore(conn)
    registry = SessionRegistry(default_program_factory, allow_db_provider_dispatch=True, session_store=store, owner_id=OWNER,
                               instance_id='w#1', lease_ttl=60.0)
    _OPEN['registries'].append(registry)
    session, _ = registry.start(StartSessionRequest(session_id=SID, task_id='t', idea='idea',
                                                    model_route='ollama', initial_model='phi4'))
    before = store.load_published(SID, OWNER)
    head_before = store.journal_head(SID, OWNER)

    # Buffer a signal, then inject a failure AFTER the journal insert and
    # BEFORE the pointer commit, through the registry's own publish path.
    journal = session.program.progress_journal
    journal.append(phase_key='intent', stage='interview_turn', label='late', status='started')
    real = store.publish_step

    def boom(claim, control, **kwargs):
        kwargs['fault_hook'] = lambda: (_ for _ in ()).throw(RuntimeError('injected mid-transaction failure'))
        return real(claim, control, **kwargs)

    monkeypatch.setattr(store, 'publish_step', boom)
    session.sequence += 1
    with pytest.raises(RuntimeError, match='injected'):
        registry._save(session)

    # Nothing committed: control, pointer and journal head are exactly the
    # pre-attempt state, and the buffered page was NOT acknowledged.
    assert store.load_published(SID, OWNER) == before
    assert store.journal_head(SID, OWNER) == head_before
    assert len(journal.pending()) == 1

    # The retry commits the same step cleanly: no partial page, no conflict.
    monkeypatch.setattr(store, 'publish_step', real)
    registry._save(session)
    after = store.load_published(SID, OWNER)
    assert after['control']['sequence'] == before['control']['sequence'] + 1
    assert store.journal_head(SID, OWNER) == head_before + 1
    assert journal.pending() == []


def test_pg_application_crash_after_checkpoint_resumes_last_published(tmp_path, fake_ollama, fake_models):
    conn = make_conn(tmp_path)
    store = PgSessionStore(conn)
    registry = SessionRegistry(default_program_factory, allow_db_provider_dispatch=True, session_store=store, owner_id=OWNER,
                               instance_id='w#1', lease_ttl=1.0)
    _OPEN['registries'].append(registry)
    session, _ = registry.start(StartSessionRequest(session_id=SID, task_id='t', idea='idea',
                                                    model_route='ollama', initial_model='phi4'))
    published = store.load_published(SID, OWNER)['published_checkpoint_id']

    # The worker runs the founder's answer; the graph checkpoints the new leg
    # into Postgres - and the process dies BEFORE the step publishes. The
    # saver now holds checkpoints newer than the published pointer.
    app = session.program
    from langchain_core.messages import HumanMessage
    app.graph.invoke({'messages': [HumanMessage(content='customers book online from a phone')]}, app.cfg)
    latest = app.saver.get_tuple(app.cfg)
    assert latest is not None and latest.config['configurable']['checkpoint_id'] != published

    registry.close()
    time.sleep(1.2)
    registry2 = SessionRegistry(default_program_factory, allow_db_provider_dispatch=True, session_store=PgSessionStore(conn),
                                owner_id=OWNER, instance_id='w#2', lease_ttl=60.0)
    _OPEN['registries'].append(registry2)
    restored = registry2.get(SID)
    cfg = restored.program.cfg['configurable']
    assert cfg.get('checkpoint_id') == published, 'restore pins the published pointer, never saver-latest'
    values = restored.program.graph.get_state(restored.program.cfg).values
    text = str(values.get('messages', ''))
    assert 'customers book online from a phone' not in text, 'unpublished work is not served after a crash'


def test_remote_owner_stop_while_other_instance_blocked_mid_advance(tmp_path, fake_ollama, monkeypatch):
    import threading

    # Instance w#1 serves the session and is blocked inside a model call.
    entered = threading.Event()
    release = threading.Event()

    class BlockingFake(FakeModel):
        def invoke(self, messages):
            entered.set()
            release.wait(timeout=30)
            return super().invoke(messages)

    blocking = BlockingFake(['{"question": "q?", "covered": [], "thin": [], "skipped": [], "complete": false}'])
    box = {}

    def build(stage, **kwargs):
        if stage not in box:
            box[stage] = blocking if stage == 'probing' else FakeModel(
                ['{"is_web_app": true, "archetype": "saas_crud", "confidence": 0.9, "reason": "web app"}'] * 4)
        return box[stage]

    monkeypatch.setattr(model_config, 'build_chat_model', build)

    conn_a = make_conn(tmp_path)
    conn_b = make_conn(tmp_path, init=False)
    registry1 = SessionRegistry(default_program_factory, allow_db_provider_dispatch=True, session_store=PgSessionStore(conn_a),
                                owner_id=OWNER, instance_id='w#1', lease_ttl=60.0)
    _OPEN['registries'].append(registry1)
    store_b = PgSessionStore(conn_b)
    registry2 = SessionRegistry(default_program_factory, allow_db_provider_dispatch=True, session_store=store_b,
                                owner_id=OWNER, instance_id='w#2', lease_ttl=60.0)
    _OPEN['registries'].append(registry2)

    started = {}
    thread = threading.Thread(
        target=lambda: started.update(
            {'result': registry1.start(StartSessionRequest(session_id=SID, task_id='t', idea='idea',
                                                           model_route='ollama', initial_model='phi4'))}),
        daemon=True)
    thread.start()
    assert entered.wait(timeout=30), 'first advance never reached the model'
    head_during = store_b.journal_head(SID, OWNER)
    before = store_b.load_published(SID, OWNER)

    # A remote authenticated owner without the claim stops the session while
    # the other instance's advance is still in flight: one atomic transaction,
    # no lease, no waiting for the worker.
    registry2.stop(SID)
    mid = store_b.load_published(SID, OWNER)
    assert mid['control']['stopped'] is True and mid['generation'] == before['generation'] + 1

    # Release the model: the late advance finishes but its commit dies at the
    # fence - no new step, no journal page, no pointer move, no result served.
    release.set()
    thread.join(timeout=30)
    assert not thread.is_alive()
    after = store_b.load_published(SID, OWNER)
    assert after['control']['sequence'] == mid['control']['sequence']
    assert after['published_checkpoint_id'] == mid['published_checkpoint_id']
    assert store_b.journal_head(SID, OWNER) == head_during
    session1 = started['result'][0]
    assert session1.pg_stopped

    # Takeover honors the stopped marker and constructs nothing: the restored
    # session triggers no model build at all.
    models_before = dict(box)
    registry1.close()
    restored = registry2.get(SID)
    assert restored.pg_stopped and restored.program is None
    assert set(box) == set(models_before), 'no provider dispatch after Stop'


def test_stop_never_blocked_by_local_publication_failure(tmp_path, fake_ollama, fake_models, monkeypatch):
    conn = make_conn(tmp_path)
    store = PgSessionStore(conn)
    registry = SessionRegistry(default_program_factory, allow_db_provider_dispatch=True, session_store=store, owner_id=OWNER,
                               instance_id='w#1', lease_ttl=60.0)
    _OPEN['registries'].append(registry)
    registry.start(StartSessionRequest(session_id=SID, task_id='t', idea='idea',
                                       model_route='ollama', initial_model='phi4'))
    generation = store.load_published(SID, OWNER)['generation']

    # Even when this process can no longer publish session state at all, the
    # owner-authorized stop still lands atomically.
    def broken_save(session):
        raise RuntimeError('publication path down')

    monkeypatch.setattr(registry, '_save', broken_save)
    registry.stop(SID)
    row = store.load_published(SID, OWNER)
    assert row['control']['stopped'] is True and row['generation'] == generation + 1


def test_stale_lease_caller_stop_and_cross_owner_rejection(tmp_path, fake_ollama, fake_models):
    import time as _time

    conn_a = make_conn(tmp_path)
    conn_b = make_conn(tmp_path, init=False)
    registry1 = SessionRegistry(default_program_factory, allow_db_provider_dispatch=True, session_store=PgSessionStore(conn_a),
                                owner_id=OWNER, instance_id='w#1', lease_ttl=0.5)
    _OPEN['registries'].append(registry1)
    session, _ = registry1.start(StartSessionRequest(session_id=SID, task_id='t', idea='idea',
                                                     model_route='ollama', initial_model='phi4'))
    registry1.close()  # heartbeats halted; the lease runs out
    _time.sleep(0.7)
    registry2 = SessionRegistry(default_program_factory, allow_db_provider_dispatch=True, session_store=PgSessionStore(conn_b),
                                owner_id=OWNER, instance_id='w#2', lease_ttl=60.0)
    _OPEN['registries'].append(registry2)
    registry2.get(SID)  # live holder now

    # The stale instance's owner-authorized stop still wins: Stop is an owner
    # decision, not a lease privilege.
    registry1.stop(SID)
    row = PgSessionStore(conn_b).load_published(SID, OWNER)
    assert row['control']['stopped'] is True
    # The current holder's writes are fenced from now on.
    with pytest.raises(StopRequested):
        registry2._store.publish_step(registry2.get(SID).pg_claim,
                                      dict(row['control'], sequence=row['control']['sequence'] + 1))
    # A foreign owner cannot stop - or even detect - the session.
    foreign = PgSessionStore(conn_b)
    with pytest.raises(KeyError):
        foreign.stop(SID, '00000000-0000-0000-0000-000000000002')


def test_db_mode_rejects_production_factory_before_any_provider_call(tmp_path, monkeypatch):
    # PR A invariant: database persistence + the production program factory is
    # refused at construction, before any preflight or model call. Outbound is
    # monkeypatched to prove zero provider traffic on the rejected path.
    calls = []
    def record(*a, **k):
        calls.append((a, k))
        raise AssertionError('outbound provider call attempted')
    monkeypatch.setattr('httpx.get', record)
    monkeypatch.setattr('httpx.post', record)
    import socket
    monkeypatch.setattr(socket.socket, 'connect', record)
    store = PgSessionStore(make_conn(tmp_path))
    with pytest.raises(ValueError, match='cannot use the production program factory'):
        SessionRegistry(default_program_factory, session_store=store, owner_id=OWNER, instance_id='g#1')
    assert calls == []


def test_db_mode_custom_factory_cannot_bypass_provider_guard(tmp_path, monkeypatch):
    # A generic injected factory is not a bypass: without the internal opt-in,
    # the registry blocks preflight/model construction/invocation before any
    # outbound call, whatever the factory builds.
    calls = []
    def record(*a, **k):
        calls.append((a, k))
        raise AssertionError('outbound provider call attempted')
    monkeypatch.setattr('httpx.get', record)
    monkeypatch.setattr('httpx.post', record)
    from interview_engine.application import InterviewApplication
    from interview_engine.errors import UserFacingError
    def real_factory(session_id, request):
        return InterviewApplication(session_id, request.task_id, request.idea,
                                    artifact_root=str(tmp_path / 'artifacts'),
                                    model_route='ollama', initial_model='phi4')
    registry = SessionRegistry(real_factory, session_store=PgSessionStore(make_conn(tmp_path)),
                               owner_id=OWNER, instance_id='g#2')
    _OPEN['registries'].append(registry)
    with pytest.raises(UserFacingError, match='does not call providers'):
        registry.start(StartSessionRequest(task_id='t', idea='i', model_route='ollama'))
    assert calls == []


def test_registry_crash_mid_advance_staged_pause_no_rebill(tmp_path, fake_ollama, fake_models):
    """Full session-restore E2E through the registry: start publishes with every
    provider call staged; a crash mid-advance leaves a dispatched-without-response
    row (unknown outcome, possibly billed); the restored session pins the published
    pointer and the replayed logical call PAUSES with attempt evidence - no new
    provider call, no new ledger attempt, never a blind rerun. Drives the program's
    advance directly (the registry's respond control pipeline is a separate layer)."""
    import interview_engine.attempt_admission as aa
    from test_pg_staged_responses import FakeGateway
    from vmodel_contracts.interactive import InputResponse
    from interview_engine.errors import UserFacingError
    conn = make_conn(tmp_path)
    store = PgSessionStore(conn)
    registry = SessionRegistry(default_program_factory, allow_db_provider_dispatch=True, session_store=store, owner_id=OWNER,
                               instance_id='w#1', lease_ttl=1.0)
    _OPEN['registries'].append(registry)
    gw = FakeGateway()
    with aa.admission_gateway(gw, required=True):
        session, _ = registry.start(StartSessionRequest(session_id=SID, task_id='t', idea='idea',
                                                        model_route='ollama', initial_model='phi4'))
    # Every provider call of the first advance is staged with its durable
    # execution point; the saver hook advanced the point past 'bootstrap'.
    with store.conn.cursor() as cur:
        cur.execute("SELECT stage, execution_point, response FROM staged_provider_responses WHERE session_id=%s", (SID,))
        staged = cur.fetchall()
    assert staged and all(r['response'] is not None for r in staged), staged
    reserves_after_start = len(gw.reserved)

    # Crash during the NEXT advance: the provider call raises mid-flight, after
    # its staged row was dispatched and the ledger attempt reserved.
    class Boom(Exception):
        pass

    def crashing_invoke(messages):
        raise Boom('crash during provider call')

    fake_models['probing'].invoke = crashing_invoke  # the graph cached the instance
    open_input = session.program.open_input
    answer = InputResponse(input_id=open_input.input_id, actor_id='founder', value='customers book online')
    with aa.admission_gateway(gw, required=True):
        with pytest.raises(Boom):
            session.program.advance(answer)
    assert len(gw.uncertain_calls) == 1
    assert len(gw.reserved) == reserves_after_start + 1
    # The crashed call was staged under the session's PUBLISHED execution point
    # (never an unpublished intermediate checkpoint), open with response NULL.
    with store.conn.cursor() as cur:
        cur.execute("SELECT execution_point, response FROM staged_provider_responses WHERE session_id=%s AND response IS NULL", (SID,))
        open_rows = cur.fetchall()
    assert len(open_rows) == 1 and open_rows[0]['execution_point'] != 'bootstrap', open_rows

    registry.close()
    time.sleep(1.2)
    registry2 = SessionRegistry(default_program_factory, allow_db_provider_dispatch=True, session_store=PgSessionStore(conn),
                                owner_id=OWNER, instance_id='w#2', lease_ttl=60.0)
    _OPEN['registries'].append(registry2)
    restored = registry2.get(SID)
    probing_before = len(fake_models['probing'].replies)
    with aa.admission_gateway(gw, required=True):
        with pytest.raises(UserFacingError, match='Reconcile|result is missing'):
            restored.program.advance(answer)
    assert len(gw.reserved) == reserves_after_start + 1, 'resume reserved a new attempt for an unknown outcome'
    assert len(fake_models['probing'].replies) == probing_before, 'provider re-called during the paused replay'
