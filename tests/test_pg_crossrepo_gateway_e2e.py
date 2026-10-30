from conftest import managed_pool
"""Cross-repo acceptance: the interview engine's staged paid calls against the
REAL delivery-api WorkerAdmissionGateway + provider-attempt ledger, over the
REAL migration chain (pg_harness), through the registry respond path and the
actual HTTP surface.

Serving vs dispatched work: session START is the serving flow - the engine owns
the engine_sessions binding row and its opening provider calls are not
dispatched (exactly the pre-dispatch mode the engine runs in today; the cloud
dispatch guard is unchanged). Every RESPOND advance here runs under the real
gateway, which is the paid path PR B hardens.

Legs (per PR B acceptance):
a. Crash AFTER the provider result was staged but BEFORE settle (process death:
   the settle handoff AND the best-effort uncertain mark both fail): the replay
   settles the EXISTING attempt from the staged trusted usage BEFORE the graph
   gets the result - one attempt, settled once, no new reserve.
b. Re-settling the settled attempt from the same staged usage is a no-op: the
   ledger is byte-identical afterwards.
c. Crash DURING the provider call: the attempt lands 'uncertain' on the real
   ledger, the staged row stays open, and the replay pauses - no new attempt,
   no provider re-call, no auto-settle of the unknown outcome.
d. HTTP answer idempotency across a full restart with the real gateway.
"""
import os
import sys
import time
import uuid
from pathlib import Path

import pytest

pgserver = pytest.importorskip('pgserver', reason='pgserver not installed')
psycopg = pytest.importorskip('psycopg', reason='psycopg not installed')
import psycopg.rows

DELIVERY = os.getenv('DELIVERY_TEST_DELIVERY_SOURCE')
pytestmark = pytest.mark.skipif(not DELIVERY, reason='requires DELIVERY_TEST_DELIVERY_SOURCE (delivery-api checkout)')

from vmodel_contracts.interactive import StartSessionRequest, InputResponse

import interview_engine.attempt_admission as aa
from interview_engine.errors import UserFacingError
from interview_engine.http_service import SessionRegistry, default_program_factory, create_app
from interview_engine.pg_session_store import PgSessionStore
from test_http_service import TOKEN
from test_pg_application_wiring import fake_ollama, fake_models, FakeModel, OWNER, _OPEN

Q = {'provider': 'fake', 'model': 'fake', 'price_version': 'test-only', 'bound_version': 'test-only',
     'input_max': 10, 'output_max': 1000, 'expense_input': 1, 'expense_output': 2, 'retail_input': 3, 'retail_output': 4}
A_ID = '00000000-0000-0000-0000-000000000001'


@pytest.fixture
def crossrepo(tmp_path, fake_ollama, fake_models, monkeypatch):
    """Real migration chain (DA pg_harness) + real DA store + engine store over
    the SAME database; a fresh session id per test (the harness DB persists)."""
    monkeypatch.syspath_prepend(str(Path(DELIVERY) / 'tests'))
    monkeypatch.syspath_prepend(str(Path(DELIVERY) / 'src'))
    import pg_harness
    pg_harness.start()
    dsn = pg_harness.STATE.get('dsn')
    if not dsn:
        pytest.skip('pg_harness unavailable: ' + pg_harness.STATE.get('reason', ''))
    from delivery_api.postgres_store import PostgresStore
    from delivery_api.producer_admission import WorkerAdmissionGateway
    from delivery_api.tenancy import TenantScope, tenant_context

    # Every scripted model reply carries billable usage for the real settlement path.
    orig_invoke = FakeModel.invoke
    def invoke_with_usage(self, messages):
        r = orig_invoke(self, messages)
        r.usage_metadata = {'input_tokens': 2, 'output_tokens': 3, 'total_tokens': 5}
        return r
    monkeypatch.setattr(FakeModel, 'invoke', invoke_with_usage)

    da = PostgresStore(dsn.replace('user=sandbox', 'user=delivery_app'))
    owner = str(uuid.uuid4())  # per-test: the ledger hold is owner-scoped (an
    # owner with an uncertain attempt is refused new reserves until reconciled)
    scope = TenantScope(owner, owner)
    with psycopg.connect(dsn) as c:
        c.execute('INSERT INTO auth.users(id,email,email_confirmed_at) VALUES (%s,%s,now()) ON CONFLICT DO NOTHING',
                  (owner, owner + '@example.test'))
        c.execute('INSERT INTO auth.sessions(id,user_id) VALUES (%s,%s) ON CONFLICT DO NOTHING', (owner, owner))
        c.execute('INSERT INTO admission_limits VALUES (%s,%s,1000,300000,700000,300000,700000,60,100,20) ON CONFLICT(owner_id) DO NOTHING',
                  (owner, 'test-only'))
    sid = str(uuid.uuid4())
    with tenant_context(scope):
        tid = da.create_task('crossrepo-' + uuid.uuid4().hex)['task_id']
        da.set_engine_session(tid,sid)

    def quote_resolver(**kw):
        return ({**Q, 'provider': kw['provider'], 'model': kw['model'],
                 'input_max': 10, 'output_max': kw['max_output_tokens']}, 'digest-' + kw['stage'])

    def bind_and_claim():
        """Production dispatch order: the engine start owns the binding row, the
        DA task binds to it, then the ledger job is claimed."""
        with tenant_context(scope):
            da.set_engine_session(tid, sid)
            job = da.enqueue_worker(tid, 'phase:needs')
            da.claim_worker(job)
        return job

    class _TxGateway:
        """WorkerAdmissionGateway called the way the producer runner calls it:
        each ledger operation inside the store's scoped transaction."""
        def __init__(self, job):
            self._gw = WorkerAdmissionGateway(da, job, require_prepaid=False,
                                              resolve_quote=quote_resolver,
                                              measure_usage=lambda response: {k: response.usage_metadata[k]
                                                                              for k in ('input_tokens', 'output_tokens')})
        def __getattr__(self, name):
            fn = getattr(self._gw, name)
            if name not in ('reserve', 'reserve_serialized', 'start', 'settle', 'uncertain', 'settle_staged'):
                return fn
            def call(*a, **kw):
                with da._postgres.transaction():
                    return fn(*a, **kw)
            return call

    def make_gateway(job):
        return _TxGateway(job)

    engine_conn = psycopg.connect(dsn, autocommit=False, row_factory=psycopg.rows.dict_row)
    engine_store = PgSessionStore(managed_pool(engine_conn.info.dsn),claims=(owner,scope.session_id))
    _OPEN['conns'].append(engine_conn)
    from interview_engine.pg_checkpoint import bootstrap_checkpoint_schema
    bootstrap = psycopg.connect(dsn, autocommit=True)
    bootstrap_checkpoint_schema(bootstrap)
    bootstrap.close()
    yield {'da': da, 'scope': scope, 'tenant_context': tenant_context,
           'bind_and_claim': bind_and_claim, 'make_gateway': make_gateway,
           'engine_store': engine_store, 'tid': tid, 'sid': sid, 'models': fake_models,
           'owner': owner}


def _registry(store, instance, owner, ttl=60.0):
    r = SessionRegistry(default_program_factory, allow_db_provider_dispatch=True,
                        session_store=store, owner_id=owner, instance_id=instance, lease_ttl=ttl)
    _OPEN['registries'].append(r)
    return r


def _start(registry, crossrepo):
    session, _ = registry.start(StartSessionRequest(session_id=crossrepo['sid'], task_id=crossrepo['tid'],
                                                    idea='idea', model_route='ollama', initial_model='phi4'))
    return session


def _attempts(da, job, scope=None, tenant_context=None):
    assert scope is not None, 'ledger reads need the tenant scope (RLS)'
    with tenant_context(scope):
        with da._postgres.transaction() as c:
            return [dict(r) for r in c.execute("SELECT attempt_id, state, usage FROM provider_attempts WHERE job_id=%s",
                                               (job,)).fetchall()]


def _answer(session, value='customers book online'):
    open_input = session.program.open_input
    return InputResponse(input_id=open_input.input_id, actor_id='founder', value=value)


class CrashOnSettleHandoff:
    """Process-death proxy: the settle handoff fails AND the best-effort
    uncertain mark fails with it - the attempt stays 'dispatched' on the ledger
    with its staged row complete: the after-complete/before-settle crash window."""
    def __init__(self, real):
        self.real = real
        self.armed = True
    def __getattr__(self, name):
        return getattr(self.real, name)
    def settle(self, attempt, response):
        if self.armed:
            self.armed = False
            raise RuntimeError('process died in the settle handoff')
        return self.real.settle(attempt, response)
    def uncertain(self, attempt):
        raise RuntimeError('process dead: the uncertain mark never landed')


def test_replay_settles_existing_attempt_once_then_zero_ledger_changes(crossrepo):
    da, scope, tenant_context = crossrepo['da'], crossrepo['scope'], crossrepo['tenant_context']
    store, models = crossrepo['engine_store'], crossrepo['models']
    registry = _registry(store, 'w#1', crossrepo['owner'], ttl=1.0)
    session = _start(registry, crossrepo)
    job = crossrepo['bind_and_claim']()
    real_gw = crossrepo['make_gateway'](job)
    crashing = CrashOnSettleHandoff(real_gw)
    assert _attempts(da, job, scope, tenant_context) == [], 'no dispatched work before the first respond'

    # The respond advance's provider call stages+completes; the settle handoff dies.
    answer = _answer(session)
    with tenant_context(scope):
        with aa.admission_gateway(crashing, required=True):
            with pytest.raises(RuntimeError, match='settle handoff'):
                registry.respond(crossrepo['sid'], answer)
    replies_before = len(models['probing'].replies)  # the crashed call consumed one
    dispatched = [r for r in _attempts(da, job, scope, tenant_context) if r['state'] == 'dispatched']
    assert len(dispatched) == 1, 'exactly one attempt left dispatched by the crash window'

    # Restart: the replay settles the EXISTING attempt from staged usage BEFORE
    # the graph sees the result; no new attempt, no provider re-call for it.
    registry.close()
    time.sleep(1.2)
    registry2 = _registry(PgSessionStore(store.pool,claims=store.claims), 'w#2', crossrepo['owner'])
    with tenant_context(scope):
        with aa.admission_gateway(real_gw, required=True):
            registry2.respond(crossrepo['sid'], answer)
    rows = _attempts(da, job, scope, tenant_context)
    assert len(rows) == 1, 'the replay reserved nothing new: ' + repr(rows)
    assert rows[0]['state'] == 'settled' and rows[0]['usage'] == {'input_tokens': 2, 'output_tokens': 3}, \
        'the crashed attempt settled from the STAGED usage during replay'
    assert len(models['probing'].replies) == replies_before, 'the crashed call was not re-billed to the provider'

    # (b) Re-settling the settled attempt from the same staged usage: no-op.
    with tenant_context(scope):
        real_gw.settle_staged(rows[0]['attempt_id'], {'input_tokens': 2, 'output_tokens': 3, 'total_tokens': 5})
    assert _attempts(da, job, scope, tenant_context) == rows, 'settled replay changed the ledger'


def test_crash_during_provider_call_marks_uncertain_and_replay_pauses(crossrepo):
    da, scope, tenant_context = crossrepo['da'], crossrepo['scope'], crossrepo['tenant_context']
    store, models = crossrepo['engine_store'], crossrepo['models']
    registry = _registry(store, 'w#1', crossrepo['owner'], ttl=1.0)
    session = _start(registry, crossrepo)
    job = crossrepo['bind_and_claim']()
    real_gw = crossrepo['make_gateway'](job)

    class Boom(Exception):
        pass
    original_invoke = models['probing'].invoke
    models['probing'].invoke = lambda m: (_ for _ in ()).throw(Boom('provider crashed mid-call'))
    answer = _answer(session)
    with tenant_context(scope):
        with aa.admission_gateway(real_gw, required=True):
            with pytest.raises(Boom):
                registry.respond(crossrepo['sid'], answer)
    replies_before = len(models['probing'].replies)  # Boom precedes the provider call
    rows = _attempts(da, job, scope, tenant_context)
    assert len(rows) == 1 and rows[0]['state'] == 'uncertain', \
        'the crashed call reserved exactly one attempt and it landed uncertain on the REAL ledger'

    # Restart: the replay pauses with attempt evidence - no new attempt, no
    # provider call, no auto-settle of the unknown outcome.
    registry.close()
    time.sleep(1.2)
    registry2 = _registry(PgSessionStore(store.pool,claims=store.claims), 'w#2', crossrepo['owner'])
    with tenant_context(scope):
        with aa.admission_gateway(real_gw, required=True):
            with pytest.raises(UserFacingError, match='Reconcile|result is missing'):
                registry2.respond(crossrepo['sid'], answer)
    rows = _attempts(da, job, scope, tenant_context)
    assert len(rows) == 1 and rows[0]['state'] == 'uncertain', 'the paused replay reserved nothing and settled nothing'
    assert len(models['probing'].replies) == replies_before, 'the provider was not re-called'


def test_http_answer_idempotency_across_restart(crossrepo):
    """The HTTP surface with the real gateway installed the way the worker's
    process wiring installs it (per-request execution context). Session start
    stays serving-mode (unbilled opening question); every respond runs paid
    through the gateway. The committed answer replayed after a full restart is
    accepted idempotently: no double advance, no ledger change."""
    da, scope, tenant_context = crossrepo['da'], crossrepo['scope'], crossrepo['tenant_context']
    store = crossrepo['engine_store']
    registry = _registry(store, 'w#1', crossrepo['owner'], ttl=1.0)
    box = {'gw': None}

    class GatewayASGI:
        """Installs the admission gateway into each dispatched request's execution
        context (the worker's process-level seam); session start stays serving-mode."""
        def __init__(self, app):
            self.app = app
        async def __call__(self, asgi_scope, receive, send):
            if asgi_scope['type'] != 'http':
                await self.app(asgi_scope, receive, send)
                return
            serving = asgi_scope['method'] == 'POST' and asgi_scope['path'] == '/v1/sessions'
            with tenant_context(scope):
                if serving:
                    await self.app(asgi_scope, receive, send)
                    return
                assert box['gw'] is not None, 'dispatched request before the job was claimed'
                with aa.admission_gateway(box['gw'], required=True):
                    await self.app(asgi_scope, receive, send)

    from fastapi.testclient import TestClient
    sid = crossrepo['sid']

    def make_client(reg):
        return TestClient(GatewayASGI(create_app(reg, service_token=TOKEN)),
                          headers={'X-Interview-Engine-Token': TOKEN})

    def respond(client, input_id):
        return client.post(f'/v1/sessions/{sid}/inputs/{input_id}',
                           json={'protocol_version': 1, 'input_id': input_id,
                                 'value': 'customers book online', 'actor_id': 'founder'})

    client = make_client(registry)
    started = client.post('/v1/sessions', json={'protocol_version': 1, 'session_id': sid,
                                                'task_id': crossrepo['tid'], 'idea': 'booking idea',
                                                'model_route': 'ollama', 'initial_model': 'phi4'})
    assert started.status_code == 201, started.text
    open_input = started.json()['snapshot']['open_input']
    job = crossrepo['bind_and_claim']()
    box['gw'] = crossrepo['make_gateway'](job)
    assert _attempts(da, job, scope, tenant_context) == [], 'session start is serving-mode: no ledger attempts'
    first = respond(client, open_input['input_id'])
    assert first.status_code == 200, first.text
    committed = store.load_published(sid, crossrepo['owner'])['control']['sequence']
    ledger_after_commit = _attempts(da, job, scope, tenant_context)
    assert ledger_after_commit and all(r['state'] == 'settled' for r in ledger_after_commit)

    replay = respond(client, open_input['input_id'])
    assert replay.status_code == 200, replay.text
    assert store.load_published(sid, crossrepo['owner'])['control']['sequence'] == committed

    registry.close()
    time.sleep(1.2)
    registry2 = _registry(PgSessionStore(store.pool,claims=store.claims), 'w#2', crossrepo['owner'])
    client2 = make_client(registry2)
    replayed = respond(client2, open_input['input_id'])
    assert replayed.status_code == 200, replayed.text
    assert store.load_published(sid, crossrepo['owner'])['control']['sequence'] == committed, \
        'a committed answer replayed after restart must never double-advance'
    assert _attempts(da, job, scope, tenant_context) == ledger_after_commit, 'the idempotent replay changed the ledger'
