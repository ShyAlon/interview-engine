"""Hermetic fence/lease reproduction for the 2026-10-25 "Starting Interview" stall.

Live evidence: the engine answered POST /v1/sessions with
RUNNING/open_input=null/sequence=1 after two ~50s provider calls, then went
silent. These tests pin, with a controllable in-memory PG store:

1. A provider advance LONGER than the 30s lease TTL is safe while the
   heartbeat thread renews (own spawned connection, short transactions; no
   transaction or connection is shared with, or held across, the advance).
2. If the lease lapses mid-advance, the fenced post-advance publish raises
   StopRequested, _save_unfenced swallows it and sets pg_stopped - and the
   start response then shows STOP_REQUESTED. It can NEVER show the observed
   RUNNING shape: the fence path is eliminated as today's cause.
3. A program whose advance() returns None produces EXACTLY the observed
   RUNNING/open_input=null/sequence=1 response - the shape is pinned, and the
   deployed application's first advance has no code path that returns None.
4. Heartbeat failures log WARNING (reaches stderr via logging.lastResort even
   with no configured handler), so a DB-outage lapse WOULD be visible in
   captured docker logs; a starved heartbeat (alive=False never evaluated)
   logs nothing.
"""
import threading,time
import pytest
from uuid import uuid4
from vmodel_contracts.interactive import InputKind,InputRequest,ProcessStatus,StartSessionRequest
from interview_engine.http_service import SessionRegistry
from interview_engine.pg_session_store import SessionClaim
from interview_engine.stop_fence import StopRequested


class FakePgStore:
    """In-memory lease semantics: expiry is real wall-clock, heartbeat extends
    only an unexpired lease, publish_step fences on expiry - the same failure
    shape as pg_session_store, without a database."""
    def __init__(self,*,block_heartbeat=False,heartbeat_raises=False,heartbeat_alive=True):
        self.expiry=0.0;self.heartbeats=0;self.publishes=[];self.lock=threading.Lock()
        self.gate=threading.Event();self.gate.set() if not block_heartbeat else self.gate.clear()
        self.heartbeat_raises=heartbeat_raises;self.heartbeat_alive=heartbeat_alive
    def create(self,session_id,task_id,owner):pass
    def claim(self,session_id,owner,instance,ttl):
        with self.lock:self.expiry=time.time()+ttl
        return SessionClaim(session_id,owner,1,1,instance)
    def heartbeat_many(self,claims,ttl):
        lost=[]
        for claim in claims:
            if not self.gate.wait(timeout=10):raise RuntimeError("test gate")
            if self.heartbeat_raises:raise RuntimeError("database unreachable")
            with self.lock:
                self.heartbeats+=1
                now=time.time()
                if not self.heartbeat_alive or now>=self.expiry:lost.append(claim)
                else:self.expiry=now+ttl
        return lost
    def close(self):pass
    def publish_step(self,claim,control,**kwargs):
        with self.lock:
            if time.time()>=self.expiry:raise StopRequested("session lease fence rejected the publication")
            self.publishes.append(control["sequence"])


class FakeProgram:
    model_map=None
    finished=False
    def __init__(self,delay=0.0,result="question"):
        self.delay=delay;self.result=result
    def advance(self):
        if self.delay:time.sleep(self.delay)
        if isinstance(self.result,Exception):raise self.result
        if self.result is None:return None
        return InputRequest(input_id=uuid4(),kind=InputKind.TEXT,prompt=self.result)


def registry(store,program,ttl=0.3):
    # Shared heartbeat loop at the production ttl:beat ratio (one loop per
    # registry beats every held lease on one borrowed connection per tick).
    return SessionRegistry(program_factory=lambda sid,req:program,session_store=store,
        owner_id="owner-1",instance_id="test-instance",lease_ttl=ttl,allow_db_provider_dispatch=True,
        heartbeat_interval=max(ttl/3.0,0.05))

def start(reg):
    return reg.start(StartSessionRequest(task_id="task-1",idea="an idea",model_route="ollama"))


def test_long_advance_keeps_lease_while_heartbeat_renews():
    # Advance outlives the TTL twice over; renewal on the heartbeat's own
    # thread/connection must keep every fenced write valid.
    store=FakePgStore()
    session,_=start(registry(store,FakeProgram(delay=0.6),ttl=0.3))
    assert session.status==ProcessStatus.WAITING_INPUT and session.open_input is not None
    assert store.publishes==[1,2]          # pre-advance and post-advance publishes both fenced OK
    assert store.heartbeats>=2             # renewal ran DURING the blocked advance
    assert session.pg_stopped is False


def test_lease_lapse_fences_publish_and_surfaces_stop_overlay_not_running():
    # Heartbeat starved past expiry: the post-advance publish fences, is
    # swallowed by _save_unfenced, and the response overlays STOP_REQUESTED.
    # The 2026-10-25 live response was RUNNING - so today's stall did NOT come
    # through the lease fence path.
    store=FakePgStore(block_heartbeat=True)
    session,_=start(registry(store,FakeProgram(delay=0.8),ttl=0.3))
    store.gate.set()
    assert session.pg_stopped is True
    assert session.snapshot().process_status==ProcessStatus.STOP_REQUESTED
    assert store.publishes==[1]            # pre-advance publish landed; post-advance fenced out


def test_stop_requested_from_advance_warns_and_marks_stopped(caplog):
    # 2026-10-25 instrumentation (user directive: triage logging on the start
    # path): the except-StopRequested return in start, previously the
    # silent-success defect pinned here, now emits a stop.swallowed WARNING.
    # It still ALWAYS pairs with pg_stopped, so the response snapshot says
    # STOP_REQUESTED, never the observed RUNNING.
    store=FakePgStore()
    with caplog.at_level("DEBUG"):
        session,_=start(registry(store,FakeProgram(result=StopRequested("session lease lost (stop or takeover)"))))
    assert session.pg_stopped is True
    assert session.snapshot().process_status==ProcessStatus.STOP_REQUESTED
    warnings=[r for r in caplog.records if r.levelname=="WARNING"]
    assert any("stop.swallowed" in r.getMessage() and "reason=stop-requested" in r.getMessage() for r in warnings),\
        [r.getMessage() for r in caplog.records]


def test_swallowed_stop_logs_the_real_cause(caplog):
    # 2026-10-26 live stall: the swallow hid which fence tripped behind
    # "reason=stop-requested". The warning must now carry the swallowed
    # exception's own type and message.
    store=FakePgStore()
    with caplog.at_level("DEBUG"):
        session,_=start(registry(store,FakeProgram(result=StopRequested("session lease lost (stop or takeover)"))))
    warnings=[r.getMessage() for r in caplog.records if r.levelname=="WARNING"]
    assert any("stop.swallowed" in m and "cause=StopRequested: session lease lost (stop or takeover)" in m for m in warnings),warnings


def test_none_advance_produces_the_exact_observed_shape():
    # A None return from advance() yields RUNNING/open_input=null/sequence=1 -
    # precisely the live 2026-10-25 response. The deployed application's first
    # advance has no path that returns None (every loop branch returns an
    # input or raises), so this shape is unreachable with the tagged code.
    store=FakePgStore()
    session,_=start(registry(store,FakeProgram(result=None)))
    snap=session.snapshot()
    assert snap.process_status==ProcessStatus.RUNNING and snap.next_step==10 and snap.open_input is None
    assert snap.sequence==1 and snap.route_version==0


def test_heartbeat_failure_logs_warning_visible_on_stderr(caplog):
    store=FakePgStore(heartbeat_raises=True)
    import logging
    with caplog.at_level(logging.WARNING,logger="interview_engine.http"):
        start(registry(store,FakeProgram(delay=0.0),ttl=0.2))
        time.sleep(0.6)                    # let several beats fail
    assert any("lease heartbeat failed" in r.getMessage() for r in caplog.records)


def test_heartbeat_alive_false_marks_stopped_and_overlays_stop():
    store=FakePgStore(heartbeat_alive=False)
    session,_=start(registry(store,FakeProgram(delay=0.0),ttl=0.3))
    time.sleep(0.3)                        # one beat is enough
    assert session.pg_stopped is True
    assert session.snapshot().process_status==ProcessStatus.STOP_REQUESTED


def test_heartbeat_start_logs_session_and_timing(caplog):
    # The heartbeat thread was invisible: a parked session showed no heartbeat
    # evidence at all. Starting it must log the session and the beat cadence.
    import logging
    store=FakePgStore()
    with caplog.at_level(logging.INFO,logger="interview_engine.http"):
        session,_=start(registry(store,FakeProgram(delay=0.0),ttl=0.3))
    infos=[r.getMessage() for r in caplog.records if r.levelname=="INFO"]
    assert any("lease heartbeat started" in m and "session=" in m and "interval=" in m for m in infos),infos


def test_heartbeat_lost_lease_logs_the_takeover(caplog):
    # 2026-10-26 stall: the alive=False path marked the session stopped in
    # silence. That is the one line that would have named the cause.
    import logging
    store=FakePgStore(heartbeat_alive=False)
    with caplog.at_level(logging.INFO,logger="interview_engine.http"):
        session,_=start(registry(store,FakeProgram(delay=0.0),ttl=0.3))
        time.sleep(0.3)
    warnings=[r.getMessage() for r in caplog.records if r.levelname=="WARNING"]
    assert any("lease heartbeat lost" in m and "session=" in m for m in warnings),warnings
    assert session.pg_stopped is True


def test_heartbeat_loop_survives_repeated_failures(caplog):
    # The shared loop beats every session in the process: it must NOT give up
    # after consecutive failures (that would silently un-beat the healthy
    # sessions too). Each failed tick logs; the loop keeps running; the lease
    # lapse itself is reported when a later tick observes it.
    import logging
    store=FakePgStore(heartbeat_raises=True)
    with caplog.at_level(logging.INFO,logger="interview_engine.http"):
        session,_=start(registry(store,FakeProgram(delay=0.0),ttl=0.2))
        time.sleep(0.6)
        reg_loop_alive=True
    failures=[r for r in caplog.records if "lease heartbeat failed" in r.getMessage()]
    assert len(failures)>=2,"shared loop must keep beating after failures, logging each one"
    assert session.pg_stopped is False  # transport failure is not a lost lease; writes fail closed at the fence
