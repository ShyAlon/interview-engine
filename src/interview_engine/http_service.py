"""HTTP boundary for the interview engine.

This module owns transport and session/input lifecycle only. Interview domain
work calls ``SessionRegistry.publish``; it never reads HTTP or presentation
text. Control decisions use protocol enums and identifiers exclusively.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import contextmanager, nullcontext
import json
import threading
import time
from pathlib import Path
import hmac
import logging
import os
from threading import RLock
from uuid import UUID, uuid4
from vmodel_contracts.runtime import artifacts_root
from .tenant_worker import worker_root

from fastapi import Depends, FastAPI, Header, HTTPException
from .errors import UserFacingError
from .tenant_worker import current_owner as _tenant_owner

log=logging.getLogger("interview_engine.http")

from vmodel_contracts.interactive import (
    EngineSnapshot, EventType, InputRequest, InputResponse, InteractiveEvent,
    NextStep, ProcessStatus, ProtocolError, ProtocolErrorCode,
    StartSessionRequest, StartSessionResponse, SubmitInputResponse,
    ModelRouteSwitchRequest, ModelRouteSwitchResponse, RouteChangePayload,
)


@dataclass
class EngineSession:
    session_id: str
    task_id: str
    phase_id: int = 0
    sequence: int = 0
    status: ProcessStatus = ProcessStatus.CREATED
    next_step: NextStep = NextStep.RUN
    open_input: InputRequest | None = None
    responses: dict[UUID, InputResponse] = field(default_factory=dict)
    program: object | None = None
    request: StartSessionRequest | None = None
    route_version: int = 0
    route_history: list[RouteChangePayload] = field(default_factory=list)
    write_lock: RLock = field(default_factory=RLock, repr=False, compare=False)
    published: EngineSnapshot | None = field(default=None, repr=False, compare=False)
    stop_fence: object | None = field(default=None, repr=False, compare=False)
    pg_claim: object | None = field(default=None, repr=False, compare=False)
    pg_stopped: bool = field(default=False, repr=False, compare=False)

    def snapshot(self) -> EngineSnapshot:
        stopped=self.stop_fence.state if self.stop_fence is not None else ('STOP_REQUESTED' if self.pg_stopped else None)
        return EngineSnapshot(session_id=self.session_id,task_id=self.task_id,sequence=self.sequence,
            process_status=ProcessStatus[stopped] if stopped else self.status,next_step=NextStep.STOP if stopped else self.next_step,open_input=None if stopped else self.open_input,
            model_route=getattr(self.program,"model_route",None) or (self.request.model_route if self.request else None),
            model_map=getattr(self.program,"model_map",None),route_version=self.route_version,route_history=self.route_history)


# Pool connection budget (Supabase pooler, session mode: 15 clients max).
# delivery-api pins POOL_OPEN_MAX=6; this service holds per SERVED session:
# registry serving conn, lease heartbeat conn, application base conn, and the
# saver's write conn (lazy since 2026-10-26: only during fenced writes).
# Reads of parked sessions hold none. Two concurrent sessions: 6 + 2*4 = 14 <= 15.
ENGINE_PEAK_SESSION_CONNS = 4


class SessionRegistry:
    """Thread-safe application port behind the HTTP adapter."""
    def __init__(self,program_factory=None,state_dir:Path|str|None=None,*,session_store=None,owner_id=None,instance_id:str|None=None,lease_ttl:float=150.0,heartbeat_interval:float=60.0,allow_db_provider_dispatch:bool=False):
        if state_dir is not None and session_store is not None:
            raise ValueError('choose file or database session persistence, not both')
        if session_store is not None and program_factory is default_program_factory and not allow_db_provider_dispatch:
            # PR A invariant: the database persistence path never dispatches to
            # a provider before staged responses land (PR B). The production
            # factory preflights and calls real models, so the combination is
            # refused at construction - before any preflight or model call can
            # run. Tests opt in explicitly; no config/env/HTTP route enables
            # database persistence at all (constructor-only wiring).
            raise ValueError('database session persistence cannot use the production program factory until staged responses land (PR B); pass an explicit isolated program_factory')
        self._allow_db_dispatch=allow_db_provider_dispatch
        self._sessions: dict[str,EngineSession]={}
        self._program_factory=program_factory
        self._state_dir=Path(state_dir) if state_dir is not None else None
        self._store=session_store
        self._owner_id=owner_id
        if instance_id is None:
            import socket
            instance_id=f"{socket.gethostname()}#{os.getpid()}#{uuid4().hex[:8]}"
        self._instance=instance_id   # execution-instance lease identity, never a bare hostname
        self._lease_ttl=lease_ttl
        self._heartbeat_interval=heartbeat_interval
        self._hb_thread=None
        self._hb_halt=None
        self._lock=RLock()

    def _ensure_heartbeat_loop(self) -> None:
        # ONE heartbeat loop per registry for every held lease: a lease beat is
        # one short UPDATE, so all sessions share a tick on one borrowed
        # connection instead of pinning a connection per session per thread.
        # What the beat protects: it extends lease_expires_at so a long advance
        # keeps its lease while the TTL stays short, and it notices stop or
        # takeover promptly (a lost beat marks the session stopped locally).
        # Fencing itself never depends on the beat: every fenced write
        # re-checks (generation, epoch, expiry) in the same statement.
        if self._store is None:return
        with self._lock:
            if self._hb_thread is not None and self._hb_thread.is_alive():return
            self._hb_halt=threading.Event()
            self._hb_thread=threading.Thread(target=self._heartbeat_loop,daemon=True,name=f"lease-heartbeat-{self._instance}")
            self._hb_thread.start()

    def _heartbeat_loop(self) -> None:
        interval=self._heartbeat_interval
        while not self._hb_halt.wait(interval):
            with self._lock:
                claims=[s.pg_claim for s in self._sessions.values()
                        if s.pg_claim is not None and not s.pg_stopped]
            if not claims:continue
            try:
                lost=self._store.heartbeat_many(claims,self._lease_ttl)
            except Exception:
                # Transient failure: the leases lapse on their own and a later
                # tick reports them lost; fenced writes fail closed regardless.
                log.warning('lease heartbeat failed instance=%s - beat skipped; leases lapse on their own and fenced writes fail closed',self._instance,exc_info=True)
                continue
            if lost:
                with self._lock:
                    for claim in lost:
                        session=self._sessions.get(claim.session_id)
                        if session is not None and not session.pg_stopped:
                            log.warning('session lease heartbeat lost session=%s - the lease is no longer held; session marked stopped and fenced writes will fail closed',claim.session_id)
                            session.pg_stopped=True

    def close(self)->None:
        # Stop serving: halt the shared heartbeat loop, join it with a bounded
        # wait, and leave no connection checked out. Leases expire on their
        # own; a live session can then be claimed by another instance. Fences
        # close every later write from this process even before expiry.
        with self._lock:sessions=list(self._sessions.values())
        if self._hb_halt is not None:self._hb_halt.set()
        if self._hb_thread is not None:
            self._hb_thread.join(timeout=5.0)
            if self._hb_thread.is_alive():log.warning('lease heartbeat loop did not join instance=%s',self._instance)
        for session in sessions:
            # Voluntary lease release at a committed boundary: the write lock
            # waits out any in-flight turn, so another host can claim
            # immediately without resuming mid-call state. Stop stays
            # terminal - release changes only who may serve next.
            claim=getattr(session,'pg_claim',None)
            if claim is not None and self._store is not None:
                try:
                    with session.write_lock:
                        self._store.release(claim)
                except Exception:
                    log.warning('lease.release.failed session=%s',session.session_id,exc_info=True)
        # Programs hold no connections of their own: the saver borrows from
        # the process pool per operation. The store closes the pool only when
        # it built it (standalone wiring); an injected shared pool is closed
        # by its owner.
        if self._store is not None:
            try:self._store.close()
            except Exception:log.warning('session store close failed',exc_info=True)

    def _pg_owner(self)->str:
        # Owner identity comes from the authenticated tenant context (or the
        # registry's configured local owner) - never from a client-supplied id.
        owner=_tenant_owner() or self._owner_id
        if owner is None:raise ProtocolFailure(ProtocolErrorCode.INVALID_RESPONSE,'database session persistence requires an owner scope')
        return str(owner)

    @staticmethod
    def _publish(session:EngineSession)->EngineSnapshot:
        # Publish a distinct validated value whenever control state changes.
        # HTTP readers receive their own copy and cannot mutate this cache.
        session.published=session.snapshot().model_copy(deep=True)
        return session.published

    def snapshot(self,session_id:str)->EngineSnapshot:
        # Active control reads do not touch an in-flight graph or its write lock.
        session=self.get(session_id,serve=False)
        if session.published is None:raise KeyError(session_id)
        if session.stop_fence is not None and session.stop_fence.state:return session.snapshot()
        if session.pg_stopped:return session.snapshot()
        return session.published.model_copy(deep=True)

    def start(self, request: StartSessionRequest) -> tuple[EngineSession,InteractiveEvent]:
        session=EngineSession(session_id=str(request.session_id or uuid4()),task_id=request.task_id,sequence=1,
            status=ProcessStatus.RUNNING,next_step=NextStep.RUN,request=request)
        log.info("session.start.accept session=%s task=%s route=%s",session.session_id,request.task_id,request.model_route)
        from .tenant_worker import _owner
        if _owner.get() is not None:
            import re
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}',request.task_id):
                raise ProtocolFailure(ProtocolErrorCode.INVALID_RESPONSE,'invalid scoped task identity')
            request=request.model_copy(update={'session_id':UUID(session.session_id)})
            session.request=request
        # Reserve the ID immediately, but don't advertise it until preflight
        # succeeds. No model call runs under the global registry lock.
        with self._lock:
            if session.session_id in self._sessions or (self._state_path(session.session_id) is not None and self._state_path(session.session_id).exists()):
                raise ProtocolFailure(ProtocolErrorCode.INVALID_RESPONSE,"session ID already exists")
            self._sessions[session.session_id]=session
        if self._store is not None:
            owner=self._pg_owner()
            t_create=time.monotonic()
            try:self._store.create(session.session_id,session.task_id,owner)
            except Exception as exc:
                with self._lock:self._sessions.pop(session.session_id,None)
                if 'UniqueViolation' in type(exc).__name__ or 'duplicate key' in str(exc):
                    log.info("pg.registry.refused session=%s reason=duplicate-id",session.session_id)
                    raise ProtocolFailure(ProtocolErrorCode.INVALID_RESPONSE,"session ID already exists")
                raise
            log.info("pg.registry.created session=%s task=%s dur_ms=%d",session.session_id,session.task_id,int((time.monotonic()-t_create)*1000))
            claim=self._store.claim(session.session_id,owner,self._instance,self._lease_ttl)
            if claim is None:
                with self._lock:self._sessions.pop(session.session_id,None)
                log.info("pg.lease.refused session=%s owner=%s reason=lease-held",session.session_id,owner)
                raise ProtocolFailure(ProtocolErrorCode.INVALID_RESPONSE,"session lease held by another live instance")
            session.pg_claim=claim
            log.info("pg.lease.claimed session=%s owner=%s generation=%s epoch=%s lease_ttl=%s dur_ms=%d",session.session_id,owner,claim.generation,claim.epoch,self._lease_ttl,int((time.monotonic()-t_create)*1000))
            log.info("lease heartbeat started session=%s interval=%.1f ttl=%.1f",session.session_id,self._heartbeat_interval,self._lease_ttl)
            self._ensure_heartbeat_loop()
        self._ensure_stop_fence(session)
        started=self._event(session,EventType.RUN_STARTED)
        try:
            with session.write_lock:
                if self._program_factory is default_program_factory:
                    from . import session_models
                    session_models.validate(session_models.resolve(request.model_route,request.initial_model))
                    log.info("model.route.preflight.accept task=%s route=%s",request.task_id,request.model_route)
                # Preflight has passed; publish a RUNNING snapshot before
                # constructing or invoking the potentially slow program.
                self._publish(session)
                log.info("session.published session=%s seq=%s status=preflight-running",session.session_id,session.sequence)
                if self._program_factory is not None:
                    t_construct=time.monotonic()
                    log.info("program.construct.begin session=%s",session.session_id)
                    with self._dispatch_guard():
                        if self._store is not None and self._program_factory is default_program_factory:
                            session.program=self._prepare_program(session,lambda:self._default_pg_program(session,request))
                        else:
                            session.program=self._prepare_program(session,lambda:self._program_factory(session.session_id,request))
                    log.info("model.route.resolved session=%s task=%s route=%s stages=%s",session.session_id,request.task_id,request.model_route,json.dumps(getattr(session.program,"model_map",None)))
                    log.info("program.construct.end session=%s dur_ms=%d",session.session_id,int((time.monotonic()-t_construct)*1000))
                    self._save(session)  # factory/pinned model is now ready, before long advance
                    pending=self._advance(session)
                    with self._control_commit(session):
                        if pending is not None:
                            session.sequence+=1;session.open_input=pending;session.status=ProcessStatus.WAITING_INPUT;session.next_step=NextStep.WAIT_INPUT
                            log.info("open_input.set session=%s seq=%s status=WAITING_INPUT",session.session_id,session.sequence)
                        self._save(session)
                        self._publish(session)
                else:
                    with self._control_commit(session):
                        self._save(session)
                        self._publish(session)
            log.info("start.return session=%s seq=%s status=%s next_step=%s open_input=%s",session.session_id,session.sequence,int(session.status),int(session.next_step),'null' if session.open_input is None else 'set')
            return session,started
        except __import__('interview_engine.stop_fence',fromlist=['StopRequested']).StopRequested as exc:
            # The reason string alone used to hide WHICH fence tripped; the
            # swallowed exception's own message names the raise site.
            log.warning("stop.swallowed session=%s task=%s reason=stop-requested cause=%s: %s - start returns success with open_input=null",session.session_id,request.task_id,type(exc).__name__,str(exc)[:300])
            session.open_input=None
            return session,started
        except Exception as exc:
            if session.stop_fence is not None and session.stop_fence.requested:
                log.warning("stop.swallowed session=%s task=%s reason=fence-requested cause=%s: %s - start returns success with open_input=null",session.session_id,request.task_id,type(exc).__name__,str(exc)[:300])
                session.open_input=None
                return session,started
            log.exception("model.route.start.failed session=%s task=%s selected=%s",session.session_id,request.task_id,request.model_route)
            with self._lock:
                self._sessions.pop(session.session_id,None)
            raise

    def get(self, session_id: str, *, serve: bool = True) -> EngineSession:
        # serve=False is the READ path (snapshot/status): restore the staged
        # publication in memory only - never claim the lease, never start a
        # heartbeat, never construct a graph, never publish. A read of a parked
        # session must cost zero pool connections and leave the lease alone;
        # the claimless restore is never cached, so a later serve claims first.
        with self._lock:
            session=self._sessions.get(session_id)
            if session is None or (serve and self._store is not None and getattr(session,'pg_claim',None) is None):
                session=self._load(session_id,serve=serve)
                if session is not None and serve:self._sessions[session_id]=session
            if session is None: raise KeyError(session_id)
            if serve and session.published is None and (self._store is not None or (self._state_path(session_id) is not None and self._state_path(session_id).exists())):
                self._publish(session)
            return session

    def publish(self, session_id: str, event_type: EventType, *, phase_id: int,
                status: ProcessStatus, next_step: NextStep,
                input_request: InputRequest | None=None) -> InteractiveEvent:
        session=self.get(session_id)
        with session.write_lock:
            with self._control_commit(session):
                session.sequence+=1;session.phase_id=phase_id
                session.status=status;session.next_step=next_step;session.open_input=input_request;self._save(session)
                if session.pg_stopped:
                    from .stop_fence import StopRequested
                    raise StopRequested('session stop requested; publish rejected')
                self._publish(session)
                return self._event(session,event_type,input_request=input_request)

    def respond(self, session_id: str, response: InputResponse) -> tuple[InteractiveEvent,EngineSnapshot]:
        session=self.get(session_id)
        if (session.stop_fence is not None and session.stop_fence.requested) or session.pg_stopped:
            raise ProtocolFailure(ProtocolErrorCode.INVALID_RESPONSE,'session stop requested; inputs are disabled')
        with session.write_lock:
            if (session.stop_fence is not None and session.stop_fence.requested) or session.pg_stopped:
                raise ProtocolFailure(ProtocolErrorCode.INVALID_RESPONSE,'session stop requested; inputs are disabled')
            if response.input_id in session.responses:
                if session.responses[response.input_id] == response:
                    return self._event(session,EventType.INPUT_ACCEPTED),session.published or self._publish(session)
                raise ProtocolFailure(ProtocolErrorCode.DUPLICATE_RESPONSE,"input already has a different response")
            if session.open_input is None:raise ProtocolFailure(ProtocolErrorCode.UNKNOWN_INPUT,"session has no open input")
            if session.open_input.input_id != response.input_id:raise ProtocolFailure(ProtocolErrorCode.STALE_INPUT,"input is no longer open")
            if response.expected_sequence is not None and response.expected_sequence != session.sequence:
                raise ProtocolFailure(ProtocolErrorCode.INVALID_SEQUENCE,"session sequence changed; refresh the open input")
            try:
                with self._control_commit(session):
                    prior=(session.sequence,session.status,session.next_step,session.open_input)
                    session.sequence+=1;session.status=ProcessStatus.RUNNING;session.next_step=NextStep.RUN
                    # Acceptance is atomic with Stop, never held over execution.
                    session.open_input=None
                    self._publish(session)
                    accepted=self._event(session,EventType.INPUT_ACCEPTED)
            except __import__('interview_engine.stop_fence',fromlist=['StopRequested']).StopRequested:
                raise ProtocolFailure(ProtocolErrorCode.INVALID_RESPONSE,'session stop requested; inputs are disabled')
            try:
                pending=self._advance(session,response) if session.program is not None else None
            except __import__('interview_engine.stop_fence',fromlist=['StopRequested']).StopRequested:
                session.open_input=None
                return accepted,session.snapshot()
            except Exception:
                if session.stop_fence is not None and session.stop_fence.requested:
                    session.open_input=None
                    return accepted,session.snapshot()
                # Provably pre-commit: the acceptance block mutates memory only,
                # and publish_step is reachable solely via _save_unfenced from
                # registry methods, never from inside _advance - so the durable
                # control still holds the PRIOR step. Re-publishing it tripped the
                # sequence CAS and masked the real error; instead verify the
                # durable state agrees, reconcile from it if the store says the
                # step somehow landed, and let the ORIGINAL error propagate.
                if self._store is not None and session.pg_claim is not None:
                    durable=self._store.load_published(session.session_id,session.pg_claim.owner_id)
                    if durable is not None and durable['control'].get('sequence',prior[0])!=prior[0]:
                        with self._lock:self._sessions.pop(session.session_id,None)
                        self.get(session.session_id)
                        raise
                    session.sequence,session.status,session.next_step,session.open_input=prior
                    self._publish(session)
                    raise
                try:
                    with self._control_commit(session):
                        session.sequence,session.status,session.next_step,session.open_input=prior
                        self._save(session)
                        self._publish(session)
                except __import__('interview_engine.stop_fence',fromlist=['StopRequested']).StopRequested:
                    session.open_input=None
                    return accepted,session.snapshot()
                raise
            try:
                with self._control_commit(session):
                    session.responses[response.input_id]=response;session.open_input=None
                    if session.program is not None:
                        session.sequence+=1;session.open_input=pending
                        if session.program.finished:
                            session.status=ProcessStatus.COMPLETED;session.next_step=NextStep.COMPLETE
                        else:
                            session.status=ProcessStatus.WAITING_INPUT;session.next_step=NextStep.WAIT_INPUT
                    try:
                        self._save(session)
                    except __import__('interview_engine.stop_fence',fromlist=['StopRequested']).StopRequested:
                        raise
                    except Exception:
                        # Ambiguous commit: the step transaction may have landed
                        # while its ack failed (or journal acknowledge raised after
                        # the commit). The durable store is authoritative: if it
                        # holds this step, reconcile memory from it and serve the
                        # committed outcome - never reset to prior memory, never
                        # let a retry double-advance.
                        if self._store is not None and session.pg_claim is not None:
                            durable=self._store.load_published(session.session_id,session.pg_claim.owner_id)
                            if durable is not None and durable['control'].get('sequence',0)>=session.sequence:
                                with self._lock:self._sessions.pop(session.session_id,None)
                                reconciled=self.get(session.session_id)
                                return accepted,reconciled.published or self._publish(reconciled)
                            # The store disproves the in-memory step: the tail
                            # already mutated responses/open_input/sequence, so
                            # this session object can no longer serve idempotency
                            # honestly. Evict; the next get reloads the durable
                            # truth (answer unrecorded, question open) and the
                            # client retry re-advances from the published pointer.
                            with self._lock:self._sessions.pop(session.session_id,None)
                        raise
                    if session.pg_stopped:
                        session.open_input=None
                        return accepted,session.snapshot()
                    return accepted,self._publish(session)
            except __import__('interview_engine.stop_fence',fromlist=['StopRequested']).StopRequested:
                session.open_input=None
                return accepted,session.snapshot()

    def switch_route(self, session_id:str, request:ModelRouteSwitchRequest)->tuple[InteractiveEvent,EngineSnapshot]:
        session=self.get(session_id)
        if session.stop_fence is not None and session.stop_fence.requested:
            raise ProtocolFailure(ProtocolErrorCode.ROUTE_SWITCH_CONFLICT,'stopped session route changes disabled')
        with session.write_lock:
            if session.stop_fence is not None and session.stop_fence.requested:
                raise ProtocolFailure(ProtocolErrorCode.ROUTE_SWITCH_CONFLICT,'stopped session route changes disabled')
            for item in session.route_history:
                if item.idempotency_key == request.idempotency_key:
                    if item.to_route != request.model_route or item.to_model_map["probing"][1] != request.model or item.actor_id != request.actor_id:
                        raise ProtocolFailure(ProtocolErrorCode.ROUTE_SWITCH_CONFLICT,"idempotency key reused with different request")
                    return self._event(session,EventType.ROUTE_CHANGED,route_change=item,sequence=item.session_sequence),session.published or self._publish(session)
            if session.status != ProcessStatus.WAITING_INPUT or session.open_input is None or session.next_step != NextStep.WAIT_INPUT:
                raise ProtocolFailure(ProtocolErrorCode.ROUTE_SWITCH_CONFLICT,"session is not at a stable input boundary")
            if session.sequence != request.expected_sequence or session.open_input.input_id != request.expected_input_id:
                raise ProtocolFailure(ProtocolErrorCode.ROUTE_SWITCH_CONFLICT,"input or session sequence changed; refresh and retry")
            program=session.program
            if program is None or not getattr(program,'model_map',None):
                raise ProtocolFailure(ProtocolErrorCode.ROUTE_SWITCH_CONFLICT,"session has no trustworthy pinned model")
            from . import session_models
            target=session_models.resolve(request.model_route,request.model)
            if {stage:tuple(pair) for stage,pair in target.items()} == {stage:tuple(pair) for stage,pair in program.model_map.items()}:
                raise ProtocolFailure(ProtocolErrorCode.ROUTE_SWITCH_CONFLICT,"session already uses this model")
            # The immutable source map may have come from a legacy session. A
            # candidate is built only after target preflight and checkpoint read.
            session_models.validate(target,request.model_route)
            if self._program_factory is default_program_factory:
                from .application import InterviewApplication
                candidate=self._prepare_program(session,lambda:InterviewApplication(session.session_id,session.task_id,session.request.idea,
                    artifact_root=str(self._state_dir) if self._state_dir is not None else str(worker_root(artifacts_root(os.environ.get("VMODEL_ARTIFACTS_DIR")))),model_route=request.model_route,pinned_models=target,
                    pg_store=self._store,pg_claim=session.pg_claim))
                # Read the same durable graph thread; never invoke a model during preflight.
                checkpoint=candidate.graph.get_state(candidate.cfg)
                if not checkpoint or not checkpoint.values:
                    raise ProtocolFailure(ProtocolErrorCode.ROUTE_SWITCH_CONFLICT,"no durable graph checkpoint for the open input")
                candidate.result=checkpoint.values
            elif hasattr(program,'candidate_for_route'):
                candidate=self._prepare_program(session,lambda:program.candidate_for_route(request.model_route,target))
            else:
                raise ProtocolFailure(ProtocolErrorCode.ROUTE_SWITCH_CONFLICT,"program does not support model switching")
            candidate.open_input=session.open_input
            candidate.state=None
            candidate.finished=False
            version=session.route_version+1
            if hasattr(candidate,'usage_log'):candidate.usage_log.route_version=version
            payload=RouteChangePayload(actor_id=request.actor_id,from_route=program.model_route,
                to_route=request.model_route,from_model_map=program.model_map,to_model_map=target,
                route_version=version,session_sequence=session.sequence+1,idempotency_key=request.idempotency_key)
            try:
                with self._control_commit(session):
                    previous=(session.program,session.route_version,session.sequence)
                    session.program=candidate;session.route_version=version;session.sequence+=1;session.route_history.append(payload)
                    try:self._save(session)
                    except Exception:
                        session.route_history.pop();session.program,session.route_version,session.sequence=previous
                        raise
                    event=self._event(session,EventType.ROUTE_CHANGED,route_change=payload)
                    self._publish(session)
                    log.info('model.route.changed session=%s task=%s from=%s to=%s from_map=%s to_map=%s route_version=%s',
                        session.session_id,session.task_id,payload.from_route,payload.to_route,json.dumps(payload.from_model_map),json.dumps(payload.to_model_map),version)
                    return event,session.published
            except __import__('interview_engine.stop_fence',fromlist=['StopRequested']).StopRequested:
                raise ProtocolFailure(ProtocolErrorCode.ROUTE_SWITCH_CONFLICT,'stopped session route changes disabled')

    def progress(self, session_id: str, after: int = 0, limit: int = 100):
        # Never acquire the registry's long-lived graph/control lock here.
        if self._store is not None:
            from vmodel_contracts.progress_signal import ProgressPage, ProgressSignal
            published=self._store.load_published(session_id,self._pg_owner())
            if published is None or published['control'] is None:
                # Unpublished and foreign rows are indistinguishable: no
                # existence oracle through the progress API.
                raise KeyError(session_id)
            rows=self._store.read_journal(session_id,self._pg_owner(),after,limit)
            events=[ProgressSignal.model_validate(r['page']) for r in rows]
            return ProgressPage(events=events, next_cursor=events[-1].cursor if events else after)
        from .progress_journal import ProgressJournal
        self._state_path(session_id)  # validate cloud UUID before any journal path
        root = self._state_dir
        if root is None or not (root / session_id / "interview" / "progress.sqlite").exists():
            raise KeyError(session_id)
        from vmodel_contracts.progress_signal import ProgressPage
        return ProgressJournal(root, session_id, "", create=False).read(after, limit)

    def _ensure_stop_fence(self,session):
        if session.stop_fence is None and self._state_dir is not None:
            from .stop_fence import StopFence
            path=self._state_path(session.session_id).with_name('stop-request.json')
            session.stop_fence=StopFence(path)

    @staticmethod
    @contextmanager
    def _control_commit(session):
        # Short marker lock, never held over model execution. A result that
        # returned before Stop still cannot commit after the marker lands.
        fenced=session.stop_fence is not None
        log.info("control_commit.enter session=%s fenced=%s",session.session_id,fenced)
        if not fenced:
            yield
            log.info("control_commit.exit session=%s fenced=False",session.session_id)
            return
        with session.stop_fence.marker_guard():
            session.stop_fence.check(session.stop_fence.generation)
            yield
            log.info("control_commit.exit session=%s fenced=True",session.session_id)

    def _dispatch_guard(self):
        # Database persistence never dispatches providers before PR B, no
        # matter which program factory was injected. The default production
        # factory is refused at construction outright; this guard catches any
        # other factory's program the moment it tries to preflight, build a
        # chat model, or invoke one. The internal test opt-in disables it.
        if self._store is not None and not self._allow_db_dispatch:
            from .model_config import db_provider_dispatch_blocked
            return db_provider_dispatch_blocked()
        return nullcontext()

    @staticmethod
    def _prepare_program(session,build):
        # Constructors/preflight can touch per-session files. Count them as
        # live writers so Delete never unlocks while they are still running.
        if session.stop_fence is None:return build()
        with session.stop_fence.running() as generation:
            program=build()
            session.stop_fence.check(generation)
            return program

    def _advance(self,session,response=None):
        log.info("advance.begin session=%s seq=%s",session.session_id,session.sequence)
        t_advance=time.monotonic()
        with self._dispatch_guard():
            pending=self._advance_guarded(session,response)
        log.info("advance.end session=%s seq=%s dur_ms=%d pending=%s",session.session_id,session.sequence,int((time.monotonic()-t_advance)*1000),'present' if pending is not None else 'none')
        return pending

    def _advance_guarded(self,session,response=None):
        if session.stop_fence is None:
            try:return session.program.advance() if response is None else session.program.advance(response)
            except Exception as exc:
                # A graph write that dies at the database fence means this
                # session lost its lease (stop or takeover): say so on the
                # session instead of leaving it looking live.
                from .stop_fence import StopRequested
                if self._store is not None and isinstance(exc,StopRequested):session.pg_stopped=True
                raise
        with session.stop_fence.running() as generation:
            result=session.program.advance() if response is None else session.program.advance(response)
            session.stop_fence.check(generation)
            return result

    def stop(self,session_id):
        if self._store is not None:
            owner=self._pg_owner()
            # One owner-authorized transaction: the quiesced control (when this
            # process serves the session) and the generation bump commit
            # together. No execution lease is needed, no worker advance is
            # awaited, and a failed worker-side publication cannot prevent the
            # stop. After the bump every fenced write from the old generation
            # fails, so a takeover never resumes a stopped session.
            with self._lock:session=self._sessions.get(session_id)
            control=None
            if session is not None:
                # Quiescing is a new step: clearing the open input changes the
                # control payload, so it carries a NEWER sequence. The CAS
                # publish rejects a same-sequence payload change as a
                # conflicting replay.
                session.sequence+=1
                session.open_input=None
                control=self._serialize(session)
            self._store.stop(session_id,owner,control=control)
            if session is not None:
                session.pg_stopped=True
                return session.snapshot()
            row=self._store.load_published(session_id,owner)
            if row is None or row.get('control') is None:
                return EngineSnapshot(session_id=session_id,task_id='',sequence=0,
                    process_status=ProcessStatus.STOP_REQUESTED,next_step=NextStep.STOP)
            stopped_session=self._restore_payload(dict(row['control']),session_id)
            stopped_session.pg_stopped=True;stopped_session.open_input=None
            return stopped_session.snapshot()
        session=self.get(session_id);self._ensure_stop_fence(session)
        if session.stop_fence is None:
            raise ProtocolFailure(ProtocolErrorCode.INVALID_RESPONSE,'durable session storage is required for Stop')
        with session.stop_fence.marker_guard():
            session.stop_fence.request_stop()
            # Stop may arrive during construction, before the first committed
            # HTTP identity exists. Persist only identity/control snapshot for
            # stopped recovery; loading it never constructs or resumes a graph.
            path=self._state_path(session.session_id)
            if path is not None and not path.exists():self._save_unfenced(session)
        # No long execution lock or graph access. Snapshot overlays the marker.
        return session.snapshot()

    def reconcile_stopped(self,session_id):
        # Explicit same-host recovery check, never a timeout-based guess. The
        # writer lock covers accepted provider threads through actual exit.
        if self._store is not None:
            session=self.get(session_id)
            if not session.pg_stopped:
                raise ProtocolFailure(ProtocolErrorCode.INVALID_RESPONSE,'durable stop request required')
            return session.snapshot()
        session=self.get(session_id);fence=session.stop_fence
        if fence is None or not fence.state:
            raise ProtocolFailure(ProtocolErrorCode.INVALID_RESPONSE,'durable stop request required')
        from .stop_fence import StopRequested
        try:fence.confirm_quiescent_after_recovery(no_live_worker=True)
        except StopRequested:return session.snapshot()
        return session.snapshot()

    def _state_path(self,session_id:str)->Path|None:
        from .tenant_worker import _owner
        if _owner.get() is not None:
            try:
                if str(UUID(session_id))!=session_id:raise ValueError()
            except (ValueError,TypeError,AttributeError):
                raise KeyError(session_id)
        return self._state_dir/session_id/'interview'/'http-session.json' if self._state_dir is not None else None

    def _save(self,session:EngineSession)->None:
        if session.stop_fence is not None:
            with session.stop_fence.marker_guard():
                if session.stop_fence.requested:return
                return self._save_unfenced(session)
        return self._save_unfenced(session)

    @staticmethod
    def _serialize(session:EngineSession)->dict:
        return {"session_id":session.session_id,"task_id":session.task_id,"phase_id":session.phase_id,"sequence":session.sequence,
            "status":int(session.status),"next_step":int(session.next_step),"open_input":session.open_input.model_dump(mode="json") if session.open_input else None,
            "responses":[response.model_dump(mode="json") for response in session.responses.values()],"request":session.request.model_dump(mode="json") if session.request else None,
            "model_map":getattr(session.program,"model_map",None),"route_version":session.route_version,
            "route_history":[item.model_dump(mode="json") for item in session.route_history]}

    def _save_unfenced(self,session:EngineSession)->None:
        if self._store is not None:
            # One publish transaction per step: control state, the advance's
            # buffered journal pages and the graph checkpoint pointer commit
            # together. A stop or takeover that already landed rejects the
            # write, and the save becomes the same no-op the file fence
            # produces after its marker.
            from .stop_fence import StopRequested
            program=session.program
            journal=getattr(program,'progress_journal',None)
            pages=journal.pending() if hasattr(journal,'pending') else None
            checkpoint_id=None
            graph=getattr(program,'graph',None)
            if graph is not None:
                # Publish the checkpoint this step produced, not the resume
                # anchor: a restored program's cfg is pinned to the published
                # pointer (authoritative restore), and get_state on a pinned
                # config returns that pin - so reclaimed sessions used to
                # re-publish their resume checkpoint forever (2026-10-23
                # cross-host continuation loop). Provenance, never
                # saver-latest: the advance records the checkpoint it actually
                # committed (durability='exit'), so an unpublished orphan from
                # a crashed prior epoch can never be selected. The fallback
                # reads the pinned cfg: on a reclaimed session that has not
                # run a turn yet it yields the restore anchor, which simply
                # republishes the current pointer; on a fresh session the cfg
                # is unpinned and yields the first turn's checkpoint.
                checkpoint_id=getattr(program,'produced_checkpoint_id',None)
                if checkpoint_id is None:
                    state=graph.get_state(program.cfg)
                    if state is not None:checkpoint_id=state.config['configurable'].get('checkpoint_id')
            t_publish=time.monotonic()
            try:
                self._store.publish_step(session.pg_claim,self._serialize(session),
                    journal_pages=pages,
                    published_thread_id=session.session_id,
                    published_checkpoint_id=checkpoint_id)
                if pages is not None:journal.acknowledge(len(pages))
                log.info("session.saved session=%s seq=%s pages=%s checkpoint=%s dur_ms=%d outcome=published",session.session_id,session.sequence,len(pages) if pages is not None else 0,'yes' if checkpoint_id else 'no',int((time.monotonic()-t_publish)*1000))
            except StopRequested:
                session.pg_stopped=True
                log.warning("session.save.fenced session=%s seq=%s - publish rejected (stop or takeover); marked stopped",session.session_id,session.sequence)
            return
        path=self._state_path(session.session_id)
        if path is None:return
        path.parent.mkdir(parents=True,exist_ok=True)
        payload=self._serialize(session)
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(payload));temporary.replace(path)

    def _load(self,session_id:str,*,serve:bool=True)->EngineSession|None:
        if self._store is not None:return self._load_pg(session_id,serve=serve)
        path=self._state_path(session_id)
        if path is None or not path.exists():return None
        payload=json.loads(path.read_text())
        session=self._restore_payload(payload,session_id)
        self._ensure_stop_fence(session)
        if session.stop_fence is not None and session.stop_fence.requested:
            session.open_input=None
            if not serve:session.published=session.snapshot()
            return session  # No restore, graph construction or implicit resume.
        if not serve:return self._read_restore(session,payload,check_legacy_pin=True)
        return self._restore_program(session,payload)

    def _read_restore(self,session:EngineSession,payload:dict,*,check_legacy_pin:bool)->EngineSession:
        # Read-only restore, rebuilt in memory from the published control
        # payload only: no claim, heartbeat, graph or write. The legacy
        # admission contract is preserved where the old read path enforced it:
        # a legacy session with no pinned model route is unreadable (the
        # snapshot route turns this into 422). File mode always enforced it
        # (reads constructed); pg mode enforced it under the default factory.
        if check_legacy_pin and not payload.get("model_map"):
            raise UserFacingError("legacy session has no pinned model route.","It predates per-session routing.","Start a new session.")
        # The served snapshot reads model_route/model_map off the constructed
        # program; a read constructs nothing, so carry the pinned map on a
        # stand-in. This restore is never cached or served (get(serve=False)).
        from types import SimpleNamespace
        session.program=SimpleNamespace(model_route=None,model_map=payload.get("model_map"))
        session.published=session.snapshot()
        return session

    def _load_pg(self,session_id:str,*,serve:bool=True)->EngineSession|None:
        # Claim-before-serve: no claim, no restore. The winner restores only
        # from the staged publication (control payload + pointer), never from
        # saver-latest checkpoint rows.
        owner=self._pg_owner()
        claim=None
        if serve:
            claim=self._store.claim(session_id,owner,self._instance,self._lease_ttl)
            if claim is None:return None
        row=self._store.load_published(session_id,owner)
        if row is None or row.get('control') is None:return None
        payload=dict(row['control'])
        session=self._restore_payload(payload,session_id)
        if payload.get('stopped'):
            session.pg_stopped=True
            session.open_input=None
            if not serve:
                session.published=session.snapshot()
                return session
            session.pg_claim=claim
            return session  # Stopped sessions never construct or resume a graph.
        if not serve:return self._read_restore(session,payload,check_legacy_pin=self._program_factory is default_program_factory)
        session.pg_claim=claim
        log.info("lease heartbeat started session=%s interval=%.1f ttl=%.1f",session.session_id,self._heartbeat_interval,self._lease_ttl)
        self._ensure_heartbeat_loop()
        return self._restore_program(session,payload,published_checkpoint_id=row.get('published_checkpoint_id'))

    def _default_pg_program(self,session,request,pinned=None):
        if not self._allow_db_dispatch:
            raise UserFacingError("database session cannot dispatch providers yet.","Staged responses land in PR B.","Run this session in file persistence mode.")
        from .application import InterviewApplication
        # Same artifact root as the file-backed factory: database persistence
        # changes where SESSION STATE lives, never where validated artifacts
        # land - the delivery app's terminal reconcile reads them from the
        # configured artifacts dir under the serving owner's tenant scope.
        root=worker_root(artifacts_root(os.environ.get("VMODEL_ARTIFACTS_DIR")))
        return InterviewApplication(session.session_id,request.task_id,request.idea,
            artifact_root=str(root),model_route=request.model_route,pinned_models=pinned,
            initial_model=request.initial_model,pg_store=self._store,pg_claim=session.pg_claim)

    def _restore_payload(self,payload:dict,session_id:str)->EngineSession:
        request=StartSessionRequest.model_validate(payload["request"])
        from .tenant_worker import _owner
        if _owner.get() is not None and (payload['session_id']!=session_id or str(request.session_id)!=session_id or payload['task_id']!=request.task_id):
            raise KeyError('persisted session binding mismatch')
        session=EngineSession(session_id=payload["session_id"],task_id=payload["task_id"],phase_id=payload["phase_id"],sequence=payload["sequence"],
            status=ProcessStatus(payload["status"]),next_step=NextStep(payload["next_step"]),
            open_input=InputRequest.model_validate(payload["open_input"]) if payload["open_input"] else None,request=request,
            route_version=payload.get("route_version",0),route_history=[RouteChangePayload.model_validate(item) for item in payload.get("route_history",[])])
        session.responses={response.input_id:response for response in (InputResponse.model_validate(item) for item in payload["responses"])}
        return session

    def _restore_program(self,session:EngineSession,payload:dict,published_checkpoint_id:str|None=None)->EngineSession:
        session_id=session.session_id;request=session.request
        if self._program_factory is default_program_factory:
            from . import session_models
            pinned=payload.get("model_map")
            if not pinned:
                raise UserFacingError("legacy session has no pinned model route.","It predates per-session routing.","Start a new session.")
            try:session_models.validate(pinned,preflight=False)
            except UserFacingError:
                log.warning("model.route.restore.reject session=%s route=%s stages=%s",session_id,request.model_route,json.dumps(pinned),exc_info=True)
                raise
        if self._program_factory is not None:
            current_route=session.route_history[-1].to_route if session.route_history else request.model_route
            if self._store is not None and self._program_factory is default_program_factory:
                with self._dispatch_guard():
                    session.program=self._default_pg_program(session,request.model_copy(update={"model_route":current_route}),
                        pinned=payload.get('model_map') or None)
                # Resume from the published pointer, never saver-latest.
                if published_checkpoint_id is not None:
                    session.program.cfg={'configurable':{'thread_id':session.session_id,'checkpoint_ns':'','checkpoint_id':published_checkpoint_id}}
            else:
                session.program=self._program_factory(session.session_id,request.model_copy(update={"model_route":current_route}))
            session.program.open_input=session.open_input
            if session.open_input is not None and hasattr(session.program,'graph'):
                checkpoint=session.program.graph.get_state(session.program.cfg)
                if not checkpoint or not checkpoint.values:
                    raise UserFacingError("saved session has no graph checkpoint.","The open input cannot be resumed safely.","Inspect the checkpoint and restore from backup.")
                session.program.result=checkpoint.values
                session.program.state=None
                context=session.open_input.context
                purpose=context.get('decision_purpose')
                if purpose=='semantic_review':
                    from vmodel_contracts.semantic_review import SemanticReview
                    current=checkpoint.values.get('semantic_review')
                    SemanticReview.model_validate(current)
                    if current!=context.get('evidence',{}).get('semantic_review'):
                        raise UserFacingError('saved semantic review changed.','Its checkpoint no longer matches the open review.','Inspect the saved review before resuming.')
                if purpose in ('conops_concept','interview_approval') and hasattr(session.program,'_decision_context'):
                    try:restored=session.program._decision_context(purpose)
                    except (ValueError,KeyError) as exc:
                        raise UserFacingError('saved decision evidence no longer matches its checkpoint.',
                            'The review cannot safely resume under its old input ID.','Inspect the archived review before recovering this session.') from exc
                    if restored.get('evidence_revision')!=context.get('evidence_revision'):
                        raise UserFacingError('saved decision evidence changed.',
                            'The checkpoint changed during an interrupted action.','Inspect the archived review before recovering this session.')

            if hasattr(session.program,"usage_log"):session.program.usage_log.route_version=session.route_version
            log.info("model.route.restored session=%s task=%s route=%s stages=%s",session_id,request.task_id,request.model_route,json.dumps(getattr(session.program,"model_map",None)))
        return session

    @staticmethod
    def _event(session: EngineSession,event_type: EventType,*,input_request: InputRequest|None=None,route_change:RouteChangePayload|None=None,sequence:int|None=None)->InteractiveEvent:
        return InteractiveEvent(event_id=uuid4(),session_id=session.session_id,task_id=session.task_id,
            phase_id=session.phase_id,sequence=sequence if sequence is not None else session.sequence,event_type=event_type,
            process_status=session.status,next_step=session.next_step,input_request=input_request,route_change=route_change)


class ProtocolFailure(Exception):
    def __init__(self,code:ProtocolErrorCode,detail:str):self.code,self.detail=code,detail


def create_app(registry:SessionRegistry|None=None,*,service_token:str|None=None)->FastAPI:
    sessions=registry or SessionRegistry()
    from contextlib import asynccontextmanager
    @asynccontextmanager
    async def _lifespan(_app):
        yield
        # Stop serving: heartbeats halt and held leases release at committed
        # boundaries, so the next host takes over without waiting out the TTL.
        close=getattr(sessions,'close',None)
        if close is not None:close()
    app=FastAPI(title="Requi.re interview engine",version="1",lifespan=_lifespan)
    expected_token=service_token if service_token is not None else os.environ.get("INTERVIEW_ENGINE_TOKEN","local-interview-engine-token")
    if not expected_token:raise RuntimeError("INTERVIEW_ENGINE_TOKEN must not be empty")

    def authenticate(x_interview_engine_token:str|None=Header(default=None)):
        if x_interview_engine_token is None or not hmac.compare_digest(x_interview_engine_token,expected_token):
            raise HTTPException(401,"invalid interview engine service token")


    if isinstance(sessions,SessionRegistry) and sessions._state_dir is not None and not os.environ.get("INTERVIEW_ENGINE_QUESTION_QUALITY_CONFIG"):
        from . import question_config
        question_config.set_store(question_config.ConfigStore(sessions._state_dir/"question_quality.json"))

    @app.get("/v1/question-quality-config",dependencies=[Depends(authenticate)])
    def get_question_quality_config():
        from . import question_config
        return {"config_version":question_config.CONFIG_VERSION,"config":question_config.current(),
                "defaults":question_config.defaults()}

    @app.put("/v1/question-quality-config",dependencies=[Depends(authenticate)])
    def put_question_quality_config(body:dict):
        from . import question_config
        try:saved=question_config.store().put(body.get("config",body) if isinstance(body,dict) else body)
        except question_config.ConfigError as exc:raise HTTPException(422,str(exc))
        return {"config_version":question_config.CONFIG_VERSION,"config":saved}

    @app.get("/v1/model-routes/default",dependencies=[Depends(authenticate)])
    def default_model_route():
        from . import session_models
        try:
            route,model=session_models.initial_default()
            return {"route":route,"model":model}
        except UserFacingError as exc:raise HTTPException(422,ProtocolError(code=ProtocolErrorCode.INVALID_RESPONSE,detail=exc.render()).model_dump(mode="json"))

    @app.get("/v1/model-routes/ollama/models",dependencies=[Depends(authenticate)])
    def local_model_options():
        from . import model_config
        import httpx
        endpoint=model_config._BASE_URLS["ollama"].rstrip("/")+"/models"
        try:
            response=httpx.get(endpoint,timeout=2);response.raise_for_status()
            models=sorted({entry["id"] for entry in response.json().get("data",[]) if isinstance(entry,dict) and isinstance(entry.get("id"),str) and entry["id"]})
        except (httpx.HTTPError,ValueError,KeyError) as exc:
            raise HTTPException(503,ProtocolError(code=ProtocolErrorCode.INTERNAL_ERROR,detail=f"Ollama models could not be listed: {exc}",retryable=True).model_dump(mode="json"))
        return {"models":models}

    @app.get("/v1/model-routes/{route}",dependencies=[Depends(authenticate)])
    def preview_model_route(route:str,initial_model:str|None=None):
        """Read-only HTTP route preview from the same resolver used at Start."""
        from . import session_models
        try:
            models=session_models.resolve(route,initial_model)
            session_models.validate(models,route)
        except UserFacingError as exc:
            raise HTTPException(422,ProtocolError(code=ProtocolErrorCode.INVALID_RESPONSE,detail=exc.render(),retryable=False).model_dump(mode="json"))
        return {"route":route,"model_map":models}

    @app.post("/v1/sessions",response_model=StartSessionResponse,status_code=201,dependencies=[Depends(authenticate)])
    def start(request:StartSessionRequest):
        try:session,event=sessions.start(request)
        except ProtocolFailure as failure:
            raise HTTPException(409,ProtocolError(code=failure.code,detail=failure.detail).model_dump(mode="json"))
        except UserFacingError as exc:
            log.warning("interview session start needs action task=%s error=%s",request.task_id,exc.render().replace("\n"," | "))
            raise HTTPException(422,ProtocolError(code=ProtocolErrorCode.INVALID_RESPONSE,detail=exc.render(),retryable=False).model_dump(mode="json"))
        except Exception as exc:
            log.exception("interview session start failed task=%s",request.task_id)
            # Operator-only failure diagnostics travel on a response header so
            # the protocol error contract is untouched; the block is sanitized
            # by construction and bounded, and the customer-facing detail keeps
            # its minimal-disclosure shape. Delivery records it on the task's
            # failure event for operator Debug surfaces only.
            headers={}
            try:
                from . import pg_diagnostics
                import base64 as _b64, json as _json
                raw=_json.dumps(pg_diagnostics.build_operator_block(exc),separators=(",",":")).encode()
                if len(raw)<=1800:
                    headers["X-Engine-Operator-Diagnostics"]=_b64.urlsafe_b64encode(raw).decode("ascii")
            except Exception:
                headers={}
            raise HTTPException(503,ProtocolError(code=ProtocolErrorCode.INTERNAL_ERROR,detail=f"engine program failed during start: {type(exc).__name__}",retryable=True).model_dump(mode="json"),headers=headers)
        return StartSessionResponse(snapshot=session.snapshot(),accepted_event=event)

    @app.get("/v1/sessions/{session_id}/progress",dependencies=[Depends(authenticate)])
    def progress(session_id:str,after:int=0,limit:int=100):
        if after<0 or limit<1 or limit>500:raise HTTPException(422,"invalid progress cursor or page size")
        try:return sessions.progress(session_id,after,limit)
        except KeyError:
            log.info("session.progress.no_journal session=%s",session_id)
            raise HTTPException(404,"session has no progress journal")

    @app.get("/v1/sessions/{session_id}",response_model=EngineSnapshot,dependencies=[Depends(authenticate)])
    def snapshot(session_id:str):
        try:return sessions.snapshot(session_id)
        except KeyError:raise HTTPException(404,ProtocolError(code=ProtocolErrorCode.UNKNOWN_INPUT,detail="session not found").model_dump(mode="json"))
        except UserFacingError as exc:
            if exc.what == "legacy session has no pinned model route.":
                log.info("model.route.snapshot.legacy_unpinned session=%s: preserving historical session without pinned routing",session_id)
            else:
                log.warning("model.route.snapshot.reject session=%s reason=%s",session_id,exc.render().replace("\n"," | "))
            raise HTTPException(422,ProtocolError(code=ProtocolErrorCode.INVALID_RESPONSE,detail=exc.render(),retryable=False).model_dump(mode="json"))

    @app.post("/v1/sessions/{session_id}/stop",response_model=EngineSnapshot,dependencies=[Depends(authenticate)])
    def stop_session(session_id:UUID):
        # Trusted service-token boundary, owner-scoped by tenant middleware.
        # No graph lock, cancellation, deletion or implicit resume.
        log.info("session.stop.request session=%s",session_id)
        try:out=sessions.stop(str(session_id))
        except KeyError:
            log.info("session.stop.refused session=%s reason=not-found",session_id)
            raise HTTPException(404,ProtocolError(code=ProtocolErrorCode.UNKNOWN_INPUT,detail="session not found").model_dump(mode="json"))
        except ProtocolFailure as failure:
            log.info("session.stop.refused session=%s code=%s",session_id,failure.code)
            raise HTTPException(409,ProtocolError(code=failure.code,detail=failure.detail).model_dump(mode="json"))
        log.info("session.stop.recorded session=%s status=%s",session_id,getattr(out.process_status,'name',out.process_status))
        return out

    @app.post("/v1/sessions/{session_id}/stop/reconcile",response_model=EngineSnapshot,dependencies=[Depends(authenticate)])
    def reconcile_stop(session_id:UUID):
        log.info("session.stop.reconcile.request session=%s",session_id)
        try:out=sessions.reconcile_stopped(str(session_id))
        except KeyError:
            log.info("session.stop.reconcile.refused session=%s reason=not-found",session_id)
            raise HTTPException(404,ProtocolError(code=ProtocolErrorCode.UNKNOWN_INPUT,detail="session not found").model_dump(mode="json"))
        except ProtocolFailure as failure:
            log.info("session.stop.reconcile.refused session=%s code=%s",session_id,failure.code)
            raise HTTPException(409,ProtocolError(code=failure.code,detail=failure.detail).model_dump(mode="json"))
        log.info("session.stop.reconcile.recorded session=%s status=%s",session_id,getattr(out.process_status,'name',out.process_status))
        return out

    @app.post("/v1/sessions/{session_id}/model-route",response_model=ModelRouteSwitchResponse,dependencies=[Depends(authenticate)])
    def switch_model_route(session_id:str,request:ModelRouteSwitchRequest):
        try:event,state=sessions.switch_route(session_id,request)
        except KeyError:raise HTTPException(404,ProtocolError(code=ProtocolErrorCode.UNKNOWN_INPUT,detail="session not found").model_dump(mode="json"))
        except __import__('interview_engine.stop_fence',fromlist=['StopRequested']).StopRequested:
            raise HTTPException(409,ProtocolError(code=ProtocolErrorCode.ROUTE_SWITCH_CONFLICT,detail="session stop requested; route change discarded").model_dump(mode="json"))
        except ProtocolFailure as failure:
            log.warning('model.route.switch.reject session=%s reason=%s',session_id,failure.detail)
            raise HTTPException(409,ProtocolError(code=failure.code,detail=failure.detail).model_dump(mode="json"))
        except UserFacingError as exc:
            log.warning('model.route.switch.reject session=%s reason=%s',session_id,exc.render())
            raise HTTPException(422,ProtocolError(code=ProtocolErrorCode.INVALID_RESPONSE,detail=exc.render()).model_dump(mode="json"))
        return ModelRouteSwitchResponse(accepted_event=event,snapshot=state)

    @app.post("/v1/sessions/{session_id}/inputs/{input_id}",response_model=SubmitInputResponse,dependencies=[Depends(authenticate)])
    def submit(session_id:str,input_id:UUID,response:InputResponse):
        if input_id != response.input_id:
            raise HTTPException(422,ProtocolError(code=ProtocolErrorCode.INVALID_RESPONSE,detail="path and body input IDs differ").model_dump(mode="json"))
        try:event,state=sessions.respond(session_id,response)
        except KeyError:raise HTTPException(404,ProtocolError(code=ProtocolErrorCode.UNKNOWN_INPUT,detail="session not found").model_dump(mode="json"))
        except ProtocolFailure as failure:
            status=409 if failure.code in (ProtocolErrorCode.STALE_INPUT,ProtocolErrorCode.DUPLICATE_RESPONSE,ProtocolErrorCode.INVALID_SEQUENCE) else 422
            raise HTTPException(status,ProtocolError(code=failure.code,detail=failure.detail).model_dump(mode="json"))
        except UserFacingError as exc:
            log.warning("interview input needs action session=%s input=%s error=%s",session_id,input_id,exc.render().replace("\n"," | "),exc_info=True)
            raise HTTPException(422,ProtocolError(code=ProtocolErrorCode.INVALID_RESPONSE,detail=exc.render(),retryable=False).model_dump(mode="json"))
        except Exception as exc:
            log.exception("interview input advance failed session=%s input=%s",session_id,input_id)
            raise HTTPException(503,ProtocolError(code=ProtocolErrorCode.INTERNAL_ERROR,detail=f"engine program failed while advancing input: {type(exc).__name__}",retryable=True).model_dump(mode="json"))
        return SubmitInputResponse(accepted_event=event,snapshot=state)

    app.state.sessions=sessions
    return app


def default_program_factory(session_id,request):
    from .application import InterviewApplication
    from .tenant_worker import worker_root,_owner
    root=worker_root(artifacts_root(os.environ.get("VMODEL_ARTIFACTS_DIR")))
    path=root/session_id/'interview'/'http-session.json'
    pinned=None
    if path.exists():
        payload=json.loads(path.read_text());pinned=payload.get('model_map')
        if not pinned:raise UserFacingError('legacy session has no pinned model route.','It predates per-session routing.','Keep its existing artifacts; start a new session rather than resuming under a different model.')
    program=InterviewApplication(session_id,request.task_id,request.idea,artifact_root=str(root),model_route=request.model_route,pinned_models=pinned,initial_model=request.initial_model)
    log.info("model.route.effective session=%s task=%s selected=%s stages=%s",session_id,request.task_id,request.model_route,json.dumps(program.model_map))
    return program

SESSIONS_DSN_ENV='INTERVIEW_ENGINE_SESSIONS_DSN'
OWNER_ID_ENV='INTERVIEW_ENGINE_OWNER_ID'
SESSION_ID_ENV='INTERVIEW_ENGINE_SESSION_ID'
FILE_SESSIONS_ENV='INTERVIEW_ENGINE_SESSIONS_FILE'
FILE_SESSIONS_MSG=('database session persistence is the default: set '+SESSIONS_DSN_ENV+
    ' (plus '+OWNER_ID_ENV+' and '+SESSION_ID_ENV+' outside tenant mode), or explicitly choose file persistence with '+
    FILE_SESSIONS_ENV+'=1')

from vmodel_contracts.runtime import artifacts_root

def _build_service_app():
    """Service entry wiring, built when the ASGI server actually starts. Database
    session persistence is the default; file persistence requires the explicit
    opt-in; a missing choice fails the boot, never silently falls back."""
    sessions_dsn=os.environ.get(SESSIONS_DSN_ENV)
    if os.environ.get('INTERVIEW_ENGINE_TENANTS')=='1':
        if not os.environ.get('INTERVIEW_ENGINE_TOKEN') or os.environ.get('INTERVIEW_ENGINE_TOKEN')=='local-interview-engine-token':
            raise RuntimeError('Tenant engine requires a configured private service token')
        from .tenant_worker import TenantRegistryRouter,TenantWorkerMiddleware
        if not sessions_dsn and os.environ.get(FILE_SESSIONS_ENV)!='1':
            raise RuntimeError(FILE_SESSIONS_MSG)
        service=create_app(TenantRegistryRouter(SessionRegistry,default_program_factory,artifacts_root(),session_dsn=sessions_dsn,allow_file_sessions=os.environ.get(FILE_SESSIONS_ENV)=='1'))
        service.add_middleware(TenantWorkerMiddleware,base=artifacts_root(),require_session=bool(sessions_dsn))
        return service
    if sessions_dsn:
        # Database session persistence is the default once the DSN is set.
        # Staged responses (PR B) landed, so the documented
        # constructor block on provider dispatch is lifted for exactly this wiring.
        from .tenant_worker import pg_session_store
        from .pg_pool import build_pool
        owner_id=os.environ.get(OWNER_ID_ENV)
        if not owner_id:
            raise RuntimeError(OWNER_ID_ENV+' is required with '+SESSIONS_DSN_ENV+' outside tenant mode: database session persistence needs an owner scope')
        session_id=os.environ.get(SESSION_ID_ENV)
        if not session_id:
            raise RuntimeError(SESSION_ID_ENV+' is required with '+SESSIONS_DSN_ENV+' outside tenant mode: database session persistence runs under the real auth session - synthetic session authority is never substituted')
        # Standalone deployment: this process builds and owns its pool.
        pool=build_pool(sessions_dsn)
        return create_app(SessionRegistry(default_program_factory,session_store=pg_session_store(pool,owner_id,session_id,close_pool=True),owner_id=owner_id,allow_db_provider_dispatch=True))
    # File persistence only when explicitly chosen; otherwise fail clear at boot.
    if os.environ.get(FILE_SESSIONS_ENV)!='1':
        raise RuntimeError(FILE_SESSIONS_MSG)
    return create_app(SessionRegistry(default_program_factory,state_dir=artifacts_root()))


class _LazyServiceApp:
    """Builds the real app on the server's first ASGI (lifespan) call: importing
    this module for its classes has no startup side effects, while a misconfigured
    service still fails at boot, before it serves anything."""
    def __init__(self):
        self._service=None
    async def __call__(self,scope,receive,send):
        if self._service is None:
            self._service=_build_service_app()
        await self._service(scope,receive,send)


app=_LazyServiceApp()
