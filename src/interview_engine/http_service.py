"""HTTP boundary for the interview engine.

This module owns transport and session/input lifecycle only. Interview domain
work calls ``SessionRegistry.publish``; it never reads HTTP or presentation
text. Control decisions use protocol enums and identifiers exclusively.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import contextmanager
import json
import threading
import time
from pathlib import Path
import hmac
import logging
import os
from threading import RLock
from uuid import UUID, uuid4

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
    pg_heartbeat_halt: object | None = field(default=None, repr=False, compare=False)
    pg_heartbeat_thread: object | None = field(default=None, repr=False, compare=False)

    def snapshot(self) -> EngineSnapshot:
        stopped=self.stop_fence.state if self.stop_fence is not None else ('STOP_REQUESTED' if self.pg_stopped else None)
        return EngineSnapshot(session_id=self.session_id,task_id=self.task_id,sequence=self.sequence,
            process_status=ProcessStatus[stopped] if stopped else self.status,next_step=NextStep.STOP if stopped else self.next_step,open_input=None if stopped else self.open_input,
            model_route=getattr(self.program,"model_route",None) or (self.request.model_route if self.request else None),
            model_map=getattr(self.program,"model_map",None),route_version=self.route_version,route_history=self.route_history)


class SessionRegistry:
    """Thread-safe application port behind the HTTP adapter."""
    def __init__(self,program_factory=None,state_dir:Path|str|None=None,*,session_store=None,owner_id=None,instance_id:str|None=None,lease_ttl:float=30.0):
        if state_dir is not None and session_store is not None:
            raise ValueError('choose file or database session persistence, not both')
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
        self._lock=RLock()

    def _start_heartbeat(self,session:EngineSession)->None:
        # While this process serves a session, keep its lease alive so a long
        # advance cannot be taken over mid-flight. A lost heartbeat (stop or
        # takeover) marks the session stopped: every later fenced write then
        # fails closed at the database. Own connection per thread (psycopg
        # connections are not thread-safe).
        if self._store is None or session.pg_claim is None:return
        store=self._store.spawn()
        claim=session.pg_claim
        halt=threading.Event()
        session.pg_heartbeat_halt=halt
        interval=max(self._lease_ttl/3.0,0.05)
        def beat():
            # The lease expiry in every fence statement is the authority: this
            # loop only extends a live lease. Retries never revive an expired
            # one (the heartbeat's WHERE clause requires it unexpired), and a
            # blocked loop lets it lapse - writes then fail closed regardless
            # of what this thread believed.
            failures=0
            try:
                while not session.pg_stopped and not halt.is_set():
                    if halt.wait(interval):return   # wakes immediately on close()
                    if session.pg_stopped:return
                    try:alive=store.heartbeat(claim,self._lease_ttl)
                    except Exception:
                        failures+=1
                        log.warning('session lease heartbeat failed session=%s failures=%d',claim.session_id,failures,exc_info=True)
                        if failures>=5:return   # lease expires on its own; fences close
                        continue
                    failures=0
                    if not alive:
                        session.pg_stopped=True
                        return
            finally:
                store.conn.close()
        session.pg_heartbeat_thread=threading.Thread(target=beat,daemon=True,name=f"lease-heartbeat-{claim.session_id}")
        session.pg_heartbeat_thread.start()

    def close(self)->None:
        # Stop serving: halt every heartbeat, join each thread with a bounded
        # wait, and leave no connection in use. Leases expire on their own; a
        # live session can then be claimed by another instance. Fences close
        # every later write from this process even before expiry.
        with self._lock:sessions=list(self._sessions.values())
        deadline=time.monotonic()+max(2*self._lease_ttl,1.0)
        for session in sessions:
            if session.pg_heartbeat_halt is not None:session.pg_heartbeat_halt.set()
        for session in sessions:
            thread=session.pg_heartbeat_thread
            if thread is not None:
                remaining=deadline-time.monotonic()
                thread.join(timeout=max(remaining,0.05))
                if thread.is_alive():log.warning('lease heartbeat did not join session=%s',session.session_id)
        for session in sessions:
            # Close program-held database connections: no socket outlives the
            # registry that owns it.
            saver=getattr(session.program,'saver',None)
            if saver is not None:
                try:
                    if hasattr(saver,'close'):saver.close()
                    conn=getattr(saver,'conn',None)
                    if conn is not None:conn.close()
                except Exception:log.warning('program saver close failed session=%s',session.session_id,exc_info=True)

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
        session=self.get(session_id)
        if session.published is None:raise KeyError(session_id)
        if session.stop_fence is not None and session.stop_fence.state:return session.snapshot()
        if session.pg_stopped:return session.snapshot()
        return session.published.model_copy(deep=True)

    def start(self, request: StartSessionRequest) -> tuple[EngineSession,InteractiveEvent]:
        session=EngineSession(session_id=str(request.session_id or uuid4()),task_id=request.task_id,sequence=1,
            status=ProcessStatus.RUNNING,next_step=NextStep.RUN,request=request)
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
            try:self._store.create(session.session_id,session.task_id,owner)
            except Exception as exc:
                with self._lock:self._sessions.pop(session.session_id,None)
                if 'UniqueViolation' in type(exc).__name__ or 'duplicate key' in str(exc):
                    raise ProtocolFailure(ProtocolErrorCode.INVALID_RESPONSE,"session ID already exists")
                raise
            claim=self._store.claim(session.session_id,owner,self._instance,self._lease_ttl)
            if claim is None:
                with self._lock:self._sessions.pop(session.session_id,None)
                raise ProtocolFailure(ProtocolErrorCode.INVALID_RESPONSE,"session lease held by another live instance")
            session.pg_claim=claim
            self._start_heartbeat(session)
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
                if self._program_factory is not None:
                    if self._store is not None and self._program_factory is default_program_factory:
                        session.program=self._prepare_program(session,lambda:self._default_pg_program(session,request))
                    else:
                        session.program=self._prepare_program(session,lambda:self._program_factory(session.session_id,request))
                    log.info("model.route.resolved session=%s task=%s route=%s stages=%s",session.session_id,request.task_id,request.model_route,json.dumps(getattr(session.program,"model_map",None)))
                    self._save(session)  # factory/pinned model is now ready, before long advance
                    pending=self._advance(session)
                    with self._control_commit(session):
                        if pending is not None:
                            session.sequence+=1;session.open_input=pending;session.status=ProcessStatus.WAITING_INPUT;session.next_step=NextStep.WAIT_INPUT
                        self._save(session)
                        self._publish(session)
                else:
                    with self._control_commit(session):
                        self._save(session)
                        self._publish(session)
            return session,started
        except __import__('interview_engine.stop_fence',fromlist=['StopRequested']).StopRequested:
            session.open_input=None
            return session,started
        except Exception:
            if session.stop_fence is not None and session.stop_fence.requested:
                session.open_input=None
                return session,started
            log.exception("model.route.start.failed session=%s task=%s selected=%s",session.session_id,request.task_id,request.model_route)
            with self._lock:
                self._sessions.pop(session.session_id,None)
            raise

    def get(self, session_id: str) -> EngineSession:
        with self._lock:
            session=self._sessions.get(session_id)
            if session is None:
                session=self._load(session_id)
                if session is not None:self._sessions[session_id]=session
            if session is None: raise KeyError(session_id)
            if session.published is None and (self._store is not None or (self._state_path(session_id) is not None and self._state_path(session_id).exists())):
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
                    self._save(session)
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
                    artifact_root=str(self._state_dir) if self._state_dir is not None else None,model_route=request.model_route,pinned_models=target,
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
        if session.stop_fence is None:
            yield
            return
        with session.stop_fence.marker_guard():
            session.stop_fence.check(session.stop_fence.generation)
            yield

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
                state=graph.get_state(program.cfg)
                if state is not None:checkpoint_id=state.config['configurable'].get('checkpoint_id')
            try:
                self._store.publish_step(session.pg_claim,self._serialize(session),
                    journal_pages=pages,
                    published_thread_id=session.session_id,
                    published_checkpoint_id=checkpoint_id)
                if pages is not None:journal.acknowledge(len(pages))
            except StopRequested:session.pg_stopped=True
            return
        path=self._state_path(session.session_id)
        if path is None:return
        path.parent.mkdir(parents=True,exist_ok=True)
        payload=self._serialize(session)
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(payload));temporary.replace(path)

    def _load(self,session_id:str)->EngineSession|None:
        if self._store is not None:return self._load_pg(session_id)
        path=self._state_path(session_id)
        if path is None or not path.exists():return None
        payload=json.loads(path.read_text())
        session=self._restore_payload(payload,session_id)
        self._ensure_stop_fence(session)
        if session.stop_fence is not None and session.stop_fence.requested:
            session.open_input=None
            return session  # No restore, graph construction or implicit resume.
        return self._restore_program(session,payload)

    def _load_pg(self,session_id:str)->EngineSession|None:
        # Claim-before-serve: no claim, no restore. The winner restores only
        # from the staged publication (control payload + pointer), never from
        # saver-latest checkpoint rows.
        owner=self._pg_owner()
        claim=self._store.claim(session_id,owner,self._instance,self._lease_ttl)
        if claim is None:return None
        row=self._store.load_published(session_id,owner)
        if row is None or row.get('control') is None:return None
        payload=dict(row['control'])
        session=self._restore_payload(payload,session_id)
        session.pg_claim=claim
        if payload.get('stopped'):
            session.pg_stopped=True
            session.open_input=None
            return session  # Stopped sessions never construct or resume a graph.
        self._start_heartbeat(session)
        return self._restore_program(session,payload,published_checkpoint_id=row.get('published_checkpoint_id'))

    def _default_pg_program(self,session,request,pinned=None):
        from .application import InterviewApplication
        return InterviewApplication(session.session_id,request.task_id,request.idea,
            artifact_root=None,model_route=request.model_route,pinned_models=pinned,
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
    sessions=registry or SessionRegistry();app=FastAPI(title="Requi.re interview engine",version="1")
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
            raise HTTPException(503,ProtocolError(code=ProtocolErrorCode.INTERNAL_ERROR,detail=f"engine program failed during start: {type(exc).__name__}",retryable=True).model_dump(mode="json"))
        return StartSessionResponse(snapshot=session.snapshot(),accepted_event=event)

    @app.get("/v1/sessions/{session_id}/progress",dependencies=[Depends(authenticate)])
    def progress(session_id:str,after:int=0,limit:int=100):
        if after<0 or limit<1 or limit>500:raise HTTPException(422,"invalid progress cursor or page size")
        try:return sessions.progress(session_id,after,limit)
        except KeyError:raise HTTPException(404,"session has no progress journal")

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
        try:return sessions.stop(str(session_id))
        except KeyError:raise HTTPException(404,ProtocolError(code=ProtocolErrorCode.UNKNOWN_INPUT,detail="session not found").model_dump(mode="json"))
        except ProtocolFailure as failure:
            raise HTTPException(409,ProtocolError(code=failure.code,detail=failure.detail).model_dump(mode="json"))

    @app.post("/v1/sessions/{session_id}/stop/reconcile",response_model=EngineSnapshot,dependencies=[Depends(authenticate)])
    def reconcile_stop(session_id:UUID):
        try:return sessions.reconcile_stopped(str(session_id))
        except KeyError:raise HTTPException(404,ProtocolError(code=ProtocolErrorCode.UNKNOWN_INPUT,detail="session not found").model_dump(mode="json"))
        except ProtocolFailure as failure:
            raise HTTPException(409,ProtocolError(code=failure.code,detail=failure.detail).model_dump(mode="json"))

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

from vmodel_contracts.runtime import artifacts_root
if os.environ.get('INTERVIEW_ENGINE_TENANTS')=='1':
    if not os.environ.get('INTERVIEW_ENGINE_TOKEN') or os.environ.get('INTERVIEW_ENGINE_TOKEN')=='local-interview-engine-token':
        raise RuntimeError('Tenant engine requires a configured private service token')
    from .tenant_worker import TenantRegistryRouter,TenantWorkerMiddleware
    app=create_app(TenantRegistryRouter(SessionRegistry,default_program_factory,artifacts_root()))
    app.add_middleware(TenantWorkerMiddleware,base=artifacts_root())
else:
    app=create_app(SessionRegistry(default_program_factory,state_dir=artifacts_root()))
