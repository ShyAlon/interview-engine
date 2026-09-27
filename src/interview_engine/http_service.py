"""HTTP boundary for the interview engine.

This module owns transport and session/input lifecycle only. Interview domain
work calls ``SessionRegistry.publish``; it never reads HTTP or presentation
text. Control decisions use protocol enums and identifiers exclusively.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import hmac
import logging
import os
from threading import RLock
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, Header, HTTPException
from .errors import UserFacingError

log=logging.getLogger("interview_engine.http")

from vmodel_contracts.interactive import (
    EngineSnapshot, EventType, InputRequest, InputResponse, InteractiveEvent,
    NextStep, ProcessStatus, ProtocolError, ProtocolErrorCode,
    StartSessionRequest, StartSessionResponse, SubmitInputResponse,
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

    def snapshot(self) -> EngineSnapshot:
        return EngineSnapshot(session_id=self.session_id,task_id=self.task_id,sequence=self.sequence,
            process_status=self.status,next_step=self.next_step,open_input=self.open_input)


class SessionRegistry:
    """Thread-safe application port behind the HTTP adapter."""
    def __init__(self,program_factory=None,state_dir:Path|str|None=None):
        self._sessions: dict[str,EngineSession]={}
        self._program_factory=program_factory
        self._state_dir=Path(state_dir) if state_dir is not None else None
        self._lock=RLock()

    def start(self, request: StartSessionRequest) -> tuple[EngineSession,InteractiveEvent]:
        with self._lock:
            session=EngineSession(session_id=str(uuid4()),task_id=request.task_id,sequence=1,
                status=ProcessStatus.RUNNING,next_step=NextStep.RUN,request=request)
            # Reject unavailable routes before recording a session or paying for a call.
            if self._program_factory is default_program_factory:
                from . import session_models
                try:session_models.validate(session_models.resolve(request.model_route))
                except UserFacingError:
                    log.warning("model.route.preflight.reject task=%s route=%s",request.task_id,request.model_route,exc_info=True)
                    raise
                log.info("model.route.preflight.accept task=%s route=%s",request.task_id,request.model_route)
            started=self._event(session,EventType.RUN_STARTED)
            try:
                if self._program_factory is not None:
                    session.program=self._program_factory(session.session_id,request)
                    log.info("model.route.resolved session=%s task=%s route=%s stages=%s",session.session_id,request.task_id,request.model_route,json.dumps(getattr(session.program,"model_map",None)))
                    pending=session.program.advance()
                    if pending is not None:
                        session.sequence+=1;session.open_input=pending;session.status=ProcessStatus.WAITING_INPUT;session.next_step=NextStep.WAIT_INPUT
            except Exception:
                log.exception("model.route.start.failed session=%s task=%s selected=%s",session.session_id,request.task_id,request.model_route)
                raise
            self._save(session)
            self._sessions[session.session_id]=session
            return session,started

    def get(self, session_id: str) -> EngineSession:
        with self._lock:
            session=self._sessions.get(session_id)
            if session is None:
                session=self._load(session_id)
                if session is not None:self._sessions[session_id]=session
            if session is None: raise KeyError(session_id)
            return session

    def publish(self, session_id: str, event_type: EventType, *, phase_id: int,
                status: ProcessStatus, next_step: NextStep,
                input_request: InputRequest | None=None) -> InteractiveEvent:
        with self._lock:
            session=self.get(session_id);session.sequence+=1;session.phase_id=phase_id
            session.status=status;session.next_step=next_step;session.open_input=input_request;self._save(session)
            return self._event(session,event_type,input_request=input_request)

    def respond(self, session_id: str, response: InputResponse) -> tuple[InteractiveEvent,EngineSnapshot]:
        with self._lock:
            session=self.get(session_id)
            if response.input_id in session.responses:
                if session.responses[response.input_id] == response:
                    return self._event(session,EventType.INPUT_ACCEPTED),session.snapshot()
                raise ProtocolFailure(ProtocolErrorCode.DUPLICATE_RESPONSE,"input already has a different response")
            if session.open_input is None:raise ProtocolFailure(ProtocolErrorCode.UNKNOWN_INPUT,"session has no open input")
            if session.open_input.input_id != response.input_id:raise ProtocolFailure(ProtocolErrorCode.STALE_INPUT,"input is no longer open")
            prior=(session.sequence,session.status,session.next_step,session.open_input)
            session.sequence+=1;session.status=ProcessStatus.RUNNING;session.next_step=NextStep.RUN
            accepted=self._event(session,EventType.INPUT_ACCEPTED)
            try:
                pending=session.program.advance(response) if session.program is not None else None
            except Exception:
                session.sequence,session.status,session.next_step,session.open_input=prior
                raise
            session.responses[response.input_id]=response;session.open_input=None
            if session.program is not None:
                session.sequence+=1;session.open_input=pending
                if session.program.finished:
                    session.status=ProcessStatus.COMPLETED;session.next_step=NextStep.COMPLETE
                else:
                    session.status=ProcessStatus.WAITING_INPUT;session.next_step=NextStep.WAIT_INPUT
            self._save(session)
            return accepted,session.snapshot()

    def _state_path(self,session_id:str)->Path|None:
        return self._state_dir/session_id/'interview'/'http-session.json' if self._state_dir is not None else None

    def _save(self,session:EngineSession)->None:
        path=self._state_path(session.session_id)
        if path is None:return
        path.parent.mkdir(parents=True,exist_ok=True)
        payload={"session_id":session.session_id,"task_id":session.task_id,"phase_id":session.phase_id,"sequence":session.sequence,
            "status":int(session.status),"next_step":int(session.next_step),"open_input":session.open_input.model_dump(mode="json") if session.open_input else None,
            "responses":[response.model_dump(mode="json") for response in session.responses.values()],"request":session.request.model_dump(mode="json") if session.request else None,
            "model_map":getattr(session.program,"model_map",None)}
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(payload));temporary.replace(path)

    def _load(self,session_id:str)->EngineSession|None:
        path=self._state_path(session_id)
        if path is None or not path.exists():return None
        payload=json.loads(path.read_text());request=StartSessionRequest.model_validate(payload["request"])
        session=EngineSession(session_id=payload["session_id"],task_id=payload["task_id"],phase_id=payload["phase_id"],sequence=payload["sequence"],
            status=ProcessStatus(payload["status"]),next_step=NextStep(payload["next_step"]),
            open_input=InputRequest.model_validate(payload["open_input"]) if payload["open_input"] else None,request=request)
        session.responses={response.input_id:response for response in (InputResponse.model_validate(item) for item in payload["responses"])}
        if self._program_factory is default_program_factory:
            from . import session_models
            pinned=payload.get("model_map")
            if not pinned:
                raise UserFacingError("legacy session has no pinned model route.","It predates per-session routing.","Start a new session.")
            try:session_models.validate(pinned)
            except UserFacingError:
                log.warning("model.route.restore.reject session=%s route=%s stages=%s",session_id,request.model_route,json.dumps(pinned),exc_info=True)
                raise
        if self._program_factory is not None:
            session.program=self._program_factory(session.session_id,request)
            session.program.open_input=session.open_input
            log.info("model.route.restored session=%s task=%s route=%s stages=%s",session_id,request.task_id,request.model_route,json.dumps(getattr(session.program,"model_map",None)))
        return session

    @staticmethod
    def _event(session: EngineSession,event_type: EventType,*,input_request: InputRequest|None=None)->InteractiveEvent:
        return InteractiveEvent(event_id=uuid4(),session_id=session.session_id,task_id=session.task_id,
            phase_id=session.phase_id,sequence=session.sequence,event_type=event_type,
            process_status=session.status,next_step=session.next_step,input_request=input_request)


class ProtocolFailure(Exception):
    def __init__(self,code:ProtocolErrorCode,detail:str):self.code,self.detail=code,detail


def create_app(registry:SessionRegistry|None=None,*,service_token:str|None=None)->FastAPI:
    sessions=registry or SessionRegistry();app=FastAPI(title="Requi.re interview engine",version="1")
    expected_token=service_token if service_token is not None else os.environ.get("INTERVIEW_ENGINE_TOKEN","local-interview-engine-token")
    if not expected_token:raise RuntimeError("INTERVIEW_ENGINE_TOKEN must not be empty")

    def authenticate(x_interview_engine_token:str|None=Header(default=None)):
        if x_interview_engine_token is None or not hmac.compare_digest(x_interview_engine_token,expected_token):
            raise HTTPException(401,"invalid interview engine service token")


    @app.post("/v1/sessions",response_model=StartSessionResponse,status_code=201,dependencies=[Depends(authenticate)])
    def start(request:StartSessionRequest):
        try:session,event=sessions.start(request)
        except UserFacingError as exc:
            log.warning("interview session start needs action task=%s error=%s",request.task_id,exc.render().replace("\n"," | "))
            raise HTTPException(422,ProtocolError(code=ProtocolErrorCode.INVALID_RESPONSE,detail=exc.render(),retryable=False).model_dump(mode="json"))
        except Exception as exc:
            log.exception("interview session start failed task=%s",request.task_id)
            raise HTTPException(503,ProtocolError(code=ProtocolErrorCode.INTERNAL_ERROR,detail=f"engine program failed during start: {type(exc).__name__}",retryable=True).model_dump(mode="json"))
        return StartSessionResponse(snapshot=session.snapshot(),accepted_event=event)

    @app.get("/v1/sessions/{session_id}",response_model=EngineSnapshot,dependencies=[Depends(authenticate)])
    def snapshot(session_id:str):
        try:return sessions.get(session_id).snapshot()
        except KeyError:raise HTTPException(404,ProtocolError(code=ProtocolErrorCode.UNKNOWN_INPUT,detail="session not found").model_dump(mode="json"))
        except UserFacingError as exc:
            log.warning("model.route.snapshot.reject session=%s reason=%s",session_id,exc.render(),exc_info=True)
            raise HTTPException(422,ProtocolError(code=ProtocolErrorCode.INVALID_RESPONSE,detail=exc.render(),retryable=False).model_dump(mode="json"))

    @app.post("/v1/sessions/{session_id}/inputs/{input_id}",response_model=SubmitInputResponse,dependencies=[Depends(authenticate)])
    def submit(session_id:str,input_id:UUID,response:InputResponse):
        if input_id != response.input_id:
            raise HTTPException(422,ProtocolError(code=ProtocolErrorCode.INVALID_RESPONSE,detail="path and body input IDs differ").model_dump(mode="json"))
        try:event,state=sessions.respond(session_id,response)
        except KeyError:raise HTTPException(404,ProtocolError(code=ProtocolErrorCode.UNKNOWN_INPUT,detail="session not found").model_dump(mode="json"))
        except ProtocolFailure as failure:
            status=409 if failure.code in (ProtocolErrorCode.STALE_INPUT,ProtocolErrorCode.DUPLICATE_RESPONSE) else 422
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
    path=artifacts_root()/session_id/'interview'/'http-session.json'
    pinned=None
    if path.exists():
        payload=json.loads(path.read_text());pinned=payload.get('model_map')
        if not pinned:raise UserFacingError('legacy session has no pinned model route.','It predates per-session routing.','Keep its existing artifacts; start a new session rather than resuming under a different model.')
    program=InterviewApplication(session_id,request.task_id,request.idea,model_route=request.model_route,pinned_models=pinned)
    log.info("model.route.effective session=%s task=%s selected=%s stages=%s",session_id,request.task_id,request.model_route,json.dumps(program.model_map))
    return program

from vmodel_contracts.runtime import artifacts_root
app=create_app(SessionRegistry(default_program_factory,state_dir=artifacts_root()))
