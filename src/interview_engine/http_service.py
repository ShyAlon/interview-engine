"""HTTP boundary for the interview engine.

This module owns transport and session/input lifecycle only. Interview domain
work calls ``SessionRegistry.publish``; it never reads HTTP or presentation
text. Control decisions use protocol enums and identifiers exclusively.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException

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

    def snapshot(self) -> EngineSnapshot:
        return EngineSnapshot(session_id=self.session_id,task_id=self.task_id,sequence=self.sequence,
            process_status=self.status,next_step=self.next_step,open_input=self.open_input)


class SessionRegistry:
    """Thread-safe application port behind the HTTP adapter."""
    def __init__(self,program_factory=None):
        self._sessions: dict[str,EngineSession]={}
        self._program_factory=program_factory
        self._lock=RLock()

    def start(self, request: StartSessionRequest) -> tuple[EngineSession,InteractiveEvent]:
        with self._lock:
            session=EngineSession(session_id=str(uuid4()),task_id=request.task_id,sequence=1,
                status=ProcessStatus.RUNNING,next_step=NextStep.RUN)
            self._sessions[session.session_id]=session
            started=self._event(session,EventType.RUN_STARTED)
            if self._program_factory is not None:
                session.program=self._program_factory(session.session_id,request)
                pending=session.program.advance()
                if pending is not None:
                    session.sequence+=1;session.open_input=pending;session.status=ProcessStatus.WAITING_INPUT;session.next_step=NextStep.WAIT_INPUT
            return session,started

    def get(self, session_id: str) -> EngineSession:
        with self._lock:
            session=self._sessions.get(session_id)
            if session is None: raise KeyError(session_id)
            return session

    def publish(self, session_id: str, event_type: EventType, *, phase_id: int,
                status: ProcessStatus, next_step: NextStep,
                input_request: InputRequest | None=None) -> InteractiveEvent:
        with self._lock:
            session=self.get(session_id);session.sequence+=1;session.phase_id=phase_id
            session.status=status;session.next_step=next_step;session.open_input=input_request
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
            session.responses[response.input_id]=response;session.open_input=None;session.sequence+=1
            session.status=ProcessStatus.RUNNING;session.next_step=NextStep.RUN
            accepted=self._event(session,EventType.INPUT_ACCEPTED)
            if session.program is not None:
                pending=session.program.advance(response)
                session.sequence+=1;session.open_input=pending
                if session.program.finished:
                    session.status=ProcessStatus.COMPLETED;session.next_step=NextStep.COMPLETE
                else:
                    session.status=ProcessStatus.WAITING_INPUT;session.next_step=NextStep.WAIT_INPUT
            return accepted,session.snapshot()

    @staticmethod
    def _event(session: EngineSession,event_type: EventType,*,input_request: InputRequest|None=None)->InteractiveEvent:
        return InteractiveEvent(event_id=uuid4(),session_id=session.session_id,task_id=session.task_id,
            phase_id=session.phase_id,sequence=session.sequence,event_type=event_type,
            process_status=session.status,next_step=session.next_step,input_request=input_request)


class ProtocolFailure(Exception):
    def __init__(self,code:ProtocolErrorCode,detail:str):self.code,self.detail=code,detail


def create_app(registry:SessionRegistry|None=None)->FastAPI:
    sessions=registry or SessionRegistry();app=FastAPI(title="Requi.re interview engine",version="1")

    @app.post("/v1/sessions",response_model=StartSessionResponse,status_code=201)
    def start(request:StartSessionRequest):
        session,event=sessions.start(request);return StartSessionResponse(snapshot=session.snapshot(),accepted_event=event)

    @app.get("/v1/sessions/{session_id}",response_model=EngineSnapshot)
    def snapshot(session_id:str):
        try:return sessions.get(session_id).snapshot()
        except KeyError:raise HTTPException(404,ProtocolError(code=ProtocolErrorCode.UNKNOWN_INPUT,detail="session not found").model_dump(mode="json"))

    @app.post("/v1/sessions/{session_id}/inputs/{input_id}",response_model=SubmitInputResponse)
    def submit(session_id:str,input_id:UUID,response:InputResponse):
        if input_id != response.input_id:
            raise HTTPException(422,ProtocolError(code=ProtocolErrorCode.INVALID_RESPONSE,detail="path and body input IDs differ").model_dump(mode="json"))
        try:event,state=sessions.respond(session_id,response)
        except KeyError:raise HTTPException(404,ProtocolError(code=ProtocolErrorCode.UNKNOWN_INPUT,detail="session not found").model_dump(mode="json"))
        except ProtocolFailure as failure:
            status=409 if failure.code in (ProtocolErrorCode.STALE_INPUT,ProtocolErrorCode.DUPLICATE_RESPONSE) else 422
            raise HTTPException(status,ProtocolError(code=failure.code,detail=failure.detail).model_dump(mode="json"))
        return SubmitInputResponse(accepted_event=event,snapshot=state)

    app.state.sessions=sessions
    return app


def default_program_factory(session_id,request):
    from .application import InterviewApplication
    return InterviewApplication(session_id,request.task_id,request.idea)

app=create_app(SessionRegistry(default_program_factory))
