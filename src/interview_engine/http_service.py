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

    def snapshot(self) -> EngineSnapshot:
        return EngineSnapshot(session_id=self.session_id,task_id=self.task_id,sequence=self.sequence,
            process_status=self.status,next_step=self.next_step,open_input=self.open_input,
            model_route=getattr(self.program,"model_route",None) or (self.request.model_route if self.request else None),
            model_map=getattr(self.program,"model_map",None),route_version=self.route_version,route_history=self.route_history)


class SessionRegistry:
    """Thread-safe application port behind the HTTP adapter."""
    def __init__(self,program_factory=None,state_dir:Path|str|None=None):
        self._sessions: dict[str,EngineSession]={}
        self._program_factory=program_factory
        self._state_dir=Path(state_dir) if state_dir is not None else None
        self._lock=RLock()

    def start(self, request: StartSessionRequest) -> tuple[EngineSession,InteractiveEvent]:
        with self._lock:
            session=EngineSession(session_id=str(request.session_id or uuid4()),task_id=request.task_id,sequence=1,
                status=ProcessStatus.RUNNING,next_step=NextStep.RUN,request=request)
            if session.session_id in self._sessions or (self._state_path(session.session_id) is not None and self._state_path(session.session_id).exists()):
                raise ProtocolFailure(ProtocolErrorCode.INVALID_RESPONSE,"session ID already exists")
            # Reject unavailable routes before recording a session or paying for a call.
            if self._program_factory is default_program_factory:
                from . import session_models
                try:session_models.validate(session_models.resolve(request.model_route,request.initial_model))
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
            if response.expected_sequence is not None and response.expected_sequence != session.sequence:
                raise ProtocolFailure(ProtocolErrorCode.INVALID_SEQUENCE,"session sequence changed; refresh the open input")
            prior=(session.sequence,session.status,session.next_step,session.open_input)
            session.sequence+=1;session.status=ProcessStatus.RUNNING;session.next_step=NextStep.RUN
            accepted=self._event(session,EventType.INPUT_ACCEPTED)
            try:
                pending=session.program.advance(response) if session.program is not None else None
            except Exception:
                session.sequence,session.status,session.next_step,session.open_input=prior
                # The last committed input remains actionable after a failed
                # model call. Preserve the typed boundary and checkpoint pin.
                self._save(session)
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

    def switch_route(self, session_id:str, request:ModelRouteSwitchRequest)->tuple[InteractiveEvent,EngineSnapshot]:
        with self._lock:
            session=self.get(session_id)
            for item in session.route_history:
                if item.idempotency_key == request.idempotency_key:
                    if item.to_route != request.model_route or item.to_model_map["probing"][1] != request.model or item.actor_id != request.actor_id:
                        raise ProtocolFailure(ProtocolErrorCode.ROUTE_SWITCH_CONFLICT,"idempotency key reused with different request")
                    return self._event(session,EventType.ROUTE_CHANGED,route_change=item,sequence=item.session_sequence),session.snapshot()
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
                candidate=InterviewApplication(session.session_id,session.task_id,session.request.idea,
                    model_route=request.model_route,pinned_models=target)
                # Read the same durable graph thread; never invoke a model during preflight.
                checkpoint=candidate.graph.get_state(candidate.cfg)
                if not checkpoint or not checkpoint.values:
                    raise ProtocolFailure(ProtocolErrorCode.ROUTE_SWITCH_CONFLICT,"no durable graph checkpoint for the open input")
                candidate.result=checkpoint.values
            elif hasattr(program,'candidate_for_route'):
                candidate=program.candidate_for_route(request.model_route,target)
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
            previous=(session.program,session.route_version,session.sequence)
            session.program=candidate;session.route_version=version;session.sequence+=1;session.route_history.append(payload)
            try:self._save(session)
            except Exception:
                session.route_history.pop();session.program,session.route_version,session.sequence=previous
                raise
            event=self._event(session,EventType.ROUTE_CHANGED,route_change=payload)
            log.info('model.route.changed session=%s task=%s from=%s to=%s from_map=%s to_map=%s route_version=%s',
                session.session_id,session.task_id,payload.from_route,payload.to_route,json.dumps(payload.from_model_map),json.dumps(payload.to_model_map),version)
            return event,session.snapshot()

    def progress(self, session_id: str, after: int = 0, limit: int = 100):
        # Never acquire the registry's long-lived graph/control lock here.
        from .progress_journal import ProgressJournal
        root = self._state_dir
        if root is None or not (root / session_id / "interview" / "progress.sqlite").exists():
            raise KeyError(session_id)
        from vmodel_contracts.progress_signal import ProgressPage
        return ProgressJournal(root, session_id, "", create=False).read(after, limit)

    def _state_path(self,session_id:str)->Path|None:
        return self._state_dir/session_id/'interview'/'http-session.json' if self._state_dir is not None else None

    def _save(self,session:EngineSession)->None:
        path=self._state_path(session.session_id)
        if path is None:return
        path.parent.mkdir(parents=True,exist_ok=True)
        payload={"session_id":session.session_id,"task_id":session.task_id,"phase_id":session.phase_id,"sequence":session.sequence,
            "status":int(session.status),"next_step":int(session.next_step),"open_input":session.open_input.model_dump(mode="json") if session.open_input else None,
            "responses":[response.model_dump(mode="json") for response in session.responses.values()],"request":session.request.model_dump(mode="json") if session.request else None,
            "model_map":getattr(session.program,"model_map",None),"route_version":session.route_version,
            "route_history":[item.model_dump(mode="json") for item in session.route_history]}
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(payload));temporary.replace(path)

    def _load(self,session_id:str)->EngineSession|None:
        path=self._state_path(session_id)
        if path is None or not path.exists():return None
        payload=json.loads(path.read_text());request=StartSessionRequest.model_validate(payload["request"])
        session=EngineSession(session_id=payload["session_id"],task_id=payload["task_id"],phase_id=payload["phase_id"],sequence=payload["sequence"],
            status=ProcessStatus(payload["status"]),next_step=NextStep(payload["next_step"]),
            open_input=InputRequest.model_validate(payload["open_input"]) if payload["open_input"] else None,request=request,
            route_version=payload.get("route_version",0),route_history=[RouteChangePayload.model_validate(item) for item in payload.get("route_history",[])])
        session.responses={response.input_id:response for response in (InputResponse.model_validate(item) for item in payload["responses"])}
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
            session.program=self._program_factory(session.session_id,request.model_copy(update={"model_route":current_route}))
            session.program.open_input=session.open_input
            if session.open_input is not None and hasattr(session.program,'graph'):
                checkpoint=session.program.graph.get_state(session.program.cfg)
                if not checkpoint or not checkpoint.values:
                    raise UserFacingError("saved session has no graph checkpoint.","The open input cannot be resumed safely.","Inspect the checkpoint and restore from backup.")
                session.program.result=checkpoint.values
                session.program.state=None
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
        try:return sessions.get(session_id).snapshot()
        except KeyError:raise HTTPException(404,ProtocolError(code=ProtocolErrorCode.UNKNOWN_INPUT,detail="session not found").model_dump(mode="json"))
        except UserFacingError as exc:
            log.warning("model.route.snapshot.reject session=%s reason=%s",session_id,exc.render(),exc_info=True)
            raise HTTPException(422,ProtocolError(code=ProtocolErrorCode.INVALID_RESPONSE,detail=exc.render(),retryable=False).model_dump(mode="json"))

    @app.post("/v1/sessions/{session_id}/model-route",response_model=ModelRouteSwitchResponse,dependencies=[Depends(authenticate)])
    def switch_model_route(session_id:str,request:ModelRouteSwitchRequest):
        try:event,state=sessions.switch_route(session_id,request)
        except KeyError:raise HTTPException(404,ProtocolError(code=ProtocolErrorCode.UNKNOWN_INPUT,detail="session not found").model_dump(mode="json"))
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
    path=artifacts_root()/session_id/'interview'/'http-session.json'
    pinned=None
    if path.exists():
        payload=json.loads(path.read_text());pinned=payload.get('model_map')
        if not pinned:raise UserFacingError('legacy session has no pinned model route.','It predates per-session routing.','Keep its existing artifacts; start a new session rather than resuming under a different model.')
    program=InterviewApplication(session_id,request.task_id,request.idea,model_route=request.model_route,pinned_models=pinned,initial_model=request.initial_model)
    log.info("model.route.effective session=%s task=%s selected=%s stages=%s",session_id,request.task_id,request.model_route,json.dumps(program.model_map))
    return program

from vmodel_contracts.runtime import artifacts_root
app=create_app(SessionRegistry(default_program_factory,state_dir=artifacts_root()))
