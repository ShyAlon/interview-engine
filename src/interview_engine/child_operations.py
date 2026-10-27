"""One typed registry operation; refs loaded by trusted owner-bound wiring.

No HTTP server, executable payload or environment authority. A child gives up
its lease and heartbeat before returning a durable published snapshot.
"""
from uuid import uuid4
from requi_child_contract import InterviewStart,InterviewAnswer,InterviewAdvance,InterviewRoute,require_binding


def execute_operation(operation,*,trusted_binding,store,load_request,load_answer,program_factory=None,broker=None):
    from contextlib import nullcontext
    from .http_service import SessionRegistry,default_program_factory
    from .tenant_worker import _owner,_session
    from .wire_binding import broker_clients
    from .child_step import restored_child_step
    require_binding(operation,trusted_binding)
    b=operation.binding;body=operation.operation
    if store.claims!=(str(b.owner_id),str(b.auth_session_id)):
        raise PermissionError('original Auth-session transaction claims required')
    production=program_factory is None or program_factory is default_program_factory
    if production and broker is None:raise PermissionError('production child requires parent broker')
    factory=default_program_factory if production else program_factory
    with broker_clients(broker) if broker is not None else nullcontext():
        if isinstance(body,InterviewStart):
            request=load_request(body.request_ref)
            from vmodel_contracts.interactive import StartSessionRequest
            if (not isinstance(request,StartSessionRequest) or str(request.session_id)!=b.session_id
                or request.task_id!=b.task_id):raise PermissionError('durable start reference binding differs')
            registry=SessionRegistry(factory,session_store=store,owner_id=str(b.owner_id),
                instance_id='child#'+str(uuid4()),allow_db_provider_dispatch=production)
            claim=None;owner=_owner.set(str(b.owner_id));auth=_session.set(str(b.auth_session_id))
            try:
                claim=store.claim_start_expected(b,registry._instance,registry._lease_ttl)
                # Registry start still owns all publication/stop/graph behavior.
                # Its store adapter cannot create or take a second lease.
                class StartStore:
                    def __getattr__(self,name):return getattr(store,name)
                    def create(self,sid,task,owner):
                        if (sid,task,owner)!=(b.session_id,b.task_id,str(b.owner_id)):
                            raise PermissionError('bootstrap identity changed')
                    def claim(self,sid,owner,instance,ttl):
                        if (sid,owner,instance)!=(b.session_id,str(b.owner_id),registry._instance):
                            raise PermissionError('bootstrap claim changed')
                        return claim
                registry._store=StartStore()
                session,_=registry.start(request)
                return session.snapshot()
            finally:
                try:
                    registry.close()
                    if claim is not None and not store.release(claim):raise PermissionError('child lease release lost')
                finally:_session.reset(auth);_owner.reset(owner)
        if not isinstance(body,(InterviewAnswer,InterviewAdvance,InterviewRoute)):raise ValueError('interview operation required')
        with restored_child_step(operation,trusted_binding=b,store=store,program_factory=factory,broker=broker) as step:
            if isinstance(body,InterviewRoute):
                from vmodel_contracts.interactive import ModelRouteSwitchRequest
                request=load_request(body.request_ref)
                if (not isinstance(request,ModelRouteSwitchRequest) or request.expected_sequence!=body.expected_sequence
                    or str(request.expected_input_id)!=body.wait_id or request.model_route!=b.provider):
                    raise PermissionError('durable route request binding differs')
                _,snapshot=step.registry.switch_route(b.session_id,request)
                return snapshot
            if isinstance(body,InterviewAnswer):
                from vmodel_contracts.interactive import InputResponse
                response=load_answer(body.answer_ref)
                if (not isinstance(response,InputResponse) or str(response.input_id)!=body.wait_id
                    or response.expected_sequence!=body.expected_sequence):
                    raise PermissionError('durable answer reference binding differs')
                _,snapshot=step.registry.respond(b.session_id,response)
                return snapshot
            from vmodel_contracts.interactive import ProcessStatus,NextStep
            session=step.session
            if session.status!=ProcessStatus.RUNNING or session.open_input is not None:
                raise PermissionError('advance requires published running control with no founder input')
            pending=step.registry._advance(session)
            with step.registry._control_commit(session):
                session.sequence+=1;session.open_input=pending
                session.status=ProcessStatus.WAITING_INPUT if pending is not None else ProcessStatus.COMPLETED
                session.next_step=NextStep.WAIT_INPUT if pending is not None else NextStep.COMPLETE
                step.registry._save(session)
                return step.registry._publish(session)
