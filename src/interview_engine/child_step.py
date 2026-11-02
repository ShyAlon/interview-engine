"""Inert typed restore seam, constructor-only and fake-provider gated.

No HTTP/runtime activation. The child claims an unleased published session
with compare-and-claim; it never reuses a parent's live graph or lease. Money
broker integration and durable operation dispatch remain later phases.
"""
from contextlib import contextmanager
import hashlib
import json
from uuid import uuid4
from requi_child_contract import InterviewAdvance,InterviewAnswer,InterviewRoute,require_binding


def published_digest(row):
    """Hash the exact published control + pointer, never saver-latest."""
    data={key:row.get(key) for key in ('control','published_thread_id','published_checkpoint_ns','published_checkpoint_id')}
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


@contextmanager
def restored_child_step(operation,*,trusted_binding,store,program_factory,broker=None):
    from .http_service import SessionRegistry,default_program_factory
    from .tenant_worker import _owner,_session
    require_binding(operation,trusted_binding)
    if (program_factory is None or program_factory is default_program_factory) and broker is None:
        raise PermissionError('child money boundary not installed; production factory refused')
    binding=operation.binding
    if store.claims!=(str(binding.owner_id),str(binding.auth_session_id)):
        raise PermissionError('original Auth-session transaction claims required')
    if not isinstance(operation.operation,(InterviewAdvance,InterviewAnswer,InterviewRoute)):
        raise ValueError('only published interview restore is implemented in phase 2')
    registry=SessionRegistry(program_factory,session_store=store,owner_id=str(binding.owner_id),
        instance_id='child#'+str(uuid4()),allow_db_provider_dispatch=broker is not None,progress_mode="observational")
    claim=None;owner_token=_owner.set(str(binding.owner_id));session_token=_session.set(str(binding.auth_session_id))
    try:
        claim,row=store.claim_published_expected(binding,operation.operation,registry._instance,registry._lease_ttl)
        session=registry._restore_payload(dict(row['control']),binding.session_id)
        session.pg_claim=claim
        # Register BEFORE constructing: failure still releases the won lease.
        registry._sessions[session.session_id]=session
        registry._ensure_heartbeat_loop()
        session=registry._restore_program(session,dict(row['control']),published_checkpoint_id=row.get('published_checkpoint_id'))
        registry._publish(session)
        yield ChildStep(registry,session)
    finally:
        try:
            registry.close()
            if claim is not None and not store.release(claim):
                raise PermissionError('child lease no longer held at release; reconcile before success')
        finally:_session.reset(session_token);_owner.reset(owner_token)


class ChildStep:
    def __init__(self,registry,session):self.registry=registry;self.session=session
