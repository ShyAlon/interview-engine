"""Warm claimed-session reuse across sequential child operations.

A warm child claims the published session once (compare-and-claim, epoch+1),
restores the graph, then serves later operations against the LIVE in-memory
session instead of releasing and re-claiming per turn. While the lease is
held, every durable publication is fenced by that lease (publish_step), so the
in-memory control state IS the authoritative published state: follow-up
validation compares the operation's expected evidence with the live session,
not a fresh row read. Sequence equality binds the parent's post-turn row read
to this child's own last publish, because the held lease makes this child the
only possible writer between the two.

close() releases the lease exactly once; the child exits afterwards.
"""
from contextlib import ExitStack
from uuid import uuid4
from requi_child_contract import InterviewAdvance,InterviewAnswer,InterviewRoute,require_binding


class WarmChildStep:
    """A claimed, restored, heartbeating session held across operations."""

    def __init__(self,operation,*,trusted_binding,store,program_factory,broker=None):
        from .http_service import SessionRegistry,default_program_factory
        from .tenant_worker import _owner,_session
        from .wire_binding import broker_clients
        from contextlib import nullcontext
        require_binding(operation,trusted_binding)
        if (program_factory is None or program_factory is default_program_factory) and broker is None:
            raise PermissionError('child money boundary not installed; production factory refused')
        binding=operation.binding
        if store.claims!=(str(binding.owner_id),str(binding.auth_session_id)):
            raise PermissionError('original Auth-session transaction claims required')
        if not isinstance(operation.operation,(InterviewAdvance,InterviewAnswer,InterviewRoute)):
            raise ValueError('only published interview restore is warm-servable')
        self._stack=ExitStack()
        self._store=store
        registry=None;claim=None;tokens_set=False
        try:
            self._stack.enter_context(broker_clients(broker) if broker is not None else nullcontext())
            registry=SessionRegistry(program_factory,session_store=store,owner_id=str(binding.owner_id),
                instance_id='child#'+str(uuid4()),allow_db_provider_dispatch=broker is not None,
                progress_mode="observational")
            self._owner_token=_owner.set(str(binding.owner_id))
            self._session_token=_session.set(str(binding.auth_session_id))
            tokens_set=True
            claim,row=store.claim_published_expected(binding,operation.operation,registry._instance,registry._lease_ttl)
            session=registry._restore_payload(dict(row['control']),binding.session_id)
            session.pg_claim=claim
            # Register BEFORE constructing: failure still releases the won lease.
            registry._sessions[session.session_id]=session
            registry._ensure_heartbeat_loop()
            session=registry._restore_program(session,dict(row['control']),published_checkpoint_id=row.get('published_checkpoint_id'))
            registry._publish(session)
            self.registry=registry;self.session=session;self._claim=claim
            self._binding=trusted_binding
        except BaseException:
            try:
                released=registry.close() if registry is not None else {}
                if claim is not None:
                    # Registered sessions are released by registry.close(). Only
                    # a failure before registration needs a direct release.
                    ok=(released.get(claim.session_id) if claim.session_id in registry._sessions
                        else store.release(claim))
                    if not ok:
                        raise PermissionError('warm child lease release lost at setup') from None
            finally:
                if tokens_set:
                    _session.reset(self._session_token);_owner.reset(self._owner_token)
                self._stack.close()
            raise

    def check(self,operation,*,trusted_binding):
        """Validate a follow-up operation against the live published state.

        The held lease fences every writer, so live session state is the
        published row: matching expected_sequence/wait_id/review_revision
        proves the parent built this operation from this child's own last
        publish. A mismatch means the parent's view diverged - refuse.
        """
        require_binding(operation,trusted_binding)
        body=operation.operation;binding=operation.binding
        if not isinstance(body,(InterviewAdvance,InterviewAnswer,InterviewRoute)):
            raise ValueError('only published interview operations are warm-servable')
        session=self.session
        if session.pg_claim is None or session.pg_claim.epoch!=self._claim.epoch:
            raise PermissionError('warm child lease lost')
        if session.pg_stopped or (session.stop_fence is not None and session.stop_fence.requested):
            raise PermissionError('warm child session stopped')
        if binding.generation!=self._claim.generation:
            raise PermissionError('warm child generation changed')
        if session.sequence!=body.expected_sequence:
            raise PermissionError('warm child published sequence changed')
        if isinstance(body,(InterviewAnswer,InterviewRoute)):
            pending=session.open_input
            if pending is None or str(pending.input_id)!=body.wait_id:
                raise PermissionError('warm child published wait changed')
        if isinstance(body,InterviewAnswer):
            context=(session.open_input.context or {}) if session.open_input is not None else {}
            revision=context.get('evidence_revision') or 'engine:'+binding.session_id+':'+str(session.sequence)
            if revision!=body.review_revision:
                raise PermissionError('warm child published review changed')
        return self

    def close(self):
        """Release the lease exactly once, then unwind tokens and clients."""
        claim=self._claim
        if claim is None:return
        try:
            released=self.registry.close()
            ok=released[claim.session_id] if claim.session_id in released else self._store.release(claim)
            if not ok:
                raise PermissionError('warm child lease no longer held at release; reconcile before success')
        finally:
            self._claim=None
            from .tenant_worker import _owner,_session
            _session.reset(self._session_token);_owner.reset(self._owner_token)
            self._stack.close()
